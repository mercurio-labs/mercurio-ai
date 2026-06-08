use serde_json::Value;

use crate::{ChatCompletionArtifact, ChatCompletionResponse, ReasoningProviderStatus};

pub(crate) fn chat_completion_response(
    raw_message: String,
    provider: ReasoningProviderStatus,
) -> ChatCompletionResponse {
    let (message, artifacts, overlay) = extract_chat_completion_artifacts(&raw_message);
    ChatCompletionResponse {
        message,
        provider,
        artifacts,
        overlay,
    }
}

fn extract_chat_completion_artifacts(
    raw_message: &str,
) -> (String, Vec<ChatCompletionArtifact>, Option<Value>) {
    let mut visible = String::new();
    let mut artifacts = Vec::new();
    let mut overlay = None;
    let mut rest = raw_message;

    while let Some(fence_start) = rest.find("```") {
        visible.push_str(&rest[..fence_start]);
        let after_open = &rest[fence_start + 3..];
        let Some(line_end) = after_open.find('\n') else {
            visible.push_str(&rest[fence_start..]);
            rest = "";
            break;
        };
        let info = after_open[..line_end].trim();
        let content_start = line_end + 1;
        let after_info = &after_open[content_start..];
        let Some(fence_end) = after_info.find("```") else {
            visible.push_str(&rest[fence_start..]);
            rest = "";
            break;
        };
        let content = &after_info[..fence_end];
        let consumed = fence_start + 3 + content_start + fence_end + 3;
        let captured = if info
            .split_whitespace()
            .any(|part| part.eq_ignore_ascii_case("diagram"))
        {
            if let Some(spec) = serde_json::from_str::<Value>(content.trim())
                .ok()
                .and_then(validate_chat_diagram_spec)
            {
                artifacts.push(ChatCompletionArtifact::Diagram { spec });
                true
            } else {
                false
            }
        } else if info
            .split_whitespace()
            .any(|part| part.eq_ignore_ascii_case("matrix"))
        {
            if let Some(spec) = serde_json::from_str::<Value>(content.trim())
                .ok()
                .and_then(validate_chat_matrix_spec)
            {
                artifacts.push(ChatCompletionArtifact::Matrix { spec });
                true
            } else {
                false
            }
        } else if info
            .split_whitespace()
            .any(|part| part.eq_ignore_ascii_case("view_action"))
        {
            if let Some(value) = serde_json::from_str::<Value>(content.trim())
                .ok()
                .and_then(normalize_chat_view_overlay)
            {
                overlay = Some(value);
                true
            } else {
                false
            }
        } else {
            false
        };

        if !captured {
            visible.push_str(&rest[fence_start..consumed]);
        }
        rest = &rest[consumed..];
    }

    visible.push_str(rest);
    (
        collapse_chat_blank_lines(visible.trim()),
        artifacts,
        overlay,
    )
}

fn validate_chat_diagram_spec(value: Value) -> Option<Value> {
    let object = value.as_object()?;
    if object.get("version").and_then(Value::as_i64) != Some(1) {
        return None;
    }
    if object
        .get("kind")
        .and_then(Value::as_str)?
        .trim()
        .is_empty()
    {
        return None;
    }
    if object
        .get("title")
        .and_then(Value::as_str)?
        .trim()
        .is_empty()
    {
        return None;
    }
    Some(Value::Object(object.clone()))
}

fn validate_chat_matrix_spec(value: Value) -> Option<Value> {
    let object = value.as_object()?;
    if object.get("version").and_then(Value::as_i64) != Some(1) {
        return None;
    }
    if object
        .get("title")
        .and_then(Value::as_str)?
        .trim()
        .is_empty()
    {
        return None;
    }
    if !object.get("rows").is_some_and(Value::is_array)
        || !object.get("columns").is_some_and(Value::is_array)
        || !object.get("cells").is_some_and(Value::is_array)
    {
        return None;
    }
    Some(Value::Object(object.clone()))
}

fn normalize_chat_view_overlay(value: Value) -> Option<Value> {
    let object = value.as_object()?;
    let mut normalized = serde_json::Map::new();

    if let Some(highlight) = object.get("highlight") {
        let values = highlight
            .as_array()?
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| Value::String(value.to_string()))
            .collect::<Vec<_>>();
        if !values.is_empty() {
            normalized.insert("highlightedNodeIds".to_string(), Value::Array(values));
        }
    }

    if let Some(annotate) = object.get("annotate") {
        let entries = annotate
            .as_object()?
            .iter()
            .filter_map(|(key, value)| {
                let label = value.as_str()?.trim();
                if key.trim().is_empty() || label.is_empty() {
                    return None;
                }
                Some((key.clone(), Value::String(label.to_string())))
            })
            .collect::<serde_json::Map<_, _>>();
        if !entries.is_empty() {
            normalized.insert("annotationsByNodeId".to_string(), Value::Object(entries));
        }
    }

    if let Some(dim_others) = object.get("dim_others").and_then(Value::as_bool) {
        normalized.insert("dimUnhighlighted".to_string(), Value::Bool(dim_others));
    }

    Some(Value::Object(normalized))
}

fn collapse_chat_blank_lines(input: &str) -> String {
    let mut output = String::new();
    let mut blank_count = 0usize;
    for line in input.lines() {
        if line.trim().is_empty() {
            blank_count += 1;
            if blank_count <= 1 {
                output.push('\n');
            }
            continue;
        }
        blank_count = 0;
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(line.trim_end());
    }
    output.trim().to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::ReasoningProviderKind;

    use super::*;

    fn provider() -> ReasoningProviderStatus {
        ReasoningProviderStatus {
            kind: ReasoningProviderKind::Heuristic,
            provider_label: "test".to_string(),
            detail: "test".to_string(),
            structured_outputs: false,
            model_label: None,
        }
    }

    #[test]
    fn lifts_diagram_artifact_and_strips_visible_json() {
        let response = chat_completion_response(
            "Here is the diagram.\n```diagram\n{\"version\":1,\"kind\":\"graph\",\"title\":\"T\",\"root\":\"Vehicle\"}\n```\nDone."
                .to_string(),
            provider(),
        );

        assert_eq!(response.message, "Here is the diagram.\n\nDone.");
        assert_eq!(
            response.artifacts,
            vec![ChatCompletionArtifact::Diagram {
                spec: json!({
                    "version": 1,
                    "kind": "graph",
                    "title": "T",
                    "root": "Vehicle"
                })
            }]
        );
        assert!(response.overlay.is_none());
    }

    #[test]
    fn leaves_invalid_artifact_fence_visible() {
        let response = chat_completion_response(
            "Broken.\n```matrix\n{\"version\":1,\"title\":\"Missing arrays\"}\n```".to_string(),
            provider(),
        );

        assert!(response.message.contains("```matrix"));
        assert!(response.artifacts.is_empty());
    }

    #[test]
    fn normalizes_view_action_overlay() {
        let response = chat_completion_response(
            "Focus this.\n```view_action\n{\"highlight\":[\"a\",\" \",\"b\"],\"annotate\":{\"a\":\"Hot path\"},\"dim_others\":true}\n```"
                .to_string(),
            provider(),
        );

        assert_eq!(response.message, "Focus this.");
        assert_eq!(
            response.overlay,
            Some(json!({
                "highlightedNodeIds": ["a", "b"],
                "annotationsByNodeId": {
                    "a": "Hot path"
                },
                "dimUnhighlighted": true
            }))
        );
    }
}
