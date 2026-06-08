use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::provider::{AnthropicContentItem, OpenAiContentItem};
use crate::{
    ANTHROPIC_VERSION, AnthropicMessageResponse, ChatCompletionRequest, ChatMessageRole,
    DEFAULT_HTTP_TIMEOUT_SECS, OpenAiStructuredResponse, chat_developer_prompt,
};

pub(crate) fn http_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| Client::new())
}

pub(crate) fn request_openai_structured_json(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    schema_name: &str,
    schema: Value,
    developer_prompt: &str,
    user_prompt: &str,
) -> Result<Value, String> {
    let body = json!({
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": developer_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": true,
                "schema": schema,
            }
        }
    });

    let response = client
        .post(base_url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let body_text = response.text().map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(format!("AI provider request failed: {status} {body_text}"));
    }

    let envelope: OpenAiStructuredResponse =
        serde_json::from_str(&body_text).map_err(|error| error.to_string())?;
    let output_text = extract_output_text(&envelope)?;
    serde_json::from_str(&output_text).map_err(|error| error.to_string())
}

pub(crate) fn request_openai_text(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    request: &ChatCompletionRequest,
) -> Result<String, String> {
    let mut input = Vec::new();
    input.push(json!({
        "role": "developer",
        "content": chat_developer_prompt(request),
    }));
    input.extend(request.messages.iter().map(|message| {
        json!({
            "role": chat_role_name(&message.role),
            "content": message.content,
        })
    }));
    if input.is_empty() {
        return Err("Chat request must include at least one message.".to_string());
    }

    let body = json!({
        "model": model,
        "input": input,
    });

    let response = client
        .post(base_url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let body_text = response.text().map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(format!("AI provider request failed: {status} {body_text}"));
    }

    let envelope: OpenAiStructuredResponse =
        serde_json::from_str(&body_text).map_err(|error| error.to_string())?;
    extract_output_text(&envelope).map(|value| value.trim().to_string())
}

pub(crate) fn request_anthropic_structured_json(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    schema_name: &str,
    schema: Value,
    developer_prompt: &str,
    user_blocks: Vec<Value>,
) -> Result<Value, String> {
    let tool_name = format!("emit_{schema_name}");
    let body = json!({
        "model": model,
        "max_tokens": 4096,
        "system": [
            {
                "type": "text",
                "text": developer_prompt,
                "cache_control": { "type": "ephemeral" }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": user_blocks
            }
        ],
        "tools": [
            {
                "name": tool_name,
                "description": format!("Emit the `{schema_name}` JSON payload exactly matching the supplied input schema."),
                "input_schema": schema
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": tool_name
        }
    });

    let response = client
        .post(base_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .json(&body)
        .send()
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let body_text = response.text().map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(format!(
            "Anthropic provider request failed: {status} {body_text}"
        ));
    }

    let envelope: AnthropicMessageResponse =
        serde_json::from_str(&body_text).map_err(|error| error.to_string())?;
    extract_anthropic_tool_input(&envelope)
}

pub(crate) fn request_anthropic_text(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    request: &ChatCompletionRequest,
) -> Result<String, String> {
    let system_blocks = anthropic_chat_system_blocks(request);
    let messages = anthropic_chat_messages(request)?;
    let body = json!({
        "model": model,
        "max_tokens": 4096,
        "system": system_blocks,
        "messages": messages,
    });

    let response = client
        .post(base_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .json(&body)
        .send()
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let body_text = response.text().map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(format!(
            "Anthropic provider request failed: {status} {body_text}"
        ));
    }

    let envelope: AnthropicMessageResponse =
        serde_json::from_str(&body_text).map_err(|error| error.to_string())?;
    extract_anthropic_text(&envelope)
}

fn anthropic_chat_system_blocks(request: &ChatCompletionRequest) -> Vec<Value> {
    vec![json!({
        "type": "text",
        "text": chat_developer_prompt(request),
        "cache_control": { "type": "ephemeral" }
    })]
}

fn anthropic_chat_messages(request: &ChatCompletionRequest) -> Result<Vec<Value>, String> {
    let mut messages = Vec::new();
    for message in &request.messages {
        match message.role {
            ChatMessageRole::Developer => {
                messages.push(json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": format!("Developer instruction:\n{}", message.content)
                        }
                    ]
                }));
            }
            ChatMessageRole::Assistant => {
                messages.push(json!({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": message.content
                        }
                    ]
                }));
            }
            ChatMessageRole::User => {
                messages.push(json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message.content
                        }
                    ]
                }));
            }
        }
    }
    if messages.is_empty() {
        return Err("Chat request must include at least one message.".to_string());
    }
    Ok(messages)
}

fn chat_role_name(role: &ChatMessageRole) -> &'static str {
    match role {
        ChatMessageRole::Developer => "developer",
        ChatMessageRole::Assistant => "assistant",
        ChatMessageRole::User => "user",
    }
}

pub(crate) fn extract_output_text(response: &OpenAiStructuredResponse) -> Result<String, String> {
    for output in &response.output {
        for content in &output.content {
            match content {
                OpenAiContentItem::OutputText { text } => return Ok(text.clone()),
                OpenAiContentItem::Refusal { refusal } => {
                    return Err(format!("model refused structured response: {refusal}"));
                }
                OpenAiContentItem::Other => {}
            }
        }
    }

    Err("no output_text item found in AI provider response".to_string())
}

pub(crate) fn extract_anthropic_tool_input(
    response: &AnthropicMessageResponse,
) -> Result<Value, String> {
    response
        .content
        .iter()
        .find_map(|content| match content {
            AnthropicContentItem::ToolUse { input } => Some(input.clone()),
            _ => None,
        })
        .ok_or_else(|| "no tool_use item found in Anthropic provider response".to_string())
}

pub(crate) fn extract_anthropic_text(
    response: &AnthropicMessageResponse,
) -> Result<String, String> {
    let text = response
        .content
        .iter()
        .filter_map(|content| match content {
            AnthropicContentItem::Text { text } => Some(text.trim()),
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        Err("no text item found in Anthropic provider response".to_string())
    } else {
        Ok(text)
    }
}
