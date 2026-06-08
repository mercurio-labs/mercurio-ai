use serde_json::json;

use crate::{
    AskMercurioArtifact, AskMercurioCitation, AskMercurioProjectContext, AskMercurioTask,
    ProposalDraft,
};

pub fn classify_ask_mercurio_task(prompt: &str) -> AskMercurioTask {
    let prompt = prompt.to_ascii_lowercase();
    if is_requirements_view_prompt(&prompt) {
        AskMercurioTask::ViewRequest
    } else if prompt.contains("diagram")
        || prompt.contains("draw")
        || prompt.contains("visual")
        || prompt.contains("graph")
    {
        AskMercurioTask::DiagramRequest
    } else if prompt.contains("proposal")
        || prompt.contains("pull request")
        || prompt.contains(" pr")
        || prompt.contains("pr ")
        || prompt.contains("merge request")
    {
        AskMercurioTask::PrDraft
    } else if prompt.contains("design")
        || prompt.contains("why")
        || prompt.contains("how should")
        || prompt.contains("tradeoff")
        || prompt.contains("architecture")
    {
        AskMercurioTask::DesignQuestion
    } else {
        AskMercurioTask::General
    }
}

fn is_requirements_view_prompt(prompt: &str) -> bool {
    (prompt.contains("requirement") || prompt.contains("requirements"))
        && (prompt.contains("table")
            || prompt.contains("view")
            || prompt.contains("matrix")
            || prompt.contains("show")
            || prompt.contains("list"))
}

pub(crate) fn ask_mercurio_developer_context(task: &AskMercurioTask) -> String {
    let task_detail = match task {
        AskMercurioTask::DesignQuestion => {
            "Answer the design question using only supplied Mercurio project evidence. Cite relevant element or artifact ids."
        }
        AskMercurioTask::DiagramRequest => {
            "Explain the diagram intent briefly. The application may attach a validated diagram_spec artifact separately."
        }
        AskMercurioTask::ViewRequest => {
            "Explain the requested semantic view briefly. The application may attach a validated requirements_view artifact separately."
        }
        AskMercurioTask::PrDraft => {
            "Draft a Mercurio proposal only. Do not claim that branches, commits, files, or pull requests were created."
        }
        AskMercurioTask::General => {
            "Answer as Ask Mercurio for model-aware engineering work. Stay grounded in supplied project evidence."
        }
    };
    format!(
        "You are Ask Mercurio. {task_detail} Be concise, engineering-focused, and explicit when evidence is missing. If context includes `Metamodel lookup result:` lines, treat them as authoritative KIR evidence and do not substitute generic SysML, UML, or modeling-language knowledge."
    )
}

pub(crate) fn ask_mercurio_citations(
    project: Option<&AskMercurioProjectContext>,
    response: &str,
) -> Vec<AskMercurioCitation> {
    let mut citations = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    if let Some(project) = project {
        let citation = AskMercurioCitation {
            label: project
                .project_name
                .as_deref()
                .unwrap_or(&project.project_id)
                .to_string(),
            target_type: "project".to_string(),
            target_id: project.project_id.clone(),
        };
        seen.insert((citation.target_type.clone(), citation.target_id.clone()));
        citations.push(citation);
        if let Some(artifact_id) = &project.artifact_id {
            let citation = AskMercurioCitation {
                label: "Latest semantic artifact".to_string(),
                target_type: "artifact".to_string(),
                target_id: artifact_id.clone(),
            };
            seen.insert((citation.target_type.clone(), citation.target_id.clone()));
            citations.push(citation);
        }
        if let Some(root_id) = &project.diagram_root_id
            && response.contains(root_id)
        {
            let citation = AskMercurioCitation {
                label: project
                    .diagram_root_label
                    .as_deref()
                    .unwrap_or(root_id)
                    .to_string(),
                target_type: "element".to_string(),
                target_id: root_id.clone(),
            };
            seen.insert((citation.target_type.clone(), citation.target_id.clone()));
            citations.push(citation);
        }
    }
    for token in response
        .split_whitespace()
        .filter_map(normalize_response_element_hint)
        .filter(|token| token.contains('.') || token.contains("::"))
    {
        if citations.len() >= 6 {
            break;
        }
        let target_id = token.replace("::", ".");
        if !seen.insert(("element_hint".to_string(), target_id.clone())) {
            continue;
        }
        citations.push(AskMercurioCitation {
            label: target_id.clone(),
            target_type: "element_hint".to_string(),
            target_id,
        });
    }
    citations
}

fn normalize_response_element_hint(token: &str) -> Option<String> {
    let trimmed = token.trim_matches(|ch: char| {
        !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == ':')
    });
    if trimmed.is_empty()
        || trimmed.starts_with("http")
        || trimmed.ends_with('.')
        || trimmed
            .chars()
            .filter(|ch| *ch == '.' || *ch == ':')
            .count()
            == 0
    {
        return None;
    }
    Some(trimmed.to_string())
}

pub(crate) fn ask_mercurio_artifacts(
    task: &AskMercurioTask,
    project: Option<&AskMercurioProjectContext>,
    prompt: &str,
) -> Vec<AskMercurioArtifact> {
    match task {
        AskMercurioTask::DiagramRequest => vec![AskMercurioArtifact::DiagramSpec(json!({
            "version": 1,
            "kind": "dependency_graph",
            "title": diagram_title(prompt),
            "description": "Draft diagram generated from Ask Mercurio request.",
            "root": project.and_then(|project| project.diagram_root_id.as_deref()),
            "rootLabel": project.and_then(|project| project.diagram_root_label.as_deref()),
            "query": {
                "relations": ["specializes", "contains", "references"],
                "direction": "both",
                "depth": 2,
                "include_libraries": false,
                "include_user_model": true
            },
            "layout": {
                "direction": "right"
            },
            "style": {}
        }))],
        AskMercurioTask::ViewRequest => vec![AskMercurioArtifact::RequirementsView(json!({
            "version": 1,
            "kind": "requirements_table",
            "title": requirements_view_title(prompt),
            "description": "Requirements table generated from the current Mercurio semantic graph.",
            "renderer": "table",
            "endpoint": "/api/views/requirements-table"
        }))],
        AskMercurioTask::PrDraft => vec![AskMercurioArtifact::ProposalDraft(ProposalDraft {
            title: pr_title(prompt),
            body: pr_body(project, prompt),
            suggested_base_branch: Some("main".to_string()),
            suggested_head_branch: Some(pr_head_branch(prompt)),
            checklist: vec![
                "Link the proposal to affected semantic elements.".to_string(),
                "Review semantic impact against the latest indexed artifact.".to_string(),
                "Run project validation before preparing source-control changes.".to_string(),
            ],
            linked_semantic_elements: Vec::new(),
        })],
        _ => Vec::new(),
    }
}

fn requirements_view_title(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        "Requirements Table".to_string()
    } else {
        format!(
            "Requirements View: {}",
            trimmed.chars().take(56).collect::<String>()
        )
    }
}

fn diagram_title(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        "Mercurio Diagram".to_string()
    } else {
        format!("Diagram: {}", trimmed.chars().take(60).collect::<String>())
    }
}

fn pr_title(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        "Update Mercurio model".to_string()
    } else {
        format!("Draft: {}", trimmed.chars().take(64).collect::<String>())
    }
}

fn pr_head_branch(prompt: &str) -> String {
    let normalized = prompt
        .split_whitespace()
        .take(6)
        .flat_map(|word| word.chars())
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    let slug = normalized
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    format!(
        "ask-mercurio/{}",
        if slug.is_empty() {
            "model-update"
        } else {
            &slug
        }
    )
}

fn pr_body(project: Option<&AskMercurioProjectContext>, prompt: &str) -> String {
    let mut body = String::new();
    body.push_str("## Summary\n");
    body.push_str("- Draft proposal prepared by Ask Mercurio.\n");
    body.push_str("- Requested change: ");
    body.push_str(if prompt.trim().is_empty() {
        "model update"
    } else {
        prompt.trim()
    });
    body.push_str("\n\n## Evidence\n");
    if let Some(project) = project {
        body.push_str(&format!("- Project: {}\n", project.project_id));
        if let Some(artifact_id) = &project.artifact_id {
            body.push_str(&format!("- Semantic artifact: {artifact_id}\n"));
        }
        if let Some(commit) = &project.commit {
            body.push_str(&format!("- Base commit: {commit}\n"));
        }
    } else {
        body.push_str("- No selected project context was attached.\n");
    }
    body.push_str("\n## Validation\n- Run semantic compile and review impact before preparing a branch or PR.\n");
    body
}
