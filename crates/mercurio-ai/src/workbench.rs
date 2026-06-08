use std::collections::BTreeMap;

use serde_json::Value;

use crate::{
    AiWorkbenchMode, AiWorkbenchRequest, AiWorkbenchResponse, AiWorkspaceInput,
    ChatCompletionRequest, ReasoningProvider, ReasoningProviderConfigOverrides,
    ReasoningProviderSecretOverrides, SemanticAgentRunRequest, SemanticAgentRunStatus,
    SemanticAgentToolKind, SemanticAgentToolMode, complete_configured_chat,
    configured_reasoning_provider, default_model_quality_profile,
    design_intent_to_semantic_goal_spec, latest_user_content, run_semantic_mutation_agent,
};

pub fn run_configured_workbench_interaction(
    config: ReasoningProviderConfigOverrides,
    secrets: ReasoningProviderSecretOverrides,
    request: &AiWorkbenchRequest,
) -> Result<AiWorkbenchResponse, String> {
    if matches!(request.mode, AiWorkbenchMode::Exploration) {
        let files = request
            .workspace
            .as_ref()
            .map(|workspace| {
                let snapshots = if workspace.source_snapshots.is_empty() {
                    &workspace.dirty_snapshots
                } else {
                    &workspace.source_snapshots
                };
                snapshots
                    .iter()
                    .map(|snapshot| (snapshot.path.clone(), snapshot.content.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        if !files.is_empty() {
            let provider = configured_reasoning_provider(config, secrets);
            let goal = request
                .intent
                .as_ref()
                .map(|intent| intent.summary.clone())
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| latest_user_content(&request.messages).to_string());
            let run = run_semantic_mutation_agent(
                &provider,
                SemanticAgentRunRequest {
                    goal: goal.clone(),
                    goal_spec: request
                        .intent
                        .as_ref()
                        .map(design_intent_to_semantic_goal_spec),
                    quality_goal: Some(default_model_quality_profile().goal),
                    minimum_quality_score: Some(0.5),
                    initial_files: files,
                    focus: request.focus.clone(),
                    max_steps: 6,
                    reasoning_tools: vec![
                        SemanticAgentToolKind::RequirementCoverage,
                        SemanticAgentToolKind::SemanticImpact,
                        SemanticAgentToolKind::ModelInspection,
                    ],
                    tool_mode: SemanticAgentToolMode::Auto,
                },
            );
            let message = format!(
                "Semantic exploration {}: {}. Steps: {}.",
                match run.status {
                    SemanticAgentRunStatus::Completed => "completed",
                    SemanticAgentRunStatus::Stopped => "stopped",
                    SemanticAgentRunStatus::Failed => "failed",
                },
                run.stop_reason,
                run.steps.len()
            );
            return Ok(AiWorkbenchResponse {
                message,
                provider: provider.provider_status(),
                artifacts: Vec::new(),
                overlay: None,
                assessment: None,
                cognitive_context: None,
                proposed_actions: vec![serde_json::to_value(run).unwrap_or(Value::Null)],
            });
        }
    }

    let intent_context = request.intent.as_ref().map(|intent| {
        format!(
            "Design intent: {}; goals: {}; constraints: {}; assumptions: {}",
            intent.summary,
            intent.goals.join(", "),
            intent.constraints.join(", "),
            intent.assumptions.join(", ")
        )
    });
    let mut context = Vec::new();
    if let Some(line) = intent_context {
        context.push(line);
    }
    if !request.focus.is_empty() {
        context.push(format!(
            "Workbench focus: {}",
            request
                .focus
                .iter()
                .map(|focus| focus.qualified_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(workspace) = request.workspace.as_ref() {
        context.extend(ai_workspace_input_context_lines(workspace));
    }

    let chat_request = ChatCompletionRequest {
        messages: request.messages.clone(),
        context,
        workspace: request.workspace.clone(),
        cognitive_context: request.cognitive_context.clone(),
    };
    complete_configured_chat(config, secrets, &chat_request).map(AiWorkbenchResponse::from)
}

fn ai_workspace_input_context_lines(workspace: &AiWorkspaceInput) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(root) = workspace
        .workspace_root
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("Workspace root: {root}"));
    }
    if let Some(path) = workspace
        .active_editor_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("Active editor: {path}"));
    }
    if let Some(id) = workspace
        .selected_element_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("Selected element: {id}"));
    }
    if !workspace.dirty_snapshots.is_empty() {
        lines.push(format!(
            "Dirty editor snapshots: {}",
            workspace
                .dirty_snapshots
                .iter()
                .map(|snapshot| format!(
                    "{} ({} chars)",
                    snapshot.path,
                    snapshot.content.chars().count()
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    lines
}
