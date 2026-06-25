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
        let (files, source_diagnostic_files) = request
            .workspace
            .as_ref()
            .map(|workspace| {
                let baseline_snapshots = if workspace.source_snapshots.is_empty() {
                    workspace.dirty_snapshots.as_slice()
                } else {
                    workspace.source_snapshots.as_slice()
                };
                let files = baseline_snapshots
                    .iter()
                    .map(|snapshot| (snapshot.path.clone(), snapshot.content.clone()))
                    .collect::<BTreeMap<_, _>>();
                let source_diagnostic_files = if workspace.source_snapshots.is_empty() {
                    BTreeMap::new()
                } else {
                    workspace
                        .dirty_snapshots
                        .iter()
                        .map(|snapshot| (snapshot.path.clone(), snapshot.content.clone()))
                        .collect::<BTreeMap<_, _>>()
                };
                (files, source_diagnostic_files)
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
                exploration_agent_request(request, goal.clone(), files, source_diagnostic_files),
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
    let cognitive_context = request.cognitive_context.clone();

    let chat_request = ChatCompletionRequest {
        messages: request.messages.clone(),
        context,
        workspace: request.workspace.clone(),
        cognitive_context,
    };
    complete_configured_chat(config, secrets, &chat_request).map(AiWorkbenchResponse::from)
}

fn exploration_agent_request(
    request: &AiWorkbenchRequest,
    goal: String,
    initial_files: BTreeMap<String, String>,
    source_diagnostic_files: BTreeMap<String, String>,
) -> SemanticAgentRunRequest {
    SemanticAgentRunRequest {
        goal,
        goal_spec: request
            .intent
            .as_ref()
            .map(design_intent_to_semantic_goal_spec),
        quality_goal: Some(default_model_quality_profile().goal),
        minimum_quality_score: Some(0.5),
        initial_files,
        source_diagnostic_files,
        focus: request.focus.clone(),
        max_steps: 6,
        reasoning_tools: vec![
            SemanticAgentToolKind::RequirementCoverage,
            SemanticAgentToolKind::SemanticImpact,
            SemanticAgentToolKind::ModelInspection,
        ],
        tool_mode: SemanticAgentToolMode::Auto,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AiWorkbenchMode, AiWorkspaceDirtySnapshot, AiWorkspaceInput, DesignIntent, ElementRef,
        GoalPolicy, SemanticGoalCheck,
    };

    #[test]
    fn workbench_exploration_maps_design_intent_to_semantic_goal_spec() {
        let focus = ElementRef {
            qualified_name: "Vehicle.powertrain".to_string(),
        };
        let request = AiWorkbenchRequest {
            mode: AiWorkbenchMode::Exploration,
            messages: Vec::new(),
            intent: Some(DesignIntent {
                summary: "Improve hybrid vehicle efficiency".to_string(),
                goals: vec!["reduce energy use".to_string()],
                constraints: vec!["preserve safety requirements".to_string()],
                assumptions: vec!["Urban drive cycle".to_string()],
                metadata: BTreeMap::new(),
            }),
            focus: vec![focus],
            workspace: None,
            cognitive_context: None,
        };
        let agent_request = exploration_agent_request(
            &request,
            "Improve hybrid vehicle efficiency".to_string(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let goal_spec = agent_request
            .goal_spec
            .expect("workbench intent should produce semantic goal spec");

        assert_eq!(goal_spec.policy, GoalPolicy::Any);
        assert!(goal_spec.checks.iter().any(|check| {
            matches!(
                check,
                SemanticGoalCheck::NamedElementExists { name, kind: None }
                    if name == "ReduceEnergyUse"
            )
        }));
        assert_eq!(agent_request.focus, request.focus);
    }

    #[test]
    fn workbench_exploration_uses_clean_sources_as_baseline_and_dirty_sources_as_diagnostics() {
        let request = AiWorkbenchRequest {
            mode: AiWorkbenchMode::Exploration,
            messages: Vec::new(),
            intent: None,
            focus: Vec::new(),
            workspace: Some(AiWorkspaceInput {
                workspace_root: None,
                active_editor_path: None,
                selected_element_id: None,
                source_snapshots: vec![AiWorkspaceDirtySnapshot {
                    path: "model.sysml".to_string(),
                    content: "package Demo { part def Vehicle; }".to_string(),
                    revision: Some(1),
                }],
                dirty_snapshots: vec![AiWorkspaceDirtySnapshot {
                    path: "model.sysml".to_string(),
                    content: "package Demo { part def Vehicle".to_string(),
                    revision: Some(2),
                }],
            }),
            cognitive_context: None,
        };

        let baseline = request
            .workspace
            .as_ref()
            .unwrap()
            .source_snapshots
            .iter()
            .map(|snapshot| (snapshot.path.clone(), snapshot.content.clone()))
            .collect::<BTreeMap<_, _>>();
        let dirty = request
            .workspace
            .as_ref()
            .unwrap()
            .dirty_snapshots
            .iter()
            .map(|snapshot| (snapshot.path.clone(), snapshot.content.clone()))
            .collect::<BTreeMap<_, _>>();
        let agent_request =
            exploration_agent_request(&request, "Explore".to_string(), baseline, dirty);

        assert_eq!(
            agent_request.initial_files["model.sysml"],
            "package Demo { part def Vehicle; }"
        );
        assert_eq!(
            agent_request.source_diagnostic_files["model.sysml"],
            "package Demo { part def Vehicle"
        );
    }
}
