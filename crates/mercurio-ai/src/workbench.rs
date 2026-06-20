use std::collections::BTreeMap;

use serde_json::Value;

use crate::{
    AiWorkbenchMode, AiWorkbenchRequest, AiWorkbenchResponse, AiWorkspaceInput,
    ChatCompletionRequest, ModelRevision, ReasoningProvider, ReasoningProviderConfigOverrides,
    ReasoningProviderSecretOverrides, SemanticAgentRunRequest, SemanticAgentRunStatus,
    SemanticAgentToolKind, SemanticAgentToolMode, SemanticContextBuilder, complete_configured_chat,
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
                exploration_agent_request(request, goal.clone(), files),
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
    let model_revision = request
        .model_revision
        .clone()
        .map(|envelope| envelope.into_model_revision())
        .transpose()
        .map_err(|err| format!("invalid model revision envelope: {err}"))?;
    if let Some(revision) = model_revision.as_ref() {
        context.extend(ai_model_revision_context_lines(revision));
    }
    let cognitive_context = match (&request.cognitive_context, model_revision.as_ref()) {
        (Some(context), _) => Some(context.clone()),
        (None, Some(revision)) => {
            Some(SemanticContextBuilder::default().build_from_model_revision(
                revision,
                &request.focus,
                &[],
            ))
        }
        (None, None) => None,
    };

    let chat_request = ChatCompletionRequest {
        messages: request.messages.clone(),
        context,
        workspace: request.workspace.clone(),
        model_revision: request.model_revision.clone(),
        cognitive_context,
    };
    complete_configured_chat(config, secrets, &chat_request).map(AiWorkbenchResponse::from)
}

fn exploration_agent_request(
    request: &AiWorkbenchRequest,
    goal: String,
    initial_files: BTreeMap<String, String>,
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

fn ai_model_revision_context_lines(revision: &ModelRevision) -> Vec<String> {
    let descriptor = revision.descriptor();
    let mut lines = vec![
        format!("Model revision: {}", descriptor.id),
        format!("Model revision producer: {:?}", descriptor.producer),
        format!("Model element count: {}", descriptor.element_count),
    ];
    if let Some(profile_id) = descriptor.profile_id {
        lines.push(format!("Model profile: {profile_id}"));
    }
    if let Some(source_set) = revision.build().input_source_set.as_ref() {
        lines.push(format!(
            "Model input source set: {} ({} sources)",
            source_set.id,
            source_set.sources.len()
        ));
        if !source_set.sources.is_empty() {
            lines.push(format!(
                "Model input sources: {}",
                source_set
                    .sources
                    .iter()
                    .map(|source| source.uri.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AiWorkbenchMode, DesignIntent, ElementRef, GoalPolicy, KirDocument, KirElement,
        ModelBuildRecord, ModelRevisionProducer, SemanticGoalCheck,
    };
    use serde_json::json;

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
            model_revision: None,
            cognitive_context: None,
        };
        let agent_request = exploration_agent_request(
            &request,
            "Improve hybrid vehicle efficiency".to_string(),
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
    fn model_revision_context_lines_include_revision_provenance() {
        let revision = crate::ModelRevision::from_kir_document(
            KirDocument {
                metadata: BTreeMap::new(),
                elements: vec![KirElement {
                    id: "part.Vehicle".to_string(),
                    kind: "PartDefinition".to_string(),
                    layer: 2,
                    properties: BTreeMap::from([("declared_name".to_string(), json!("Vehicle"))]),
                }],
            },
            ModelBuildRecord::new(ModelRevisionProducer::RemotePull),
        )
        .unwrap();

        let lines = ai_model_revision_context_lines(&revision);

        assert!(lines.iter().any(|line| line.starts_with("Model revision:")));
        assert!(
            lines
                .iter()
                .any(|line| line == "Model revision producer: RemotePull")
        );
        assert!(lines.iter().any(|line| line == "Model element count: 1"));
    }
}
