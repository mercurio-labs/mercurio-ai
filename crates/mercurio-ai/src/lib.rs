use serde::{Deserialize, Serialize};
use serde_json::Value;

mod agent;
mod artifacts;
mod ask_mercurio;
mod context;
mod heuristic;
mod prompts;
mod provider;
mod provider_http;
mod provider_impls;
mod provider_prompts;
mod provider_registry;
mod provider_resolution;
mod schema;
mod workbench;
pub use agent::run_semantic_mutation_agent;
#[cfg(test)]
pub(crate) use agent::select_semantic_agent_tools;
pub(crate) use agent::semantic_agent_tool_id;
pub(crate) use artifacts::chat_completion_response;
pub use ask_mercurio::classify_ask_mercurio_task;
pub(crate) use ask_mercurio::{
    ask_mercurio_artifacts, ask_mercurio_citations, ask_mercurio_developer_context,
};
pub use context::SemanticContextBuilder;
#[cfg(test)]
pub(crate) use heuristic::request_context_has_element;
pub(crate) use prompts::chat_developer_prompt;
pub(crate) use provider::{AnthropicMessageResponse, OpenAiStructuredResponse};
pub use provider::{
    AnthropicReasoningProvider, AzureOpenAiReasoningProvider, HeuristicReasoningProvider,
    OpenAiReasoningProvider, ReasoningProvider, ReasoningProviderConfigOverrides,
    ReasoningProviderKind, ReasoningProviderSecretOverrides, ReasoningProviderStatus,
    ResolvedReasoningProvider, SemanticMutationProposalProvider,
};
#[cfg(test)]
pub(crate) use provider_http::{extract_anthropic_tool_input, extract_output_text};
pub(crate) use provider_prompts::{
    ConnectionProbeEnvelope, SemanticSummaryEnvelope, connection_probe_schema,
    parse_semantic_mutation_proposals_payload, semantic_mutation_agent_guidance,
    semantic_mutation_proposal_developer_prompt, semantic_mutation_proposal_schema,
    semantic_mutation_proposal_user_prompt, semantic_summary_developer_prompt,
    semantic_summary_schema, semantic_summary_user_prompt,
};
pub use provider_registry::provider_descriptors;
#[cfg(test)]
pub(crate) use provider_resolution::normalize_azure_openai_base_url;
use provider_resolution::{
    configured_provider_from_registry, configured_provider_kind,
    configured_provider_missing_message, heuristic_provider, resolve_reasoning_provider_from_env,
};
pub use schema::*;
pub use workbench::run_configured_workbench_interaction;

pub use mercurio_core::{
    CognitiveContext, CognitiveDiagnostic, CognitiveDiagnosticSeverity, CognitiveElement,
    CognitiveFocus, CognitiveRelationship, CoreMutationFeasibilityService, DesignIntent, Edge,
    Element, ElementRef, FeasibilityStatus, GoalEvaluation, GoalPolicy, Graph, KirDocument,
    MutationApplicationResult, MutationContext, MutationEvidence, MutationFeasibilityReport,
    MutationFeasibilityService, MutationProposal, NodeId, SemanticArtifact, SemanticElementRef,
    SemanticExpression, SemanticGoalCheck, SemanticGoalExplanation, SemanticGoalSpec,
    SemanticMutation, SemanticMutationCapabilityContext, SemanticReasoningContext,
    SemanticWorkspaceRef, SourceSpanRef, WorkspaceRevision, default_stdlib_path,
    design_intent_to_assessment_spec, design_intent_to_semantic_goal_spec, stable_digest,
};
pub use mercurio_requirements::{
    default_model_quality_profile, evaluate_semantic_goal, explain_semantic_goal,
};
pub use mercurio_sysml::{
    compile_sysml_text, enrich_sysml_semantic_reasoning_context_with_child_affordances,
    load_authoring_project_from_sysml, sysml_mutation_feasibility_service,
    sysml_semantic_mutation_capability_context,
    sysml_semantic_reasoning_context_from_authoring_project,
};

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4-mini";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1/responses";
const DEFAULT_AZURE_OPENAI_PATH: &str = "/openai/v1/responses";
const DEFAULT_ANTHROPIC_PROPOSAL_MODEL: &str = "claude-opus-4-8";
const DEFAULT_ANTHROPIC_FAST_MODEL: &str = "claude-sonnet-4-6";
const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticChangeKind {
    Added,
    Removed,
    Changed,
    Unchanged,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticChangeItem {
    pub kind: SemanticChangeKind,
    pub element_id: String,
    pub element_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changed_properties: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changed_relationships: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticSummaryRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_hint: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changed_files: Vec<String>,
    #[serde(default)]
    pub changes: Vec<SemanticChangeItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticSummaryResponse {
    pub title: String,
    pub body: Vec<String>,
    pub provider: ReasoningProviderStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AskMercurioTask {
    DesignQuestion,
    DiagramRequest,
    ViewRequest,
    #[serde(rename = "proposal_draft", alias = "pr_draft")]
    PrDraft,
    General,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AskMercurioProjectContext {
    pub project_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagram_root_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagram_root_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AskMercurioRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_hint: Option<AskMercurioTask>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AskMercurioResponse {
    pub message: String,
    pub task: AskMercurioTask,
    pub provider: ReasoningProviderStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<AskMercurioProjectContext>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<AskMercurioCitation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<AskMercurioArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AskMercurioCitation {
    pub label: String,
    pub target_type: String,
    pub target_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum AskMercurioArtifact {
    DiagramSpec(Value),
    RequirementsView(Value),
    ProposalDraft(ProposalDraft),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProposalDraft {
    pub title: String,
    pub body: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_base_branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_head_branch: Option<String>,
    pub checklist: Vec<String>,
    #[serde(default)]
    pub linked_semantic_elements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticMutationProposalRequest {
    pub design_intent: String,
    pub workspace_revision: WorkspaceRevision,
    #[serde(default)]
    pub focus: Vec<ElementRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_goal_guidance: Option<SemanticGoalExplanation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_goal_guidance: Option<SemanticGoalExplanation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_context: Option<SemanticReasoningContext>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cognitive_context: Option<CognitiveContext>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_tool_results: Vec<SemanticAgentToolResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CheckedMutationProposal {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_id: Option<String>,
    pub proposal: MutationProposal,
    pub feasibility: MutationFeasibilityReport,
    pub revision_attempted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAgentRunRequest {
    pub goal: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_spec: Option<SemanticGoalSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_goal: Option<SemanticGoalSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_quality_score: Option<f64>,
    #[serde(default)]
    pub initial_files: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub focus: Vec<ElementRef>,
    pub max_steps: usize,
    #[serde(default)]
    pub reasoning_tools: Vec<SemanticAgentToolKind>,
    #[serde(default)]
    pub tool_mode: SemanticAgentToolMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAgentRun {
    pub goal: String,
    pub status: SemanticAgentRunStatus,
    pub stop_reason: String,
    pub steps: Vec<SemanticAgentStep>,
    pub final_files: std::collections::BTreeMap<String, String>,
    pub final_workspace_revision: WorkspaceRevision,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SemanticAgentRunStatus {
    Completed,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SemanticAgentToolKind {
    RequirementCoverage,
    SemanticImpact,
    StateSimulation,
    ModelInspection,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SemanticAgentToolMode {
    Off,
    RequestedOnly,
    #[default]
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAgentToolFinding {
    pub id: String,
    pub severity: String,
    pub title: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub elements: Vec<ElementRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAgentToolResult {
    pub tool: SemanticAgentToolKind,
    pub status: String,
    pub summary: Vec<String>,
    pub findings: Vec<SemanticAgentToolFinding>,
    pub artifact: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAgentStep {
    pub index: usize,
    pub workspace_revision: WorkspaceRevision,
    pub semantic_context: SemanticReasoningContext,
    pub goal_evaluation: Option<GoalEvaluation>,
    pub quality_evaluation: Option<GoalEvaluation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<SemanticAgentToolResult>,
    pub proposals: Vec<CheckedMutationProposal>,
    pub selected_proposal_index: Option<usize>,
    pub applied: Option<MutationApplicationResult>,
    pub stop_reason: Option<String>,
}

pub fn propose_checked_semantic_mutations<P, F>(
    provider: &P,
    feasibility: &F,
    context: &MutationContext,
    request: &SemanticMutationProposalRequest,
) -> Vec<CheckedMutationProposal>
where
    P: SemanticMutationProposalProvider,
    F: MutationFeasibilityService,
{
    provider
        .propose_semantic_mutations(request)
        .into_iter()
        .map(|proposal| checked_or_revised_proposal(feasibility, context, proposal))
        .collect()
}

fn checked_or_revised_proposal<F>(
    feasibility: &F,
    context: &MutationContext,
    proposal: MutationProposal,
) -> CheckedMutationProposal
where
    F: MutationFeasibilityService,
{
    let first_report = feasibility.check(context, &proposal);
    if first_report.status != FeasibilityStatus::RequiresSupportingChanges
        || first_report.suggested_supporting_changes.is_empty()
    {
        let proposal_id = checked_proposal_id(&first_report);
        return CheckedMutationProposal {
            proposal_id,
            proposal,
            feasibility: first_report,
            revision_attempted: false,
        };
    }

    let mut revised = proposal.clone();
    let mut operations = first_report.suggested_supporting_changes.clone();
    operations.extend(proposal.operations.clone());
    revised.operations = operations;
    revised.rationale = Some(match proposal.rationale {
        Some(rationale) => format!("{rationale} Revised with core-suggested supporting changes."),
        None => "Revised with core-suggested supporting changes.".to_string(),
    });
    let revised_report = feasibility.check(context, &revised);
    let proposal_id = checked_proposal_id(&revised_report);
    CheckedMutationProposal {
        proposal_id,
        proposal: revised,
        feasibility: revised_report,
        revision_attempted: true,
    }
}

fn checked_proposal_id(report: &MutationFeasibilityReport) -> Option<String> {
    report
        .normalized_plan
        .as_ref()
        .map(|plan| plan.proposal_id.clone())
}

pub fn default_reasoning_provider() -> ResolvedReasoningProvider {
    resolve_reasoning_provider_from_env(&ReasoningProviderSecretOverrides::default())
}

pub fn default_reasoning_provider_with_secret_overrides(
    secrets: ReasoningProviderSecretOverrides,
) -> ResolvedReasoningProvider {
    resolve_reasoning_provider_from_env(&secrets)
}

pub fn default_reasoning_provider_status() -> ReasoningProviderStatus {
    default_reasoning_provider().provider_status()
}

pub fn default_reasoning_provider_status_with_secret_overrides(
    secrets: ReasoningProviderSecretOverrides,
) -> ReasoningProviderStatus {
    default_reasoning_provider_with_secret_overrides(secrets).provider_status()
}

pub fn test_default_reasoning_provider_connection() -> Result<ReasoningProviderStatus, String> {
    default_reasoning_provider().test_connection()
}

pub fn test_default_reasoning_provider_connection_with_secret_overrides(
    secrets: ReasoningProviderSecretOverrides,
) -> Result<ReasoningProviderStatus, String> {
    default_reasoning_provider_with_secret_overrides(secrets).test_connection()
}

pub fn configured_reasoning_provider(
    config: ReasoningProviderConfigOverrides,
    secrets: ReasoningProviderSecretOverrides,
) -> ResolvedReasoningProvider {
    configured_provider_from_registry(&config, &secrets).unwrap_or_else(|| {
        if config.provider.is_some() {
            ResolvedReasoningProvider::Heuristic(heuristic_provider())
        } else {
            default_reasoning_provider_with_secret_overrides(secrets)
        }
    })
}

pub fn test_configured_reasoning_provider_connection(
    config: ReasoningProviderConfigOverrides,
    secrets: ReasoningProviderSecretOverrides,
) -> Result<ReasoningProviderStatus, String> {
    if let Some(kind) = configured_provider_kind(&config) {
        let provider = configured_provider_from_registry(&config, &secrets)
            .ok_or_else(|| configured_provider_missing_message(&config, &secrets, kind))?;
        provider.test_connection()
    } else {
        default_reasoning_provider_with_secret_overrides(secrets).test_connection()
    }
}

pub fn summarize_semantic_changes(request: &SemanticSummaryRequest) -> SemanticSummaryResponse {
    default_reasoning_provider().summarize_semantic_changes(request)
}

pub fn summarize_semantic_changes_with_secret_overrides(
    request: &SemanticSummaryRequest,
    secrets: ReasoningProviderSecretOverrides,
) -> SemanticSummaryResponse {
    default_reasoning_provider_with_secret_overrides(secrets).summarize_semantic_changes(request)
}

pub fn complete_chat_with_secret_overrides(
    request: &ChatCompletionRequest,
    secrets: ReasoningProviderSecretOverrides,
) -> Result<ChatCompletionResponse, String> {
    default_reasoning_provider_with_secret_overrides(secrets).complete_chat(request)
}

pub fn ask_mercurio(
    request: &AskMercurioRequest,
    project: Option<AskMercurioProjectContext>,
    context: Vec<String>,
) -> Result<AskMercurioResponse, String> {
    ask_mercurio_with_provider(default_reasoning_provider(), request, project, context)
}

pub fn ask_mercurio_with_config(
    config: ReasoningProviderConfigOverrides,
    secrets: ReasoningProviderSecretOverrides,
    request: &AskMercurioRequest,
    project: Option<AskMercurioProjectContext>,
    context: Vec<String>,
) -> Result<AskMercurioResponse, String> {
    ask_mercurio_with_provider(
        configured_reasoning_provider(config, secrets),
        request,
        project,
        context,
    )
}

fn ask_mercurio_with_provider(
    provider: ResolvedReasoningProvider,
    request: &AskMercurioRequest,
    project: Option<AskMercurioProjectContext>,
    context: Vec<String>,
) -> Result<AskMercurioResponse, String> {
    let task = request
        .task_hint
        .clone()
        .unwrap_or_else(|| classify_ask_mercurio_task(latest_user_content(&request.messages)));
    let mut chat_context = vec![ask_mercurio_developer_context(&task)];
    chat_context.extend(context);
    let chat_request = ChatCompletionRequest {
        messages: request.messages.clone(),
        context: chat_context,
        workspace: None,
        cognitive_context: None,
    };
    let chat = match provider.complete_chat(&chat_request) {
        Ok(chat) => chat,
        Err(provider_err) => heuristic_provider()
            .complete_chat(&chat_request)
            .map_err(|fallback_err| {
                format!(
                    "configured AI provider failed ({provider_err}); heuristic fallback also failed ({fallback_err})"
                )
            })?,
    };
    let citations = ask_mercurio_citations(project.as_ref(), &chat.message);
    let artifacts = ask_mercurio_artifacts(
        &task,
        project.as_ref(),
        latest_user_content(&request.messages),
    );

    Ok(AskMercurioResponse {
        message: chat.message,
        task,
        provider: chat.provider,
        project,
        citations,
        artifacts,
    })
}

pub fn complete_configured_chat(
    config: ReasoningProviderConfigOverrides,
    secrets: ReasoningProviderSecretOverrides,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, String> {
    if let Some(kind) = configured_provider_kind(&config) {
        let provider = configured_provider_from_registry(&config, &secrets)
            .ok_or_else(|| configured_provider_missing_message(&config, &secrets, kind))?;
        provider.complete_chat(request)
    } else {
        complete_chat_with_secret_overrides(request, secrets)
    }
}

pub(crate) fn latest_user_content(messages: &[ChatMessage]) -> &str {
    messages
        .iter()
        .rev()
        .find(|message| message.role == ChatMessageRole::User)
        .map(|message| message.content.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use mercurio_core::{
        AuthoringProject, FeasibilityStatus, MutationContext, default_model_quality_profile,
    };
    use mercurio_sysml::{
        load_authoring_project_from_sysml, sysml_mutation_feasibility_service,
        sysml_semantic_reasoning_context_from_authoring_project,
    };
    use serde_json::json;

    use super::{
        AnthropicMessageResponse, CheckedMutationProposal, MutationProposal,
        OpenAiStructuredResponse, ReasoningProviderConfigOverrides,
        ReasoningProviderSecretOverrides, SemanticChangeItem, SemanticChangeKind,
        SemanticContextBuilder, SemanticMutationProposalProvider, SemanticMutationProposalRequest,
        SemanticSummaryRequest, ask_mercurio_artifacts, classify_ask_mercurio_task,
        configured_reasoning_provider, extract_anthropic_tool_input, extract_output_text,
        heuristic_provider, normalize_azure_openai_base_url,
        parse_semantic_mutation_proposals_payload, propose_checked_semantic_mutations,
        run_semantic_mutation_agent, semantic_mutation_proposal_schema,
        semantic_mutation_proposal_user_prompt, summarize_semantic_changes,
        test_configured_reasoning_provider_connection,
    };
    use crate::{
        AskMercurioArtifact, AskMercurioTask, ElementRef, ReasoningProvider, ReasoningProviderKind,
        SemanticAgentRunRequest, SemanticAgentRunStatus, SemanticAgentToolFinding,
        SemanticAgentToolKind, SemanticAgentToolResult, SemanticMutation, WorkspaceRevision,
        explain_semantic_goal,
    };

    struct FixedProposalProvider {
        proposals: Vec<MutationProposal>,
    }

    impl SemanticMutationProposalProvider for FixedProposalProvider {
        fn propose_semantic_mutations(
            &self,
            _request: &SemanticMutationProposalRequest,
        ) -> Vec<MutationProposal> {
            self.proposals.clone()
        }
    }

    struct RequestRevisionProposalProvider;

    impl SemanticMutationProposalProvider for RequestRevisionProposalProvider {
        fn propose_semantic_mutations(
            &self,
            request: &SemanticMutationProposalRequest,
        ) -> Vec<MutationProposal> {
            if crate::request_context_has_element(request, "Demo.UAVInterceptor") {
                return Vec::new();
            }
            vec![MutationProposal {
                intent: "Add UAV interceptor definition".to_string(),
                affected_elements: vec![ElementRef::new("Demo")],
                operations: vec![SemanticMutation::AddDefinition {
                    container: ElementRef::new("Demo"),
                    keyword: "part".to_string(),
                    name: "UAVInterceptor".to_string(),
                    specializes: Vec::new(),
                }],
                evidence: Vec::new(),
                rationale: None,
                workspace_revision: request.workspace_revision.clone(),
            }]
        }
    }

    fn hybrid_vehicle_project() -> AuthoringProject {
        load_authoring_project_from_sysml(BTreeMap::from([(
            "hybrid.sysml".to_string(),
            r#"
package HybridVehicle {
    part def HybridVehicle {
        part battery : BatteryPack;
    }

    part def BatteryPack;

    requirement def ImproveEfficiency;
}
"#
            .to_string(),
        )]))
        .unwrap()
    }

    #[test]
    fn heuristic_summary_counts_semantic_changes() {
        let response = summarize_semantic_changes(&SemanticSummaryRequest {
            title_hint: None,
            changed_files: vec!["models/vehicle.sysml".to_string()],
            changes: vec![SemanticChangeItem {
                kind: SemanticChangeKind::Added,
                element_id: "type.Vehicle.Battery".to_string(),
                element_kind: "PartDefinition".to_string(),
                label: Some("Battery".to_string()),
                changed_properties: Vec::new(),
                changed_relationships: Vec::new(),
                source_path: Some("models/vehicle.sysml".to_string()),
            }],
        });

        assert_eq!(response.title, "Add semantic model elements");
        assert!(response.body.iter().any(|line| line.contains("Added 1")));
    }

    #[test]
    fn heuristic_provider_is_always_testable() {
        let provider = heuristic_provider();
        let status = provider.test_connection().unwrap();
        assert!(status.structured_outputs);
    }

    #[test]
    fn heuristic_provider_returns_semantic_mutation_proposal_for_hybrid_efficiency() {
        let provider = heuristic_provider();
        let proposals = provider.propose_semantic_mutations(&SemanticMutationProposalRequest {
            design_intent: "Improve hybrid vehicle efficiency".to_string(),
            workspace_revision: WorkspaceRevision {
                fingerprint: "test-revision".to_string(),
            },
            focus: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
            task_goal_guidance: None,
            quality_goal_guidance: None,
            semantic_context: None,
            cognitive_context: None,
            reasoning_tool_results: Vec::new(),
        });

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].workspace_revision.fingerprint, "test-revision");
        assert!(proposals[0].operations.iter().any(|operation| matches!(
            operation,
            SemanticMutation::AddDefinition { name, .. }
                if name == "RegenerativeBrakingSystem"
        )));
        assert_eq!(proposals[0].operations.len(), 1);
    }

    #[test]
    fn heuristic_provider_proposes_requirements_package_from_empty_model() {
        let provider = heuristic_provider();
        let proposals = provider.propose_semantic_mutations(&SemanticMutationProposalRequest {
            design_intent: "add a requirements package with 10 requirements".to_string(),
            workspace_revision: WorkspaceRevision {
                fingerprint: "test-revision".to_string(),
            },
            focus: Vec::new(),
            task_goal_guidance: None,
            quality_goal_guidance: None,
            semantic_context: None,
            cognitive_context: None,
            reasoning_tool_results: Vec::new(),
        });

        assert_eq!(proposals.len(), 1);
        assert!(matches!(
            proposals[0].operations.as_slice(),
            [SemanticMutation::AddPackage { name, .. }] if name == "Requirements"
        ));
    }

    #[test]
    fn semantic_mutation_proposal_schema_accepts_supported_operations() {
        let schema = semantic_mutation_proposal_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["proposals"]["type"], "array");
        let schema_text = serde_json::to_string(&schema).unwrap();
        assert!(schema_text.contains("\"part\""));
        assert!(schema_text.contains("\"satisfy\""));
        assert!(!schema_text.contains("\"block\""));
    }

    #[test]
    fn semantic_mutation_prompt_includes_capability_context() {
        let mutation_context = MutationContext::from_project(hybrid_vehicle_project());
        let semantic_context = sysml_semantic_reasoning_context_from_authoring_project(
            &mutation_context.project,
            mutation_context.workspace_revision.clone(),
            vec![ElementRef::new("HybridVehicle.HybridVehicle")],
            64,
        );
        let tool_results = vec![SemanticAgentToolResult {
            tool: SemanticAgentToolKind::RequirementCoverage,
            status: "completed".to_string(),
            summary: vec!["coverage gap found".to_string()],
            findings: vec![SemanticAgentToolFinding {
                id: "coverage.req.missing_satisfy".to_string(),
                severity: "warning".to_string(),
                title: "Requirement coverage gap".to_string(),
                message: "ImproveEfficiency has no satisfy relationship.".to_string(),
                elements: vec![ElementRef::new("HybridVehicle.ImproveEfficiency")],
            }],
            artifact: json!({"uncoveredRequirementCount": 1}),
        }];
        let cognitive_context = SemanticContextBuilder::default()
            .build_from_project(
                &mutation_context.project,
                mutation_context.workspace_revision.clone(),
                &[ElementRef::new("HybridVehicle.HybridVehicle")],
                &tool_results,
            )
            .unwrap();
        let request = SemanticMutationProposalRequest {
            design_intent: "Improve efficiency".to_string(),
            workspace_revision: mutation_context.workspace_revision.clone(),
            focus: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
            task_goal_guidance: None,
            quality_goal_guidance: Some(explain_semantic_goal(
                &default_model_quality_profile().goal,
            )),
            semantic_context: Some(semantic_context),
            cognitive_context: Some(cognitive_context),
            reasoning_tool_results: tool_results,
        };
        let prompt = semantic_mutation_proposal_user_prompt(&request);

        assert!(prompt.contains("capability_context"));
        assert!(prompt.contains("sysml-v2-writable-mutation-v1"));
        assert!(prompt.contains("semantic_context"));
        assert!(prompt.contains("cognitive_context"));
        assert!(prompt.contains("grounding_rule"));
        assert!(prompt.contains("sysml-v2-authoring-context-v1"));
        assert!(prompt.contains("quality_goal_guidance"));
        assert!(prompt.contains("coverage.req.missing_satisfy"));
        assert!(prompt.contains("Every requirement element must have non-empty semantic field"));
        assert!(prompt.contains("Never use keyword `block`"));
        assert!(prompt.contains("HybridVehicle.HybridVehicle"));
    }

    #[test]
    fn semantic_context_builder_grounds_focus_neighborhood_and_tool_findings() {
        let mutation_context = MutationContext::from_project(hybrid_vehicle_project());
        let tool_results = vec![SemanticAgentToolResult {
            tool: SemanticAgentToolKind::RequirementCoverage,
            status: "completed".to_string(),
            summary: vec!["coverage gap found".to_string()],
            findings: vec![SemanticAgentToolFinding {
                id: "coverage.req.missing_satisfy".to_string(),
                severity: "warning".to_string(),
                title: "Requirement coverage gap".to_string(),
                message: "ImproveEfficiency has no satisfy relationship.".to_string(),
                elements: vec![ElementRef::new("HybridVehicle.ImproveEfficiency")],
            }],
            artifact: json!({"uncoveredRequirementCount": 1}),
        }];

        let context = SemanticContextBuilder::default()
            .build_from_project(
                &mutation_context.project,
                mutation_context.workspace_revision.clone(),
                &[ElementRef::new("HybridVehicle.HybridVehicle")],
                &tool_results,
            )
            .unwrap();

        assert!(context.focus.elements.iter().any(
            |element| element.qualified_name.as_deref() == Some("HybridVehicle.HybridVehicle")
        ));
        assert!(
            context
                .elements
                .iter()
                .any(|element| element.element.qualified_name.as_deref()
                    == Some("HybridVehicle.HybridVehicle"))
        );
        assert!(
            context
                .relationships
                .iter()
                .any(|relationship| relationship.kind == "owner")
        );
        assert_eq!(context.diagnostics.len(), 1);
        assert_eq!(context.diagnostics[0].code, "coverage.req.missing_satisfy");
        assert_eq!(context.artifacts.len(), 1);
        assert_eq!(context.artifacts[0].kind, "reasoning.requirement_coverage");
        assert!(
            context.artifacts[0]
                .element_refs
                .iter()
                .any(|element| element.qualified_name.as_deref()
                    == Some("HybridVehicle.ImproveEfficiency"))
        );
    }

    #[test]
    fn parses_provider_semantic_mutation_payload_and_pins_revision() {
        let request = SemanticMutationProposalRequest {
            design_intent: "Improve efficiency".to_string(),
            workspace_revision: WorkspaceRevision {
                fingerprint: "fresh".to_string(),
            },
            focus: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
            task_goal_guidance: None,
            quality_goal_guidance: None,
            semantic_context: None,
            cognitive_context: None,
            reasoning_tool_results: Vec::new(),
        };

        let proposals = parse_semantic_mutation_proposals_payload(
            json!({
                "proposals": [
                    {
                        "intent": "Add regenerative braking",
                        "affected_elements": [
                            { "qualified_name": "HybridVehicle.HybridVehicle" }
                        ],
                        "operations": [
                            {
                                "AddDefinition": {
                                    "container": { "qualified_name": "HybridVehicle" },
                                    "keyword": "part",
                                    "name": "RegenerativeBrakingSystem",
                                    "specializes": []
                                }
                            },
                            {
                                "AddUsage": {
                                    "container": { "qualified_name": "HybridVehicle.HybridVehicle" },
                                    "keyword": "part",
                                    "name": "regenerativeBraking",
                                    "ty": { "qualified_name": "HybridVehicle.RegenerativeBrakingSystem" },
                                    "specializes": []
                                }
                            }
                        ],
                        "evidence": [
                            {
                                "element": { "qualified_name": "HybridVehicle.BatteryPack" },
                                "summary": "Battery pack can receive recovered energy."
                            }
                        ],
                        "rationale": "Recover kinetic energy.",
                        "workspace_revision": { "fingerprint": "provider-stale" }
                    }
                ]
            }),
            &request,
        )
        .unwrap();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].workspace_revision.fingerprint, "fresh");
        assert!(matches!(
            proposals[0].operations[0],
            SemanticMutation::AddDefinition { ref name, .. }
                if name == "RegenerativeBrakingSystem"
        ));
    }

    #[test]
    fn checked_semantic_mutation_flow_accepts_feasible_ai_proposal() {
        let context = MutationContext::from_project(hybrid_vehicle_project());
        let provider = heuristic_provider();
        let checked = propose_checked_semantic_mutations(
            &provider,
            &sysml_mutation_feasibility_service(),
            &context,
            &SemanticMutationProposalRequest {
                design_intent: "Improve hybrid vehicle efficiency".to_string(),
                workspace_revision: context.workspace_revision.clone(),
                focus: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
                task_goal_guidance: None,
                quality_goal_guidance: None,
                semantic_context: None,
                cognitive_context: None,
                reasoning_tool_results: Vec::new(),
            },
        );

        assert_eq!(checked.len(), 1);
        assert!(matches!(
            checked[0],
            CheckedMutationProposal {
                revision_attempted: false,
                ..
            }
        ));
        assert_eq!(checked[0].feasibility.status, FeasibilityStatus::Allowed);
    }

    #[test]
    fn checked_semantic_mutation_flow_revises_with_supporting_changes() {
        let context = MutationContext::from_project(hybrid_vehicle_project());
        let provider = FixedProposalProvider {
            proposals: vec![MutationProposal {
                intent: "Add regenerative braking usage".to_string(),
                affected_elements: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
                operations: vec![SemanticMutation::AddUsage {
                    container: ElementRef::new("HybridVehicle.HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "regenerativeBraking".to_string(),
                    ty: Some(ElementRef::new("HybridVehicle.RegenerativeBrakingSystem")),
                    specializes: Vec::new(),
                }],
                evidence: Vec::new(),
                rationale: None,
                workspace_revision: context.workspace_revision.clone(),
            }],
        };

        let checked = propose_checked_semantic_mutations(
            &provider,
            &sysml_mutation_feasibility_service(),
            &context,
            &SemanticMutationProposalRequest {
                design_intent: "Improve hybrid vehicle efficiency".to_string(),
                workspace_revision: context.workspace_revision.clone(),
                focus: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
                task_goal_guidance: None,
                quality_goal_guidance: None,
                semantic_context: None,
                cognitive_context: None,
                reasoning_tool_results: Vec::new(),
            },
        );

        assert_eq!(checked.len(), 1);
        assert!(checked[0].revision_attempted);
        assert_eq!(checked[0].feasibility.status, FeasibilityStatus::Allowed);
        assert!(matches!(
            checked[0].proposal.operations.first(),
            Some(SemanticMutation::AddDefinition { name, .. })
                if name == "RegenerativeBrakingSystem"
        ));
    }

    #[test]
    fn semantic_agent_builds_minimal_hybrid_vehicle_from_empty_model() {
        let provider = heuristic_provider();

        let run = run_semantic_mutation_agent(
            &provider,
            SemanticAgentRunRequest {
                goal: "Create a minimal hybrid vehicle model that improves efficiency".to_string(),
                goal_spec: None,
                quality_goal: None,
                minimum_quality_score: None,
                initial_files: BTreeMap::new(),
                focus: Vec::new(),
                max_steps: 8,
                reasoning_tools: Vec::new(),
                tool_mode: crate::SemanticAgentToolMode::Off,
            },
        );

        assert_eq!(run.status, SemanticAgentRunStatus::Completed);
        assert_eq!(run.stop_reason, "goal and quality satisfied");
        assert!(run.steps.len() >= 5);
        assert!(
            run.steps
                .iter()
                .all(|step| step.applied.is_some() || step.stop_reason.is_some())
        );
        let rendered = run
            .final_files
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("package HybridVehicle"));
        assert!(rendered.contains("part def HybridVehicle"));
        assert!(rendered.contains("part def Engine"));
        assert!(rendered.contains("part def ElectricMotor"));
        assert!(rendered.contains("part def BatteryPack"));
        assert!(rendered.contains("part engine: HybridVehicle::Engine"));
        assert!(rendered.contains("part motor: HybridVehicle::ElectricMotor"));
        assert!(rendered.contains("part battery: HybridVehicle::BatteryPack"));
        assert!(rendered.contains("requirement def ImproveEfficiency"));
        assert!(rendered.contains("part def RegenerativeBrakingSystem"));
        assert!(rendered.contains("satisfy requirement ImproveEfficiency"));
    }

    #[test]
    fn semantic_agent_adds_requirements_package_with_ten_requirements() {
        let provider = heuristic_provider();

        let run = run_semantic_mutation_agent(
            &provider,
            SemanticAgentRunRequest {
                goal: "add a requirements package with 10 requirements".to_string(),
                goal_spec: None,
                quality_goal: Some(default_model_quality_profile().goal),
                minimum_quality_score: Some(1.0),
                initial_files: BTreeMap::new(),
                focus: Vec::new(),
                max_steps: 3,
                reasoning_tools: Vec::new(),
                tool_mode: crate::SemanticAgentToolMode::Off,
            },
        );

        assert_eq!(run.status, SemanticAgentRunStatus::Completed, "{run:#?}");
        assert_eq!(run.stop_reason, "goal and quality satisfied");
        let rendered = run
            .final_files
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("package Requirements"));
        assert_eq!(rendered.matches("requirement def ").count(), 10);
        assert!(rendered.contains("requirement def FunctionalPerformance"));
        assert!(rendered.contains("doc /* id: REQ-001 */"));
        assert!(rendered.contains("doc /* The system shall deliver its primary function"));
        assert!(rendered.contains("requirement def VerificationEvidence"));
        assert!(rendered.contains("doc /* id: REQ-010 */"));
    }

    #[test]
    fn semantic_agent_can_complete_without_task_goal_spec() {
        let run = run_semantic_mutation_agent(
            &RequestRevisionProposalProvider,
            SemanticAgentRunRequest {
                goal: "Create a UAV interceptor".to_string(),
                goal_spec: None,
                quality_goal: Some(default_model_quality_profile().goal),
                minimum_quality_score: Some(0.5),
                initial_files: BTreeMap::from([(
                    "model.sysml".to_string(),
                    "package Demo {}".to_string(),
                )]),
                focus: Vec::new(),
                max_steps: 4,
                reasoning_tools: Vec::new(),
                tool_mode: crate::SemanticAgentToolMode::Off,
            },
        );

        assert_eq!(run.status, SemanticAgentRunStatus::Completed);
        assert_eq!(run.stop_reason, "goal and quality satisfied");
        assert!(
            run.final_files
                .get("model.sysml")
                .is_some_and(|source| source.contains("part def UAVInterceptor"))
        );
    }

    #[test]
    fn semantic_agent_auto_runs_reasoning_tools_before_proposing() {
        let run = run_semantic_mutation_agent(
            &RequestRevisionProposalProvider,
            SemanticAgentRunRequest {
                goal: "Create a UAV interceptor and improve requirement coverage".to_string(),
                goal_spec: None,
                quality_goal: Some(default_model_quality_profile().goal),
                minimum_quality_score: Some(0.5),
                initial_files: BTreeMap::from([(
                    "model.sysml".to_string(),
                    "package Demo {}".to_string(),
                )]),
                focus: Vec::new(),
                max_steps: 1,
                reasoning_tools: Vec::new(),
                tool_mode: crate::SemanticAgentToolMode::Auto,
            },
        );

        assert!(
            run.steps
                .first()
                .is_some_and(|step| step.tool_results.iter().any(|result| {
                    result.tool == crate::SemanticAgentToolKind::RequirementCoverage
                        && !result.summary.is_empty()
                })),
            "{run:#?}"
        );
    }

    #[test]
    fn semantic_agent_auto_runs_model_inspection_for_metamodel_questions() {
        let tools = crate::select_semantic_agent_tools(&SemanticAgentRunRequest {
            goal: "What are the attributes of metamodel Element?".to_string(),
            goal_spec: None,
            quality_goal: None,
            minimum_quality_score: None,
            initial_files: BTreeMap::new(),
            focus: Vec::new(),
            max_steps: 1,
            reasoning_tools: Vec::new(),
            tool_mode: crate::SemanticAgentToolMode::Auto,
        });

        assert!(tools.contains(&crate::SemanticAgentToolKind::ModelInspection));
    }

    #[test]
    #[ignore = "requires a configured external provider and spends tokens"]
    fn provider_semantic_mutation_smoke_returns_checked_proposal() {
        let provider = crate::default_reasoning_provider();
        let status = provider.provider_status();
        assert!(
            !matches!(status.kind, ReasoningProviderKind::Heuristic),
            "set MERCURIO_AI_PROVIDER=openai or azure_openai with provider credentials"
        );

        let context = MutationContext::from_project(hybrid_vehicle_project());
        let checked = propose_checked_semantic_mutations(
            &provider,
            &sysml_mutation_feasibility_service(),
            &context,
            &SemanticMutationProposalRequest {
                design_intent:
                    "Analyze this hybrid vehicle model and propose one SysML semantic mutation that improves efficiency."
                        .to_string(),
                workspace_revision: context.workspace_revision.clone(),
                focus: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
                task_goal_guidance: None,
                quality_goal_guidance: Some(explain_semantic_goal(
                    &default_model_quality_profile().goal,
                )),
                semantic_context: None,
                cognitive_context: None,
                reasoning_tool_results: Vec::new(),
            },
        );

        assert!(!checked.is_empty(), "provider returned no proposals");
        assert!(
            checked
                .iter()
                .any(|proposal| !proposal.proposal.operations.is_empty()),
            "provider returned only empty proposals"
        );
        assert!(
            checked
                .iter()
                .all(|proposal| proposal.feasibility.checked_against == context.workspace_revision),
            "provider proposals were not checked against the current workspace revision"
        );
        assert!(
            checked.iter().any(|proposal| matches!(
                proposal.feasibility.status,
                FeasibilityStatus::Allowed | FeasibilityStatus::AllowedWithWarnings
            )),
            "provider returned proposals, but none were feasible: {checked:#?}"
        );
    }

    #[test]
    #[ignore = "requires a configured external provider, spends tokens, and prints provider output"]
    fn provider_semantic_mutation_verbose_smoke_prints_checked_proposals() {
        let provider = crate::default_reasoning_provider();
        let status = provider.provider_status();
        assert!(
            !matches!(status.kind, ReasoningProviderKind::Heuristic),
            "set MERCURIO_AI_PROVIDER=openai or azure_openai with provider credentials"
        );
        println!(
            "provider: {} ({:?}) model={}",
            status.provider_label,
            status.kind,
            status.model_label.as_deref().unwrap_or("<none>")
        );

        let context = MutationContext::from_project(hybrid_vehicle_project());
        println!(
            "workspace revision: {}",
            context.workspace_revision.fingerprint
        );

        let request = SemanticMutationProposalRequest {
            design_intent:
                "Analyze this hybrid vehicle model and propose one SysML semantic mutation that improves efficiency."
                    .to_string(),
            workspace_revision: context.workspace_revision.clone(),
            focus: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
            task_goal_guidance: None,
            quality_goal_guidance: Some(explain_semantic_goal(
                &default_model_quality_profile().goal,
            )),
            semantic_context: None,
            cognitive_context: None,
            reasoning_tool_results: Vec::new(),
        };
        println!("design intent: {}", request.design_intent);
        println!(
            "focus: {}",
            request
                .focus
                .iter()
                .map(|focus| focus.qualified_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let checked = propose_checked_semantic_mutations(
            &provider,
            &sysml_mutation_feasibility_service(),
            &context,
            &request,
        );

        println!("proposal count: {}", checked.len());
        for (index, checked_proposal) in checked.iter().enumerate() {
            println!("--- proposal {} ---", index + 1);
            println!("proposal id: {:?}", checked_proposal.proposal_id);
            println!(
                "revision attempted: {}",
                checked_proposal.revision_attempted
            );
            println!("intent: {}", checked_proposal.proposal.intent);
            println!("rationale: {:?}", checked_proposal.proposal.rationale);
            println!(
                "affected elements: {}",
                checked_proposal
                    .proposal
                    .affected_elements
                    .iter()
                    .map(|element| element.qualified_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "evidence: {}",
                checked_proposal
                    .proposal
                    .evidence
                    .iter()
                    .map(|evidence| evidence.summary.as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
            println!("operations:");
            for (operation_index, operation) in
                checked_proposal.proposal.operations.iter().enumerate()
            {
                println!("  {}. {:?}", operation_index + 1, operation);
            }
            println!("feasibility: {:?}", checked_proposal.feasibility.status);
            if !checked_proposal.feasibility.blocking_reasons.is_empty() {
                println!("blocking reasons:");
                for issue in &checked_proposal.feasibility.blocking_reasons {
                    println!(
                        "  - {:?} op={:?}: {}",
                        issue.kind, issue.operation_index, issue.message
                    );
                }
            }
            if !checked_proposal.feasibility.warnings.is_empty() {
                println!("warnings:");
                for issue in &checked_proposal.feasibility.warnings {
                    println!(
                        "  - {:?} op={:?}: {}",
                        issue.kind, issue.operation_index, issue.message
                    );
                }
            }
            if !checked_proposal
                .feasibility
                .suggested_supporting_changes
                .is_empty()
            {
                println!("suggested supporting changes:");
                for operation in &checked_proposal.feasibility.suggested_supporting_changes {
                    println!("  - {:?}", operation);
                }
            }
            if let Some(diff) = &checked_proposal.feasibility.resulting_diff {
                println!("semantic diff: {:?}", diff);
            }
        }

        assert!(!checked.is_empty(), "provider returned no proposals");
        assert!(
            checked.iter().any(|proposal| matches!(
                proposal.feasibility.status,
                FeasibilityStatus::Allowed | FeasibilityStatus::AllowedWithWarnings
            )),
            "provider returned proposals, but none were feasible"
        );
    }

    #[test]
    #[ignore = "requires a configured external provider, spends tokens, and prints agent output"]
    fn provider_semantic_agent_hybrid_vehicle_from_empty_verbose_smoke() {
        let provider = crate::default_reasoning_provider();
        let status = provider.provider_status();
        assert!(
            !matches!(status.kind, ReasoningProviderKind::Heuristic),
            "set MERCURIO_AI_PROVIDER=openai or azure_openai with provider credentials"
        );
        println!(
            "provider: {} ({:?}) model={}",
            status.provider_label,
            status.kind,
            status.model_label.as_deref().unwrap_or("<none>")
        );

        let run = run_semantic_mutation_agent(
            &provider,
            SemanticAgentRunRequest {
                goal: "Create a minimal SysML v2 semantic model of a hybrid vehicle from an empty model. Build it through small checked semantic mutations. Include a vehicle part definition, engine, electric motor, battery pack, an efficiency requirement, and a regenerative braking concept that satisfies the efficiency requirement."
                    .to_string(),
                goal_spec: None,
                quality_goal: Some(default_model_quality_profile().goal),
                minimum_quality_score: Some(0.5),
                initial_files: BTreeMap::new(),
                focus: Vec::new(),
                max_steps: 8,
                reasoning_tools: Vec::new(),
                tool_mode: crate::SemanticAgentToolMode::Auto,
            },
        );

        println!("run status: {:?}", run.status);
        println!("stop reason: {}", run.stop_reason);
        println!(
            "final workspace revision: {}",
            run.final_workspace_revision.fingerprint
        );
        println!("step count: {}", run.steps.len());

        for step in &run.steps {
            println!("--- step {} ---", step.index + 1);
            println!(
                "workspace revision: {}",
                step.workspace_revision.fingerprint
            );
            println!(
                "context: elements={} relationships={} facts={} affordances={} truncated={}",
                step.semantic_context.elements.len(),
                step.semantic_context.relationships.len(),
                step.semantic_context.facts.len(),
                step.semantic_context.affordances.len(),
                step.semantic_context.truncated
            );
            println!("proposal count: {}", step.proposals.len());
            println!(
                "selected proposal: {}",
                step.selected_proposal_index
                    .map(|index| (index + 1).to_string())
                    .unwrap_or_else(|| "<none>".to_string())
            );
            if let Some(goal) = &step.goal_evaluation {
                println!(
                    "goal: satisfied={} score={:.3} policy={:?} checks={}",
                    goal.satisfied,
                    goal.score,
                    goal.policy,
                    goal.results.len()
                );
                for (goal_index, result) in goal.results.iter().enumerate() {
                    if !result.satisfied {
                        println!(
                            "  goal check {} unsatisfied: {:?} evidence={}",
                            goal_index + 1,
                            result.check,
                            result.evidence.join(" | ")
                        );
                    }
                }
            } else {
                println!("goal: <none>");
            }
            if let Some(quality) = &step.quality_evaluation {
                println!(
                    "quality: satisfied={} score={:.3} policy={:?} checks={}",
                    quality.satisfied,
                    quality.score,
                    quality.policy,
                    quality.results.len()
                );
                for (quality_index, result) in quality.results.iter().enumerate() {
                    if !result.satisfied {
                        println!(
                            "  quality check {} unsatisfied: {:?} evidence={}",
                            quality_index + 1,
                            result.check,
                            result.evidence.join(" | ")
                        );
                    }
                }
            } else {
                println!("quality: <none>");
            }
            for (proposal_index, checked_proposal) in step.proposals.iter().enumerate() {
                println!("  proposal {}:", proposal_index + 1);
                println!("    intent: {}", checked_proposal.proposal.intent);
                println!("    rationale: {:?}", checked_proposal.proposal.rationale);
                println!("    feasibility: {:?}", checked_proposal.feasibility.status);
                println!(
                    "    revision attempted: {}",
                    checked_proposal.revision_attempted
                );
                println!("    operations:");
                for (operation_index, operation) in
                    checked_proposal.proposal.operations.iter().enumerate()
                {
                    println!("      {}. {:?}", operation_index + 1, operation);
                }
                if !checked_proposal.feasibility.blocking_reasons.is_empty() {
                    println!("    blocking reasons:");
                    for issue in &checked_proposal.feasibility.blocking_reasons {
                        println!(
                            "      - {:?} op={:?}: {}",
                            issue.kind, issue.operation_index, issue.message
                        );
                    }
                }
                if !checked_proposal.feasibility.warnings.is_empty() {
                    println!("    warnings:");
                    for issue in &checked_proposal.feasibility.warnings {
                        println!(
                            "      - {:?} op={:?}: {}",
                            issue.kind, issue.operation_index, issue.message
                        );
                    }
                }
                if let Some(diff) = &checked_proposal.feasibility.resulting_diff {
                    println!("    semantic diff: {:?}", diff);
                }
            }
            if let Some(applied) = &step.applied {
                println!("applied changed files: {:?}", applied.changed_files);
                println!(
                    "applied changed declarations: {:?}",
                    applied.changed_declarations
                );
                println!("applied semantic diff: {:?}", applied.semantic_diff);
            }
            if let Some(stop_reason) = &step.stop_reason {
                println!("step stop reason: {stop_reason}");
            }
        }

        println!("--- final files ---");
        for (path, content) in &run.final_files {
            println!("### {path}");
            println!("{content}");
        }

        assert!(
            matches!(
                run.status,
                SemanticAgentRunStatus::Completed | SemanticAgentRunStatus::Stopped
            ),
            "agent failed: {run:#?}"
        );
        assert!(
            run.steps.iter().any(|step| step.applied.is_some()),
            "provider did not produce any applied mutation"
        );
    }

    #[test]
    fn configured_azure_test_does_not_fall_back_to_heuristic() {
        let result = test_configured_reasoning_provider_connection(
            ReasoningProviderConfigOverrides {
                provider: Some(ReasoningProviderKind::AzureOpenAi),
                azure_openai_deployment: Some("test-mini".to_string()),
                azure_openai_base_url: Some("https://example.openai.azure.com".to_string()),
                ..ReasoningProviderConfigOverrides::default()
            },
            ReasoningProviderSecretOverrides::default(),
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Azure OpenAI settings are incomplete")
        );
    }

    #[test]
    fn configured_anthropic_provider_uses_separate_models() {
        let provider = configured_reasoning_provider(
            ReasoningProviderConfigOverrides {
                provider: Some(ReasoningProviderKind::Anthropic),
                anthropic_proposal_model: Some("claude-opus-4-8".to_string()),
                anthropic_fast_model: Some("claude-sonnet-4-6".to_string()),
                ..ReasoningProviderConfigOverrides::default()
            },
            ReasoningProviderSecretOverrides {
                anthropic_api_key: Some("test-key".to_string()),
                ..ReasoningProviderSecretOverrides::default()
            },
        );

        let status = provider.provider_status();
        assert_eq!(status.kind, ReasoningProviderKind::Anthropic);
        assert_eq!(
            status.model_label.as_deref(),
            Some("claude-opus-4-8 / claude-sonnet-4-6")
        );
    }

    #[test]
    fn extract_output_text_reads_structured_response() {
        let response: OpenAiStructuredResponse = serde_json::from_value(json!({
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "{\"title\":\"ok\",\"body\":[\"careful\"]}"
                        }
                    ]
                }
            ]
        }))
        .unwrap();

        let output = extract_output_text(&response).unwrap();
        assert!(output.contains("\"title\""));
    }

    #[test]
    fn extract_anthropic_tool_input_reads_structured_response() {
        let response: AnthropicMessageResponse = serde_json::from_value(json!({
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_test",
                    "name": "emit_connection_probe",
                    "input": { "ok": true }
                }
            ]
        }))
        .unwrap();

        assert_eq!(
            extract_anthropic_tool_input(&response).unwrap(),
            json!({ "ok": true })
        );
    }

    #[test]
    fn normalize_azure_openai_base_url_accepts_endpoint_or_v1_base() {
        assert_eq!(
            normalize_azure_openai_base_url("https://example.openai.azure.com"),
            "https://example.openai.azure.com/openai/v1/responses"
        );
        assert_eq!(
            normalize_azure_openai_base_url("https://example.openai.azure.com/openai/v1/"),
            "https://example.openai.azure.com/openai/v1/responses"
        );
        assert_eq!(
            normalize_azure_openai_base_url("https://example.openai.azure.com/openai/v1/responses"),
            "https://example.openai.azure.com/openai/v1/responses"
        );
    }

    #[test]
    fn ask_mercurio_classifies_supported_tasks() {
        assert_eq!(
            classify_ask_mercurio_task("Create a dependency diagram for the camera model"),
            AskMercurioTask::DiagramRequest
        );
        assert_eq!(
            classify_ask_mercurio_task("Create a requirements table view"),
            AskMercurioTask::ViewRequest
        );
        assert_eq!(
            classify_ask_mercurio_task("Draft a pull request for this update"),
            AskMercurioTask::PrDraft
        );
        assert_eq!(
            classify_ask_mercurio_task("What design tradeoff is represented here?"),
            AskMercurioTask::DesignQuestion
        );
    }

    #[test]
    fn ask_mercurio_pr_task_returns_draft_only_artifact() {
        let artifacts = ask_mercurio_artifacts(
            &AskMercurioTask::PrDraft,
            None,
            "Draft a proposal for a brake model update",
        );

        let Some(AskMercurioArtifact::ProposalDraft(draft)) = artifacts.first() else {
            panic!("expected proposal draft artifact");
        };
        assert!(draft.title.contains("Draft:"));
        assert!(
            draft
                .suggested_head_branch
                .as_deref()
                .unwrap_or_default()
                .starts_with("ask-mercurio/")
        );
        assert!(draft.body.contains("No selected project context"));
        assert!(
            draft
                .checklist
                .iter()
                .any(|item| item.contains("semantic impact"))
        );
    }

    #[test]
    fn ask_mercurio_view_task_returns_requirements_view_artifact() {
        let artifacts = ask_mercurio_artifacts(
            &AskMercurioTask::ViewRequest,
            None,
            "Show me a requirements table",
        );

        let Some(AskMercurioArtifact::RequirementsView(view)) = artifacts.first() else {
            panic!("expected requirements view artifact");
        };
        assert_eq!(view["kind"], "requirements_table");
        assert_eq!(view["endpoint"], "/api/views/requirements-table");
    }
}
