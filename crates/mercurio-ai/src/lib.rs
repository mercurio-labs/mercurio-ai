use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::time::Duration;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use mercurio_core::runtime::Runtime;
use mercurio_core::{
    AnalysisScope, CapabilityRegistry, CapabilityRunReport, CapabilityRunRequest, CapabilityTarget,
    SemanticWorkspaceSnapshot,
};
use mercurio_reasoner_api::{
    FindingSeverity, ReasoningReport, SemanticArtifactRef, SemanticContextKind, SemanticContextRef,
};
use mercurio_reference_capabilities::{
    analyze_requirement_coverage, analyze_semantic_impact, analyze_state_machine_simulation,
};

pub use mercurio_core::{
    CognitiveContext, CognitiveDiagnostic, CognitiveDiagnosticSeverity, CognitiveElement,
    CognitiveFocus, CognitiveRelationship, CoreMutationFeasibilityService, DesignIntent, Edge,
    Element, ElementRef, FeasibilityStatus, GoalEvaluation, Graph, KirDocument,
    MutationApplicationResult, MutationContext, MutationEvidence, MutationFeasibilityReport,
    MutationFeasibilityService, MutationProposal, NodeId, SemanticArtifact, SemanticElementRef,
    SemanticExpression, SemanticGoalCheck, SemanticGoalExplanation, SemanticGoalSpec,
    SemanticMutation, SemanticMutationCapabilityContext, SemanticReasoningContext,
    SemanticWorkspaceRef, SourceSpanRef, WorkspaceRevision, default_stdlib_path, stable_digest,
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
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningProviderKind {
    Heuristic,
    OpenAi,
    AzureOpenAi,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningProviderStatus {
    pub kind: ReasoningProviderKind,
    pub provider_label: String,
    pub detail: String,
    pub structured_outputs: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_label: Option<String>,
}

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
pub enum ChatMessageRole {
    Developer,
    Assistant,
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatMessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionResponse {
    pub message: String,
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

#[derive(Debug, Deserialize)]
struct OpenAiStructuredResponse {
    output: Vec<OpenAiOutputItem>,
}

#[derive(Debug, Deserialize)]
struct OpenAiOutputItem {
    #[serde(default)]
    content: Vec<OpenAiContentItem>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum OpenAiContentItem {
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct SemanticSummaryEnvelope {
    title: String,
    body: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ConnectionProbeEnvelope {
    ok: bool,
}

#[derive(Debug, Deserialize)]
struct SemanticMutationProposalEnvelope {
    proposals: Vec<MutationProposal>,
}

#[derive(Debug, Clone)]
pub enum ResolvedReasoningProvider {
    Heuristic(HeuristicReasoningProvider),
    OpenAi(OpenAiReasoningProvider),
    AzureOpenAi(AzureOpenAiReasoningProvider),
}

#[derive(Debug, Clone)]
pub struct HeuristicReasoningProvider {
    status: ReasoningProviderStatus,
}

#[derive(Debug, Clone)]
pub struct OpenAiReasoningProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    status: ReasoningProviderStatus,
    fallback: HeuristicReasoningProvider,
}

#[derive(Debug, Clone)]
pub struct AzureOpenAiReasoningProvider {
    client: Client,
    api_key: String,
    deployment: String,
    base_url: String,
    status: ReasoningProviderStatus,
    fallback: HeuristicReasoningProvider,
}

pub trait ReasoningProvider {
    fn provider_status(&self) -> ReasoningProviderStatus;

    fn test_connection(&self) -> Result<ReasoningProviderStatus, String>;

    fn summarize_semantic_changes(
        &self,
        request: &SemanticSummaryRequest,
    ) -> SemanticSummaryResponse;

    fn complete_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String>;
}

pub trait SemanticMutationProposalProvider {
    fn propose_semantic_mutations(
        &self,
        request: &SemanticMutationProposalRequest,
    ) -> Vec<MutationProposal>;
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

#[derive(Debug, Clone)]
pub struct SemanticContextBuilder {
    max_depth: usize,
    max_elements: usize,
    max_relationships: usize,
}

impl Default for SemanticContextBuilder {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_elements: 96,
            max_relationships: 192,
        }
    }
}

impl SemanticContextBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = max_elements;
        self
    }

    pub fn max_relationships(mut self, max_relationships: usize) -> Self {
        self.max_relationships = max_relationships;
        self
    }

    pub fn build_from_project(
        &self,
        project: &mercurio_core::AuthoringProject,
        workspace_revision: WorkspaceRevision,
        focus: &[ElementRef],
        reasoning_tool_results: &[SemanticAgentToolResult],
    ) -> Result<CognitiveContext, String> {
        let document = project
            .compile_kir_document()
            .map_err(|err| format!("failed to compile semantic context KIR: {err}"))?;
        self.build_from_document(&document, workspace_revision, focus, reasoning_tool_results)
    }

    pub fn build_from_document(
        &self,
        document: &KirDocument,
        workspace_revision: WorkspaceRevision,
        focus: &[ElementRef],
        reasoning_tool_results: &[SemanticAgentToolResult],
    ) -> Result<CognitiveContext, String> {
        let graph = Graph::from_document(document.clone())
            .map_err(|err| format!("failed to build semantic graph: {err}"))?;
        Ok(self.build_from_graph(
            &graph,
            workspace_revision,
            focus,
            reasoning_tool_results,
            document
                .elements
                .iter()
                .filter_map(|element| ai_string_property(&element.properties, "source_file"))
                .collect(),
        ))
    }

    pub fn build_from_graph(
        &self,
        graph: &Graph,
        workspace_revision: WorkspaceRevision,
        focus: &[ElementRef],
        reasoning_tool_results: &[SemanticAgentToolResult],
        source_files: BTreeSet<String>,
    ) -> CognitiveContext {
        let focus_nodes = resolve_focus_nodes(graph, focus);
        let mut selected_nodes = BTreeSet::new();
        let mut queue = VecDeque::new();
        let mut truncated = false;

        if focus_nodes.is_empty() {
            for element in graph.elements().iter().take(self.max_elements) {
                selected_nodes.insert(element.id);
            }
            truncated = graph.elements().len() > selected_nodes.len();
        } else {
            for node in &focus_nodes {
                selected_nodes.insert(*node);
                queue.push_back((*node, 0usize));
            }
            while let Some((node, depth)) = queue.pop_front() {
                if depth >= self.max_depth {
                    continue;
                }
                for edge in graph.outgoing_edges(node).chain(graph.incoming_edges(node)) {
                    for adjacent in [edge.source, edge.target] {
                        if selected_nodes.contains(&adjacent) {
                            continue;
                        }
                        if selected_nodes.len() >= self.max_elements {
                            truncated = true;
                            continue;
                        }
                        selected_nodes.insert(adjacent);
                        queue.push_back((adjacent, depth + 1));
                    }
                }
            }
        }

        let focus_refs = focus_nodes
            .iter()
            .filter_map(|node| graph.element(*node).map(ai_semantic_element_ref))
            .collect::<Vec<_>>();
        let cognitive_focus = if focus_refs.is_empty() {
            CognitiveFocus::workspace()
        } else {
            CognitiveFocus::elements(focus_refs)
        };

        let mut elements = graph
            .elements()
            .iter()
            .filter(|element| selected_nodes.contains(&element.id))
            .map(ai_cognitive_element)
            .collect::<Vec<_>>();
        elements.sort_by_key(|element| {
            (
                !cognitive_focus
                    .elements
                    .iter()
                    .any(|focus| focus.element_id == element.element.element_id),
                element.layer,
                element.element.element_id.clone(),
            )
        });

        let mut relationships = Vec::new();
        for (index, edge) in graph.edges().iter().enumerate() {
            let endpoint_selected =
                selected_nodes.contains(&edge.source) || selected_nodes.contains(&edge.target);
            if !endpoint_selected {
                continue;
            }
            if relationships.len() >= self.max_relationships {
                truncated = true;
                break;
            }
            let Some(source) = graph.element(edge.source) else {
                continue;
            };
            let Some(target) = graph.element(edge.target) else {
                continue;
            };
            relationships.push(ai_cognitive_relationship(index, edge, source, target));
        }

        let mut source_files = source_files;
        for element in &elements {
            for span in &element.source_spans {
                if !span.file.is_empty() {
                    source_files.insert(span.file.clone());
                }
            }
        }

        let diagnostics = cognitive_diagnostics_from_tool_results(graph, reasoning_tool_results);
        let artifacts = cognitive_artifacts_from_tool_results(graph, reasoning_tool_results);

        CognitiveContext {
            workspace: Some(SemanticWorkspaceRef {
                revision: workspace_revision,
                profile_id: Some("sysml".to_string()),
            }),
            focus: cognitive_focus,
            elements,
            relationships,
            diagnostics,
            artifacts,
            source_files: source_files.into_iter().collect(),
            history: Vec::new(),
            truncated,
        }
    }
}

pub fn run_semantic_mutation_agent<P>(
    provider: &P,
    request: SemanticAgentRunRequest,
) -> SemanticAgentRun
where
    P: SemanticMutationProposalProvider,
{
    let selected_tools = select_semantic_agent_tools(&request);
    let mut files = request.initial_files;
    let mut project = match load_authoring_project_from_sysml(files.clone()) {
        Ok(project) => project,
        Err(err) => {
            return SemanticAgentRun {
                goal: request.goal,
                status: SemanticAgentRunStatus::Failed,
                stop_reason: format!("failed to load initial SysML: {err}"),
                steps: Vec::new(),
                final_files: files,
                final_workspace_revision: WorkspaceRevision::unchecked(),
            };
        }
    };
    let feasibility = sysml_mutation_feasibility_service();
    let mut steps = Vec::new();
    let max_steps = request.max_steps.max(1);
    let goal_spec = request
        .goal_spec
        .clone()
        .or_else(|| default_semantic_agent_goal_spec(&request.goal));
    let quality_goal = request.quality_goal.clone();
    let minimum_quality_score = request.minimum_quality_score;

    for index in 0..max_steps {
        let context = MutationContext::from_project(project);
        let mut semantic_context = sysml_semantic_reasoning_context_from_authoring_project(
            &context.project,
            context.workspace_revision.clone(),
            request.focus.clone(),
            128,
        );
        enrich_sysml_semantic_reasoning_context_with_child_affordances(&mut semantic_context, 192);
        let tool_results = run_semantic_agent_tools(
            &selected_tools,
            &files,
            &context.workspace_revision,
            &request.goal,
            index,
        );
        let cognitive_context = match SemanticContextBuilder::default().build_from_project(
            &context.project,
            context.workspace_revision.clone(),
            &request.focus,
            &tool_results,
        ) {
            Ok(context) => Some(context),
            Err(err) => {
                return SemanticAgentRun {
                    goal: request.goal,
                    status: SemanticAgentRunStatus::Failed,
                    stop_reason: err,
                    steps,
                    final_files: files,
                    final_workspace_revision: context.workspace_revision,
                };
            }
        };
        let proposal_request = SemanticMutationProposalRequest {
            design_intent: request.goal.clone(),
            workspace_revision: context.workspace_revision.clone(),
            focus: request.focus.clone(),
            task_goal_guidance: goal_spec.as_ref().map(explain_semantic_goal),
            quality_goal_guidance: quality_goal.as_ref().map(explain_semantic_goal),
            semantic_context: Some(semantic_context.clone()),
            cognitive_context,
            reasoning_tool_results: tool_results.clone(),
        };
        let proposals =
            propose_checked_semantic_mutations(provider, &feasibility, &context, &proposal_request);
        let Some((selected_index, selected)) =
            proposals.iter().enumerate().find(|(_, proposal)| {
                matches!(
                    proposal.feasibility.status,
                    FeasibilityStatus::Allowed | FeasibilityStatus::AllowedWithWarnings
                ) && proposal
                    .feasibility
                    .normalized_plan
                    .as_ref()
                    .is_some_and(|plan| !plan.normalized_operations.is_empty())
            })
        else {
            let stop_reason = if proposals.is_empty() {
                "provider returned no proposals".to_string()
            } else {
                "no feasible proposal was available".to_string()
            };
            let revision = context.workspace_revision.clone();
            steps.push(SemanticAgentStep {
                index,
                workspace_revision: revision.clone(),
                semantic_context,
                goal_evaluation: evaluate_current_goal(
                    goal_spec.as_ref(),
                    &context.project,
                    &request.focus,
                ),
                quality_evaluation: evaluate_current_goal(
                    quality_goal.as_ref(),
                    &context.project,
                    &request.focus,
                ),
                tool_results,
                proposals,
                selected_proposal_index: None,
                applied: None,
                stop_reason: Some(stop_reason.clone()),
            });
            return SemanticAgentRun {
                goal: request.goal,
                status: SemanticAgentRunStatus::Stopped,
                stop_reason,
                steps,
                final_files: files,
                final_workspace_revision: revision,
            };
        };

        let plan = selected
            .feasibility
            .normalized_plan
            .as_ref()
            .expect("checked above");
        let applied = match feasibility.apply_checked_plan(&context, plan) {
            Ok(applied) => applied,
            Err(err) => {
                let stop_reason = format!("failed to apply checked plan: {}", err.message);
                let revision = context.workspace_revision.clone();
                steps.push(SemanticAgentStep {
                    index,
                    workspace_revision: revision.clone(),
                    semantic_context,
                    goal_evaluation: evaluate_current_goal(
                        goal_spec.as_ref(),
                        &context.project,
                        &request.focus,
                    ),
                    quality_evaluation: evaluate_current_goal(
                        quality_goal.as_ref(),
                        &context.project,
                        &request.focus,
                    ),
                    tool_results,
                    proposals,
                    selected_proposal_index: Some(selected_index),
                    applied: None,
                    stop_reason: Some(stop_reason.clone()),
                });
                return SemanticAgentRun {
                    goal: request.goal,
                    status: SemanticAgentRunStatus::Failed,
                    stop_reason,
                    steps,
                    final_files: files,
                    final_workspace_revision: revision,
                };
            }
        };

        files.extend(applied.edited_files.clone());
        project = match load_authoring_project_from_sysml(files.clone()) {
            Ok(project) => project,
            Err(err) => {
                let stop_reason = format!("applied mutation produced invalid SysML: {err}");
                let revision = context.workspace_revision.clone();
                steps.push(SemanticAgentStep {
                    index,
                    workspace_revision: revision.clone(),
                    semantic_context,
                    goal_evaluation: evaluate_current_goal(
                        goal_spec.as_ref(),
                        &context.project,
                        &request.focus,
                    ),
                    quality_evaluation: evaluate_current_goal(
                        quality_goal.as_ref(),
                        &context.project,
                        &request.focus,
                    ),
                    tool_results,
                    proposals,
                    selected_proposal_index: Some(selected_index),
                    applied: Some(applied),
                    stop_reason: Some(stop_reason.clone()),
                });
                return SemanticAgentRun {
                    goal: request.goal,
                    status: SemanticAgentRunStatus::Failed,
                    stop_reason,
                    steps,
                    final_files: files,
                    final_workspace_revision: revision,
                };
            }
        };

        let goal_evaluation = evaluate_current_goal(goal_spec.as_ref(), &project, &request.focus);
        let quality_evaluation =
            evaluate_current_goal(quality_goal.as_ref(), &project, &request.focus);
        let goal_satisfied = goal_evaluation
            .as_ref()
            .is_none_or(|evaluation| evaluation.satisfied);
        let quality_satisfied = minimum_quality_score.is_none_or(|minimum_score| {
            quality_evaluation
                .as_ref()
                .is_some_and(|evaluation| evaluation.score >= minimum_score)
        });
        steps.push(SemanticAgentStep {
            index,
            workspace_revision: context.workspace_revision.clone(),
            semantic_context,
            goal_evaluation,
            quality_evaluation,
            tool_results,
            proposals,
            selected_proposal_index: Some(selected_index),
            applied: Some(applied),
            stop_reason: (goal_satisfied && quality_satisfied)
                .then(|| "goal and quality satisfied".to_string())
                .or_else(|| goal_satisfied.then(|| "goal satisfied".to_string())),
        });
        if goal_satisfied && quality_satisfied {
            let final_context = MutationContext::from_project(project);
            return SemanticAgentRun {
                goal: request.goal,
                status: SemanticAgentRunStatus::Completed,
                stop_reason: "goal and quality satisfied".to_string(),
                steps,
                final_files: files,
                final_workspace_revision: final_context.workspace_revision,
            };
        }
    }

    let final_context = MutationContext::from_project(project);
    SemanticAgentRun {
        goal: request.goal,
        status: SemanticAgentRunStatus::Stopped,
        stop_reason: "max steps reached".to_string(),
        steps,
        final_files: files,
        final_workspace_revision: final_context.workspace_revision,
    }
}

fn evaluate_current_goal(
    goal: Option<&SemanticGoalSpec>,
    project: &mercurio_core::AuthoringProject,
    focus: &[ElementRef],
) -> Option<GoalEvaluation> {
    let goal = goal?;
    let context = MutationContext::from_project(project.clone());
    let semantic_context = sysml_semantic_reasoning_context_from_authoring_project(
        &context.project,
        context.workspace_revision,
        focus.to_vec(),
        128,
    );
    Some(evaluate_semantic_goal(&semantic_context, goal))
}

fn select_semantic_agent_tools(request: &SemanticAgentRunRequest) -> Vec<SemanticAgentToolKind> {
    let mut tools = request
        .reasoning_tools
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    match request.tool_mode {
        SemanticAgentToolMode::Off => return Vec::new(),
        SemanticAgentToolMode::RequestedOnly => {}
        SemanticAgentToolMode::Auto => {
            let goal = request.goal.to_ascii_lowercase();
            if goal.contains("requirement")
                || goal.contains("satisfy")
                || goal.contains("verify")
                || goal.contains("coverage")
            {
                tools.insert(SemanticAgentToolKind::RequirementCoverage);
            }
            if goal.contains("impact")
                || goal.contains("risk")
                || goal.contains("change")
                || goal.contains("affected")
            {
                tools.insert(SemanticAgentToolKind::SemanticImpact);
            }
            if goal.contains("state")
                || goal.contains("transition")
                || goal.contains("simulate")
                || goal.contains("simulation")
            {
                tools.insert(SemanticAgentToolKind::StateSimulation);
            }
            if goal.contains("metamodel")
                || goal.contains("metatype")
                || goal.contains("inspect")
                || goal.contains("lookup")
                || goal.contains("attribute")
                || goal.contains("attributes")
                || goal.contains("property")
                || goal.contains("properties")
                || goal.contains("what is")
                || goal.contains("what are")
            {
                tools.insert(SemanticAgentToolKind::ModelInspection);
            }
        }
    }
    tools.into_iter().collect()
}

fn run_semantic_agent_tools(
    tools: &[SemanticAgentToolKind],
    files: &BTreeMap<String, String>,
    workspace_revision: &WorkspaceRevision,
    goal: &str,
    step_index: usize,
) -> Vec<SemanticAgentToolResult> {
    if tools.is_empty() {
        return Vec::new();
    }
    if tools
        .iter()
        .all(|tool| *tool == SemanticAgentToolKind::ModelInspection)
    {
        return tools
            .iter()
            .copied()
            .map(|tool| run_model_inspection_tool(tool, files, goal, step_index))
            .collect();
    }
    let document = match compile_agent_kir_document(files) {
        Ok(document) => document,
        Err(err) => {
            return tools
                .iter()
                .copied()
                .map(|tool| tool_error_result(tool, err.clone()))
                .collect();
        }
    };
    let runtime = match Runtime::from_document(document) {
        Ok(runtime) => runtime,
        Err(err) => {
            return tools
                .iter()
                .copied()
                .map(|tool| tool_error_result(tool, format!("failed to build runtime: {err}")))
                .collect();
        }
    };
    let context = agent_tool_context(workspace_revision);
    tools
        .iter()
        .copied()
        .map(|tool| {
            let request_id = format!(
                "semantic_agent.step{step_index}.{}",
                semantic_agent_tool_id(tool)
            );
            let report = match tool {
                SemanticAgentToolKind::RequirementCoverage => {
                    analyze_requirement_coverage(&runtime, context.clone(), request_id)
                }
                SemanticAgentToolKind::SemanticImpact => {
                    analyze_semantic_impact(&runtime, context.clone(), request_id)
                }
                SemanticAgentToolKind::StateSimulation => {
                    analyze_state_machine_simulation(&runtime, context.clone(), request_id)
                }
                SemanticAgentToolKind::ModelInspection => {
                    return run_model_inspection_tool(tool, files, goal, step_index);
                }
            };
            tool_result_from_report(tool, report)
        })
        .collect()
}

fn run_model_inspection_tool(
    tool: SemanticAgentToolKind,
    files: &BTreeMap<String, String>,
    goal: &str,
    step_index: usize,
) -> SemanticAgentToolResult {
    let snapshot = match compile_agent_inspection_snapshot(files) {
        Ok(snapshot) => snapshot,
        Err(err) => return tool_error_result(tool, err),
    };
    let registry = CapabilityRegistry::with_foundation_builtins();
    let run_id = format!(
        "semantic_agent.step{step_index}.{}",
        semantic_agent_tool_id(tool)
    );
    let report = match registry.run(
        &snapshot,
        CapabilityRunRequest {
            run_id,
            capability_id: "foundation.inspect.model".to_string(),
            target: CapabilityTarget::Workspace,
            parameters: BTreeMap::from([
                ("query".to_string(), Value::String(goal.to_string())),
                ("limit".to_string(), Value::from(8)),
                (
                    "analysis_scope".to_string(),
                    Value::String(model_inspection_analysis_scope(goal).as_str().to_string()),
                ),
            ]),
            input_artifacts: Vec::new(),
        },
    ) {
        Ok(report) => report,
        Err(err) => return tool_error_result(tool, err.to_string()),
    };
    tool_result_from_capability_report(tool, report)
}

fn model_inspection_analysis_scope(goal: &str) -> AnalysisScope {
    let normalized = goal.to_ascii_lowercase();
    let mentions_metamodel = normalized.contains("metamodel")
        || normalized.contains("meta-model")
        || normalized.contains("kerml")
        || normalized.contains("what is element")
        || normalized.contains("element's attributes");
    let mentions_stdlib = normalized.contains("sysml library")
        || normalized.contains("standard library")
        || normalized.contains("stdlib");
    if mentions_metamodel && mentions_stdlib {
        AnalysisScope::All
    } else if mentions_metamodel {
        AnalysisScope::Metamodel
    } else if mentions_stdlib {
        AnalysisScope::Stdlib
    } else {
        AnalysisScope::AuthoredModel
    }
}

fn compile_agent_inspection_snapshot(
    files: &BTreeMap<String, String>,
) -> Result<SemanticWorkspaceSnapshot, String> {
    let stdlib = KirDocument::from_path(Path::new(&default_stdlib_path()))
        .map_err(|err| format!("failed to load bundled stdlib: {err}"))?;
    let mut documents = vec![stdlib.clone()];
    for (path, content) in files {
        let document = compile_sysml_text(content, path, &stdlib)
            .map_err(|err| format!("failed to compile {path}: {err}"))?;
        documents.push(document);
    }
    let document = KirDocument::merge(documents)
        .map_err(|err| format!("failed to merge KIR documents: {err}"))?;
    SemanticWorkspaceSnapshot::from_document_with_profile(document, Some("sysml".to_string()))
        .map_err(|err| format!("failed to build inspection snapshot: {err}"))
}

fn compile_agent_kir_document(files: &BTreeMap<String, String>) -> Result<KirDocument, String> {
    let stdlib = KirDocument::from_path(Path::new(&default_stdlib_path()))
        .map_err(|err| format!("failed to load bundled stdlib: {err}"))?;
    let mut documents = Vec::new();
    for (path, content) in files {
        let document = compile_sysml_text(content, path, &stdlib)
            .map_err(|err| format!("failed to compile {path}: {err}"))?;
        documents.push(document);
    }
    KirDocument::merge(documents).map_err(|err| format!("failed to merge KIR documents: {err}"))
}

fn agent_tool_context(workspace_revision: &WorkspaceRevision) -> SemanticContextRef {
    SemanticContextRef {
        context_id: "semantic_agent".to_string(),
        kind: SemanticContextKind::DraftOverlay {
            overlay_digest: workspace_revision.fingerprint.clone(),
        },
        artifact: SemanticArtifactRef {
            artifact_key: format!("semantic-agent:{}", workspace_revision.fingerprint),
            kir_schema_version: "mercurio.kir.v1".to_string(),
            source_authority: Some("semantic_agent".to_string()),
            source_revision: Some(workspace_revision.fingerprint.clone()),
        },
    }
}

fn tool_result_from_report(
    tool: SemanticAgentToolKind,
    report: ReasoningReport,
) -> SemanticAgentToolResult {
    let summary = reasoning_report_summary(&report);
    let findings = report
        .findings
        .iter()
        .take(12)
        .map(|finding| SemanticAgentToolFinding {
            id: finding.id.clone(),
            severity: severity_label(&finding.severity).to_string(),
            title: finding.title.clone(),
            message: finding.message.clone(),
            elements: finding
                .elements
                .iter()
                .map(|element| {
                    ElementRef::new(
                        element
                            .qualified_name
                            .clone()
                            .unwrap_or_else(|| element.element_id.clone()),
                    )
                })
                .collect(),
        })
        .collect();
    let artifact = json!({
        "requestId": report.request_id,
        "capability": report.capability.id,
        "status": report.status.clone(),
        "artifacts": report.artifacts,
        "evidenceNodeCount": report.evidence.nodes.len(),
        "evidenceEdgeCount": report.evidence.edges.len(),
    });
    SemanticAgentToolResult {
        tool,
        status: serde_json::to_value(&report.status)
            .ok()
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_else(|| format!("{:?}", report.status)),
        summary,
        findings,
        artifact,
    }
}

fn tool_result_from_capability_report(
    tool: SemanticAgentToolKind,
    report: CapabilityRunReport,
) -> SemanticAgentToolResult {
    let summary = capability_report_summary(&report);
    let findings = report
        .insights
        .iter()
        .take(12)
        .map(|insight| SemanticAgentToolFinding {
            id: insight.id.clone(),
            severity: format!("{:?}", insight.severity).to_ascii_lowercase(),
            title: format!("{:?}", insight.kind),
            message: insight.claim.clone(),
            elements: vec![ElementRef::new(
                insight
                    .subject
                    .qualified_name
                    .clone()
                    .unwrap_or_else(|| insight.subject.element_id.clone()),
            )],
        })
        .collect();
    let artifact = json!({
        "runId": report.run_id,
        "capability": report.capability_id,
        "status": format!("{:?}", report.status),
        "artifacts": report.artifacts,
        "evidenceNodeCount": report.evidence.nodes.len(),
        "evidenceEdgeCount": report.evidence.edges.len(),
    });
    SemanticAgentToolResult {
        tool,
        status: format!("{:?}", report.status).to_ascii_lowercase(),
        summary,
        findings,
        artifact,
    }
}

fn reasoning_report_summary(report: &ReasoningReport) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "{} returned {:?} with {} finding(s).",
        report.capability.name,
        report.status,
        report.findings.len()
    ));
    for artifact in &report.artifacts {
        if let Some(object) = artifact.payload.as_object() {
            let parts = object
                .iter()
                .take(6)
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>();
            if !parts.is_empty() {
                lines.push(format!("{}: {}", artifact.kind, parts.join(", ")));
            }
        }
    }
    for finding in report.findings.iter().take(5) {
        lines.push(format!(
            "[{}] {}: {}",
            severity_label(&finding.severity),
            finding.title,
            finding.message
        ));
    }
    lines
}

fn capability_report_summary(report: &CapabilityRunReport) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "{} returned {:?} with {} insight(s).",
        report.capability_id,
        report.status,
        report.insights.len()
    ));
    for artifact in &report.artifacts {
        if let Some(object) = artifact.payload.as_object() {
            let parts = object
                .iter()
                .take(8)
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>();
            if !parts.is_empty() {
                lines.push(format!("{}: {}", artifact.kind, parts.join(", ")));
            }
        }
    }
    for insight in report.insights.iter().take(5) {
        lines.push(insight.claim.clone());
    }
    lines
}

fn tool_error_result(tool: SemanticAgentToolKind, message: String) -> SemanticAgentToolResult {
    SemanticAgentToolResult {
        tool,
        status: "error".to_string(),
        summary: vec![message.clone()],
        findings: vec![SemanticAgentToolFinding {
            id: format!("semantic_agent.tool_error.{}", semantic_agent_tool_id(tool)),
            severity: "error".to_string(),
            title: "Reasoning tool failed".to_string(),
            message,
            elements: Vec::new(),
        }],
        artifact: json!({}),
    }
}

fn semantic_agent_tool_id(tool: SemanticAgentToolKind) -> &'static str {
    match tool {
        SemanticAgentToolKind::RequirementCoverage => "requirement_coverage",
        SemanticAgentToolKind::SemanticImpact => "semantic_impact",
        SemanticAgentToolKind::StateSimulation => "state_simulation",
        SemanticAgentToolKind::ModelInspection => "model_inspection",
    }
}

fn resolve_focus_nodes(graph: &Graph, focus: &[ElementRef]) -> Vec<NodeId> {
    let mut nodes = Vec::new();
    let mut seen = BTreeSet::new();
    for element_ref in focus {
        if let Some(element) = resolve_element_ref(graph, element_ref)
            && seen.insert(element.id)
        {
            nodes.push(element.id);
        }
    }
    nodes
}

fn resolve_element_ref<'a>(graph: &'a Graph, element_ref: &ElementRef) -> Option<&'a Element> {
    let name = element_ref.qualified_name.as_str();
    graph
        .element_by_element_id(name)
        .or_else(|| {
            graph.elements().iter().find(|element| {
                ai_string_property(&element.properties.to_btree_map(), "qualified_name").as_deref()
                    == Some(name)
            })
        })
        .or_else(|| {
            graph.elements().iter().find(|element| {
                ai_string_property(&element.properties.to_btree_map(), "declared_name").as_deref()
                    == Some(name)
            })
        })
        .or_else(|| {
            graph
                .elements()
                .iter()
                .find(|element| element.element_id.ends_with(name))
        })
}

fn resolve_tool_element_ref(graph: &Graph, element_ref: &ElementRef) -> SemanticElementRef {
    resolve_element_ref(graph, element_ref)
        .map(ai_semantic_element_ref)
        .unwrap_or_else(|| SemanticElementRef {
            element_id: element_ref.qualified_name.clone(),
            qualified_name: Some(element_ref.qualified_name.clone()),
            label: element_ref
                .qualified_name
                .rsplit(['.', ':', '/'])
                .find(|part| !part.is_empty())
                .map(ToOwned::to_owned),
        })
}

fn ai_cognitive_element(element: &Element) -> CognitiveElement {
    let properties = element.properties.to_btree_map();
    CognitiveElement {
        element: ai_semantic_element_ref(element),
        kind: element.kind.to_string(),
        metatype: ai_string_property(&properties, "metatype")
            .or_else(|| ai_string_property(&properties, "type")),
        layer: element.layer,
        attributes: properties.clone(),
        source_spans: ai_source_span_for_properties(&properties)
            .into_iter()
            .collect(),
    }
}

fn ai_cognitive_relationship(
    index: usize,
    edge: &Edge,
    source: &Element,
    target: &Element,
) -> CognitiveRelationship {
    CognitiveRelationship {
        id: format!("kir.edge.{index}"),
        kind: edge.relation.to_string(),
        source: ai_semantic_element_ref(source),
        target: ai_semantic_element_ref(target),
    }
}

fn ai_semantic_element_ref(element: &Element) -> SemanticElementRef {
    let properties = element.properties.to_btree_map();
    SemanticElementRef {
        element_id: element.element_id.clone(),
        qualified_name: ai_string_property(&properties, "qualified_name"),
        label: ai_string_property(&properties, "declared_name")
            .or_else(|| ai_string_property(&properties, "name"))
            .or_else(|| {
                element
                    .element_id
                    .rsplit(['.', ':', '/'])
                    .find(|part| !part.is_empty())
                    .map(ToOwned::to_owned)
            }),
    }
}

fn cognitive_diagnostics_from_tool_results(
    graph: &Graph,
    tool_results: &[SemanticAgentToolResult],
) -> Vec<CognitiveDiagnostic> {
    tool_results
        .iter()
        .flat_map(|result| {
            result.findings.iter().map(|finding| {
                let element = finding
                    .elements
                    .first()
                    .map(|element_ref| resolve_tool_element_ref(graph, element_ref));
                CognitiveDiagnostic {
                    code: finding.id.clone(),
                    severity: cognitive_severity_from_label(&finding.severity),
                    message: format!(
                        "{} [{}]: {}",
                        finding.title,
                        semantic_agent_tool_id(result.tool),
                        finding.message
                    ),
                    element,
                    source_spans: Vec::new(),
                }
            })
        })
        .collect()
}

fn cognitive_artifacts_from_tool_results(
    graph: &Graph,
    tool_results: &[SemanticAgentToolResult],
) -> Vec<SemanticArtifact> {
    tool_results
        .iter()
        .enumerate()
        .map(|(index, result)| {
            let element_refs = result
                .findings
                .iter()
                .flat_map(|finding| finding.elements.iter())
                .map(|element_ref| resolve_tool_element_ref(graph, element_ref))
                .fold(Vec::new(), |mut refs, element_ref| {
                    if !refs
                        .iter()
                        .any(|seen: &SemanticElementRef| seen.element_id == element_ref.element_id)
                    {
                        refs.push(element_ref);
                    }
                    refs
                });
            let payload = json!(result);
            let bytes = serde_json::to_vec(&payload).unwrap_or_default();
            SemanticArtifact {
                id: format!(
                    "semantic_agent.tool.{index}.{}",
                    semantic_agent_tool_id(result.tool)
                ),
                kind: format!("reasoning.{}", semantic_agent_tool_id(result.tool)),
                schema: "mercurio.ai.semantic_agent_tool_result.v1".to_string(),
                digest: stable_digest([(
                    semantic_agent_tool_id(result.tool).as_bytes(),
                    bytes.as_slice(),
                )]),
                element_refs,
                payload,
            }
        })
        .collect()
}

fn cognitive_severity_from_label(label: &str) -> CognitiveDiagnosticSeverity {
    match label.to_ascii_lowercase().as_str() {
        "error" | "critical" => CognitiveDiagnosticSeverity::Error,
        "warning" | "warn" => CognitiveDiagnosticSeverity::Warning,
        _ => CognitiveDiagnosticSeverity::Info,
    }
}

fn ai_source_span_for_properties(properties: &BTreeMap<String, Value>) -> Option<SourceSpanRef> {
    let direct = properties.get("source_span");
    let metadata = properties.get("metadata");
    let span = direct.or_else(|| metadata.and_then(|metadata| metadata.get("source_span")))?;
    let file = properties
        .get("source_file")
        .and_then(Value::as_str)
        .or_else(|| {
            metadata
                .and_then(|metadata| metadata.get("source_file"))
                .and_then(Value::as_str)
        })
        .or_else(|| span.get("file").and_then(Value::as_str))
        .unwrap_or("");
    Some(SourceSpanRef {
        file: file.to_string(),
        start_line: span
            .get("start_line")
            .or_else(|| span.get("startLine"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        start_col: span
            .get("start_col")
            .or_else(|| span.get("startCol"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        end_line: span
            .get("end_line")
            .or_else(|| span.get("endLine"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        end_col: span
            .get("end_col")
            .or_else(|| span.get("endCol"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
    })
}

fn ai_string_property(properties: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    properties
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn severity_label(severity: &FindingSeverity) -> &'static str {
    match severity {
        FindingSeverity::Info => "info",
        FindingSeverity::Warning => "warning",
        FindingSeverity::Error => "error",
        FindingSeverity::Critical => "critical",
    }
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

#[derive(Debug, Clone, Default)]
pub struct ReasoningProviderSecretOverrides {
    pub openai_api_key: Option<String>,
    pub azure_openai_api_key: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ReasoningProviderConfigOverrides {
    pub provider: Option<ReasoningProviderKind>,
    pub openai_model: Option<String>,
    pub openai_base_url: Option<String>,
    pub azure_openai_deployment: Option<String>,
    pub azure_openai_base_url: Option<String>,
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
    match config.provider {
        Some(ReasoningProviderKind::Heuristic) => {
            ResolvedReasoningProvider::Heuristic(heuristic_provider())
        }
        Some(ReasoningProviderKind::AzureOpenAi) => {
            azure_openai_provider_from_config(&config, &secrets)
                .map(ResolvedReasoningProvider::AzureOpenAi)
                .unwrap_or_else(|| ResolvedReasoningProvider::Heuristic(heuristic_provider()))
        }
        Some(ReasoningProviderKind::OpenAi) => openai_provider_from_config(&config, &secrets)
            .map(ResolvedReasoningProvider::OpenAi)
            .unwrap_or_else(|| ResolvedReasoningProvider::Heuristic(heuristic_provider())),
        _ => default_reasoning_provider_with_secret_overrides(secrets),
    }
}

pub fn test_configured_reasoning_provider_connection(
    config: ReasoningProviderConfigOverrides,
    secrets: ReasoningProviderSecretOverrides,
) -> Result<ReasoningProviderStatus, String> {
    match config.provider {
        Some(ReasoningProviderKind::AzureOpenAi) => {
            let provider =
                azure_openai_provider_from_config(&config, &secrets).ok_or_else(|| {
                    let mut missing = Vec::new();
                    if config
                        .azure_openai_deployment
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .is_none()
                    {
                        missing.push("deployment");
                    }
                    if config
                        .azure_openai_base_url
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .is_none()
                    {
                        missing.push("base URL");
                    }
                    if secrets
                        .azure_openai_api_key
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .is_none()
                    {
                        missing.push("stored API key");
                    }
                    format!(
                        "Azure OpenAI settings are incomplete. Missing {}.",
                        missing.join(", ")
                    )
                })?;
            provider.test_connection()
        }
        Some(ReasoningProviderKind::OpenAi) => {
            let provider = openai_provider_from_config(&config, &secrets).ok_or_else(|| {
                let mut missing = Vec::new();
                if config
                    .openai_model
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .is_none()
                {
                    missing.push("model");
                }
                if config
                    .openai_base_url
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .is_none()
                {
                    missing.push("base URL");
                }
                if secrets
                    .openai_api_key
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .is_none()
                {
                    missing.push("stored API key");
                }
                format!(
                    "OpenAI settings are incomplete. Missing {}.",
                    missing.join(", ")
                )
            })?;
            provider.test_connection()
        }
        _ => default_reasoning_provider_with_secret_overrides(secrets).test_connection(),
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
    };
    let chat = provider
        .complete_chat(&chat_request)
        .unwrap_or_else(|_| heuristic_provider().complete_chat(&chat_request).unwrap());
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
    match config.provider {
        Some(ReasoningProviderKind::AzureOpenAi) => {
            let provider =
                azure_openai_provider_from_config(&config, &secrets).ok_or_else(|| {
                    configured_provider_missing_message(
                        &config,
                        &secrets,
                        ReasoningProviderKind::AzureOpenAi,
                    )
                })?;
            provider.complete_chat(request)
        }
        Some(ReasoningProviderKind::OpenAi) => {
            let provider = openai_provider_from_config(&config, &secrets).ok_or_else(|| {
                configured_provider_missing_message(
                    &config,
                    &secrets,
                    ReasoningProviderKind::OpenAi,
                )
            })?;
            provider.complete_chat(request)
        }
        Some(ReasoningProviderKind::Heuristic) => heuristic_provider().complete_chat(request),
        _ => complete_chat_with_secret_overrides(request, secrets),
    }
}

impl ReasoningProvider for ResolvedReasoningProvider {
    fn provider_status(&self) -> ReasoningProviderStatus {
        match self {
            Self::Heuristic(provider) => provider.provider_status(),
            Self::OpenAi(provider) => provider.provider_status(),
            Self::AzureOpenAi(provider) => provider.provider_status(),
        }
    }

    fn test_connection(&self) -> Result<ReasoningProviderStatus, String> {
        match self {
            Self::Heuristic(provider) => provider.test_connection(),
            Self::OpenAi(provider) => provider.test_connection(),
            Self::AzureOpenAi(provider) => provider.test_connection(),
        }
    }

    fn summarize_semantic_changes(
        &self,
        request: &SemanticSummaryRequest,
    ) -> SemanticSummaryResponse {
        match self {
            Self::Heuristic(provider) => provider.summarize_semantic_changes(request),
            Self::OpenAi(provider) => provider.summarize_semantic_changes(request),
            Self::AzureOpenAi(provider) => provider.summarize_semantic_changes(request),
        }
    }

    fn complete_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        match self {
            Self::Heuristic(provider) => provider.complete_chat(request),
            Self::OpenAi(provider) => provider.complete_chat(request),
            Self::AzureOpenAi(provider) => provider.complete_chat(request),
        }
    }
}

impl ReasoningProvider for HeuristicReasoningProvider {
    fn provider_status(&self) -> ReasoningProviderStatus {
        self.status.clone()
    }

    fn test_connection(&self) -> Result<ReasoningProviderStatus, String> {
        Ok(self.status.clone())
    }

    fn summarize_semantic_changes(
        &self,
        request: &SemanticSummaryRequest,
    ) -> SemanticSummaryResponse {
        heuristic_semantic_summary(request, self.status.clone())
    }

    fn complete_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        Ok(heuristic_chat_completion(request, self.status.clone()))
    }
}

impl SemanticMutationProposalProvider for HeuristicReasoningProvider {
    fn propose_semantic_mutations(
        &self,
        request: &SemanticMutationProposalRequest,
    ) -> Vec<MutationProposal> {
        heuristic_semantic_mutation_proposals(request)
    }
}

impl SemanticMutationProposalProvider for ResolvedReasoningProvider {
    fn propose_semantic_mutations(
        &self,
        request: &SemanticMutationProposalRequest,
    ) -> Vec<MutationProposal> {
        match self {
            Self::Heuristic(provider) => provider.propose_semantic_mutations(request),
            Self::OpenAi(provider) => provider
                .propose_semantic_mutations_via_openai(request)
                .unwrap_or_else(|_| provider.fallback.propose_semantic_mutations(request)),
            Self::AzureOpenAi(provider) => provider
                .propose_semantic_mutations_via_azure(request)
                .unwrap_or_else(|_| provider.fallback.propose_semantic_mutations(request)),
        }
    }
}

impl ReasoningProvider for OpenAiReasoningProvider {
    fn provider_status(&self) -> ReasoningProviderStatus {
        self.status.clone()
    }

    fn test_connection(&self) -> Result<ReasoningProviderStatus, String> {
        let payload = self.request_structured_json(
            "connection_probe",
            connection_probe_schema(),
            "Return JSON only. Respond with {\"ok\":true}.",
            "Confirm that the configured reasoning provider is reachable.",
        )?;
        let envelope: ConnectionProbeEnvelope =
            serde_json::from_value(payload).map_err(|error| error.to_string())?;
        if envelope.ok {
            Ok(self.status.clone())
        } else {
            Err("OpenAI provider returned an invalid connection probe response.".to_string())
        }
    }

    fn summarize_semantic_changes(
        &self,
        request: &SemanticSummaryRequest,
    ) -> SemanticSummaryResponse {
        match self.summarize_via_openai(request) {
            Ok(response) => response,
            Err(_) => self.fallback.summarize_semantic_changes(request),
        }
    }

    fn complete_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        self.complete_chat_via_openai(request)
    }
}

impl ReasoningProvider for AzureOpenAiReasoningProvider {
    fn provider_status(&self) -> ReasoningProviderStatus {
        self.status.clone()
    }

    fn test_connection(&self) -> Result<ReasoningProviderStatus, String> {
        let payload = self.request_structured_json(
            "connection_probe",
            connection_probe_schema(),
            "Return JSON only. Respond with {\"ok\":true}.",
            "Confirm that the configured Azure OpenAI reasoning provider is reachable.",
        )?;
        let envelope: ConnectionProbeEnvelope =
            serde_json::from_value(payload).map_err(|error| error.to_string())?;
        if envelope.ok {
            Ok(self.status.clone())
        } else {
            Err("Azure OpenAI provider returned an invalid connection probe response.".to_string())
        }
    }

    fn summarize_semantic_changes(
        &self,
        request: &SemanticSummaryRequest,
    ) -> SemanticSummaryResponse {
        match self.summarize_via_azure(request) {
            Ok(response) => response,
            Err(_) => self.fallback.summarize_semantic_changes(request),
        }
    }

    fn complete_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        self.complete_chat_via_azure(request)
    }
}

impl OpenAiReasoningProvider {
    fn propose_semantic_mutations_via_openai(
        &self,
        request: &SemanticMutationProposalRequest,
    ) -> Result<Vec<MutationProposal>, String> {
        let payload = self.request_structured_json(
            "semantic_mutation_proposals",
            semantic_mutation_proposal_schema(),
            semantic_mutation_proposal_developer_prompt(),
            &semantic_mutation_proposal_user_prompt(request),
        )?;
        parse_semantic_mutation_proposals_payload(payload, request)
    }

    fn summarize_via_openai(
        &self,
        request: &SemanticSummaryRequest,
    ) -> Result<SemanticSummaryResponse, String> {
        let payload = self.request_structured_json(
            "semantic_change_summary",
            semantic_summary_schema(),
            semantic_summary_developer_prompt(),
            &semantic_summary_user_prompt(request),
        )?;
        let envelope: SemanticSummaryEnvelope =
            serde_json::from_value(payload).map_err(|error| error.to_string())?;
        Ok(SemanticSummaryResponse {
            title: envelope.title.trim().to_string(),
            body: envelope
                .body
                .into_iter()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty())
                .collect(),
            provider: self.status.clone(),
        })
    }

    fn request_structured_json(
        &self,
        schema_name: &str,
        schema: Value,
        developer_prompt: &str,
        user_prompt: &str,
    ) -> Result<Value, String> {
        request_openai_structured_json(
            &self.client,
            &self.base_url,
            &self.api_key,
            &self.model,
            schema_name,
            schema,
            developer_prompt,
            user_prompt,
        )
    }

    fn complete_chat_via_openai(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        let message = request_openai_text(
            &self.client,
            &self.base_url,
            &self.api_key,
            &self.model,
            request,
        )?;
        Ok(ChatCompletionResponse {
            message,
            provider: self.status.clone(),
        })
    }
}

impl AzureOpenAiReasoningProvider {
    fn propose_semantic_mutations_via_azure(
        &self,
        request: &SemanticMutationProposalRequest,
    ) -> Result<Vec<MutationProposal>, String> {
        let payload = self.request_structured_json(
            "semantic_mutation_proposals",
            semantic_mutation_proposal_schema(),
            semantic_mutation_proposal_developer_prompt(),
            &semantic_mutation_proposal_user_prompt(request),
        )?;
        parse_semantic_mutation_proposals_payload(payload, request)
    }

    fn summarize_via_azure(
        &self,
        request: &SemanticSummaryRequest,
    ) -> Result<SemanticSummaryResponse, String> {
        let payload = self.request_structured_json(
            "semantic_change_summary",
            semantic_summary_schema(),
            semantic_summary_developer_prompt(),
            &semantic_summary_user_prompt(request),
        )?;
        let envelope: SemanticSummaryEnvelope =
            serde_json::from_value(payload).map_err(|error| error.to_string())?;
        Ok(SemanticSummaryResponse {
            title: envelope.title.trim().to_string(),
            body: envelope
                .body
                .into_iter()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty())
                .collect(),
            provider: self.status.clone(),
        })
    }

    fn request_structured_json(
        &self,
        schema_name: &str,
        schema: Value,
        developer_prompt: &str,
        user_prompt: &str,
    ) -> Result<Value, String> {
        request_openai_structured_json(
            &self.client,
            &self.base_url,
            &self.api_key,
            &self.deployment,
            schema_name,
            schema,
            developer_prompt,
            user_prompt,
        )
    }

    fn complete_chat_via_azure(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        let message = request_openai_text(
            &self.client,
            &self.base_url,
            &self.api_key,
            &self.deployment,
            request,
        )?;
        Ok(ChatCompletionResponse {
            message,
            provider: self.status.clone(),
        })
    }
}

fn resolve_reasoning_provider_from_env(
    secrets: &ReasoningProviderSecretOverrides,
) -> ResolvedReasoningProvider {
    let requested = std::env::var("MERCURIO_AI_PROVIDER")
        .or_else(|_| std::env::var("MERCURIO_REASONING_PROVIDER"))
        .unwrap_or_default()
        .to_ascii_lowercase();

    if requested == "azure_openai" || requested == "azure-openai" {
        if let Some(provider) = azure_openai_provider_from_env(secrets) {
            return ResolvedReasoningProvider::AzureOpenAi(provider);
        }
    }

    if (requested == "openai" || requested.is_empty())
        && let Some(provider) = openai_provider_from_env(secrets)
    {
        return ResolvedReasoningProvider::OpenAi(provider);
    }

    ResolvedReasoningProvider::Heuristic(heuristic_provider())
}

fn heuristic_provider() -> HeuristicReasoningProvider {
    HeuristicReasoningProvider {
        status: ReasoningProviderStatus {
            kind: ReasoningProviderKind::Heuristic,
            provider_label: "Heuristic".to_string(),
            detail: "Local deterministic summaries; no external provider configured.".to_string(),
            structured_outputs: true,
            model_label: None,
        },
    }
}

fn openai_provider_from_env(
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<OpenAiReasoningProvider> {
    let api_key = secrets
        .openai_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var("OPENAI_API_KEY")
                .or_else(|_| std::env::var("MERCURIO_OPENAI_API_KEY"))
                .ok()
                .filter(|value| !value.trim().is_empty())
        })?;
    let model = std::env::var("MERCURIO_OPENAI_MODEL")
        .or_else(|_| std::env::var("OPENAI_MODEL"))
        .unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string());
    let base_url = std::env::var("MERCURIO_OPENAI_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_OPENAI_BASE_URL.to_string());

    Some(OpenAiReasoningProvider {
        client: http_client(),
        api_key,
        model: model.clone(),
        base_url,
        status: ReasoningProviderStatus {
            kind: ReasoningProviderKind::OpenAi,
            provider_label: "OpenAI".to_string(),
            detail: "OpenAI Responses API configured from environment.".to_string(),
            structured_outputs: true,
            model_label: Some(model),
        },
        fallback: heuristic_provider(),
    })
}

fn openai_provider_from_config(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<OpenAiReasoningProvider> {
    let api_key = secrets
        .openai_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)?;
    let model = config
        .openai_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_OPENAI_MODEL)
        .to_string();
    let base_url = config
        .openai_base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_OPENAI_BASE_URL)
        .to_string();

    Some(OpenAiReasoningProvider {
        client: http_client(),
        api_key,
        model: model.clone(),
        base_url,
        status: ReasoningProviderStatus {
            kind: ReasoningProviderKind::OpenAi,
            provider_label: "OpenAI".to_string(),
            detail:
                "OpenAI Responses API configured from application settings and stored credential."
                    .to_string(),
            structured_outputs: true,
            model_label: Some(model),
        },
        fallback: heuristic_provider(),
    })
}

fn azure_openai_provider_from_env(
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<AzureOpenAiReasoningProvider> {
    let api_key = secrets
        .azure_openai_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var("AZURE_OPENAI_API_KEY")
                .or_else(|_| std::env::var("MERCURIO_AZURE_OPENAI_API_KEY"))
                .ok()
                .filter(|value| !value.trim().is_empty())
        })?;
    let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT")
        .or_else(|_| std::env::var("MERCURIO_AZURE_OPENAI_DEPLOYMENT"))
        .ok()
        .filter(|value| !value.trim().is_empty())?;
    let base_url = std::env::var("MERCURIO_AZURE_OPENAI_BASE_URL")
        .or_else(|_| std::env::var("AZURE_OPENAI_BASE_URL"))
        .or_else(|_| std::env::var("AZURE_OPENAI_ENDPOINT"))
        .ok()
        .map(|value| normalize_azure_openai_base_url(&value))?;

    Some(AzureOpenAiReasoningProvider {
        client: http_client(),
        api_key,
        deployment: deployment.clone(),
        base_url,
        status: ReasoningProviderStatus {
            kind: ReasoningProviderKind::AzureOpenAi,
            provider_label: "Azure OpenAI".to_string(),
            detail: "Azure OpenAI Responses API configured from environment.".to_string(),
            structured_outputs: true,
            model_label: Some(deployment),
        },
        fallback: heuristic_provider(),
    })
}

fn azure_openai_provider_from_config(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<AzureOpenAiReasoningProvider> {
    let api_key = secrets
        .azure_openai_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)?;
    let deployment = config
        .azure_openai_deployment
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())?
        .to_string();
    let base_url = config
        .azure_openai_base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(normalize_azure_openai_base_url)?;

    Some(AzureOpenAiReasoningProvider {
        client: http_client(),
        api_key,
        deployment: deployment.clone(),
        base_url,
        status: ReasoningProviderStatus {
            kind: ReasoningProviderKind::AzureOpenAi,
            provider_label: "Azure OpenAI".to_string(),
            detail:
                "Azure OpenAI Responses API configured from application settings and stored credential."
                    .to_string(),
            structured_outputs: true,
            model_label: Some(deployment),
        },
        fallback: heuristic_provider(),
    })
}

fn http_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| Client::new())
}

fn request_openai_structured_json(
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

fn request_openai_text(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    request: &ChatCompletionRequest,
) -> Result<String, String> {
    let mut input = Vec::new();
    if !request.context.is_empty() {
        input.push(json!({
            "role": "developer",
            "content": format!(
                "Use this Mercurio model context as the authoritative current workspace state.\n\
                 Critical grounding rules:\n\
                 - Treat live editor/workspace snapshots as newer and more authoritative than chat history, compiled metadata, or prior assistant messages.\n\
                 - Do not claim a file, package, requirement, element, relationship, diagram, commit, or edit exists unless it appears in the supplied current workspace context or validated tool output.\n\
                 - If prior chat says something was created but the current workspace snapshot does not contain it, say it is not present in the current model.\n\
                 - If context includes `Metamodel lookup result:` lines, treat them as authoritative KIR evidence and do not substitute generic SysML, UML, or modeling-language knowledge.\n\
                 - For questions asking what is in the current model, summarize only elements present in the current workspace context and state when expected evidence is missing.\n\n{}",
                request.context.join("\n")
            ),
        }));
    }
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

fn chat_role_name(role: &ChatMessageRole) -> &'static str {
    match role {
        ChatMessageRole::Developer => "developer",
        ChatMessageRole::Assistant => "assistant",
        ChatMessageRole::User => "user",
    }
}

fn configured_provider_missing_message(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
    provider: ReasoningProviderKind,
) -> String {
    let mut missing = Vec::new();
    match provider {
        ReasoningProviderKind::AzureOpenAi => {
            if config
                .azure_openai_deployment
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push("deployment");
            }
            if config
                .azure_openai_base_url
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push("base URL");
            }
            if secrets
                .azure_openai_api_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push("stored API key");
            }
            format!(
                "Azure OpenAI settings are incomplete. Missing {}.",
                missing.join(", ")
            )
        }
        ReasoningProviderKind::OpenAi => {
            if config
                .openai_model
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push("model");
            }
            if config
                .openai_base_url
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push("base URL");
            }
            if secrets
                .openai_api_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push("stored API key");
            }
            format!(
                "OpenAI settings are incomplete. Missing {}.",
                missing.join(", ")
            )
        }
        ReasoningProviderKind::Heuristic => "No external AI provider is configured.".to_string(),
    }
}

fn heuristic_semantic_summary(
    request: &SemanticSummaryRequest,
    provider: ReasoningProviderStatus,
) -> SemanticSummaryResponse {
    let added = request
        .changes
        .iter()
        .filter(|change| change.kind == SemanticChangeKind::Added)
        .count();
    let removed = request
        .changes
        .iter()
        .filter(|change| change.kind == SemanticChangeKind::Removed)
        .count();
    let changed = request
        .changes
        .iter()
        .filter(|change| change.kind == SemanticChangeKind::Changed)
        .count();
    let title = request
        .title_hint
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| semantic_summary_title(added, changed, removed));

    let mut body = Vec::new();
    if !request.changed_files.is_empty() {
        body.push(format!(
            "Updated {} file(s): {}",
            request.changed_files.len(),
            request
                .changed_files
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if added > 0 {
        body.push(format!("Added {added} semantic element(s)."));
    }
    if changed > 0 {
        body.push(format!("Changed {changed} semantic element(s)."));
    }
    if removed > 0 {
        body.push(format!("Removed {removed} semantic element(s)."));
    }
    body.extend(request.changes.iter().take(6).map(describe_change_item));
    if body.is_empty() {
        body.push("No semantic changes were supplied.".to_string());
    }

    SemanticSummaryResponse {
        title,
        body,
        provider,
    }
}

fn heuristic_semantic_mutation_proposals(
    request: &SemanticMutationProposalRequest,
) -> Vec<MutationProposal> {
    let intent = request.design_intent.to_ascii_lowercase();
    if !(intent.contains("hybrid") || intent.contains("efficiency")) {
        return Vec::new();
    }

    if request.semantic_context.is_none() && request.cognitive_context.is_none() {
        return vec![heuristic_regenerative_braking_proposal(request)];
    }

    if !request_context_has_element(request, "HybridVehicle") {
        return vec![MutationProposal {
            intent: "Create the hybrid vehicle model package".to_string(),
            affected_elements: vec![ElementRef::new("HybridVehicle")],
            operations: vec![SemanticMutation::AddPackage {
                target_file: "hybrid_vehicle.sysml".to_string(),
                name: "HybridVehicle".to_string(),
            }],
            evidence: vec![MutationEvidence {
                element: None,
                summary: "The model needs a package before domain elements can be owned."
                    .to_string(),
            }],
            rationale: Some(
                "A package is the stable namespace for the generated hybrid vehicle model."
                    .to_string(),
            ),
            workspace_revision: request.workspace_revision.clone(),
        }];
    }

    if !request_context_has_element(request, "HybridVehicle.HybridVehicle") {
        return vec![MutationProposal {
            intent: "Add the core hybrid vehicle element and efficiency requirement".to_string(),
            affected_elements: vec![ElementRef::new("HybridVehicle")],
            operations: vec![
                SemanticMutation::AddDefinition {
                    container: ElementRef::new("HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "HybridVehicle".to_string(),
                    specializes: Vec::new(),
                },
                SemanticMutation::AddDefinition {
                    container: ElementRef::new("HybridVehicle"),
                    keyword: "requirement".to_string(),
                    name: "ImproveEfficiency".to_string(),
                    specializes: Vec::new(),
                },
            ],
            evidence: vec![MutationEvidence {
                element: Some(ElementRef::new("HybridVehicle")),
                summary: "The package exists and can own the vehicle definition and requirement."
                    .to_string(),
            }],
            rationale: Some(
                "The vehicle definition and efficiency requirement establish the model root and design objective."
                    .to_string(),
            ),
            workspace_revision: request.workspace_revision.clone(),
        }];
    }

    if !request_context_has_element(request, "HybridVehicle.Engine") {
        return vec![MutationProposal {
            intent: "Add the major hybrid powertrain subsystem definitions".to_string(),
            affected_elements: vec![ElementRef::new("HybridVehicle")],
            operations: vec![
                SemanticMutation::AddDefinition {
                    container: ElementRef::new("HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "Engine".to_string(),
                    specializes: Vec::new(),
                },
                SemanticMutation::AddDefinition {
                    container: ElementRef::new("HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "ElectricMotor".to_string(),
                    specializes: Vec::new(),
                },
                SemanticMutation::AddDefinition {
                    container: ElementRef::new("HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "BatteryPack".to_string(),
                    specializes: Vec::new(),
                },
            ],
            evidence: vec![MutationEvidence {
                element: Some(ElementRef::new("HybridVehicle.HybridVehicle")),
                summary: "A hybrid vehicle needs combustion, electric drive, and energy storage subsystems."
                    .to_string(),
            }],
            rationale: Some(
                "These subsystem definitions provide reusable types for the vehicle composition."
                    .to_string(),
            ),
            workspace_revision: request.workspace_revision.clone(),
        }];
    }

    if !request_context_has_element(request, "HybridVehicle.HybridVehicle.engine") {
        return vec![MutationProposal {
            intent: "Compose the hybrid vehicle from the major subsystem usages".to_string(),
            affected_elements: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
            operations: vec![
                SemanticMutation::AddUsage {
                    container: ElementRef::new("HybridVehicle.HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "engine".to_string(),
                    ty: Some(ElementRef::new("HybridVehicle.Engine")),
                    specializes: Vec::new(),
                },
                SemanticMutation::AddUsage {
                    container: ElementRef::new("HybridVehicle.HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "motor".to_string(),
                    ty: Some(ElementRef::new("HybridVehicle.ElectricMotor")),
                    specializes: Vec::new(),
                },
                SemanticMutation::AddUsage {
                    container: ElementRef::new("HybridVehicle.HybridVehicle"),
                    keyword: "part".to_string(),
                    name: "battery".to_string(),
                    ty: Some(ElementRef::new("HybridVehicle.BatteryPack")),
                    specializes: Vec::new(),
                },
            ],
            evidence: vec![MutationEvidence {
                element: Some(ElementRef::new("HybridVehicle.HybridVehicle")),
                summary: "The vehicle definition exists and can own typed subsystem usages."
                    .to_string(),
            }],
            rationale: Some(
                "Typed usages connect reusable subsystem definitions to the vehicle architecture."
                    .to_string(),
            ),
            workspace_revision: request.workspace_revision.clone(),
        }];
    }

    vec![heuristic_regenerative_braking_proposal(request)]
}

fn heuristic_regenerative_braking_proposal(
    request: &SemanticMutationProposalRequest,
) -> MutationProposal {
    let operations =
        if !request_context_has_element(request, "HybridVehicle.RegenerativeBrakingSystem") {
            vec![SemanticMutation::AddDefinition {
                container: ElementRef::new("HybridVehicle"),
                keyword: "part".to_string(),
                name: "RegenerativeBrakingSystem".to_string(),
                specializes: Vec::new(),
            }]
        } else if !request_context_has_element(
            request,
            "HybridVehicle.HybridVehicle.regenerativeBraking",
        ) {
            vec![SemanticMutation::AddUsage {
                container: ElementRef::new("HybridVehicle.HybridVehicle"),
                keyword: "part".to_string(),
                name: "regenerativeBraking".to_string(),
                ty: Some(ElementRef::new("HybridVehicle.RegenerativeBrakingSystem")),
                specializes: Vec::new(),
            }]
        } else {
            vec![SemanticMutation::AddRelationship {
                kind: "satisfy".to_string(),
                source: ElementRef::new("HybridVehicle.RegenerativeBrakingSystem"),
                target: ElementRef::new("HybridVehicle.ImproveEfficiency"),
            }]
        };

    MutationProposal {
        intent: "Improve hybrid vehicle efficiency through regenerative braking".to_string(),
        affected_elements: vec![ElementRef::new("HybridVehicle.HybridVehicle")],
        operations,
        evidence: vec![
            MutationEvidence {
                element: Some(ElementRef::new("HybridVehicle.BatteryPack")),
                summary: "Battery storage exists and can receive recovered braking energy."
                    .to_string(),
            },
            MutationEvidence {
                element: Some(ElementRef::new("HybridVehicle.ElectricMotor")),
                summary: "Electric drive components can participate in energy recovery."
                    .to_string(),
            },
        ],
        rationale: Some(
            "Regenerative braking is a model-level efficiency improvement because it recovers kinetic energy and traces directly to the efficiency requirement."
                .to_string(),
        ),
        workspace_revision: request.workspace_revision.clone(),
    }
}

fn request_context_has_element(request: &SemanticMutationProposalRequest, element: &str) -> bool {
    request.cognitive_context.as_ref().is_some_and(|context| {
        context.elements.iter().any(|item| {
            item.element.element_id == element
                || item.element.qualified_name.as_deref() == Some(element)
                || item.element.label.as_deref() == Some(element)
        })
    }) || request.semantic_context.as_ref().is_some_and(|context| {
        context
            .elements
            .iter()
            .any(|item| item.element.qualified_name == element)
    })
}

fn default_semantic_agent_goal_spec(goal: &str) -> Option<SemanticGoalSpec> {
    let goal = goal.to_ascii_lowercase();
    if !(goal.contains("hybrid") || goal.contains("efficiency")) {
        return None;
    }
    Some(SemanticGoalSpec {
        policy: mercurio_core::GoalPolicy::All,
        checks: vec![
            SemanticGoalCheck::ElementExists {
                element: ElementRef::new("HybridVehicle"),
                kind: Some("package".to_string()),
            },
            SemanticGoalCheck::AnyOf {
                checks: vec![
                    SemanticGoalCheck::ElementExists {
                        element: ElementRef::new("HybridVehicle.HybridVehicle"),
                        kind: Some("part".to_string()),
                    },
                    SemanticGoalCheck::ElementExists {
                        element: ElementRef::new("HybridVehicle.Vehicle"),
                        kind: Some("part".to_string()),
                    },
                    SemanticGoalCheck::NamedElementExists {
                        name: "Vehicle".to_string(),
                        kind: Some("part".to_string()),
                    },
                ],
            },
            SemanticGoalCheck::NamedElementExists {
                name: "Engine".to_string(),
                kind: Some("part".to_string()),
            },
            SemanticGoalCheck::NamedElementExists {
                name: "ElectricMotor".to_string(),
                kind: Some("part".to_string()),
            },
            SemanticGoalCheck::NamedElementExists {
                name: "BatteryPack".to_string(),
                kind: Some("part".to_string()),
            },
            SemanticGoalCheck::AnyOf {
                checks: vec![
                    SemanticGoalCheck::ElementExists {
                        element: ElementRef::new("HybridVehicle.ImproveEfficiency"),
                        kind: Some("requirement".to_string()),
                    },
                    SemanticGoalCheck::ElementExists {
                        element: ElementRef::new("HybridVehicle.EfficiencyRequirement"),
                        kind: Some("requirement".to_string()),
                    },
                    SemanticGoalCheck::NamedElementExists {
                        name: "ImproveEfficiency".to_string(),
                        kind: Some("requirement".to_string()),
                    },
                    SemanticGoalCheck::NamedElementExists {
                        name: "EfficiencyRequirement".to_string(),
                        kind: Some("requirement".to_string()),
                    },
                ],
            },
            SemanticGoalCheck::AnyOf {
                checks: vec![
                    SemanticGoalCheck::NamedElementExists {
                        name: "RegenerativeBrakingSystem".to_string(),
                        kind: Some("part".to_string()),
                    },
                    SemanticGoalCheck::NamedElementExists {
                        name: "RegenerativeBraking".to_string(),
                        kind: Some("part".to_string()),
                    },
                    SemanticGoalCheck::NamedElementExists {
                        name: "RegenerativeBraking".to_string(),
                        kind: Some("action".to_string()),
                    },
                ],
            },
            SemanticGoalCheck::AnyOf {
                checks: vec![
                    SemanticGoalCheck::NamedRelationshipExists {
                        source_name: "RegenerativeBrakingSystem".to_string(),
                        kind: "satisfy".to_string(),
                        target_name: "ImproveEfficiency".to_string(),
                    },
                    SemanticGoalCheck::NamedRelationshipExists {
                        source_name: "RegenerativeBraking".to_string(),
                        kind: "satisfy".to_string(),
                        target_name: "EfficiencyRequirement".to_string(),
                    },
                    SemanticGoalCheck::NamedRelationshipExists {
                        source_name: "RegenerativeBraking".to_string(),
                        kind: "satisfy".to_string(),
                        target_name: "ImproveEfficiency".to_string(),
                    },
                ],
            },
        ],
    })
}

fn heuristic_chat_completion(
    request: &ChatCompletionRequest,
    provider: ReasoningProviderStatus,
) -> ChatCompletionResponse {
    let latest = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == ChatMessageRole::User)
        .map(|message| message.content.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("your request");
    let context = if request.context.is_empty() {
        "No model context was supplied.".to_string()
    } else {
        format!("I received {} context item(s).", request.context.len())
    };
    ChatCompletionResponse {
        message: format!(
            "I received \"{latest}\". {context} Configure OpenAI or Azure OpenAI in Settings to generate provider-backed answers."
        ),
        provider,
    }
}

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

fn latest_user_content(messages: &[ChatMessage]) -> &str {
    messages
        .iter()
        .rev()
        .find(|message| message.role == ChatMessageRole::User)
        .map(|message| message.content.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("")
}

fn ask_mercurio_developer_context(task: &AskMercurioTask) -> String {
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

fn ask_mercurio_citations(
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

fn ask_mercurio_artifacts(
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

fn semantic_summary_title(added: usize, changed: usize, removed: usize) -> String {
    match (added, changed, removed) {
        (0, 0, 0) => "Summarize semantic model state".to_string(),
        (_, 0, 0) if added > 0 => "Add semantic model elements".to_string(),
        (0, _, 0) if changed > 0 => "Update semantic model elements".to_string(),
        (0, 0, _) if removed > 0 => "Remove semantic model elements".to_string(),
        _ => "Update semantic model structure".to_string(),
    }
}

fn describe_change_item(change: &SemanticChangeItem) -> String {
    let label = change.label.as_deref().unwrap_or(&change.element_id);
    let kind = match change.kind {
        SemanticChangeKind::Added => "Added",
        SemanticChangeKind::Removed => "Removed",
        SemanticChangeKind::Changed => "Changed",
        SemanticChangeKind::Unchanged => "Unchanged",
    };
    let mut detail = format!("{kind} {label} ({})", change.element_kind);
    if !change.changed_properties.is_empty() {
        detail.push_str(&format!(
            "; properties: {}",
            change.changed_properties.join(", ")
        ));
    }
    if !change.changed_relationships.is_empty() {
        detail.push_str(&format!(
            "; relationships: {}",
            change.changed_relationships.join(", ")
        ));
    }
    detail
}

fn semantic_summary_developer_prompt() -> &'static str {
    "Write a concise engineering change summary from the supplied semantic diff. \
     Return JSON only. Do not invent changes that are not present. Prefer domain \
     language from element labels and kinds. Keep the title under 72 characters."
}

fn semantic_summary_user_prompt(request: &SemanticSummaryRequest) -> String {
    serde_json::to_string_pretty(request).unwrap_or_else(|_| "{}".to_string())
}

fn semantic_mutation_proposal_developer_prompt() -> &'static str {
    "Return semantic SysML mutation proposals as JSON only. Propose operations in terms \
     of stable semantic elements and qualified names, not prose patches. Do not invent \
     source text edits. Treat cognitive_context as the authoritative semantic grounding: \
     it contains KIR element ids, graph neighborhoods, diagnostics, and reasoning artifacts. \
     Use semantic_context only as compatibility affordance data. Cite finding ids, artifact ids, \
     and element refs in evidence when they support a proposal. Do not reconstruct semantic truth \
     from staged source text when structured context is available. Use only supported operation tags, keywords, and relationship \
     kinds from the supplied capability context and schema. Use dot-qualified ElementRef \
     names exactly as they appear in semantic_context.elements; do not use :: separators \
     inside ElementRef. Do not propose adding an element that already appears in \
     semantic_context.elements. Prefer one coherent batch of 2 to 5 non-empty operations \
     when the required containers and types already exist. For an empty model, create only \
     the root package first. Use RemoveDeclaration or RemoveUsage for cleanup when \
     simplification is requested and the target exists. Requirement definitions should have explicit id and text \
     attributes; use SetAttribute on existing requirement elements to fill missing fields. \
     Core feasibility will reject impossible changes."
}

fn semantic_mutation_proposal_user_prompt(request: &SemanticMutationProposalRequest) -> String {
    serde_json::to_string_pretty(&json!({
        "capability_context": sysml_semantic_mutation_capability_context(),
        "agent_guidance": {
            "element_ref_format": "Use dot-qualified names such as HybridVehicle.Vehicle, never HybridVehicle::Vehicle.",
            "current_state_rule": "Treat cognitive_context.elements and semantic_context.elements as already existing. Do not re-add them.",
            "grounding_rule": "Use cognitive_context as authoritative structured context. Prefer KIR element_id for evidence identity, and use qualified_name only as ElementRef display/input metadata.",
            "citation_rule": "When reasoning_tool_results or cognitive_context.diagnostics identify a relevant finding, cite its id and related element in proposal evidence.",
            "operation_rule": "Every proposal must contain at least one operation. Empty proposals are ignored.",
            "quality_rule": "When a requirement already exists without id or text, prefer SetAttribute operations for id and text before adding more requirements.",
            "batching_rule": "Batch related operations only when their containers and referenced types already exist in the current semantic context.",
            "affordance_rule": "Prefer operations supported by semantic_context.affordances for the target element."
        },
        "request": request,
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

fn parse_semantic_mutation_proposals_payload(
    payload: Value,
    request: &SemanticMutationProposalRequest,
) -> Result<Vec<MutationProposal>, String> {
    let envelope: SemanticMutationProposalEnvelope =
        serde_json::from_value(payload).map_err(|error| error.to_string())?;
    Ok(envelope
        .proposals
        .into_iter()
        .map(|mut proposal| {
            proposal.workspace_revision = request.workspace_revision.clone();
            proposal
        })
        .collect())
}

fn extract_output_text(response: &OpenAiStructuredResponse) -> Result<String, String> {
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

fn semantic_summary_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "title": { "type": "string" },
            "body": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["title", "body"]
    })
}

fn semantic_mutation_proposal_schema() -> Value {
    let capability_context = sysml_semantic_mutation_capability_context();
    let element_ref = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "qualified_name": { "type": "string" }
        },
        "required": ["qualified_name"]
    });
    let workspace_revision = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "fingerprint": { "type": "string" }
        },
        "required": ["fingerprint"]
    });
    let evidence = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "element": {
                "anyOf": [
                    element_ref.clone(),
                    { "type": "null" }
                ]
            },
            "summary": { "type": "string" }
        },
        "required": ["element", "summary"]
    });
    let element_ref_array = json!({
        "type": "array",
        "items": element_ref.clone()
    });
    let semantic_expression = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "Text": { "type": "string" }
        },
        "required": ["Text"]
    });
    let definition_keyword = json!({
        "type": "string",
        "enum": capability_context.definition_keywords
    });
    let usage_keyword = json!({
        "type": "string",
        "enum": capability_context.usage_keywords
    });
    let operation = json!({
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "AddPackage": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "target_file": { "type": "string" },
                            "name": { "type": "string" }
                        },
                        "required": ["target_file", "name"]
                    }
                },
                "required": ["AddPackage"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "AddDefinition": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "container": element_ref.clone(),
                            "keyword": definition_keyword.clone(),
                            "name": { "type": "string" },
                            "specializes": element_ref_array.clone()
                        },
                        "required": ["container", "keyword", "name", "specializes"]
                    }
                },
                "required": ["AddDefinition"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "AddUsage": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "container": element_ref.clone(),
                            "keyword": usage_keyword.clone(),
                            "name": { "type": "string" },
                            "ty": {
                                "anyOf": [
                                    element_ref.clone(),
                                    { "type": "null" }
                                ]
                            },
                            "specializes": element_ref_array.clone()
                        },
                        "required": ["container", "keyword", "name", "ty", "specializes"]
                    }
                },
                "required": ["AddUsage"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "AddRelationship": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": capability_context.relationship_kinds
                            },
                            "source": element_ref.clone(),
                            "target": element_ref.clone()
                        },
                        "required": ["kind", "source", "target"]
                    }
                },
                "required": ["AddRelationship"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "RemoveDeclaration": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone()
                        },
                        "required": ["element"]
                    }
                },
                "required": ["RemoveDeclaration"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "RemoveUsage": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone()
                        },
                        "required": ["element"]
                    }
                },
                "required": ["RemoveUsage"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "RemoveRelationship": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": capability_context.relationship_kinds
                            },
                            "source": element_ref.clone(),
                            "target": element_ref.clone()
                        },
                        "required": ["kind", "source", "target"]
                    }
                },
                "required": ["RemoveRelationship"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "RenameDeclaration": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone(),
                            "new_name": { "type": "string" }
                        },
                        "required": ["element", "new_name"]
                    }
                },
                "required": ["RenameDeclaration"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "UpdateUsageType": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone(),
                            "ty": {
                                "anyOf": [
                                    element_ref.clone(),
                                    { "type": "null" }
                                ]
                            }
                        },
                        "required": ["element", "ty"]
                    }
                },
                "required": ["UpdateUsageType"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "SetExpression": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone(),
                            "expression": {
                                "anyOf": [
                                    semantic_expression,
                                    { "type": "null" }
                                ]
                            }
                        },
                        "required": ["element", "expression"]
                    }
                },
                "required": ["SetExpression"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "UpdateSpecializations": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone(),
                            "specializes": element_ref_array.clone()
                        },
                        "required": ["element", "specializes"]
                    }
                },
                "required": ["UpdateSpecializations"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "MoveDeclaration": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone(),
                            "destination": element_ref.clone()
                        },
                        "required": ["element", "destination"]
                    }
                },
                "required": ["MoveDeclaration"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "SetAttribute": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "element": element_ref.clone(),
                            "attribute": { "type": "string" },
                            "value": true
                        },
                        "required": ["element", "attribute", "value"]
                    }
                },
                "required": ["SetAttribute"]
            }
        ]
    });
    let proposal = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "intent": { "type": "string" },
            "affected_elements": {
                "type": "array",
                "items": element_ref
            },
            "operations": {
                "type": "array",
                "items": operation
            },
            "evidence": {
                "type": "array",
                "items": evidence
            },
            "rationale": {
                "anyOf": [
                    { "type": "string" },
                    { "type": "null" }
                ]
            },
            "workspace_revision": workspace_revision
        },
        "required": [
            "intent",
            "affected_elements",
            "operations",
            "evidence",
            "rationale",
            "workspace_revision"
        ]
    });
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "proposals": {
                "type": "array",
                "items": proposal
            }
        },
        "required": ["proposals"]
    })
}

fn connection_probe_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "ok": { "type": "boolean" }
        },
        "required": ["ok"]
    })
}

fn normalize_azure_openai_base_url(value: &str) -> String {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.ends_with("/openai/v1/responses") {
        return trimmed.to_string();
    }
    if trimmed.ends_with("/openai/v1") {
        return format!("{trimmed}/responses");
    }
    format!("{trimmed}{DEFAULT_AZURE_OPENAI_PATH}")
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
        CheckedMutationProposal, MutationProposal, OpenAiStructuredResponse,
        ReasoningProviderConfigOverrides, ReasoningProviderSecretOverrides, SemanticChangeItem,
        SemanticChangeKind, SemanticContextBuilder, SemanticMutationProposalProvider,
        SemanticMutationProposalRequest, SemanticSummaryRequest, ask_mercurio_artifacts,
        classify_ask_mercurio_task, extract_output_text, heuristic_provider,
        normalize_azure_openai_base_url, parse_semantic_mutation_proposals_payload,
        propose_checked_semantic_mutations, run_semantic_mutation_agent,
        semantic_mutation_proposal_schema, semantic_mutation_proposal_user_prompt,
        summarize_semantic_changes, test_configured_reasoning_provider_connection,
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
