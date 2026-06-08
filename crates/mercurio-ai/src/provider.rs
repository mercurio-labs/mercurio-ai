use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    ChatCompletionRequest, ChatCompletionResponse, MutationProposal,
    SemanticMutationProposalRequest, SemanticSummaryRequest, SemanticSummaryResponse,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningProviderKind {
    Heuristic,
    OpenAi,
    AzureOpenAi,
    Anthropic,
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

#[derive(Debug, Clone, Default)]
pub struct ReasoningProviderSecretOverrides {
    pub openai_api_key: Option<String>,
    pub azure_openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ReasoningProviderConfigOverrides {
    pub provider: Option<ReasoningProviderKind>,
    pub openai_model: Option<String>,
    pub openai_base_url: Option<String>,
    pub azure_openai_deployment: Option<String>,
    pub azure_openai_base_url: Option<String>,
    pub anthropic_proposal_model: Option<String>,
    pub anthropic_fast_model: Option<String>,
    pub anthropic_base_url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiStructuredResponse {
    pub(crate) output: Vec<OpenAiOutputItem>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiOutputItem {
    #[serde(default)]
    pub(crate) content: Vec<OpenAiContentItem>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum OpenAiContentItem {
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicMessageResponse {
    #[serde(default)]
    pub(crate) content: Vec<AnthropicContentItem>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicContentItem {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { input: Value },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone)]
pub enum ResolvedReasoningProvider {
    Heuristic(HeuristicReasoningProvider),
    OpenAi(OpenAiReasoningProvider),
    AzureOpenAi(AzureOpenAiReasoningProvider),
    Anthropic(AnthropicReasoningProvider),
}

#[derive(Debug, Clone)]
pub struct HeuristicReasoningProvider {
    pub(crate) status: ReasoningProviderStatus,
}

#[derive(Debug, Clone)]
pub struct OpenAiReasoningProvider {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) status: ReasoningProviderStatus,
    pub(crate) fallback: HeuristicReasoningProvider,
}

#[derive(Debug, Clone)]
pub struct AzureOpenAiReasoningProvider {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    pub(crate) deployment: String,
    pub(crate) base_url: String,
    pub(crate) status: ReasoningProviderStatus,
    pub(crate) fallback: HeuristicReasoningProvider,
}

#[derive(Debug, Clone)]
pub struct AnthropicReasoningProvider {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    pub(crate) proposal_model: String,
    pub(crate) fast_model: String,
    pub(crate) base_url: String,
    pub(crate) status: ReasoningProviderStatus,
    pub(crate) fallback: HeuristicReasoningProvider,
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
