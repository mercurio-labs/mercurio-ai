use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    CognitiveContext, DesignIntent, ElementRef, ModelRevisionEnvelope, ReasoningProviderStatus,
};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<AiWorkspaceInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<ModelRevisionEnvelope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cognitive_context: Option<CognitiveContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionResponse {
    pub message: String,
    pub provider: ReasoningProviderStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ChatCompletionArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatCompletionArtifact {
    Diagram { spec: Value },
    Matrix { spec: Value },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiWorkspaceInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_editor_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_element_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dirty_snapshots: Vec<AiWorkspaceDirtySnapshot>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_snapshots: Vec<AiWorkspaceDirtySnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AiWorkspaceDirtySnapshot {
    pub path: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AiWorkbenchMode {
    Pairing,
    Assessment,
    Exploration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AiWorkbenchRequest {
    pub mode: AiWorkbenchMode,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<DesignIntent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub focus: Vec<ElementRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<AiWorkspaceInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<ModelRevisionEnvelope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cognitive_context: Option<CognitiveContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AiWorkbenchResponse {
    pub message: String,
    pub provider: ReasoningProviderStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ChatCompletionArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assessment: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cognitive_context: Option<CognitiveContext>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proposed_actions: Vec<Value>,
}

impl From<ChatCompletionResponse> for AiWorkbenchResponse {
    fn from(response: ChatCompletionResponse) -> Self {
        Self {
            message: response.message,
            provider: response.provider,
            artifacts: response.artifacts,
            overlay: response.overlay,
            assessment: None,
            cognitive_context: None,
            proposed_actions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AiProviderDescriptor {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credential_schema: Vec<AiProviderFieldDescriptor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub settings_schema: Vec<AiProviderFieldDescriptor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub default_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AiProviderFieldDescriptor {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default)]
    pub secret: bool,
    #[serde(default)]
    pub required: bool,
}
