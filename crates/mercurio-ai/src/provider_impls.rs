use serde_json::{Value, json};

use mercurio_sysml::sysml_semantic_mutation_capability_context;

use crate::heuristic::{
    heuristic_chat_completion, heuristic_semantic_mutation_proposals, heuristic_semantic_summary,
};
use crate::provider_http::{
    request_anthropic_structured_json, request_anthropic_text, request_openai_structured_json,
    request_openai_text,
};
use crate::{
    AnthropicReasoningProvider, AzureOpenAiReasoningProvider, ChatCompletionRequest,
    ChatCompletionResponse, ConnectionProbeEnvelope, HeuristicReasoningProvider, MutationProposal,
    OpenAiReasoningProvider, ReasoningProvider, ReasoningProviderStatus, ResolvedReasoningProvider,
    SemanticMutationProposalProvider, SemanticMutationProposalRequest, SemanticSummaryEnvelope,
    SemanticSummaryRequest, SemanticSummaryResponse, chat_completion_response,
    connection_probe_schema, parse_semantic_mutation_proposals_payload,
    semantic_mutation_agent_guidance, semantic_mutation_proposal_developer_prompt,
    semantic_mutation_proposal_schema, semantic_mutation_proposal_user_prompt,
    semantic_summary_developer_prompt, semantic_summary_schema, semantic_summary_user_prompt,
};

impl ReasoningProvider for ResolvedReasoningProvider {
    fn provider_status(&self) -> ReasoningProviderStatus {
        match self {
            Self::Heuristic(provider) => provider.provider_status(),
            Self::OpenAi(provider) => provider.provider_status(),
            Self::AzureOpenAi(provider) => provider.provider_status(),
            Self::Anthropic(provider) => provider.provider_status(),
        }
    }

    fn test_connection(&self) -> Result<ReasoningProviderStatus, String> {
        match self {
            Self::Heuristic(provider) => provider.test_connection(),
            Self::OpenAi(provider) => provider.test_connection(),
            Self::AzureOpenAi(provider) => provider.test_connection(),
            Self::Anthropic(provider) => provider.test_connection(),
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
            Self::Anthropic(provider) => provider.summarize_semantic_changes(request),
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
            Self::Anthropic(provider) => provider.complete_chat(request),
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
            Self::Anthropic(provider) => provider
                .propose_semantic_mutations_via_anthropic(request)
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

impl ReasoningProvider for AnthropicReasoningProvider {
    fn provider_status(&self) -> ReasoningProviderStatus {
        self.status.clone()
    }

    fn test_connection(&self) -> Result<ReasoningProviderStatus, String> {
        let payload = self.request_structured_json(
            &self.fast_model,
            "connection_probe",
            connection_probe_schema(),
            "Return JSON only. Respond with {\"ok\":true}.",
            vec![json!({
                "type": "text",
                "text": "Confirm that the configured Anthropic reasoning provider is reachable."
            })],
        )?;
        let envelope: ConnectionProbeEnvelope =
            serde_json::from_value(payload).map_err(|error| error.to_string())?;
        if envelope.ok {
            Ok(self.status.clone())
        } else {
            Err("Anthropic provider returned an invalid connection probe response.".to_string())
        }
    }

    fn summarize_semantic_changes(
        &self,
        request: &SemanticSummaryRequest,
    ) -> SemanticSummaryResponse {
        match self.summarize_via_anthropic(request) {
            Ok(response) => response,
            Err(_) => self.fallback.summarize_semantic_changes(request),
        }
    }

    fn complete_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        self.complete_chat_via_anthropic(request)
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
        Ok(chat_completion_response(message, self.status.clone()))
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
        Ok(chat_completion_response(message, self.status.clone()))
    }
}

impl AnthropicReasoningProvider {
    fn propose_semantic_mutations_via_anthropic(
        &self,
        request: &SemanticMutationProposalRequest,
    ) -> Result<Vec<MutationProposal>, String> {
        let payload = self.request_structured_json(
            &self.proposal_model,
            "semantic_mutation_proposals",
            semantic_mutation_proposal_schema(),
            semantic_mutation_proposal_developer_prompt(),
            anthropic_semantic_mutation_message_blocks(request),
        )?;
        parse_semantic_mutation_proposals_payload(payload, request)
    }

    fn summarize_via_anthropic(
        &self,
        request: &SemanticSummaryRequest,
    ) -> Result<SemanticSummaryResponse, String> {
        let payload = self.request_structured_json(
            &self.fast_model,
            "semantic_change_summary",
            semantic_summary_schema(),
            semantic_summary_developer_prompt(),
            vec![json!({
                "type": "text",
                "text": semantic_summary_user_prompt(request)
            })],
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
        model: &str,
        schema_name: &str,
        schema: Value,
        developer_prompt: &str,
        user_blocks: Vec<Value>,
    ) -> Result<Value, String> {
        request_anthropic_structured_json(
            &self.client,
            &self.base_url,
            &self.api_key,
            model,
            schema_name,
            schema,
            developer_prompt,
            user_blocks,
        )
    }

    fn complete_chat_via_anthropic(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        let message = request_anthropic_text(
            &self.client,
            &self.base_url,
            &self.api_key,
            &self.fast_model,
            request,
        )?;
        Ok(chat_completion_response(message, self.status.clone()))
    }
}

fn anthropic_semantic_mutation_message_blocks(
    request: &SemanticMutationProposalRequest,
) -> Vec<Value> {
    let mut blocks = Vec::new();
    let stable_context = json!({
        "capability_context": sysml_semantic_mutation_capability_context(),
        "semantic_context": request.semantic_context,
        "cognitive_context": request.cognitive_context,
    });
    blocks.push(json!({
        "type": "text",
        "text": serde_json::to_string_pretty(&stable_context).unwrap_or_else(|_| "{}".to_string()),
        "cache_control": { "type": "ephemeral" }
    }));
    let dynamic_request = json!({
        "agent_guidance": semantic_mutation_agent_guidance(),
        "request": {
            "design_intent": request.design_intent,
            "workspace_revision": request.workspace_revision,
            "focus": request.focus,
            "task_goal_guidance": request.task_goal_guidance,
            "quality_goal_guidance": request.quality_goal_guidance,
            "reasoning_tool_results": request.reasoning_tool_results,
        }
    });
    blocks.push(json!({
        "type": "text",
        "text": serde_json::to_string_pretty(&dynamic_request).unwrap_or_else(|_| "{}".to_string())
    }));
    blocks
}
