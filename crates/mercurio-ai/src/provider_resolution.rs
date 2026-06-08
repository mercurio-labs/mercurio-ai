use crate::provider_http::http_client;
use crate::provider_registry::provider_descriptors;
use crate::{
    AnthropicReasoningProvider, AzureOpenAiReasoningProvider, DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_FAST_MODEL, DEFAULT_ANTHROPIC_PROPOSAL_MODEL, DEFAULT_AZURE_OPENAI_PATH,
    DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL, HeuristicReasoningProvider,
    OpenAiReasoningProvider, ReasoningProviderConfigOverrides, ReasoningProviderKind,
    ReasoningProviderSecretOverrides, ReasoningProviderStatus, ResolvedReasoningProvider,
};

type ConfiguredProviderFactory = fn(
    &ReasoningProviderConfigOverrides,
    &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider>;
type EnvProviderFactory =
    fn(&ReasoningProviderSecretOverrides) -> Option<ResolvedReasoningProvider>;

pub(crate) struct ProviderImplementationDescriptor {
    pub kind: ReasoningProviderKind,
    pub env_aliases: &'static [&'static str],
    pub from_config: ConfiguredProviderFactory,
    pub from_env: EnvProviderFactory,
}

pub(crate) fn provider_implementations() -> &'static [ProviderImplementationDescriptor] {
    &[
        ProviderImplementationDescriptor {
            kind: ReasoningProviderKind::OpenAi,
            env_aliases: &["openai"],
            from_config: configured_openai_provider,
            from_env: env_openai_provider,
        },
        ProviderImplementationDescriptor {
            kind: ReasoningProviderKind::AzureOpenAi,
            env_aliases: &["azure_openai", "azure-openai"],
            from_config: configured_azure_openai_provider,
            from_env: env_azure_openai_provider,
        },
        ProviderImplementationDescriptor {
            kind: ReasoningProviderKind::Anthropic,
            env_aliases: &["anthropic", "claude"],
            from_config: configured_anthropic_provider,
            from_env: env_anthropic_provider,
        },
    ]
}

pub(crate) fn configured_provider_from_registry(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider> {
    let kind = config.provider.as_ref()?;
    if matches!(kind, ReasoningProviderKind::Heuristic) {
        return Some(ResolvedReasoningProvider::Heuristic(heuristic_provider()));
    }
    provider_implementations()
        .iter()
        .find(|implementation| &implementation.kind == kind)
        .and_then(|implementation| (implementation.from_config)(config, secrets))
}

pub(crate) fn configured_provider_kind(
    config: &ReasoningProviderConfigOverrides,
) -> Option<ReasoningProviderKind> {
    let kind = config.provider.as_ref()?;
    if matches!(kind, ReasoningProviderKind::Heuristic)
        || provider_implementations()
            .iter()
            .any(|implementation| &implementation.kind == kind)
    {
        Some(kind.clone())
    } else {
        None
    }
}

fn configured_openai_provider(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider> {
    openai_provider_from_config(config, secrets).map(ResolvedReasoningProvider::OpenAi)
}

fn configured_azure_openai_provider(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider> {
    azure_openai_provider_from_config(config, secrets).map(ResolvedReasoningProvider::AzureOpenAi)
}

fn configured_anthropic_provider(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider> {
    anthropic_provider_from_config(config, secrets).map(ResolvedReasoningProvider::Anthropic)
}

fn env_openai_provider(
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider> {
    openai_provider_from_env(secrets).map(ResolvedReasoningProvider::OpenAi)
}

fn env_azure_openai_provider(
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider> {
    azure_openai_provider_from_env(secrets).map(ResolvedReasoningProvider::AzureOpenAi)
}

fn env_anthropic_provider(
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<ResolvedReasoningProvider> {
    anthropic_provider_from_env(secrets).map(ResolvedReasoningProvider::Anthropic)
}

pub(crate) fn resolve_reasoning_provider_from_env(
    secrets: &ReasoningProviderSecretOverrides,
) -> ResolvedReasoningProvider {
    let requested = std::env::var("MERCURIO_AI_PROVIDER")
        .or_else(|_| std::env::var("MERCURIO_REASONING_PROVIDER"))
        .unwrap_or_default()
        .to_ascii_lowercase();

    if let Some(implementation) = provider_implementation_for_env(&requested)
        && let Some(provider) = (implementation.from_env)(secrets)
    {
        return provider;
    }

    ResolvedReasoningProvider::Heuristic(heuristic_provider())
}

fn provider_implementation_for_env(
    requested: &str,
) -> Option<&'static ProviderImplementationDescriptor> {
    let normalized = requested.trim();
    if normalized.is_empty() {
        return provider_implementations()
            .iter()
            .find(|implementation| implementation.kind == ReasoningProviderKind::OpenAi);
    }
    provider_implementations().iter().find(|implementation| {
        implementation
            .env_aliases
            .iter()
            .any(|alias| *alias == normalized)
    })
}

pub(crate) fn heuristic_provider() -> HeuristicReasoningProvider {
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

pub(crate) fn openai_provider_from_env(
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

pub(crate) fn openai_provider_from_config(
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

pub(crate) fn azure_openai_provider_from_env(
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

pub(crate) fn azure_openai_provider_from_config(
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

pub(crate) fn anthropic_provider_from_env(
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<AnthropicReasoningProvider> {
    let api_key = secrets
        .anthropic_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("MERCURIO_ANTHROPIC_API_KEY"))
                .ok()
                .filter(|value| !value.trim().is_empty())
        })?;
    let proposal_model = std::env::var("MERCURIO_ANTHROPIC_PROPOSAL_MODEL")
        .or_else(|_| std::env::var("ANTHROPIC_PROPOSAL_MODEL"))
        .or_else(|_| std::env::var("MERCURIO_CLAUDE_PROPOSAL_MODEL"))
        .unwrap_or_else(|_| DEFAULT_ANTHROPIC_PROPOSAL_MODEL.to_string());
    let fast_model = std::env::var("MERCURIO_ANTHROPIC_FAST_MODEL")
        .or_else(|_| std::env::var("ANTHROPIC_FAST_MODEL"))
        .or_else(|_| std::env::var("MERCURIO_CLAUDE_FAST_MODEL"))
        .unwrap_or_else(|_| DEFAULT_ANTHROPIC_FAST_MODEL.to_string());
    let base_url = std::env::var("MERCURIO_ANTHROPIC_BASE_URL")
        .or_else(|_| std::env::var("ANTHROPIC_BASE_URL"))
        .unwrap_or_else(|_| DEFAULT_ANTHROPIC_BASE_URL.to_string());

    Some(AnthropicReasoningProvider {
        client: http_client(),
        api_key,
        proposal_model: proposal_model.clone(),
        fast_model: fast_model.clone(),
        base_url,
        status: anthropic_status(
            format!("{proposal_model} / {fast_model}"),
            "Anthropic Messages API configured from environment.",
        ),
        fallback: heuristic_provider(),
    })
}

pub(crate) fn anthropic_provider_from_config(
    config: &ReasoningProviderConfigOverrides,
    secrets: &ReasoningProviderSecretOverrides,
) -> Option<AnthropicReasoningProvider> {
    let api_key = secrets
        .anthropic_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)?;
    let proposal_model = config
        .anthropic_proposal_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_ANTHROPIC_PROPOSAL_MODEL)
        .to_string();
    let fast_model = config
        .anthropic_fast_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_ANTHROPIC_FAST_MODEL)
        .to_string();
    let base_url = config
        .anthropic_base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_ANTHROPIC_BASE_URL)
        .to_string();

    Some(AnthropicReasoningProvider {
        client: http_client(),
        api_key,
        proposal_model: proposal_model.clone(),
        fast_model: fast_model.clone(),
        base_url,
        status: anthropic_status(
            format!("{proposal_model} / {fast_model}"),
            "Anthropic Messages API configured from application settings and stored credential.",
        ),
        fallback: heuristic_provider(),
    })
}

fn anthropic_status(model_label: String, detail: &str) -> ReasoningProviderStatus {
    ReasoningProviderStatus {
        kind: ReasoningProviderKind::Anthropic,
        provider_label: "Anthropic".to_string(),
        detail: detail.to_string(),
        structured_outputs: true,
        model_label: Some(model_label),
    }
}

pub(crate) fn configured_provider_missing_message(
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
                missing.push(provider_setting_label(&provider, "deployment"));
            }
            if config
                .azure_openai_base_url
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push(provider_setting_label(&provider, "baseUrl"));
            }
            if secrets
                .azure_openai_api_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push(provider_secret_label(&provider));
            }
            provider_missing_message(&provider, missing)
        }
        ReasoningProviderKind::OpenAi => {
            if config
                .openai_model
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push(provider_setting_label(&provider, "model"));
            }
            if config
                .openai_base_url
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push(provider_setting_label(&provider, "baseUrl"));
            }
            if secrets
                .openai_api_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push(provider_secret_label(&provider));
            }
            provider_missing_message(&provider, missing)
        }
        ReasoningProviderKind::Anthropic => {
            if secrets
                .anthropic_api_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            {
                missing.push(provider_secret_label(&provider));
            }
            provider_missing_message(&provider, missing)
        }
        ReasoningProviderKind::Heuristic => "No external AI provider is configured.".to_string(),
    }
}

fn provider_missing_message(provider: &ReasoningProviderKind, missing: Vec<String>) -> String {
    format!(
        "{} settings are incomplete. Missing {}.",
        provider_label(provider),
        missing.join(", ")
    )
}

fn provider_label(provider: &ReasoningProviderKind) -> String {
    provider_descriptor(provider)
        .map(|descriptor| descriptor.label)
        .unwrap_or_else(|| match provider {
            ReasoningProviderKind::Heuristic => "Heuristic".to_string(),
            ReasoningProviderKind::OpenAi => "OpenAI".to_string(),
            ReasoningProviderKind::AzureOpenAi => "Azure OpenAI".to_string(),
            ReasoningProviderKind::Anthropic => "Anthropic".to_string(),
        })
}

fn provider_setting_label(provider: &ReasoningProviderKind, field_id: &str) -> String {
    provider_descriptor(provider)
        .and_then(|descriptor| {
            descriptor
                .settings_schema
                .into_iter()
                .find(|field| field.id == field_id)
        })
        .map(|field| field.label)
        .unwrap_or_else(|| field_id.to_string())
}

fn provider_secret_label(provider: &ReasoningProviderKind) -> String {
    provider_descriptor(provider)
        .and_then(|descriptor| descriptor.credential_schema.into_iter().next())
        .map(|field| format!("stored {}", field.label.to_ascii_lowercase()))
        .unwrap_or_else(|| "stored API key".to_string())
}

fn provider_descriptor(provider: &ReasoningProviderKind) -> Option<crate::AiProviderDescriptor> {
    let provider_id = match provider {
        ReasoningProviderKind::Heuristic => return None,
        ReasoningProviderKind::OpenAi => "openai",
        ReasoningProviderKind::AzureOpenAi => "azure_openai",
        ReasoningProviderKind::Anthropic => "anthropic",
    };
    provider_descriptors()
        .into_iter()
        .find(|descriptor| descriptor.id == provider_id)
}

pub(crate) fn normalize_azure_openai_base_url(value: &str) -> String {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.ends_with("/openai/v1/responses") {
        return trimmed.to_string();
    }
    if trimmed.ends_with("/openai/v1") {
        return format!("{trimmed}/responses");
    }
    format!("{trimmed}{DEFAULT_AZURE_OPENAI_PATH}")
}
