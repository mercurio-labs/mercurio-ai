use crate::{
    AiProviderDescriptor, AiProviderFieldDescriptor, DEFAULT_ANTHROPIC_FAST_MODEL,
    DEFAULT_ANTHROPIC_PROPOSAL_MODEL, DEFAULT_OPENAI_MODEL,
};

pub fn provider_descriptors() -> Vec<AiProviderDescriptor> {
    vec![
        AiProviderDescriptor {
            id: "openai".to_string(),
            label: "OpenAI".to_string(),
            capabilities: vec![
                "chat".to_string(),
                "assessment".to_string(),
                "mutation".to_string(),
                "structured_outputs".to_string(),
            ],
            credential_schema: vec![secret_field()],
            settings_schema: vec![
                field("model", "Model", "model", true),
                field("baseUrl", "Base URL", "url", true),
            ],
            default_models: vec![DEFAULT_OPENAI_MODEL.to_string()],
        },
        AiProviderDescriptor {
            id: "azure_openai".to_string(),
            label: "Azure OpenAI".to_string(),
            capabilities: vec![
                "chat".to_string(),
                "assessment".to_string(),
                "mutation".to_string(),
            ],
            credential_schema: vec![secret_field()],
            settings_schema: vec![
                field("deployment", "Deployment", "model", true),
                field("baseUrl", "Base URL", "url", true),
            ],
            default_models: Vec::new(),
        },
        AiProviderDescriptor {
            id: "anthropic".to_string(),
            label: "Anthropic".to_string(),
            capabilities: vec![
                "chat".to_string(),
                "assessment".to_string(),
                "mutation".to_string(),
            ],
            credential_schema: vec![secret_field()],
            settings_schema: vec![
                field("proposalModel", "Proposal model", "model", true),
                field("fastModel", "Fast model", "model", true),
                field("baseUrl", "Base URL", "url", true),
            ],
            default_models: vec![
                DEFAULT_ANTHROPIC_PROPOSAL_MODEL.to_string(),
                DEFAULT_ANTHROPIC_FAST_MODEL.to_string(),
            ],
        },
    ]
}

fn secret_field() -> AiProviderFieldDescriptor {
    AiProviderFieldDescriptor {
        id: "apiKey".to_string(),
        label: "API key".to_string(),
        kind: Some("secret".to_string()),
        secret: true,
        required: true,
    }
}

fn field(id: &str, label: &str, kind: &str, required: bool) -> AiProviderFieldDescriptor {
    AiProviderFieldDescriptor {
        id: id.to_string(),
        label: label.to_string(),
        kind: Some(kind.to_string()),
        secret: false,
        required,
    }
}
