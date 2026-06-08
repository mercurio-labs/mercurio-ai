use serde::Deserialize;
use serde_json::{Value, json};

use mercurio_sysml::sysml_semantic_mutation_capability_context;

use crate::{
    MutationProposal, SemanticMutationProposalRequest, SemanticSummaryRequest, WorkspaceRevision,
};

#[derive(Debug, Deserialize)]
pub(crate) struct SemanticSummaryEnvelope {
    pub(crate) title: String,
    pub(crate) body: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ConnectionProbeEnvelope {
    pub(crate) ok: bool,
}

#[derive(Debug, Deserialize)]
struct SemanticMutationProposalEnvelope {
    proposals: Vec<MutationProposal>,
}

pub(crate) fn semantic_summary_developer_prompt() -> &'static str {
    "Write a concise engineering change summary from the supplied semantic diff. \
     Return JSON only. Do not invent changes that are not present. Prefer domain \
     language from element labels and kinds. Keep the title under 72 characters."
}

pub(crate) fn semantic_summary_user_prompt(request: &SemanticSummaryRequest) -> String {
    serde_json::to_string_pretty(request).unwrap_or_else(|_| "{}".to_string())
}

pub(crate) fn semantic_mutation_proposal_developer_prompt() -> &'static str {
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

pub(crate) fn semantic_mutation_proposal_user_prompt(
    request: &SemanticMutationProposalRequest,
) -> String {
    serde_json::to_string_pretty(&json!({
        "capability_context": sysml_semantic_mutation_capability_context(),
        "agent_guidance": semantic_mutation_agent_guidance(),
        "request": request,
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

pub(crate) fn semantic_mutation_agent_guidance() -> Value {
    json!({
        "element_ref_format": "Use dot-qualified names such as HybridVehicle.Vehicle, never HybridVehicle::Vehicle.",
        "current_state_rule": "Treat cognitive_context.elements and semantic_context.elements as already existing. Do not re-add them.",
        "grounding_rule": "Use cognitive_context as authoritative structured context. Prefer KIR element_id for evidence identity, and use qualified_name only as ElementRef display/input metadata.",
        "citation_rule": "When reasoning_tool_results or cognitive_context.diagnostics identify a relevant finding, cite its id and related element in proposal evidence.",
        "operation_rule": "Every proposal must contain at least one operation. Empty proposals are ignored.",
        "quality_rule": "When a requirement already exists without id or text, prefer SetAttribute operations for id and text before adding more requirements.",
        "batching_rule": "Batch related operations only when their containers and referenced types already exist in the current semantic context.",
        "affordance_rule": "Prefer operations supported by semantic_context.affordances for the target element."
    })
}

pub(crate) fn parse_semantic_mutation_proposals_payload(
    payload: Value,
    request: &SemanticMutationProposalRequest,
) -> Result<Vec<MutationProposal>, String> {
    let envelope: SemanticMutationProposalEnvelope =
        serde_json::from_value(payload).map_err(|error| error.to_string())?;
    Ok(envelope
        .proposals
        .into_iter()
        .map(|mut proposal| {
            proposal.workspace_revision = WorkspaceRevision {
                fingerprint: request.workspace_revision.fingerprint.clone(),
            };
            proposal
        })
        .collect())
}

pub(crate) fn semantic_summary_schema() -> Value {
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

pub(crate) fn semantic_mutation_proposal_schema() -> Value {
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

pub(crate) fn connection_probe_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "ok": { "type": "boolean" }
        },
        "required": ["ok"]
    })
}
