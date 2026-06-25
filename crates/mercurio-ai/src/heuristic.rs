use crate::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessageRole, ElementRef, MutationEvidence,
    MutationProposal, ReasoningProviderStatus, SemanticChangeItem, SemanticChangeKind,
    SemanticMutation, SemanticMutationProposalRequest, SemanticSummaryRequest,
    SemanticSummaryResponse, chat_completion_response,
};
use mercurio_core::SemanticElementKind;

const DEFAULT_REQUIREMENT_COUNT: usize = 10;

pub(crate) fn heuristic_semantic_summary(
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

pub(crate) fn heuristic_semantic_mutation_proposals(
    request: &SemanticMutationProposalRequest,
) -> Vec<MutationProposal> {
    let intent = request.design_intent.to_ascii_lowercase();
    if is_requirements_package_request(&intent) {
        return heuristic_requirements_package_proposals(request);
    }

    if !(intent.contains("hybrid") || intent.contains("efficiency")) {
        return Vec::new();
    }

    if request.semantic_context.is_none() && request.cognitive_context.is_none() {
        return vec![heuristic_regenerative_braking_proposal(request)];
    }

    if !request_context_has_element(request, "HybridVehicle") {
        return vec![MutationProposal {
            intent: "Create the hybrid vehicle model package".to_string(),
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
            operations: vec![
                add_element_definition("HybridVehicle", "PartDefinition", "HybridVehicle"),
                add_element_definition("HybridVehicle", "RequirementDefinition", "ImproveEfficiency"),
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
            operations: vec![
                add_element_definition("HybridVehicle", "PartDefinition", "Engine"),
                add_element_definition("HybridVehicle", "PartDefinition", "ElectricMotor"),
                add_element_definition("HybridVehicle", "PartDefinition", "BatteryPack"),
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
            operations: vec![
                add_element_usage(
                    "HybridVehicle.HybridVehicle",
                    "PartUsage",
                    "engine",
                    Some("HybridVehicle.Engine"),
                ),
                add_element_usage(
                    "HybridVehicle.HybridVehicle",
                    "PartUsage",
                    "motor",
                    Some("HybridVehicle.ElectricMotor"),
                ),
                add_element_usage(
                    "HybridVehicle.HybridVehicle",
                    "PartUsage",
                    "battery",
                    Some("HybridVehicle.BatteryPack"),
                ),
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

fn is_requirements_package_request(intent: &str) -> bool {
    intent.contains("requirement")
        && intent.contains("package")
        && (intent.contains("10") || intent.contains("ten"))
}

fn heuristic_requirements_package_proposals(
    request: &SemanticMutationProposalRequest,
) -> Vec<MutationProposal> {
    if !request_context_has_element(request, "Requirements") {
        return vec![MutationProposal {
            intent: "Create requirements package".to_string(),
            operations: vec![SemanticMutation::AddPackage {
                target_file: "requirements.sysml".to_string(),
                name: "Requirements".to_string(),
            }],
            evidence: vec![MutationEvidence {
                element: None,
                summary: "The request asks for a dedicated requirements package.".to_string(),
            }],
            rationale: Some(
                "A package provides the namespace that can own the requested requirements."
                    .to_string(),
            ),
            workspace_revision: request.workspace_revision.clone(),
        }];
    }

    let missing = default_requirement_specs()
        .into_iter()
        .filter(|spec| {
            !request_context_has_element(request, &format!("Requirements.{}", spec.name))
        })
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Vec::new();
    }

    let mut operations = Vec::new();
    for spec in missing {
        let element = ElementRef::new(format!("Requirements.{}", spec.name));
        operations.push(add_element_definition(
            "Requirements",
            "RequirementDefinition",
            spec.name,
        ));
        operations.push(SemanticMutation::SetAttribute {
            element: element.clone(),
            attribute: "id".to_string(),
            value: serde_json::Value::String(spec.id.to_string()),
        });
        operations.push(SemanticMutation::SetAttribute {
            element,
            attribute: "text".to_string(),
            value: serde_json::Value::String(spec.text.to_string()),
        });
    }

    vec![MutationProposal {
        intent: format!("Add {DEFAULT_REQUIREMENT_COUNT} requirement definitions"),
        operations,
        evidence: vec![MutationEvidence {
            element: Some(ElementRef::new("Requirements")),
            summary: "The package exists and can own requirement definitions.".to_string(),
        }],
        rationale: Some(
            "Each requirement is created as a SysML requirement definition and given explicit id and text attributes."
                .to_string(),
        ),
        workspace_revision: request.workspace_revision.clone(),
    }]
}

#[derive(Clone, Copy)]
struct RequirementSpec {
    name: &'static str,
    id: &'static str,
    text: &'static str,
}

fn default_requirement_specs() -> [RequirementSpec; DEFAULT_REQUIREMENT_COUNT] {
    [
        RequirementSpec {
            name: "FunctionalPerformance",
            id: "REQ-001",
            text: "The system shall deliver its primary function under nominal operating conditions.",
        },
        RequirementSpec {
            name: "UserSafety",
            id: "REQ-002",
            text: "The system shall protect users from hazardous operating states.",
        },
        RequirementSpec {
            name: "FaultDetection",
            id: "REQ-003",
            text: "The system shall detect and report critical faults.",
        },
        RequirementSpec {
            name: "RecoveryBehavior",
            id: "REQ-004",
            text: "The system shall enter a controlled recovery mode after a critical fault.",
        },
        RequirementSpec {
            name: "DataIntegrity",
            id: "REQ-005",
            text: "The system shall preserve integrity of operational data across transactions.",
        },
        RequirementSpec {
            name: "Availability",
            id: "REQ-006",
            text: "The system shall remain available during planned operating periods.",
        },
        RequirementSpec {
            name: "Maintainability",
            id: "REQ-007",
            text: "The system shall support diagnostic maintenance without replacing unrelated components.",
        },
        RequirementSpec {
            name: "Interoperability",
            id: "REQ-008",
            text: "The system shall exchange required data with external interfaces using documented contracts.",
        },
        RequirementSpec {
            name: "EnvironmentalTolerance",
            id: "REQ-009",
            text: "The system shall operate within the specified environmental envelope.",
        },
        RequirementSpec {
            name: "VerificationEvidence",
            id: "REQ-010",
            text: "The system shall maintain verification evidence for each allocated requirement.",
        },
    ]
}

fn heuristic_regenerative_braking_proposal(
    request: &SemanticMutationProposalRequest,
) -> MutationProposal {
    let operations =
        if !request_context_has_element(request, "HybridVehicle.RegenerativeBrakingSystem") {
            vec![add_element_definition(
                "HybridVehicle",
                "PartDefinition",
                "RegenerativeBrakingSystem",
            )]
        } else if !request_context_has_element(
            request,
            "HybridVehicle.HybridVehicle.regenerativeBraking",
        ) {
            vec![add_element_usage(
                "HybridVehicle.HybridVehicle",
                "PartUsage",
                "regenerativeBraking",
                Some("HybridVehicle.RegenerativeBrakingSystem"),
            )]
        } else {
            vec![SemanticMutation::AddRelationship {
                kind: "satisfy".to_string(),
                source: ElementRef::new("HybridVehicle.RegenerativeBrakingSystem"),
                target: ElementRef::new("HybridVehicle.ImproveEfficiency"),
            }]
        };

    MutationProposal {
        intent: "Improve hybrid vehicle efficiency through regenerative braking".to_string(),
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

fn add_element_definition(container: &str, metaclass: &str, name: &str) -> SemanticMutation {
    SemanticMutation::AddElement {
        container: ElementRef::new(container),
        kind: SemanticElementKind::new(metaclass),
        name: name.to_string(),
        ty: None,
        specializes: Vec::new(),
        properties: Default::default(),
    }
}

fn add_element_usage(
    container: &str,
    metaclass: &str,
    name: &str,
    ty: Option<&str>,
) -> SemanticMutation {
    SemanticMutation::AddElement {
        container: ElementRef::new(container),
        kind: SemanticElementKind::new(metaclass),
        name: name.to_string(),
        ty: ty.map(ElementRef::new),
        specializes: Vec::new(),
        properties: Default::default(),
    }
}

pub(crate) fn request_context_has_element(
    request: &SemanticMutationProposalRequest,
    element: &str,
) -> bool {
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

pub(crate) fn heuristic_chat_completion(
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
    chat_completion_response(
        format!(
            "I received \"{latest}\". {context} Configure OpenAI or Azure OpenAI in Settings to generate provider-backed answers."
        ),
        provider,
    )
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
