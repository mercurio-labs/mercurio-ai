use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Value, json};

use mercurio_core::runtime::Runtime;
use mercurio_core::{
    AnalysisScope, CapabilityRegistry, CapabilityRunReport, CapabilityRunRequest, CapabilityTarget,
    ElementRef, FeasibilityStatus, GoalEvaluation, KirDocument, MutationContext, SemanticGoalCheck,
    SemanticGoalSpec, SemanticWorkspaceSnapshot, WorkspaceRevision, default_stdlib_path,
};
use mercurio_reasoner_api::{
    FindingSeverity, ReasoningReport, SemanticArtifactRef, SemanticContextKind, SemanticContextRef,
};
use mercurio_reference_capabilities::{
    analyze_requirement_coverage, analyze_semantic_impact, analyze_state_machine_simulation,
};
use mercurio_requirements::{evaluate_semantic_goal, explain_semantic_goal};
use mercurio_sysml::{
    compile_sysml_text, enrich_sysml_semantic_reasoning_context_with_child_affordances,
    load_authoring_project_from_sysml, sysml_mutation_feasibility_service,
    sysml_semantic_reasoning_context_from_authoring_project,
};

use crate::{
    SemanticAgentRun, SemanticAgentRunRequest, SemanticAgentRunStatus, SemanticAgentStep,
    SemanticAgentToolFinding, SemanticAgentToolKind, SemanticAgentToolMode,
    SemanticAgentToolResult, SemanticContextBuilder, SemanticMutationProposalProvider,
    SemanticMutationProposalRequest, propose_checked_semantic_mutations,
};

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

pub(crate) fn select_semantic_agent_tools(
    request: &SemanticAgentRunRequest,
) -> Vec<SemanticAgentToolKind> {
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

pub(crate) fn semantic_agent_tool_id(tool: SemanticAgentToolKind) -> &'static str {
    match tool {
        SemanticAgentToolKind::RequirementCoverage => "requirement_coverage",
        SemanticAgentToolKind::SemanticImpact => "semantic_impact",
        SemanticAgentToolKind::StateSimulation => "state_simulation",
        SemanticAgentToolKind::ModelInspection => "model_inspection",
    }
}

fn severity_label(severity: &FindingSeverity) -> &'static str {
    match severity {
        FindingSeverity::Info => "info",
        FindingSeverity::Warning => "warning",
        FindingSeverity::Error => "error",
        FindingSeverity::Critical => "critical",
    }
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
