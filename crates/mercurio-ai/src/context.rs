use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde_json::{Value, json};

use crate::{SemanticAgentToolResult, semantic_agent_tool_id};
use mercurio_core::{
    CognitiveContext, CognitiveDiagnostic, CognitiveDiagnosticSeverity, CognitiveElement,
    CognitiveFocus, CognitiveRelationship, Edge, Element, ElementRef, Graph, KirDocument, NodeId,
    SemanticArtifact, SemanticElementRef, SemanticWorkspaceRef, SourceSpanRef, WorkspaceRevision,
    stable_digest,
};

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
