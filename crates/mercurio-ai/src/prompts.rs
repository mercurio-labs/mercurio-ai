use crate::ChatCompletionRequest;

pub(crate) fn chat_developer_prompt(request: &ChatCompletionRequest) -> String {
    let mut context_entries = request.context.clone();
    if let Some(cognitive_context) = request.cognitive_context.as_ref() {
        let cognitive_json = serde_json::to_string_pretty(cognitive_context).unwrap_or_default();
        context_entries.push(format!("Compiled CognitiveContext:\n{cognitive_json}"));
    }
    let context = if context_entries.is_empty() {
        "No current workspace context was supplied.".to_string()
    } else {
        context_entries.join("\n")
    };

    format!(
        "Use this Mercurio model context as the authoritative current workspace state.\n\
         Critical grounding rules:\n\
         - Treat live editor/workspace snapshots as newer and more authoritative than chat history, compiled metadata, or prior assistant messages.\n\
         - Do not claim a file, package, requirement, element, relationship, diagram, commit, or edit exists unless it appears in the supplied current workspace context or validated tool output.\n\
         - If prior chat says something was created but the current workspace snapshot does not contain it, say it is not present in the current model.\n\
         - If context includes `Metamodel lookup result:` lines, treat them as authoritative KIR evidence and do not substitute generic SysML, UML, or modeling-language knowledge.\n\
         - For questions asking what is in the current model, summarize only elements present in the current workspace context and state when expected evidence is missing.\n\n\
         Artifact contract:\n\
         - Product artifacts are typed response metadata, not visible Markdown. Keep prose readable and avoid exposing implementation JSON to the user.\n\
         - If this provider path cannot emit typed metadata directly and the user explicitly asks for a diagram, use the temporary compatibility fallback: one fenced ```diagram JSON block after prose. The boundary will lift it into a typed artifact and strip it from UI text.\n\
         - Compatibility diagram JSON must include version: 1, kind, lensId, title, root, and optional query/layout/style fields. Choose lensId from: type-hierarchy, decomposition, neighborhood, coverage, composition, references, dependencies, package-tree, impact, validation.\n\
         - If this provider path cannot emit typed metadata directly and the user explicitly asks for a matrix or table, use one fenced ```matrix JSON block after prose. Matrix JSON must include version: 1, title, rows, columns, and cells.\n\
         - If this provider path cannot emit typed metadata directly and an overlay is needed, use one fenced ```view_action JSON block after prose. Omit it when no overlay change is needed.\n\n\
         Current workspace context:\n{}",
        context
    )
}
