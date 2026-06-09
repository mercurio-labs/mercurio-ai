# mercurio-ai — Agent Orientation

AI-facing Mercurio crate workspace. Provides AI orchestration helpers that bridge Mercurio's semantic workspace to LLM-based reasoning pipelines.

---

## Contents

```
crates/mercurio-ai/    — AI orchestration crate
```

---

## Dependency Direction

Sits in the reasoning/plugin layer alongside `mercurio-plugins`. May depend on foundation and sysml crates. Must **not** be depended on by those workspaces (enforced by `mercurio-foundation/repo-boundaries.json`).

---

## Build & Test

```powershell
cargo build
cargo test
```

---

## Further Reading

- [README.md](README.md) — workspace overview
- [../mercurio-plugins/AGENTS.md](../mercurio-plugins/AGENTS.md) — plugin contracts and ABI reference
- [../mercurio-foundation/repo-boundaries.json](../mercurio-foundation/repo-boundaries.json) — dependency boundary rules
