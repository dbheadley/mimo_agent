---
description: "Use when editing AGENTS.md, .instructions.md, .agent.md, or workspace prompts for this repo. Covers the thin-router architecture, canonical .github locations, removal of obsolete entry points, and anti-patterns that recreate context bloat."
applyTo:
   - "AGENTS.md"
   - ".github/instructions/**"
   - ".github/agents/**"
   - ".github/prompts/**"
---

# Workspace Customization Rules

## Decision Tree

1. Does the rule truly apply to every task in the repo?
   - Keep it in the root `AGENTS.md` router.
2. Is the rule file-scoped or topic-scoped?
   - Put it in `.github/instructions/` with a narrow `applyTo` or a keyword-rich `description`.
3. Is the change about intent routing or role isolation?
   - Put it in `.github/agents/`.
4. Is the content mostly reference material or lookup tables?
   - Prefer an instruction file with no broad `applyTo`.
5. Does an old entry point still exist after the canonical `.github` surface is stable?
   - Migrate any remaining references and delete it instead of keeping a permanent shim.

## Directives

- Canonical workspace customization files live under `.github/instructions/` and `.github/agents/`.
- Keep the root `AGENTS.md` short and decision-tree driven.
- Avoid `applyTo: "**"` unless the content truly belongs in every request.
- Use keyword-rich `description` text so on-demand discovery works.
- Keep one concern per file and one canonical home per topic.

## Recommendations

- Use custom agents for routing and role boundaries, not for storing long reference guides.
- Use instruction files for the actual reference content that future tasks should load.
- Remove obsolete redirect files once the canonical `.github` surfaces are stable and all references are updated.