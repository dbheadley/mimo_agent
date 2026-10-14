# Plan: Routed AGENTS Architecture

Use this prompt when planning changes to the workspace customization system for mimo_pack.

## Objective

Keep the root `AGENTS.md` thin, keep canonical task-specific guidance in `.github/instructions/`, and keep routing behavior in `.github/agents/`.

## Planning Questions

1. Which rules are truly always-on and belong in the root router?
2. Which topics deserve a dedicated instruction file because they are task-specific, file-scoped, or reference-heavy?
3. Which roles deserve a custom agent because they route or isolate context rather than store reference content?
4. Which obsolete entry points can be removed completely once the canonical surfaces are stable?
5. Which topics currently exist in more than one place and need a single canonical home?

## Current Canonical Layout

- Root router: `AGENTS.md`
- Instructions: `.github/instructions/*.instructions.md`
- Agents: `.github/agents/*.agent.md`

## Recommended File Families

- Notebook workflows
- Library core rules
- Compute-oriented library rules
- Plotting rules
- Testing rules
- Data loading and time alignment reference
- Interop and utility discovery
- Workspace customization rules

## Recommended Agents

- `mimo-router`
- `mimo-new-analysis`
- `mimo-continue-analysis`
- `mimo-library-change`
- `mimo-customization`

## Anti-Patterns To Avoid

- Recreating a monolithic AGENTS file in `.github/instructions/`
- Broad `applyTo: "**"` for content that is not truly global
- Using custom agents as large reference manuals
- Keeping obsolete redirect files after the canonical `.github` layout is stable

## Deliverable

Produce a file plan that names:

1. the canonical file for each topic
2. any obsolete entry points that should be deleted or updated during migration
3. the migration order
4. the validation steps that confirm the final structure stays narrow and task-specific