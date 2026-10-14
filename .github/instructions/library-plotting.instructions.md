---
description: "Use when changing mimo_pack plotting code in plot/. Covers public/private plotting split, axes handling, labeling, colormap rules, and when plotting changes should trigger test updates."
applyTo:
  - "mimo_pack/plot/**"
---

# Plotting Rules

## Decision Tree

1. Is this the public plot entry point?
   - Validate rich input, extract metadata, and call a private plotting helper.
2. Is this the private plotting helper?
   - Keep it pure matplotlib and data-array agnostic.
3. Does the change introduce sequential, diverging, or categorical colors?
   - Use `mimo_pack.plot.colors` instead of hardcoded matplotlib defaults.
4. Does the change alter public behavior or expected rendering structure?
   - Also consult `.github/instructions/library-testing.instructions.md` for test updates.

## Directives

- Accept an optional `ax`; create one if `None`.
- Return the `Axes` or `Figure` object created by the public function.
- Use constants from `mimo_pack.consts` for dimension lookups.
- Auto-label axes and titles from dimensions and `ATTR_UNITS` when applicable.
- Keep plotting modules free of disk I/O and loader/session logic.

## Color Rules

- Use `MIMO_SEQ_CMAP` for sequential colormaps.
- Use `MIMO_DIV_CMAP` for diverging colormaps.
- Use `MIMO_PALETTE` for categorical group colors.
- Do not hardcode `viridis`, `RdBu_r`, `tab:blue`, or similar defaults in new plot code.

## Recommendations

- Reuse existing patterns in `mimo_pack.plot` for grids, labels, and legend handling.
- Test behavior at the API level instead of asserting pixel-perfect rendering.