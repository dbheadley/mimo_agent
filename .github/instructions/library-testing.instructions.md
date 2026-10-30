---
description: "Use when adding or updating tests for mimo_pack. Covers synthetic data factories, required test categories, and test file naming for analysis, plot, preprocess, and utility changes."
applyTo:
  - "mimo_pack/testing/**"
---

# Testing Rules

Tests are opt-in: write or change tests only when the user explicitly asks. Never add them as part of a code change by default.

## Decision Tree

1. Did the user explicitly ask for a test of a public function, loader, transformation, or plot behavior?
   - Add one happy-path test in the nearest existing test module; add more categories below only if requested.
2. Is the change compute-oriented?
   - Default to output type/shape; add validation, coordinate, attr, or edge-case tests only if requested.
3. Is the change in plotting?
   - Assert returned `Axes` or `Figure` objects and key labels/artists rather than pixels.
4. Do you think you need a new fixture?
   - Check `mimo_pack.testing.data` first and reuse a synthetic factory whenever possible.

## Directives

- Use synthetic data factories from `mimo_pack.testing.data`; do not load real experimental data in unit tests.
- Use fixed seeds for reproducibility.
- Keep tests close to the changed behavior and avoid widening unrelated coverage.
- Match the existing naming pattern: `test_analysis_<module>.py` or `test_<module>.py`.

## Available Test Categories (only when requested)

- Basic functionality.
- Input validation.
- Coordinate preservation when channel structure is preserved.
- Attr/provenance preservation.
- Edge cases such as short input, single channel, or boundary parameter values.

## Common Factories

- `synthetic_lfp()` for continuous LFP-like xarray data.
- `synthetic_ap()` for AP-band xarray data.
- `synthetic_spikes()` for `TsGroup` spike collections.
- `synthetic_lfp_trials()` for trial/epoch-based xarray data.
- `synthetic_iv_data()` for in-vitro current-step style data.
- `synthetic_nidq()` (or `synthetic_nidq_digital()` / `synthetic_nidq_analog()`) for NIDQ-style digital/analog xarray data.
- `synthetic_fiber_photometry()` for Doric-style fiber photometry xarray data.