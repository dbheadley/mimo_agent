---
description: "Use when adding or updating tests for mimo_pack. Covers synthetic data factories, required test categories, and test file naming for analysis, plot, preprocess, and utility changes."
applyTo:
  - "mimo_pack/testing/**"
---

# Testing Rules

## Decision Tree

1. Did a public function, loader, transformation, or plot behavior change?
   - Add or update a targeted test in the nearest existing test module or create a new focused test file.
2. Is the change compute-oriented?
   - Cover output type/shape, validation errors, coordinate preservation, attr preservation, and edge cases.
3. Is the change in plotting?
   - Assert returned `Axes` or `Figure` objects and key labels/artists rather than pixels.
4. Do you think you need a new fixture?
   - Check `mimo_pack.testing.data` first and reuse a synthetic factory whenever possible.

## Directives

- Use synthetic data factories from `mimo_pack.testing.data`; do not load real experimental data in unit tests.
- Use fixed seeds for reproducibility.
- Keep tests close to the changed behavior and avoid widening unrelated coverage.
- Match the existing naming pattern: `test_analysis_<module>.py` or `test_<module>.py`.

## Required Test Categories

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