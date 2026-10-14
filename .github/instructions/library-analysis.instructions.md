---
description: "Use when changing compute-oriented mimo_pack code in analysis/, preprocess/, math/, or util/. Covers module placement, public function structure, fs handling, provenance tracking, coordinate preservation, and validation patterns."
applyTo:
  - "mimo_pack/analysis/**"
  - "mimo_pack/preprocess/**"
  - "mimo_pack/math/**"
  - "mimo_pack/util/**"
---

# Compute-Oriented Library Rules

## Decision Tree

1. Does the code touch disk or read raw files?
   - Move that behavior to `fileio/` or `core/` instead of implementing it here.
2. Does the code use only numpy/scipy and no xarray or pynapple?
   - Keep it in `math/`.
3. Does the code clean, reshape, align, or artifact-correct raw data for downstream use?
   - Keep it in `preprocess/`.
4. Does the code compute a scientific result from rich in-memory objects?
   - Keep it in `analysis/`.
5. Is it a generic helper without domain-specific semantics?
   - Keep it in `util/`.

## Directives

- Public APIs outside `math/` should accept rich objects, not raw arrays or standalone file paths.
- Outside `math/`, extract `fs` from `data.attrs[ATTR_FS]` or a utility helper; do not accept it as a separate public parameter.
- Validate input types, dimensions, and required attrs with actionable `TypeError` and `ValueError` messages.
- Use `.values` only for the inner numeric compute step; keep alignment, coords, and attrs in xarray/pynapple.
- Append to the `description` attr when transforming a `DataArray`.
- Preserve spatial coordinates such as `COORD_X`, `COORD_Y`, and `COORD_SHANK` through transformations.

## Recommendations

- Reuse `mimo_pack.util.xarray` helpers before writing new windowing, resampling, z-scoring, channel-selection, or concatenation code.
- Favor the smallest local edit that keeps module boundaries intact.
- Keep docstrings numpy-style and include examples when the public API changes.

## Required Outcomes

- Correct module placement.
- Rich-object inputs and outputs.
- Provenance preserved in attrs.
- Matching test coverage added or updated.