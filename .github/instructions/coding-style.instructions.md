---
description: "Use when writing or editing any Python or notebook code in this workspace. Defines how much code to write: no speculative abstractions, no defensive error handling, and no tests unless requested. Overrides default over-engineering behavior."
applyTo:
  - "**/*.py"
  - "**/*.ipynb"
---

# Coding Style (Overrides Default Behavior)

Write the smallest complete change that does what was asked. These rules override your default habits and any softer wording elsewhere. They do not override the mimo data-model rules (`mimo_pack.consts` constants, rich xarray/pynapple objects, module placement).

## Rules

1. **Reuse first.** Check `mimo_pack.util` and existing code before writing anything. Prefer an existing helper, then a library in `environment.yml`, then inline code.
2. **No new function, class, or module unless it is called from 2 or more places** or the user asked for one. Inline up to about 15 lines rather than extracting a helper.
3. **No wrappers** around a single library call, and no layers that only forward arguments.
4. **No speculative parameters, options, flags, or config** for cases nobody asked for.
5. **No error handling by default.** No `try/except`, no `isinstance` checks, no defensive `None`/empty guards, and no custom exceptions. Let failures raise naturally. Exceptions: a boundary where bad input is realistic and silent wrong results are possible (user-facing public API entry points, file I/O), and then one short check for that specific failure only.
6. **No tests unless the user explicitly asks for them.** Do not create, update, or extend test files, and do not mention missing tests. If asked, write the minimum: one happy-path test per function, using `mimo_pack.testing.data` factories.
7. **No unrequested extras:** no logging, type-hint sweeps, docstring rewrites, README or markdown files, or refactors of nearby code.
8. **Comments only where the code is not self-evident.** Do not narrate what the code does.
9. **Notebooks:** few cells, linear flow, one idea per cell, and no helper functions for code used once.
10. **Loops:** prefer vectorized operations or comprehensions over explicit `for` loops when possible. When loops are necessary, keep them short and simple and have them show a progress bar if they are long-running.

## Example

```python
# Over-engineered (do not do this)
def _validate_inputs(data, band):
    if not isinstance(data, xr.DataArray):
        raise TypeError(...)
    if len(band) != 2:
        raise ValueError(...)

def _filter_band(data, band):
    try:
        return bandpass(data, *band)
    except Exception as e:
        raise RuntimeError(...) from e

def band_power(data, band=(30, 80), method="welch", verbose=False): ...

# Do this
power = bandpass(lfp, 30, 80).pipe(np.square).mean(DIM_TIME)
```

## Before You Finish

End your final message with a one-line check: list any helper, validation, `try/except`, parameter, or file you added beyond the literal request, or write "None". Remove anything you cannot justify as required by the request.
