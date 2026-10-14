"""OKLCH-based color utilities for mimo_pack plotting.

Provides perceptually-uniform colors using the OKLCH color space.
Two named colormaps (``'mimo_sequential'`` and ``'mimo_diverging'``) are
registered with matplotlib when this module is imported, so they can be
used as plain strings in any matplotlib function.

Public API
----------
MIMO_PALETTE : list of str
    8-color categorical palette (hex strings) for multi-group plots.
MIMO_SEQ_CMAP : str
    Name of the registered sequential colormap (``'mimo_sequential'``).
MIMO_DIV_CMAP : str
    Name of the registered diverging colormap (``'mimo_diverging'``).
oklch_to_hex(L, C, H) : str
    Convert a single OKLCH triplet to a hex colour string.
make_palette(n, L, C) : list of str
    Build an n-colour categorical palette from evenly-spaced OKLCH hues.

OKLCH conversion uses only numpy — no additional dependencies required.
Matrices from Björn Ottosson's OKLAB specification (2021).

Created on 2026-03-06
Drew Headley
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# OKLCH → linear sRGB conversion matrices (Ottosson 2021)
# ---------------------------------------------------------------------------

# OKLAB → LMS^(1/3)  (inverse of M2)
_M2_INV = np.array([
    [1.0,  0.3963377774,  0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
])

# LMS → linear sRGB  (inverse of M1)
_M1_INV = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
])


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma correction (IEC 61966-2-1) element-wise."""
    return np.where(c <= 0.0031308,
                    12.92 * c,
                    1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


def oklch_to_srgb(L: float, C: float, H: float) -> np.ndarray:
    """Convert an OKLCH triplet to a clipped sRGB triplet.

    Parameters
    ----------
    L : float
        Lightness in [0, 1].
    C : float
        Chroma (>= 0).  Typical range 0–0.37.
    H : float
        Hue angle in degrees [0, 360).

    Returns
    -------
    np.ndarray, shape (3,)
        sRGB values clipped to [0, 1].
    """
    h_rad = np.deg2rad(H)
    lab = np.array([L, C * np.cos(h_rad), C * np.sin(h_rad)])

    # OKLAB → LMS^(1/3) → LMS (cube)
    lms_cbrt = _M2_INV @ lab
    lms = lms_cbrt ** 3

    # LMS → linear sRGB → sRGB
    linear_rgb = _M1_INV @ lms
    srgb = _linear_to_srgb(linear_rgb)
    return np.clip(srgb, 0.0, 1.0)


def oklch_to_hex(L: float, C: float, H: float) -> str:
    """Convert an OKLCH triplet to a CSS hex colour string.

    Parameters
    ----------
    L : float
        Lightness in [0, 1].
    C : float
        Chroma (>= 0).
    H : float
        Hue angle in degrees.

    Returns
    -------
    str
        Hex string, e.g. ``'#3a86ff'``.
    """
    rgb = oklch_to_srgb(L, C, H)
    return mcolors.to_hex(rgb)


# ---------------------------------------------------------------------------
# Categorical palette
# ---------------------------------------------------------------------------

def make_palette(n: int = 16, L: float = 0.75, C: float = 0.20,
                 hue_start: float = 29.0, hue_step: float = 45.0,
                 L_dark: float = 0.60) -> list:
    """Build an n-colour categorical palette using fixed OKLCH hue steps.

    Colours are perceptually uniform in lightness and chroma.  Hues advance
    by *hue_step* degrees (default 45°) from *hue_start*.  When more colours
    are needed than fit in one revolution (360 / hue_step), additional tiers
    at darker lightness (*L_dark*) are generated with the same hue sequence.

    Parameters
    ----------
    n : int, optional
        Number of colours.  Default is 16.
    L : float, optional
        Lightness for the first tier (0–1).  Default is 0.75.
    C : float, optional
        Chroma for all colours.  Default is 0.20.
    hue_start : float, optional
        Hue angle (degrees) of the first colour.  Default is 29 (red).
    hue_step : float, optional
        Hue separation between adjacent colours in degrees.  Default is 45.
    L_dark : float, optional
        Lightness for the second tier.  Default is 0.60.

    Returns
    -------
    list of str
        Hex colour strings.

    Examples
    --------
    >>> palette = make_palette(8)
    >>> len(palette)
    8
    """
    n_per_tier = int(360 / hue_step)
    hues = [(hue_start + i * hue_step) % 360.0 for i in range(n_per_tier)]
    lightness_tiers = [L, L_dark]
    result = []
    for tier_L in lightness_tiers:
        for h in hues:
            result.append(oklch_to_hex(tier_L, C, h))
            if len(result) >= n:
                return result
    return result


# ---------------------------------------------------------------------------
# Colormap factories
# ---------------------------------------------------------------------------

def _make_sequential_cmap(hue: float = 250.0,
                           name: str = 'mimo_sequential') -> mcolors.LinearSegmentedColormap:
    """Build a sequential OKLCH colormap.

    Ramps from near-white (low L) to the given hue at high L.
    Chroma rises from 0 to *C_max* as luminance increases.

    Parameters
    ----------
    hue : float, optional
        Hue angle in degrees.  Default 250 (blue).
    name : str, optional
        Matplotlib colormap name.  Default ``'mimo_sequential'``.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    n = 256
    Ls = np.linspace(0.30, 0.92, n)
    Cs = np.linspace(0.00, 0.20, n)
    rgb = np.array([oklch_to_srgb(l, c, hue) for l, c in zip(Ls, Cs)])
    return mcolors.LinearSegmentedColormap.from_list(name, rgb, N=n)


def _make_diverging_cmap(hue_lo: float = 250.0,
                          hue_hi: float = 30.0,
                          name: str = 'mimo_diverging') -> mcolors.LinearSegmentedColormap:
    """Build a diverging OKLCH colormap.

    Goes from *hue_lo* (default blue) through neutral grey to *hue_hi*
    (default orange), with equal lightness and chroma at both ends.

    Parameters
    ----------
    hue_lo : float, optional
        Hue for the low end.  Default 250 (blue).
    hue_hi : float, optional
        Hue for the high end.  Default 30 (orange).
    name : str, optional
        Matplotlib colormap name.  Default ``'mimo_diverging'``.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    n = 256
    half = n // 2

    # low side: hue_lo, full chroma → grey
    Ls_lo = np.linspace(0.58, 0.90, half)
    Cs_lo = np.linspace(0.22, 0.00, half)
    rgb_lo = np.array([oklch_to_srgb(l, c, hue_lo) for l, c in zip(Ls_lo, Cs_lo)])

    # high side: grey → hue_hi, full chroma
    Ls_hi = np.linspace(0.90, 0.58, half)
    Cs_hi = np.linspace(0.00, 0.22, half)
    rgb_hi = np.array([oklch_to_srgb(l, c, hue_hi) for l, c in zip(Ls_hi, Cs_hi)])

    rgb = np.vstack([rgb_lo, rgb_hi])
    return mcolors.LinearSegmentedColormap.from_list(name, rgb, N=n)


# ---------------------------------------------------------------------------
# Register colormaps and build the public palette
# ---------------------------------------------------------------------------

#: Name of the registered sequential colormap.
MIMO_SEQ_CMAP: str = 'mimo_sequential'

#: Name of the registered diverging colormap.
MIMO_DIV_CMAP: str = 'mimo_diverging'

#: 16-colour categorical palette for multi-group plots (hex strings).
MIMO_PALETTE: list = make_palette(16)

# Register colormaps on import.  force=True prevents errors on re-import.
plt.colormaps.register(_make_sequential_cmap(), name=MIMO_SEQ_CMAP, force=True)
plt.colormaps.register(_make_diverging_cmap(),  name=MIMO_DIV_CMAP,  force=True)
# Also register reversed variants
plt.colormaps.register(_make_sequential_cmap().reversed(), name=MIMO_SEQ_CMAP + '_r', force=True)
plt.colormaps.register(_make_diverging_cmap().reversed(),  name=MIMO_DIV_CMAP + '_r',  force=True)
