"""Analytical Flame Transfer Function (FTF) formulas.

Provides the closed-form expression for the uniformly perturbed conical flame
response (axisymmetric, small-amplitude, quasi-steady model) used in acoustic /
combustion instability studies.

Definition (Schuller–Durox–Candel 2003):
    FTF_UCO(ω) = (2/ω²) * ( 1 - exp(i ω) + i ω )

Where:
  - ω is the angular forcing frequency (rad/s)
  - Resulting complex transfer function H relates relative flame surface (or
    heat-release) fluctuation to relative inlet velocity fluctuation.

Returned metrics:
  - gain = |H(ω)|
  - phase = arg(H(ω)) (radians by default, degrees optional)

Small-ω asymptotics:
    gain ≈ 1, phase ≈ ω/3  (radians)

High-ω limit:
    gain → 2/ω → 0, phase → π/2 (90°).

Reference:
  Schuller, T., Durox, D., Candel, S., "A Unified Model for the Occurrence of
  Instabilities in Laminar Premixed Flames", Combustion and Flame, 134 (2003)
  21–34. DOI: 10.1016/S0010-2180(02)00582-X
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union

__all__ = ["FTF_UCO", "FTF_CCO"]


def FTF_UCO(
    omega: Union[float, np.ndarray, list, tuple],
    *,
    phase_units: str = "rad",
    return_complex: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical FTF for uniformly perturbed conical flames.

    Parameters
    ----------
    omega : float or array-like
        Angular frequency values (rad/s). May include 0; the limit ω→0 gives H→0.
    phase_units : {'rad','deg'}, default 'rad'
        Units for returned phase array.
    return_complex : bool, default False
        If True, also return the complex transfer function H(ω).

    Returns
    -------
    phase : np.ndarray
        Phase of H(ω) in requested units.
    gain : np.ndarray
        Magnitude |H(ω)|.
    H : np.ndarray (complex) [only if return_complex]
        The complex transfer function values.

    Notes
    -----
    Formula: H(ω) = (2/ω²) * (1 - exp(i ω) + i ω)
    Implemented with a removable singularity at ω=0 using the series expansion
    (H ≈ 1 + i ω/3 - ω²/12 + ...) for numerical stability.
    """
    w = np.asarray(omega, dtype=float)
    H = np.zeros_like(w, dtype=complex)

    # Mask for non-zero frequencies
    nz = w != 0.0
    w_nz = w[nz]
    if w_nz.size:
        H[nz] = (2.0 / (w_nz**2)) * (1.0 - np.exp(1j * w_nz) + 1j * w_nz)

    # Series expansion for w≈0:
    # 1 - exp(iw) + i w = w^2/2 + i w^3/6 - w^4/24 + ...
    # => H = (2/w^2)*(...) ≈ 1 + i (w/3) - (w^2)/12 + ...
    zmask = ~nz
    if np.any(zmask):
        w0 = w[zmask]
        H[zmask] = 1.0 + 1j * (w0/3.0) - (w0**2)/12.0

    gain = np.abs(H)
    phase = np.angle(H)
    if phase_units == "deg":
        phase = np.degrees(phase)
    elif phase_units != "rad":
        raise ValueError("phase_units must be 'rad' or 'deg'")

    if return_complex:
        return phase, gain, H
    return phase, gain


def FTF_CCO(
    omega: Union[float, np.ndarray, list, tuple],
    alpha: Union[float, np.ndarray, list, tuple],
    *,
    phase_units: str = "rad",
    return_complex: bool = False,
    radians: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical FTF for a cosinusoidally corrugated outflow (CCO) flame.

    Formula:
        FTF_CCO(ω, α) = (2/ω²) * ( 1/(1 - cos² α) ) * [ 1 - e^{i ω} + ( e^{i ω cos² α} - 1 ) / cos² α ]

    Parameters
    ----------
    omega : float or array-like
        Angular frequencies (rad/s).
    alpha : float or array-like
        Corrugation angle. Interpreted as degrees by default; set radians=True if already in radians.
    phase_units : {'rad','deg'}
        Units for returned phase.
    return_complex : bool, default False
        If True also return complex H values.
    radians : bool, default False
        If True, treat provided alpha values as radians; else convert from degrees.

    Returns
    -------
    phase : np.ndarray
        Phase of H in chosen units.
    gain : np.ndarray
        Magnitude |H|.
    H : np.ndarray (complex) [if return_complex]

    Notes
    -----
    - Small-ω expansion: H ≈ 1 + i ((1 + cos² α)/3) ω - ((1 + 4 cos² α + cos⁴ α)/24) ω² + ...
      used for ω≈0 to remove the removable singularity and improve numerical stability.
    - expm1 provides cancellation-resistant evaluation of e^{i x} - 1 terms.
    - Raises ValueError if cos² α or sin² α are below a threshold (near singular configurations α≈π/2 or α≈0/π).
    """
    w = np.asarray(omega, dtype=float)
    # Normalize alpha input
    if radians:
        a_rad = np.asarray(alpha, dtype=float)
        a_deg = np.degrees(a_rad)
    else:
        a_deg = np.asarray(alpha, dtype=float)
        a_rad = np.deg2rad(a_deg)

    w_b, a_rad_b = np.broadcast_arrays(w, a_rad)
    _, a_deg_b = np.broadcast_arrays(w, a_deg)

    # Masks for special handling
    small_alpha = a_deg_b < 1.0          # use provided limit expression
    large_alpha = a_deg_b > 89.0         # fall back to UCO model
    base_mask = ~(small_alpha | large_alpha)

    H = np.zeros_like(w_b, dtype=complex)

    # Base formula region (regular CCO)
    if np.any(base_mask):
        w_base = w_b[base_mask]
        a_base = a_rad_b[base_mask]
        cos_a = np.cos(a_base)
        cos2 = cos_a**2
        sin2 = 1.0 - cos2
        eps = 1e-12
        if np.any(cos2 < eps):
            raise ValueError("alpha (base region) produces cos^2(alpha) ~ 0 leading to singularity")
        if np.any(sin2 < eps):
            raise ValueError("alpha (base region) produces sin^2(alpha) ~ 0 leading to singularity (1/(1-cos^2))")
        nz = w_base != 0.0
        w_nz = w_base[nz]
        H_base = np.zeros_like(w_base, dtype=complex)
        if w_nz.size:
            expm1_full = np.expm1(1j * w_nz)              # e^{iω} - 1
            expm1_scaled = np.expm1(1j * w_nz * cos2[nz]) # e^{iω cos^2 α} - 1
            term = -expm1_full + expm1_scaled / cos2[nz]
            H_base[nz] = (2.0 / (w_nz**2)) * (1.0 / sin2[nz]) * term
        zmask = ~nz
        if np.any(zmask):
            w0 = w_base[zmask]
            c2z = cos2[zmask]
            real2 = (1 + 4 * c2z + c2z**2) / 24.0
            H_base[zmask] = 1.0 + 1j * (1.0 + c2z) * w0 / 3.0 - real2 * (w0**2)
        H[base_mask] = H_base

    # Small alpha limit region
    if np.any(small_alpha):
        w_small = w_b[small_alpha]
        H_small = np.zeros_like(w_small, dtype=complex)
        nzs = w_small != 0.0
        w_small_nz = w_small[nzs]
        if w_small_nz.size:
            # Limit formula: (2/ω^2) * ( (1 - i ω) * exp(i ω) - 1 )
            core = (1.0 - 1j * w_small_nz) * np.exp(1j * w_small_nz) - 1.0
            H_small[nzs] = (2.0 / (w_small_nz**2)) * core
        # w -> 0 expansion: ( (1 - iω) e^{iω} - 1 ) ≈ ω^2/2 => H ≈ 1
        zsmall = ~nzs
        if np.any(zsmall):
            w0 = w_small[zsmall]
            # Include second-order correction similar to expansion: H ≈ 1 - w^2/12 (optional refinement)
            H_small[zsmall] = 1.0 - (w0**2)/12.0
        H[small_alpha] = H_small

    # Large alpha region -> use UCO model
    if np.any(large_alpha):
        w_large = w_b[large_alpha]
        # Reuse FTF_UCO complex output
        _, _, H_uco = FTF_UCO(w_large, return_complex=True)
        H[large_alpha] = H_uco

    gain = np.abs(H)
    phase = np.angle(H)
    if phase_units == "deg":
        phase = np.degrees(phase)
    elif phase_units != "rad":
        raise ValueError("phase_units must be 'rad' or 'deg'")

    if return_complex:
        return phase, gain, H
    return phase, gain


if __name__ == "__main__":
    # Simple demonstration / quick self-test
    test_w = np.array([0.0, 0.1, 1.0, 10.0])
    ph, g, Hc = FTF_UCO(test_w, phase_units="deg", return_complex=True)
    for wv, gv, pv in zip(test_w, g, ph):
        print(f"omega={wv:5.2f} rad/s | gain={gv:8.5f} | phase={pv:8.3f} deg")
    # Demo CCO with alpha = 0.3 rad
    ph2, g2, H2 = FTF_CCO(test_w, 0.3, phase_units="deg", return_complex=True)
    for wv, gv, pv in zip(test_w, g2, ph2):
        print(f"[CCO alpha=0.3] omega={wv:5.2f} rad/s | gain={gv:8.5f} | phase={pv:8.3f} deg")