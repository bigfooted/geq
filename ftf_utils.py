"""
Utilities to estimate the Flame Transfer Function (FTF) from unsteady G-equation
simulations using flame surface area as heat-release proxy.

Primary entry point:
- compute_ftf(area, u, t, omega0, ...): per-frequency sweep estimator

Definition:
  H(ω) = Â(ω) / Û(ω)
where A is the flame surface area (or its fluctuation) and u is the inlet velocity
(or reference input). By default, signals are detrended and normalized to their
mean (relative fluctuations), so H is dimensionless.

Notes
- This implementation uses complex projection at a specified angular frequency
  ω0 (rad/s). It is robust to windows that do not contain an integer number of
  periods.
- For broadband estimation, consider a cross-spectral method (not included here).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional


def _ensure_1d(x):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input arrays must be 1D time series")
    return x


def _hann(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    return np.hanning(n)


def compute_ftf(
    area: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    omega0: Optional[float] = None,
    *,
    frequency_hz: Optional[float] = None,
    drop_cycles: int = 3,
    normalize: str = "relative",
    detrend: bool = True,
    window: str = "hann",
    phase_units: str = "deg",
) -> Dict[str, np.ndarray]:
    """
    Estimate per-frequency Flame Transfer Function H(ω0) using complex projection.

    Parameters
    ----------
    area : array-like
        Flame surface area time series A(t)
    u : array-like
        Inlet/reference velocity time series u(t)
    t : array-like
        Time stamps (uniform or nearly uniform)
    omega0 : float, optional
        Angular forcing frequency ω0 in rad/s. Required unless frequency_hz is provided.
    frequency_hz : float, optional
        Forcing frequency f0 in Hz. If provided, omega0 = 2π f0 is used.
    drop_cycles : int, default 3
        Number of forcing cycles to discard as transient before estimation.
    normalize : {'relative','absolute'}, default 'relative'
        If 'relative', work with relative fluctuations: (x - mean(x)) / mean(x).
        If 'absolute', work with mean-subtracted absolute fluctuations: x - mean(x).
    detrend : bool, default True
        Remove mean before analysis (mean is always removed internally; normalize controls scaling).
    window : {'hann', None}, default 'hann'
        Window applied to the steady-state segment before projection.
    phase_units : {'deg','rad'}, default 'deg'
        Units for returned phase.

    Returns
    -------
    out : dict
        Keys:
          - 'omega': np.ndarray([omega0])
          - 'H': complex np.ndarray([H])
          - 'gain': np.ndarray([|H|])
          - 'phase': np.ndarray([phase]) (deg or rad)
          - 'meta': dict with preprocessing flags
    """
    area = _ensure_1d(area)
    u = _ensure_1d(u)
    t = _ensure_1d(t)
    if not (len(area) == len(u) == len(t)):
        raise ValueError("area, u, t must have same length")

    # Infer omega0
    if omega0 is None and frequency_hz is None:
        raise ValueError("Provide omega0 (rad/s) or frequency_hz (Hz)")
    if omega0 is None:
        omega0 = 2.0 * np.pi * float(frequency_hz)

    # Timebase and potential non-uniform sampling handling (simple check)
    dt_med = np.median(np.diff(t)) if len(t) > 1 else 0.0
    if len(t) > 2:
        dt_var = np.max(np.abs(np.diff(t) - dt_med))
        if dt_var > 1e-6:
            # Optional: resample uniformly. Here we assume nearly uniform sampling is provided.
            pass

    # Prepare signals: mean subtract, then scale by mean if relative
    A_mean = float(np.mean(area))
    U_mean = float(np.mean(u))
    a = area - (A_mean if detrend else 0.0)
    uu = u - (U_mean if detrend else 0.0)
    if normalize == "relative":
        if np.abs(A_mean) > 0:
            a = a / A_mean
        if np.abs(U_mean) > 0:
            uu = uu / U_mean
    elif normalize == "absolute":
        # already mean-subtracted if detrend=True
        pass
    else:
        raise ValueError("normalize must be 'relative' or 'absolute'")

    # Drop transient cycles
    T = 2.0 * np.pi / float(omega0)
    t_start = t[0] + drop_cycles * T
    mask = t >= t_start
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)  # fallback: keep all
    ts = t[mask]
    aus = a[mask]
    uus = uu[mask]

    # Window
    if window == "hann":
        w = _hann(len(ts))
    elif window is None or window == "none":
        w = np.ones_like(ts)
    else:
        raise ValueError("Unsupported window. Use 'hann' or None")

    # Complex projection at omega0
    # Using exp(+i*omega*t) convention to match theoretical FTF formulas
    # (Schuller-Durox-Candel convention: positive phase = flame response lags velocity)
    exp_vec = np.exp(1j * omega0 * ts)
    U_hat = np.sum(uus * w * exp_vec)
    A_hat = np.sum(aus * w * exp_vec)

    # Avoid division by zero
    if U_hat == 0:
        H = np.nan + 1j * np.nan
    else:
        H = A_hat / U_hat

    gain = np.array([np.abs(H)])
    phase_rad = np.array([np.angle(H)])
    if phase_units == "deg":
        phase = np.degrees(phase_rad)
    elif phase_units == "rad":
        phase = phase_rad
    else:
        raise ValueError("phase_units must be 'deg' or 'rad'")

    out = {
        "omega": np.array([float(omega0)]),
        "H": np.array([H], dtype=complex),
        "gain": gain,
        "phase": phase,
        "meta": {
            "normalize": normalize,
            "detrend": detrend,
            "window": window,
            "drop_cycles": drop_cycles,
            "phase_units": phase_units,
        },
    }
    return out


def plot_ftf_bode(
    frequencies_hz: np.ndarray,
    gains: np.ndarray,
    phases: np.ndarray,
    *,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    phase_units: str = "deg",
    ylim_gain: Optional[tuple] = None,
    ylim_phase: Optional[tuple] = None,
):
    """
    Plot FTF gain and phase versus frequency (Bode-style).

    Parameters
    ----------
    frequencies_hz : array-like
        Frequencies in Hz
    gains : array-like
        |H(f)| values
    phases : array-like
        Phase of H(f) in degrees (default) or radians if phase_units='rad'
    title : str, optional
        Figure title
    savepath : str, optional
        If provided, save the figure to this path
    phase_units : {'deg','rad'}
        Units for phases array (for axis labeling)
    ylim_gain, ylim_phase : tuple, optional
        y-axis limits for gain and phase subplots
    """
    import matplotlib.pyplot as plt
    f = np.asarray(frequencies_hz)
    g = np.asarray(gains)
    p = np.asarray(phases)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    # Gain
    ax1.semilogx(f, g, 'o-', lw=2, ms=5, color='tab:blue')
    ax1.set_ylabel('Gain |H|')
    if ylim_gain is not None:
        ax1.set_ylim(*ylim_gain)
    ax1.grid(True, which='both', alpha=0.3)
    # Phase
    ax2.semilogx(f, p, 's--', lw=2, ms=5, color='tab:orange')
    ax2.set_xlabel('Frequency (Hz)')
    ylabel = 'Phase (deg)' if phase_units == 'deg' else 'Phase (rad)'
    ax2.set_ylabel(ylabel)
    if ylim_phase is not None:
        ax2.set_ylim(*ylim_phase)
    ax2.grid(True, which='both', alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
        fig.subplots_adjust(top=0.90)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)
