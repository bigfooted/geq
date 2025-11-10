import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Restart the single-frequency FTF testcase from a saved checkpoint and run for +10 seconds.

Usage examples:
  python tests/test_ftf_linear_flame_single/test_restart_ftf_linear_flame_single.py
  python tests/test_ftf_linear_flame_single/test_restart_ftf_linear_flame_single.py ckpt=ftf_single_f50.0Hz_ckpt.npz
  python tests/test_ftf_linear_flame_single/test_restart_ftf_linear_flame_single.py f=50.0 A=0.10 K=0.1

Notes:
- If f/A/K are provided, a time-dependent velocity updater (three-region) is reconstructed.
  Otherwise, the restart proceeds with the velocities stored in the checkpoint (static continuation).
- The script saves a new checkpoint at the end of the restart window.
"""

import numpy as np
import matplotlib.pyplot as plt
from checkpoint_utils import load_checkpoint, restart_solve, save_checkpoint, CheckpointMeta
from contour_utils import compute_contour_length
from ftf_utils import compute_ftf


def create_three_region_updater(X, Y, x1=0.1, x2=0.9, U_Y=0.2, A=0.10, St=1.0, K=0.1,
                                u_y_left=0.0, u_y_right=0.0):
    Xl = X.copy(); Yl = Y.copy()
    left_mask = (Xl <= x1)
    right_mask = (Xl >= x2)
    def updater(solver, t):
        u_x = np.zeros_like(Xl)
        u_y_mid = U_Y * (1.0 + A * np.sin(St * (t - K * Yl)))
        u_y = np.where(left_mask, u_y_left, np.where(right_mask, u_y_right, u_y_mid))
        return u_x, u_y
    return updater


def test_restart_ftf_linear_flame_single(
    ckpt_path=None,
    add_seconds: float = 10.0,
    time_scheme: str = None,
    use_reinit: bool = None,
    steps_per_period: int = None,
    frequency_hz: float = None,
    A: float = 0.10,
    K: float = 0.1,
    contour_times=None,
    verbose: bool = True,
):
    # Resolve checkpoint path
    if ckpt_path is None and frequency_hz is not None:
        ckpt_path = f"ftf_single_f{float(frequency_hz):.1f}Hz_ckpt.npz"
    if ckpt_path is None:
        # Default to 5 Hz filename
        ckpt_path = "ftf_single_f5.0Hz_ckpt.npz"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run the single-frequency test first or specify ckpt=...")

    # Load checkpoint and meta
    solver0, G0, t0, meta = load_checkpoint(ckpt_path)
    if verbose:
        print(f"Loaded checkpoint: {ckpt_path} (t0={t0:.6f})  grid=({meta.nx},{meta.ny}) L=({meta.Lx},{meta.Ly})")

    # Determine run controls
    dt = meta.dt if (meta.dt is not None) else 0.001
    save_interval = meta.save_interval if (meta.save_interval is not None) else 1
    scheme = time_scheme if (time_scheme is not None) else (meta.time_scheme or 'rk3')
    if use_reinit is None:
        # Follow meta settings: disable reinit if meta.reinit_interval == 0
        reinit_interval = meta.reinit_interval or 0
        reinit_method = meta.reinit_method or 'none'
    else:
        reinit_interval = (50 if use_reinit else 0)
        reinit_method = ('fast_marching' if use_reinit else 'none')

    # Optional reconstruction of forcing updater
    velocity_updater = None
    # Prefer checkpoint meta.extra if present for an exact restart
    extra = getattr(meta, 'extra', None)
    used_source = None
    if isinstance(extra, dict) and extra.get('forcing_type') == 'three_region_sinus':
        freq_meta = extra.get('frequency_hz', None)
        A_meta = extra.get('A', None)
        K_meta = extra.get('K', None)
        u_y_mid_meta = extra.get('u_y_mid', 0.2)
        x1_meta = extra.get('x1', 0.1)
        x2_meta = extra.get('x2', 0.9)
        u_y_left_meta = extra.get('u_y_left', 0.0)
        u_y_right_meta = extra.get('u_y_right', 0.0)
        f_use = frequency_hz if (frequency_hz is not None) else freq_meta
        if f_use is not None:
            omega = 2.0 * np.pi * float(f_use)
            A_use = A if (A is not None) else (A_meta if A_meta is not None else 0.10)
            K_use = K if (K is not None) else (K_meta if K_meta is not None else 0.1)
            velocity_updater = create_three_region_updater(
                solver0.X, solver0.Y,
                x1=x1_meta, x2=x2_meta, U_Y=u_y_mid_meta,
                A=A_use, St=omega, K=K_use,
                u_y_left=u_y_left_meta, u_y_right=u_y_right_meta,
            )
            frequency_hz = f_use  # propagate for plotting labels
            used_source = 'meta.extra'
    elif frequency_hz is not None:
        omega = 2.0 * np.pi * float(frequency_hz)
        # Fallback default parameters matching original test
        x1, x2 = 0.1, 0.9
        u_y_left, u_y_mid, u_y_right = 0.0, 0.2, 0.0
        velocity_updater = create_three_region_updater(
            solver0.X, solver0.Y, x1=x1, x2=x2, U_Y=u_y_mid, A=A, St=omega, K=K,
            u_y_left=u_y_left, u_y_right=u_y_right
        )
        used_source = 'cli'
    if verbose:
        if velocity_updater is not None:
            print(f"Using reconstructed velocity updater from {used_source or 'cli'}: f={frequency_hz} Hz, A={A}, K={K}")
        else:
            print("No forcing metadata/overrides provided: continuing with velocities stored in checkpoint (static continuation).")

    # Restart for +add_seconds
    t_final = t0 + float(add_seconds)
    if verbose:
        print(f"Restarting from t={t0:.6f} to t={t_final:.6f} with dt={dt:.6f}, scheme={scheme}, reinit_interval={reinit_interval}, method={reinit_method}")

    t_start = time.time()
    G_hist, t_hist = restart_solve(
        ckpt_path,
        t_final=t_final,
        dt=dt,
        save_interval=save_interval,
        time_scheme=scheme,
        reinit_interval=reinit_interval,
        reinit_method=reinit_method,
        reinit_local=(meta.reinit_local if meta.reinit_local is not None else True),
        smooth_ic=False,
        velocity_updater=velocity_updater,
    )
    elapsed = time.time() - t_start
    if verbose:
        n_steps = int(np.ceil((t_final - t0) / dt))
        print(f"Restart completed in {elapsed:.2f} s, steps≈{n_steps}, snapshots={len(t_hist)}")

    # Post-processing: compute flame length over the restart window and create figures
    try:
        if verbose:
            print("Computing flame lengths (spatial skip N=5) for restart window...")
        spatial_skip_N = 5
        flame_lengths = [
            compute_contour_length(Gk, solver0.X, solver0.Y, iso_value=0.0, N=spatial_skip_N)
            for Gk in G_hist
        ]
        t_arr = np.array(t_hist)
        # Reference inlet signal if frequency is known
        u_ref = None
        if frequency_hz is not None:
            omega = 2.0 * np.pi * float(frequency_hz)
            # Use U_Y from meta.extra if available for exact overlay
            u_y_mid = (extra.get('u_y_mid') if isinstance(extra, dict) and extra.get('u_y_mid') is not None else 0.2)
            A_use = A if (A is not None) else (extra.get('A') if isinstance(extra, dict) else 0.10)
            K_use = K if (K is not None) else (extra.get('K') if isinstance(extra, dict) else 0.1)
            u_ref = u_y_mid * (1.0 + A_use * np.sin(omega * (t_arr - K_use * 0.0)))

        # Compute FTF over the restart window (if we have u_ref)
        restart_ftf_gain = None
        restart_ftf_phase_deg = None
        if u_ref is not None:
            try:
                ftf = compute_ftf(
                    area=np.array(flame_lengths),
                    u=u_ref,
                    t=t_arr,
                    frequency_hz=float(frequency_hz),
                    drop_cycles=0,
                    normalize='relative',
                    detrend=True,
                    window='hann',
                    phase_units='deg',
                )
                restart_ftf_gain = float(ftf['gain'][0])
                restart_ftf_phase_deg = float(ftf['phase'][0])
                if verbose:
                    print(f"Restart-window FTF: gain={restart_ftf_gain:.6f}, phase={restart_ftf_phase_deg:.2f} deg")
            except Exception as e:
                if verbose:
                    print(f"FTF computation during restart failed: {e}")

        base_stamp = f"t{t0:.2f}_to_{t_arr[-1]:.2f}"

        # Time history plot over the restart window
        fig_ts, ax_l = plt.subplots(1, 1, figsize=(9, 4))
        ax_r = ax_l.twinx()
        ln1 = ax_l.plot(t_arr, np.array(flame_lengths), 'k-', linewidth=1.5, label=f'|Γ| (skip N={spatial_skip_N})')
        # Mark time locations (every Nth saved time) with small markers
        time_skip_N = 5
        marked_idx_all = np.arange(0, len(t_arr), time_skip_N)
        ax_l.plot(t_arr[marked_idx_all], np.array(flame_lengths)[marked_idx_all], 'ro', markersize=3, label=f'sampled times (every {time_skip_N})')
        ln2 = []
        if u_ref is not None:
            ln2 = ax_r.plot(t_arr, u_ref, 'b--', linewidth=1.5, label='u_ref (mid, y=0)')
        ax_l.set_xlabel('Time (s)'); ax_l.set_ylabel('Flame length |Γ|'); ax_r.set_ylabel('Velocity u_y (m/s)')
        ax_l.grid(True, alpha=0.3)
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        if lines:
            ax_l.legend(lines, labels, loc='best')
        title_suffix = f", f={frequency_hz:.1f} Hz" if frequency_hz is not None else ""
        subtitle = ''
        if restart_ftf_gain is not None and restart_ftf_phase_deg is not None:
            subtitle = f" (FTF gain={restart_ftf_gain:.4f}, phase={restart_ftf_phase_deg:.1f}°)"
        ax_l.set_title(f'Restart window {base_stamp}{title_suffix}{subtitle}')
        fig_ts.tight_layout()
        fig_ts.savefig(f'ftf_single_time_restart_{base_stamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_ts)

        # Full time series for the restart (identical to above but kept for parity with original naming)
        fig_full, ax_full = plt.subplots(1, 1, figsize=(9, 4))
        ax_full.plot(t_arr, flame_lengths, 'k-', linewidth=1.4, label=f'|Γ| (skip N={spatial_skip_N})')
        time_skip_N_full = 5
        idx_full = np.arange(0, len(t_arr), time_skip_N_full)
        ax_full.plot(t_arr[idx_full], np.array(flame_lengths)[idx_full], 'r', label=f'sampled times (every {time_skip_N_full})')
        ax_full.set_xlabel('Time (s)'); ax_full.set_ylabel('Flame length |Γ|'); ax_full.grid(True, alpha=0.3)
        ax_full.legend(loc='best')
        ax_full.set_title(f'Full restart series {base_stamp}{title_suffix}{subtitle}')
        fig_full.tight_layout()
        fig_full.savefig(f'ftf_single_time_full_restart_{base_stamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_full)

        # Contour grid at specified times (2x3)
        if contour_times is None:
            # Default: small set near the start of restart window (mimic original spacing)
            contour_times = [t0 + 0.01 * k for k in range(6)]
        if contour_times and len(contour_times) > 0:
            times_req = list(contour_times)
            n_show = min(6, len(times_req))
            figc, axc = plt.subplots(2, 3, figsize=(15, 10))
            axes_flat = axc.flatten()
            for k in range(6):
                ax = axes_flat[k]
                if k < n_show:
                    t_req = float(times_req[k])
                    idx = int(np.argmin(np.abs(t_arr - t_req)))
                    tk = t_arr[idx]
                    Gk = G_hist[idx]
                    cf = ax.contourf(solver0.X, solver0.Y, Gk, levels=20, cmap='RdBu_r')
                    plt.colorbar(cf, ax=ax)
                    ax.contour(solver0.X, solver0.Y, Gk, levels=[0], colors='k', linewidths=2)
                    ax.set_title(f't req={t_req:.3f}s, used={tk:.3f}s')
                    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
            plt.tight_layout()
            figc.savefig(f'ftf_single_contours_restart_{base_stamp}.png', dpi=300, bbox_inches='tight')
            plt.close(figc)
    except Exception as e:
        if verbose:
            print(f"Post-processing/plotting failed: {e}")

    # Save a new checkpoint at the end of the restart
    out_path = f"ftf_single_restart_t{t0:.2f}_to_{t_hist[-1]:.2f}.npz"
    # Prepare extra metadata for exact chained restarts and analysis
    extra_out = {}
    if isinstance(extra, dict):
        extra_out.update(extra)
    extra_out.update({
        'restart_window': [float(t0), float(t_hist[-1])],
        'restart_dt': float(dt),
        'restart_steps': int(np.ceil((t_hist[-1]-t0)/dt)),
        'restart_samples': int(len(t_hist)),
        'restart_elapsed_s': float(elapsed),
    })
    if frequency_hz is not None:
        extra_out['frequency_hz'] = float(frequency_hz)
    if restart_ftf_gain is not None:
        extra_out['restart_ftf_gain'] = float(restart_ftf_gain)
    if restart_ftf_phase_deg is not None:
        extra_out['restart_ftf_phase_deg'] = float(restart_ftf_phase_deg)

    meta_out = CheckpointMeta(
        nx=solver0.nx, ny=solver0.ny, Lx=solver0.Lx, Ly=solver0.Ly,
        dt=dt, time_scheme=scheme,
        reinit_interval=reinit_interval, reinit_method=reinit_method,
        reinit_local=(meta.reinit_local if meta.reinit_local is not None else True),
        save_interval=save_interval,
        notes=f"restart from {os.path.basename(ckpt_path)}",
        extra=extra_out,
    )
    save_checkpoint(out_path, solver0, G_hist[-1], t_hist[-1], meta_out)
    if verbose:
        print(f"Saved restart checkpoint: {out_path}")

    return G_hist, t_hist


if __name__ == '__main__':
    # CLI parsing
    ckpt = None
    add_s = 10.0
    freq = None
    A = 0.10
    K = 0.1
    scheme = None
    use_reinit = None
    contour_times = None

    for arg in sys.argv[1:]:
        if arg.startswith('ckpt='):
            ckpt = arg.split('=', 1)[1]
        elif arg.startswith('t=') or arg.startswith('add='):
            try:
                add_s = float(arg.split('=', 1)[1])
            except Exception:
                pass
        elif arg.startswith('f='):
            try:
                freq = float(arg.split('=', 1)[1])
            except Exception:
                pass
        elif arg.startswith('A='):
            try:
                A = float(arg.split('=', 1)[1])
            except Exception:
                pass
        elif arg.startswith('K='):
            try:
                K = float(arg.split('=', 1)[1])
            except Exception:
                pass
        elif arg.lower() in ['euler', 'rk2', 'rk3']:
            scheme = arg.lower()
        elif arg == 'reinit' or arg == 'use_reinit':
            use_reinit = True
        elif arg == 'no_reinit':
            use_reinit = False
        elif arg.startswith('contour_times='):
            # comma-separated list of floats
            try:
                vals = arg.split('=', 1)[1]
                contour_times = [float(v) for v in vals.split(',') if v.strip()]
            except Exception:
                pass

    test_restart_ftf_linear_flame_single(
        ckpt_path=ckpt,
        add_seconds=add_s,
        time_scheme=scheme,
        use_reinit=use_reinit,
        frequency_hz=freq,
        A=A,
        K=K,
        contour_times=contour_times,
        verbose=True,
    )
