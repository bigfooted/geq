import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
"""
UPDATED Single-frequency Flame Transfer Function (FTF) test with RECOMMENDED SETTINGS.

Based on comprehensive HJ-WENO5 and Fast Marching Method analysis, this version uses:
- Proper SDF initialization (|∇G| = 1.0)
- Godunov gradient scheme (robust for flame cusps)
- RK3 time scheme (3rd-order accuracy)
- Upwind2 spatial scheme (2nd-order, robust)
- Frequent reinitialization with Fast Marching Method

Key findings from testing:
- WENO5 gradient EXPLODES for oscillating flames (|∇G| → 60,900 in 1.5s)
- Godunov gradient is STABLE and accurate for cusped flames
- Proper SDF initialization critical for numerical stability
- FMM gives |∇G| = 1.002 (only 0.17% error) - best initialization method

See OSCILLATING_FLAME_RECOMMENDATIONS.md for full analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from g_equation_solver_improved import GEquationSolver2D
from checkpoint_utils import save_checkpoint, CheckpointMeta
from contour_utils import compute_contour_length
from ftf_utils import compute_ftf
from reporting_utils import (
    print_solver_overview, print_solve_start, print_performance, print_completion
)


def initial_horizontal_flame_proper_sdf(X, Y, y_flame):
    """
    Create proper signed distance function for horizontal flame.

    For a horizontal line at y = y_flame, the exact SDF is simply:
    G(x,y) = y_flame - y

    This gives |∇G| = 1.0 everywhere (perfect initialization).
    Much simpler and more accurate than the old linear profile method.

    Parameters
    ----------
    X, Y : ndarray
        Meshgrid coordinates
    y_flame : float
        Y-coordinate of horizontal flame position

    Returns
    -------
    G : ndarray
        Signed distance function with |∇G| = 1.0
        Negative below flame (unburned), positive above (burned)
    """
    return y_flame - Y


def create_three_region_updater(X, Y, x1=0.1, x2=0.9, U_Y=0.2, A=0.1, St=1.0, K=0.1,
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


def test_ftf_linear_flame_single(
    frequency_hz=10.0,
    time_scheme='rk3',              # UPDATED: RK3 for better accuracy
    spatial_scheme='upwind2',        # UPDATED: Upwind2 (robust for cusps)
    gradient_scheme='godunov',       # UPDATED: Godunov (stable for cusps)
    use_reinit=True,                 # UPDATED: Enable reinitialization
    reinit_interval=10,              # UPDATED: Frequent (cusps degrade |∇G|)
    steps_per_period=20,
    drop_cycles=10,
    measure_cycles=10,
    verbose=True,
    A=0.05,
    K=0.1,
    contour_times=[10.0, 10.01, 10.02, 10.03, 10.04, 10.05],
    save_restart=True,
    restart_path=None,
    notes="UPDATED single-frequency FTF checkpoint with recommended settings"
):
    # Domain/grid
    nx, ny = 201, 201
    Lx, Ly = 0.2, 0.4
    S_L = 0.1

    # Velocity regions
    x1, x2 = 0.05*Lx, 0.95*Lx
    u_y_left, u_y_mid, u_y_right = 0.0, 0.2, 0.0
    y0 = 0.3*Ly

    omega = 2 * np.pi * frequency_hz
    T = 1.0 / frequency_hz
    dt = min(0.001, T / float(steps_per_period))
    t_final = (drop_cycles + measure_cycles) * T
    n_steps = int(np.ceil(t_final / dt))

    if verbose:
        flow_desc = [
            "UPDATED: Recommended settings for oscillating flame FTF",
            f"  frequency={frequency_hz:.2f} Hz (omega={omega:.3f} rad/s)",
            f"  u_y_left={u_y_left}, u_y_mid={u_y_mid}, u_y_right={u_y_right}",
            f"  A={A}, K={K}",
            f"  gradient_scheme={gradient_scheme} (robust for cusps!)",
            f"  spatial_scheme={spatial_scheme} (2nd-order robust)",
            f"  time_scheme={time_scheme} (3rd-order accurate)",
            f"  steps_per_period={steps_per_period}, drop_cycles={drop_cycles}, measure_cycles={measure_cycles}",
            f"  dt={dt:.6f} s, total steps≈{n_steps}",
        ]
        reinit_desc = (
            f"Reinitialization: LOCAL every {reinit_interval} steps (fast_marching)"
            if use_reinit else "Reinitialization: Disabled"
        )
        print_solver_overview(
            "FTF Estimation (UPDATED): Linear Flame with Recommended Settings",
            nx, ny, Lx, Ly, S_L, flow_desc, time_scheme, reinit_desc
        )

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)

    # Spatially varying S_L pin bottom sides
    mask_bottom_side = (solver.Y < 0.001) & ((solver.X < x1) | (solver.X > x2))
    solver.S_L = np.where(mask_bottom_side, 0.0, S_L)

    # Dirichlet inlet band at bottom
    bottom_band_height = min(solver.dy * 0.5, 0.001)
    pin_mask = (solver.Y < bottom_band_height)
    pin_values = np.zeros_like(solver.X); pin_values[pin_mask] = +1.0
    try:
        solver.set_pinned_region(pin_mask, pin_values)
    except AttributeError:
        pass

    # UPDATED: Proper SDF initialization (|∇G| = 1.0)
    G0 = initial_horizontal_flame_proper_sdf(solver.X, solver.Y, y0)

    # Verify initial gradient quality
    if verbose:
        grad_x = np.gradient(G0, solver.dx, axis=1)
        grad_y = np.gradient(G0, solver.dy, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        mask_interface = np.abs(G0) < 3*solver.dx
        grad_int = grad_mag[mask_interface]
        print(f"\nInitial condition quality (Proper SDF):")
        print(f"  |∇G| mean: {np.mean(grad_int):.6f}")
        print(f"  |∇G| std:  {np.std(grad_int):.6f}")
        print(f"  Drift from 1.0: {abs(np.mean(grad_int) - 1.0):.6e}")

    # Velocity updater
    velocity_updater = create_three_region_updater(
        solver.X, solver.Y, x1=x1, x2=x2, U_Y=u_y_mid, A=A, St=omega, K=K,
        u_y_left=u_y_left, u_y_right=u_y_right
    )

    if verbose:
        print_solve_start()
        print(f"Single frequency run: f={frequency_hz:.2f} Hz, dt={dt:.6f} s, steps≈{n_steps}")
        print(f"Gradient scheme: {gradient_scheme.upper()} (recommended for oscillating flames)")

    start = time.time()
    G_hist, t_hist = solver.solve(
        G0, t_final, dt,
        save_interval=1,
        time_scheme=time_scheme,
        spatial_scheme=spatial_scheme,      # UPDATED: Specify spatial scheme
        gradient_scheme=gradient_scheme,     # UPDATED: Specify gradient scheme
        reinit_interval=(reinit_interval if use_reinit else 0),
        reinit_method=('fast_marching' if use_reinit else 'none'),
        reinit_local=True,
        velocity_updater=velocity_updater,
    )
    elapsed = time.time() - start
    if verbose:
        print_performance(elapsed, n_steps, len(t_hist))

    # Check final gradient quality
    if verbose:
        G_final = G_hist[-1]
        if gradient_scheme == 'godunov':
            grad_mag_final = solver.compute_gradient_magnitude(G_final)
        else:
            grad_mag_final = solver.compute_gradient_magnitude_weno5(G_final)
        mask_interface = np.abs(G_final) < 3*solver.dx
        grad_int_final = grad_mag_final[mask_interface]
        print(f"\nFinal gradient quality:")
        print(f"  |∇G| mean: {np.mean(grad_int_final):.6f}")
        print(f"  |∇G| std:  {np.std(grad_int_final):.6f}")
        print(f"  Drift from 1.0: {abs(np.mean(grad_int_final) - 1.0):.6e}")

    # Signals
    if verbose:
        print("Computing flame lengths (spatial skip N=5)...")
    spatial_skip_N = 5
    flame_lengths = [compute_contour_length(G, solver.X, solver.Y, iso_value=0.0, N=spatial_skip_N) for G in G_hist]
    t_arr = np.array(t_hist)
    u_ref = u_y_mid * (1.0 + A * np.sin(omega * (t_arr - K * 0.0)))

    if verbose:
        print("Computing FTF...")
    ftf = compute_ftf(
        area=np.array(flame_lengths),
        u=u_ref,
        t=t_arr,
        frequency_hz=frequency_hz,
        drop_cycles=drop_cycles,
        normalize='relative',
        detrend=True,
        window='hann',
        phase_units='deg',
    )

    gain = float(ftf['gain'][0])
    phase = float(ftf['phase'][0])

    # Save plots with scheme label
    scheme_label = f"{gradient_scheme}_{spatial_scheme}_{time_scheme}"

    # Separate plot: mean flame surface proxy A as function of time
    try:
        A_arr = np.array(flame_lengths, dtype=float)
        # Running (cumulative) mean over time
        A_running_mean = np.cumsum(A_arr) / np.maximum(1, np.arange(1, len(A_arr) + 1))
        # Mean over the steady-state window used for FTF
        t_start_win = t_arr[0] + drop_cycles * T
        m_win = t_arr >= t_start_win
        A_mean_window = float(np.mean(A_arr[m_win])) if np.any(m_win) else float(np.mean(A_arr))

        fig_mean, axm = plt.subplots(1, 1, figsize=(9, 4))
        # Plot instantaneous A(t) lightly for context
        axm.plot(t_arr, A_arr, color='0.7', lw=1.0, label='A(t) (instantaneous)')
        # Plot running mean Ā(t)
        axm.plot(t_arr, A_running_mean, color='tab:green', lw=2.0, label='Ā(t) (running mean)')
        # Horizontal reference: steady-state window mean
        axm.axhline(A_mean_window, color='crimson', ls='--', lw=1.5, label='Ā (steady-state window)')
        axm.axvline(t_start_win, color='crimson', linestyle=':', linewidth=1.2, alpha=0.9, label='window start')
        axm.set_xlabel('Time (s)')
        axm.set_ylabel('Flame surface proxy A (|Γ|)')
        axm.grid(True, alpha=0.3)
        axm.legend(loc='best')
        axm.set_title(
            f'UPDATED: Mean flame surface A vs time\n'
            f'f={frequency_hz:.1f} Hz, Ā_window={A_mean_window:.6f}, scheme={gradient_scheme.upper()}'
        )
        fig_mean.tight_layout()
        fig_mean.savefig(f'ftf_single_A_mean_time_f{frequency_hz:.1f}Hz_{scheme_label}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_mean)
    except Exception:
        pass

    # Time history plot near window start
    try:
        t_start_win = t_arr[0] + drop_cycles * T
        plot_start = max(t_arr[0], t_start_win - 2.0 * T)
        m_plot = t_arr >= plot_start
        fig_ts, ax_l = plt.subplots(1, 1, figsize=(9, 4))
        ax_r = ax_l.twinx()
        ln1 = ax_l.plot(t_arr[m_plot], np.array(flame_lengths)[m_plot], 'k-', linewidth=1.5, label=f'|Γ| (skip N={spatial_skip_N})')
        # Mark time locations (every Nth saved time) with markers on the flame length curve
        time_skip_N = 5
        marked_idx_all = np.arange(0, len(t_arr), time_skip_N)
        marked_mask = np.zeros_like(t_arr, dtype=bool)
        marked_mask[marked_idx_all] = True
        marked_in_window = m_plot & marked_mask
        ax_l.plot(t_arr[marked_in_window], np.array(flame_lengths)[marked_in_window], 'ro', markersize=3, label=f'sampled times (every {time_skip_N})')
        ln2 = ax_r.plot(t_arr[m_plot], u_ref[m_plot], 'b--', linewidth=1.5, label='u_ref (mid, y=0)')
        vline = ax_l.axvline(x=t_start_win, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label='window start')
        # Reinit markers: circles on time axis at each reinitialization moment
        if use_reinit and reinit_interval > 0:
            reinit_indices = np.arange(0, len(t_arr), reinit_interval)
            reinit_times = t_arr[reinit_indices]
            # Restrict to plotted window
            reinit_window_mask = reinit_times >= plot_start
            reinit_times_plot = reinit_times[reinit_window_mask]
            if reinit_times_plot.size > 0:
                # Place markers slightly below the min flame length for visibility
                y_min_plot = np.min(np.array(flame_lengths)[m_plot])
                y_max_plot = np.max(np.array(flame_lengths)[m_plot])
                y_span = max(1e-12, y_max_plot - y_min_plot)
                y_mark = y_min_plot - 0.04 * y_span
                ax_l.plot(reinit_times_plot, np.full_like(reinit_times_plot, y_mark),
                          'o', color='orange', markersize=4, label='reinit')
                # Extend y-limit to include markers
                ax_l.set_ylim(y_mark - 0.02 * y_span, y_max_plot + 0.02 * y_span)
        ax_l.set_xlabel('Time (s)'); ax_l.set_ylabel('Flame length |Γ|'); ax_r.set_ylabel('Velocity u_y (m/s)')
        ax_l.grid(True, alpha=0.3)
        lines = ln1 + ln2 + [vline]; labels = [l.get_label() for l in lines]
        ax_l.legend(lines, labels, loc='best')
        ax_l.set_title(
            f'UPDATED ({gradient_scheme.upper()}): f={frequency_hz:.1f} Hz\n'
            f'gain={gain:.4f}, phase={phase:.1f}°, dt={dt:.6f}s'
        )
        fig_ts.tight_layout()
        fig_ts.savefig(f'ftf_single_time_f{frequency_hz:.1f}Hz_{scheme_label}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_ts)
    except Exception:
        pass

    # Full time series plot of flame length over entire simulation
    try:
        fig_full, ax_full = plt.subplots(1, 1, figsize=(9, 4))
        ax_full.plot(t_arr, flame_lengths, 'k-', linewidth=1.4, label=f'|Γ| (skip N={spatial_skip_N})')
        # Mark every 5th saved time for reference
        time_skip_N_full = 5
        idx_full = np.arange(0, len(t_arr), time_skip_N_full)
        ax_full.plot(t_arr[idx_full], np.array(flame_lengths)[idx_full], 'r', label=f'sampled times (every {time_skip_N_full})')
        ax_full.set_xlabel('Time (s)'); ax_full.set_ylabel('Flame length |Γ|'); ax_full.grid(True, alpha=0.3)
        ax_full.legend(loc='best')
        ax_full.set_title(
            f'UPDATED ({gradient_scheme.upper()}): Full time series\n'
            f'f={frequency_hz:.1f} Hz, gain={gain:.4f}, phase={phase:.1f}°'
        )
        fig_full.tight_layout()
        fig_full.savefig(f'ftf_single_time_full_f{frequency_hz:.1f}Hz_{scheme_label}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_full)
    except Exception:
        pass

    # Contour grid at specified times (2x3). If more than 6 times provided, use first 6; if fewer, leave extras blank.
    if contour_times is not None and len(contour_times) > 0:
        try:
            times_req = list(contour_times)
            n_show = min(6, len(times_req))
            figc, axc = plt.subplots(2, 3, figsize=(15, 10))
            axes_flat = axc.flatten()
            for k in range(6):
                ax = axes_flat[k]
                if k < n_show:
                    t_req = float(times_req[k])
                    # Find nearest saved time index
                    idx = int(np.argmin(np.abs(t_arr - t_req)))
                    tk = t_arr[idx]
                    Gk = G_hist[idx]
                    cf = ax.contourf(solver.X, solver.Y, Gk, levels=20, cmap='RdBu_r')
                    plt.colorbar(cf, ax=ax)
                    ax.contour(solver.X, solver.Y, Gk, levels=[0], colors='k', linewidths=2)
                    ax.set_title(f't req={t_req:.3f}s, used={tk:.3f}s')
                    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
            plt.tight_layout()
            figc.savefig(f'ftf_single_contours_f{frequency_hz:.1f}Hz_{scheme_label}.png', dpi=300, bbox_inches='tight')
            plt.close(figc)
        except Exception:
            pass

    if verbose:
        print(f"\nSingle-frequency FTF result ({gradient_scheme.upper()}): gain={gain:.6f}, phase={phase:.2f} deg")
        print_completion()

    # Save restart checkpoint of final state (optional)
    if save_restart:
        try:
            ckpt_path = restart_path or f"ftf_single_f{frequency_hz:.1f}Hz_{scheme_label}_ckpt.npz"
            meta = CheckpointMeta(
                nx=solver.nx, ny=solver.ny, Lx=solver.Lx, Ly=solver.Ly,
                dt=dt, time_scheme=time_scheme,
                reinit_interval=(reinit_interval if use_reinit else 0),
                reinit_method=('fast_marching' if use_reinit else 'none'),
                reinit_local=True,
                save_interval=1,
                notes=notes,
                extra={
                    'forcing_type': 'three_region_sinus',
                    'frequency_hz': frequency_hz,
                    'omega': omega,
                    'gradient_scheme': gradient_scheme,
                    'spatial_scheme': spatial_scheme,
                    'A': A,
                    'K': K,
                    'u_y_left': u_y_left,
                    'u_y_mid': u_y_mid,
                    'u_y_right': u_y_right,
                    'x1': x1,
                    'x2': x2,
                    'y0': y0,
                    'spatial_skip_N': spatial_skip_N,
                    'drop_cycles': drop_cycles,
                    'measure_cycles': measure_cycles,
                }
            )
            save_checkpoint(ckpt_path, solver, G_hist[-1], t_arr[-1], meta)
            if verbose:
                print(f"Checkpoint saved: {ckpt_path} (t={t_arr[-1]:.6f})")
        except Exception as e:
            if verbose:
                print(f"Failed to save checkpoint: {e}")

    return {
        'frequency_hz': frequency_hz,
        'gain': gain,
        'phase_deg': phase,
        'dt': dt,
        'steps': n_steps,
        'gradient_scheme': gradient_scheme,
        'spatial_scheme': spatial_scheme,
    }


if __name__ == '__main__':
    # CLI parsing with UPDATED defaults
    frequency = 10.0
    scheme = 'rk3'              # UPDATED: RK3 (3rd-order)
    spatial_scheme = 'upwind2'   # UPDATED: Upwind2 (2nd-order robust)
    gradient_scheme = 'godunov'  # UPDATED: Godunov (robust for cusps)
    use_reinit = True            # UPDATED: Enable reinitialization
    reinit_interval = 10         # UPDATED: Frequent (was 50)
    steps_per_period = 20
    drop_cycles = 50
    measure_cycles = 10
    contour_times = [10.0, 10.01, 10.02, 10.03, 10.04, 10.05]
    save_restart = True
    restart_path = None
    notes = "UPDATED FTF checkpoint with recommended settings"

    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2', 'rk3']:
            scheme = arg.lower()
        elif arg.lower() in ['upwind', 'upwind2', 'weno5']:
            spatial_scheme = arg.lower()
        elif arg.lower() in ['godunov', 'weno5']:
            gradient_scheme = arg.lower()
        elif arg == 'no_reinit':
            use_reinit = False
        elif arg.startswith('f='):
            try:
                frequency = float(arg.split('=')[1])
            except Exception:
                pass
        elif arg.startswith('reinit_interval='):
            try:
                reinit_interval = int(arg.split('=')[1])
            except Exception:
                pass
        elif arg.startswith('steps_per_period='):
            try:
                steps_per_period = int(arg.split('=')[1])
            except Exception:
                pass
        elif arg.startswith('drop_cycles='):
            try:
                drop_cycles = int(arg.split('=')[1])
            except Exception:
                pass
        elif arg.startswith('measure_cycles='):
            try:
                measure_cycles = int(arg.split('=')[1])
            except Exception:
                pass
        elif arg.startswith('contour_times='):
            # Parse comma-separated float list
            try:
                vals = arg.split('=', 1)[1]
                contour_times = [float(v) for v in vals.split(',') if v.strip()]
            except Exception:
                pass
        elif arg.lower() == 'no_save':
            save_restart = False
        elif arg.startswith('ckpt='):
            restart_path = arg.split('=',1)[1]
        elif arg.startswith('notes='):
            notes = arg.split('=',1)[1]

    print("\n" + "="*80)
    print("UPDATED FTF TEST WITH RECOMMENDED SETTINGS")
    print("="*80)
    print(f"frequency={frequency} Hz")
    print(f"time_scheme={scheme} (3rd-order)")
    print(f"spatial_scheme={spatial_scheme} (2nd-order robust)")
    print(f"gradient_scheme={gradient_scheme} (robust for cusps!)")
    print(f"use_reinit={use_reinit}, reinit_interval={reinit_interval}")
    print(f"\nBased on comprehensive testing:")
    print(f"  - WENO5 gradient EXPLODES for oscillating flames")
    print(f"  - Godunov gradient STABLE and accurate")
    print(f"  - Proper SDF initialization (|∇G| = 1.0) critical")
    print(f"  - See OSCILLATING_FLAME_RECOMMENDATIONS.md for details")
    print("="*80 + "\n")

    test_ftf_linear_flame_single(
        frequency_hz=frequency,
        time_scheme=scheme,
        spatial_scheme=spatial_scheme,
        gradient_scheme=gradient_scheme,
        use_reinit=use_reinit,
        reinit_interval=reinit_interval,
        steps_per_period=steps_per_period,
        drop_cycles=drop_cycles,
        measure_cycles=measure_cycles,
        verbose=True,
        contour_times=contour_times,
        save_restart=save_restart,
        restart_path=restart_path,
        notes=notes,
    )
