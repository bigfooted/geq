import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
"""
Single-frequency Flame Transfer Function (FTF) test for the non-homogeneous linear flame.
Runs one prescribed forcing frequency and computes gain/phase between inlet velocity
(reference mid-band velocity at y=0) and flame surface length (|Gamma| of G=0 contour).

Simplifications vs multi-frequency sweep:
- Only one frequency (frequency_hz)
- Optional reinitialization controlled by --no_reinit flag (still defaults ON)
- No reinit comparison diagnostic logic
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


def initial_horizontal_flame_linear_profile(X, Y, y_flame):
    Ly = float(np.max(Y))
    eps = 1e-12
    y0 = float(y_flame)
    y0_eff = max(y0, eps)
    Ly_minus_y0_eff = max(Ly - y0, eps)
    G_lower = 1.0 - (Y / y0_eff)
    G_upper = - (Y - y0) / Ly_minus_y0_eff
    return np.where(Y <= y0, G_lower, G_upper)


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
    frequency_hz=4.0,
    time_scheme='rk3',
    spatial_scheme='weno5',
    gradient_scheme='godunov',
    use_reinit=False,
    steps_per_period=50,
    drop_cycles=1,
    measure_cycles=10,
    verbose=True,
    A=0.05,
    K=0.1,
    contour_times=[2.0, 2.01, 2.02, 2.03, 2.04, 2.05],
    save_restart=True,
    restart_path=None,
    notes="single-frequency FTF checkpoint"
):
    # Domain/grid
    nx, ny = 201, 201
    Lx, Ly = 0.12, 0.25
    S_L = 0.4

    # Velocity regions
    x1, x2 = 0.03*Lx, 0.97*Lx
    u_y_left, u_y_mid, u_y_right = 0.0, 1.0, 0.0
    y0 = 0.3*Ly

    omega = 2 * np.pi * frequency_hz
    T = 1.0 / frequency_hz

    # CFL-based time step: CFL = (u_max + S_L) * dt / dx < 0.5 for stability
    dx = Lx / (nx - 1)
    u_max = u_y_mid * (1.0 + A)
    dt_cfl = 0.5 * dx / (u_max + S_L)  # Target CFL = 0.5 for safety
    dt = min(dt_cfl, T / float(steps_per_period))

    t_final = (drop_cycles + measure_cycles) * T
    n_steps = int(np.ceil(t_final / dt))

    # Auto-adjust contour_times to be within simulation time
    # Place them near the end of simulation (last ~5% of time span)
    if contour_times is None or len(contour_times) == 0 or max(contour_times) > t_final:
        # Create 6 evenly spaced times in the last 5% of simulation
        t_start = t_final * 0.95
        contour_times = np.linspace(t_start, t_final, 6).tolist()

    # Compute actual CFL for reporting
    cfl_actual = (u_max + S_L) * dt / dx

    if verbose:
        flow_desc = [
            "Single-frequency vertical forcing (three-region)",
            f"  frequency={frequency_hz:.2f} Hz (omega={omega:.3f} rad/s)",
            f"  u_y_left={u_y_left}, u_y_mid={u_y_mid}, u_y_right={u_y_right}",
            f"  A={A}, K={K}",
            f"  steps_per_period={steps_per_period}, drop_cycles={drop_cycles}, measure_cycles={measure_cycles}",
            f"  dt={dt:.6f} s (CFL={cfl_actual:.3f}), total steps≈{n_steps}, total time={t_final:.2f}s",
            f"  Contour snapshots at: {[f'{t:.3f}' for t in contour_times[:3]]}...{[f'{t:.3f}' for t in contour_times[-1:]]}"
        ]
        reinit_desc = (
            f"Reinitialization: LOCAL every 50 steps (fast_marching)"
            if use_reinit else "Reinitialization: Disabled"
        )
        print_solver_overview(
            "FTF Estimation (Single Frequency): Linear Flame (Three-Region)",
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

    # Initial G
    G0 = initial_horizontal_flame_linear_profile(solver.X, solver.Y, y0)

    # Velocity updater
    velocity_updater = create_three_region_updater(
        solver.X, solver.Y, x1=x1, x2=x2, U_Y=u_y_mid, A=A, St=omega, K=K,
        u_y_left=u_y_left, u_y_right=u_y_right
    )

    # Memory-efficient approach: compute flame lengths on-the-fly via callback
    # Store only scalar results, not full 2D fields
    spatial_skip_N = 5  # Spatial subsampling for contour computation
    flame_lengths = []
    time_points = []

    def compute_flame_length_callback(G, t, step):
        """Callback to compute flame length on-the-fly (called at each save_interval)"""
        length = compute_contour_length(G, solver.X, solver.Y, iso_value=0.0, N=spatial_skip_N)
        return length

    # Determine save_interval for good temporal resolution
    target_snapshots = 1500  # Good balance for FTF accuracy
    save_interval = max(1, int(np.ceil(n_steps / target_snapshots)))
    expected_calls = n_steps // save_interval + 1
    memory_scalars_mb = expected_calls * 8 / (1024**2)  # Just scalars: ~12 KB
    memory_old_gb = expected_calls * nx * ny * 8 / (1024**3)  # Old approach: ~440 MB

    if verbose:
        print_solve_start()
        print(f"Single frequency run: f={frequency_hz:.2f} Hz, dt={dt:.6f} s, steps≈{n_steps}")
        print(f"Memory optimization: save_interval={save_interval}, ~{expected_calls} flame length computations")
        print(f"  Scalar storage: {memory_scalars_mb:.3f} MB (vs {memory_old_gb:.2f} GB for full fields)")
        print(f"  Memory reduction: {memory_old_gb*1024/memory_scalars_mb:.0f}× smaller!")

    start = time.time()
    snapshots, t_hist, callback_results = solver.solve(
        G0, t_final, dt,
        save_interval=save_interval,
        time_scheme=time_scheme,
        spatial_scheme=spatial_scheme,
        gradient_scheme=gradient_scheme,
        reinit_interval=(50 if use_reinit else 0),
        reinit_method=('fast_marching' if use_reinit else 'none'),
        reinit_local=True,
        velocity_updater=velocity_updater,
        callback=compute_flame_length_callback,
        snapshot_times=contour_times,  # Only save these times for visualization
    )
    elapsed = time.time() - start

    # Extract flame lengths and time from callback results
    flame_lengths = callback_results
    t_arr = np.array(t_hist)

    if verbose:
        print_performance(elapsed, n_steps, len(t_hist))

    # Signals - flame lengths already computed on-the-fly
    u_ref = u_y_mid * (1.0 + A * np.sin(omega * (t_arr - K * 0.0)))

    if verbose:
        print("Computing FTF from on-the-fly flame lengths...")
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
    phase_rad = np.deg2rad(phase)

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
        # Plot running mean Ā(t)
        axm.plot(t_arr, A_running_mean, color='tab:green', lw=2.0, label='Ā(t) (running mean)')
        # Horizontal reference: steady-state window mean
        axm.axhline(A_mean_window, color='crimson', ls='--', lw=1.5, label='Ā (steady-state window)')
        axm.axvline(t_start_win, color='crimson', linestyle=':', linewidth=1.2, alpha=0.9, label='window start')
        axm.set_xlabel('Time (s)')
        axm.set_ylabel('Flame surface proxy A (|Γ|)')
        axm.grid(True, alpha=0.3)
        axm.legend(loc='best')
        axm.set_title(
            f'Mean flame surface A vs time: f={frequency_hz:.1f} Hz (ω={omega:.1f} rad/s)\n'
            f'Ā_window={A_mean_window:.6f}'
        )
        fig_mean.tight_layout()
        fig_mean.savefig(f'ftf_single_A_mean_time_f{frequency_hz:.1f}Hz.png', dpi=300, bbox_inches='tight')
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
        if use_reinit:
            reinit_interval_steps = 50
            reinit_indices = np.arange(0, len(t_arr), reinit_interval_steps)
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
            f'Single FTF run f={frequency_hz:.1f} Hz (ω={omega:.1f} rad/s): '
            f'gain={gain:.4f}, phase={phase:.1f}°, dt={dt:.6f}s'
        )
        fig_ts.tight_layout()
        fig_ts.savefig(f'ftf_single_time_f{frequency_hz:.1f}Hz.png', dpi=300, bbox_inches='tight')
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
        #ax_full.plot(t_arr[idx_full], np.array(flame_lengths)[idx_full], 'ro', markersize=3, label=f'sampled times (every {time_skip_N_full})')
        ax_full.plot(t_arr[idx_full], np.array(flame_lengths)[idx_full], 'r', label=f'sampled times (every {time_skip_N_full})')
        ax_full.set_xlabel('Time (s)'); ax_full.set_ylabel('Flame length |Γ|'); ax_full.grid(True, alpha=0.3)
        ax_full.legend(loc='best')
        ax_full.set_title(
            f'Full time series: f={frequency_hz:.1f} Hz (ω={omega:.1f} rad/s), dt={dt:.6f}s\n'
            f'gain={gain:.4f}, phase={phase:.1f}°'
        )
        fig_full.tight_layout()
        fig_full.savefig(f'ftf_single_time_full_f{frequency_hz:.1f}Hz.png', dpi=300, bbox_inches='tight')
        plt.close(fig_full)
    except Exception:
        pass

    # Contour grid at specified times (2x3). Use snapshots saved during solve.
    if snapshots is not None and len(snapshots) > 0:
        try:
            n_show = min(6, len(snapshots))
            figc, axc = plt.subplots(2, 3, figsize=(15, 10))
            axes_flat = axc.flatten()
            for k in range(6):
                ax = axes_flat[k]
                if k < n_show:
                    t_req, t_actual, Gk = snapshots[k]
                    cf = ax.contourf(solver.X, solver.Y, Gk, levels=20, cmap='RdBu_r')
                    plt.colorbar(cf, ax=ax)
                    ax.contour(solver.X, solver.Y, Gk, levels=[0], colors='k', linewidths=2)
                    ax.set_title(f't req={t_req:.3f}s, used={t_actual:.3f}s')
                    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
            plt.tight_layout()
            figc.savefig(f'ftf_single_contours_f{frequency_hz:.1f}Hz.png', dpi=300, bbox_inches='tight')
            plt.close(figc)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not create contour plots: {e}")

    if verbose:
        print(f"\nSingle-frequency FTF result: gain={gain:.6f}, phase={phase:.2f} deg ({phase_rad:.4f} rad)")
        print_completion()

    # Save restart checkpoint of final state (optional)
    # Use the final snapshot if available, otherwise current solver state
    G_final = snapshots[-1][2] if (snapshots and len(snapshots) > 0) else solver.G
    if save_restart:
        try:
            ckpt_path = restart_path or f"ftf_single_f{frequency_hz:.1f}Hz_ckpt.npz"
            meta = CheckpointMeta(
                nx=solver.nx, ny=solver.ny, Lx=solver.Lx, Ly=solver.Ly,
                dt=dt, time_scheme=time_scheme,
                reinit_interval=(50 if use_reinit else 0),
                reinit_method=('fast_marching' if use_reinit else 'none'),
                reinit_local=True,
                save_interval=1,
                notes=notes,
                extra={
                    'forcing_type': 'three_region_sinus',
                    'frequency_hz': frequency_hz,
                    'omega': omega,
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
            save_checkpoint(ckpt_path, solver, G_final, t_arr[-1], meta)
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
    }


if __name__ == '__main__':
    # CLI parsing
    frequency = 4.0
    # euler, rk2, rk3.
    scheme = 'rk3'
    # this is bad to put to true, we get jumps in the solution.
    use_reinit = False
    steps_per_period = 20
    drop_cycles = 5
    measure_cycles = 10
    contour_times = [10.0, 10.01, 10.02, 10.03, 10.04, 10.05]
    save_restart = True
    restart_path = None
    notes = "single-frequency FTF checkpoint"

    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2']:
            scheme = arg.lower()
        elif arg == 'no_reinit':
            use_reinit = False
        elif arg.startswith('f='):
            try:
                frequency = float(arg.split('=')[1])
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
    print(f"\nRunning single-frequency FTF: f={frequency} Hz, scheme={scheme}, use_reinit={use_reinit}\n")
    test_ftf_linear_flame_single(
        frequency_hz=frequency,
        time_scheme=scheme,
        use_reinit=use_reinit,
        steps_per_period=steps_per_period,
        drop_cycles=drop_cycles,
        measure_cycles=measure_cycles,
        verbose=True,
        contour_times=contour_times,
        save_restart=save_restart,
        restart_path=restart_path,
        notes=notes,
    )
