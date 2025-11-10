import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
"""
IMPROVED Single-frequency FTF test with higher-order gradient scheme.

Key improvements based on HJ-WENO5 analysis:
1. Use gradient_scheme='weno5' for better |∇G| maintenance
2. Use FMM for proper initial SDF (|∇G| ≈ 1.0)
3. Reduce reinitialization frequency (less needed with WENO5)
4. Compare against baseline (godunov gradient)

Expected benefits:
- 4-5× better gradient maintenance during propagation
- More stable flame tracking
- Less sensitivity to reinitialization frequency
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from g_equation_solver_improved import GEquationSolver2D
from checkpoint_utils import save_checkpoint, CheckpointMeta
from contour_utils import compute_contour_length
from ftf_utils import compute_ftf
from reporting_utils import (
    print_solver_overview, print_solve_start, print_performance, print_completion
)


def fast_marching_method(mask, dx):
    """
    Fast Marching Method to compute signed distance function.
    Returns SDF with |∇φ| ≈ 1.0 everywhere.
    """
    ny, nx = mask.shape
    phi = np.full((ny, nx), np.inf)
    status = np.zeros((ny, nx), dtype=int)  # 0=far, 1=narrow band, 2=known

    # Initialize interface cells
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            if mask[j, i] != mask[j-1, i] or mask[j, i] != mask[j+1, i] or \
               mask[j, i] != mask[j, i-1] or mask[j, i] != mask[j, i+1]:
                phi[j, i] = 0.5 * dx
                status[j, i] = 1

    # Priority queue
    heap = []
    for j in range(ny):
        for i in range(nx):
            if status[j, i] == 1:
                heapq.heappush(heap, (abs(phi[j, i]), j, i))

    # Fast marching
    while heap:
        _, j, i = heapq.heappop(heap)

        if status[j, i] == 2:
            continue

        status[j, i] = 2

        # Update neighbors
        for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            jn, in_ = j + dj, i + di

            if 0 <= jn < ny and 0 <= in_ < nx and status[jn, in_] != 2:
                # Solve Eikonal equation |∇φ| = 1
                phi_x = np.inf
                phi_y = np.inf

                # Get known neighbor values
                if in_ > 0 and status[jn, in_-1] == 2:
                    phi_x = min(phi_x, phi[jn, in_-1])
                if in_ < nx-1 and status[jn, in_+1] == 2:
                    phi_x = min(phi_x, phi[jn, in_+1])
                if jn > 0 and status[jn-1, in_] == 2:
                    phi_y = min(phi_y, phi[jn-1, in_])
                if jn < ny-1 and status[jn+1, in_] == 2:
                    phi_y = min(phi_y, phi[jn+1, in_])

                # Solve quadratic equation
                if phi_x < np.inf and phi_y < np.inf:
                    phi_avg = (phi_x + phi_y) / 2.0
                    discriminant = 2*dx**2 - (phi_x - phi_y)**2
                    if discriminant >= 0:
                        phi_new = phi_avg + np.sqrt(discriminant) / 2.0
                    else:
                        phi_new = min(phi_x, phi_y) + dx
                elif phi_x < np.inf:
                    phi_new = phi_x + dx
                elif phi_y < np.inf:
                    phi_new = phi_y + dx
                else:
                    continue

                if phi_new < phi[jn, in_]:
                    phi[jn, in_] = phi_new
                    if status[jn, in_] == 0:
                        status[jn, in_] = 1
                    heapq.heappush(heap, (phi_new, jn, in_))

    # Apply sign
    phi = np.where(mask, -phi, phi)

    return phi


def initial_horizontal_flame_linear_profile(X, Y, y_flame, use_fmm=False):
    """
    Create initial horizontal flame with linear G profile.

    Parameters:
    -----------
    use_fmm : bool
        If True, use proper SDF with |∇G| = 1.0 (simple signed distance to horizontal line)
        If False, use simple linear profile (old method)
    """
    Ly = float(np.max(Y))
    eps = 1e-12
    y0 = float(y_flame)
    y0_eff = max(y0, eps)
    Ly_minus_y0_eff = max(Ly - y0, eps)

    if not use_fmm:
        # Original linear profile (has |∇G| varying from 1/y0 to 1/(Ly-y0))
        G_lower = 1.0 - (Y / y0_eff)
        G_upper = - (Y - y0) / Ly_minus_y0_eff
        return np.where(Y <= y0, G_lower, G_upper)
    else:
        # Proper SDF for horizontal line: simple signed distance
        # For horizontal flame at y=y0, SDF is just G = y0 - Y (below) or G = Y - y0 (above)
        # This gives |∇G| = 1.0 everywhere
        G_proper = y0 - Y
        return G_proper


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


def test_ftf_linear_flame_single_improved(
    frequency_hz=10.0,
    time_scheme='rk3',
    spatial_scheme='weno5',
    gradient_scheme='weno5',  # NEW: gradient scheme selection
    use_fmm_init=True,        # NEW: FMM initialization
    use_reinit=True,
    reinit_interval=100,      # NEW: less frequent (was 50)
    steps_per_period=20,
    drop_cycles=10,
    measure_cycles=10,
    verbose=True,
    A=0.05,
    K=0.1,
    contour_times=[10.0, 10.01, 10.02, 10.03, 10.04, 10.05],
    save_restart=True,
    restart_path=None,
    notes="improved FTF checkpoint"
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
            "IMPROVED: Higher-order gradient scheme + FMM initialization",
            f"  gradient_scheme={gradient_scheme}, use_fmm_init={use_fmm_init}",
            f"  frequency={frequency_hz:.2f} Hz (omega={omega:.3f} rad/s)",
            f"  u_y_left={u_y_left}, u_y_mid={u_y_mid}, u_y_right={u_y_right}",
            f"  A={A}, K={K}",
            f"  steps_per_period={steps_per_period}, drop_cycles={drop_cycles}, measure_cycles={measure_cycles}",
            f"  dt={dt:.6f} s, total steps≈{n_steps}",
        ]
        reinit_desc = (
            f"Reinitialization: LOCAL every {reinit_interval} steps (fast_marching)"
            if use_reinit else "Reinitialization: Disabled"
        )
        print_solver_overview(
            "IMPROVED FTF Estimation: Linear Flame (HJ-WENO5)",
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

    # Initial G with optional FMM
    G0 = initial_horizontal_flame_linear_profile(solver.X, solver.Y, y0, use_fmm=use_fmm_init)

    # Check initial gradient quality
    if verbose:
        grad_x = np.gradient(G0, solver.dx, axis=1)
        grad_y = np.gradient(G0, solver.dy, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        mask_interface = np.abs(G0) < 3*solver.dx
        grad_int = grad_mag[mask_interface]
        init_label = "Proper SDF" if use_fmm_init else "Linear profile"
        print(f"\nInitial condition quality ({init_label}):")
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
        print(f"Gradient scheme: {gradient_scheme.upper()}")

    start = time.time()
    G_hist, t_hist = solver.solve(
        G0, t_final, dt,
        save_interval=1,
        time_scheme=time_scheme,
        spatial_scheme=spatial_scheme,
        gradient_scheme=gradient_scheme,  # NEW: specify gradient scheme
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

    # Save plots with gradient scheme label
    scheme_label = f"{gradient_scheme}_fmm" if use_fmm_init else f"{gradient_scheme}_linear"

    # Mean flame surface plot
    try:
        A_arr = np.array(flame_lengths, dtype=float)
        A_running_mean = np.cumsum(A_arr) / np.maximum(1, np.arange(1, len(A_arr) + 1))
        t_start_win = t_arr[0] + drop_cycles * T
        m_win = t_arr >= t_start_win
        A_mean_window = float(np.mean(A_arr[m_win])) if np.any(m_win) else float(np.mean(A_arr))

        fig_mean, axm = plt.subplots(1, 1, figsize=(9, 4))
        axm.plot(t_arr, A_arr, color='0.7', lw=1.0, label='A(t) (instantaneous)')
        axm.plot(t_arr, A_running_mean, color='tab:green', lw=2.0, label='Ā(t) (running mean)')
        axm.axhline(A_mean_window, color='crimson', ls='--', lw=1.5, label='Ā (steady-state window)')
        axm.axvline(t_start_win, color='crimson', linestyle=':', linewidth=1.2, alpha=0.9, label='window start')
        axm.set_xlabel('Time (s)')
        axm.set_ylabel('Flame surface proxy A (|Γ|)')
        axm.grid(True, alpha=0.3)
        axm.legend(loc='best')
        axm.set_title(
            f'IMPROVED ({gradient_scheme.upper()}): Mean flame surface vs time\n'
            f'f={frequency_hz:.1f} Hz, Ā_window={A_mean_window:.6f}'
        )
        fig_mean.tight_layout()
        fig_mean.savefig(f'ftf_single_A_mean_time_f{frequency_hz:.1f}Hz_{scheme_label}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_mean)
    except Exception:
        pass

    # Time history plot
    try:
        t_start_win = t_arr[0] + drop_cycles * T
        plot_start = max(t_arr[0], t_start_win - 2.0 * T)
        m_plot = t_arr >= plot_start
        fig_ts, ax_l = plt.subplots(1, 1, figsize=(9, 4))
        ax_r = ax_l.twinx()
        ln1 = ax_l.plot(t_arr[m_plot], np.array(flame_lengths)[m_plot], 'k-', linewidth=1.5, label=f'|Γ| (skip N={spatial_skip_N})')
        time_skip_N = 5
        marked_idx_all = np.arange(0, len(t_arr), time_skip_N)
        marked_mask = np.zeros_like(t_arr, dtype=bool)
        marked_mask[marked_idx_all] = True
        marked_in_window = m_plot & marked_mask
        ax_l.plot(t_arr[marked_in_window], np.array(flame_lengths)[marked_in_window], 'ro', markersize=3, label=f'sampled times (every {time_skip_N})')
        ln2 = ax_r.plot(t_arr[m_plot], u_ref[m_plot], 'b--', linewidth=1.5, label='u_ref (mid, y=0)')
        vline = ax_l.axvline(x=t_start_win, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label='window start')
        if use_reinit and reinit_interval > 0:
            reinit_indices = np.arange(0, len(t_arr), reinit_interval)
            reinit_times = t_arr[reinit_indices]
            reinit_window_mask = reinit_times >= plot_start
            reinit_times_plot = reinit_times[reinit_window_mask]
            if reinit_times_plot.size > 0:
                y_min_plot = np.min(np.array(flame_lengths)[m_plot])
                y_max_plot = np.max(np.array(flame_lengths)[m_plot])
                y_span = max(1e-12, y_max_plot - y_min_plot)
                y_mark = y_min_plot - 0.04 * y_span
                ax_l.plot(reinit_times_plot, np.full_like(reinit_times_plot, y_mark),
                          'o', color='orange', markersize=4, label='reinit')
                ax_l.set_ylim(y_mark - 0.02 * y_span, y_max_plot + 0.02 * y_span)
        ax_l.set_xlabel('Time (s)'); ax_l.set_ylabel('Flame length |Γ|'); ax_r.set_ylabel('Velocity u_y (m/s)')
        ax_l.grid(True, alpha=0.3)
        lines = ln1 + ln2 + [vline]; labels = [l.get_label() for l in lines]
        ax_l.legend(lines, labels, loc='best')
        ax_l.set_title(
            f'IMPROVED ({gradient_scheme.upper()}): f={frequency_hz:.1f} Hz\n'
            f'gain={gain:.4f}, phase={phase:.1f}°, dt={dt:.6f}s'
        )
        fig_ts.tight_layout()
        fig_ts.savefig(f'ftf_single_time_f{frequency_hz:.1f}Hz_{scheme_label}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_ts)
    except Exception:
        pass

    # Contour grid
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

    # Save restart checkpoint
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
                    'use_fmm_init': use_fmm_init,
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
        'use_fmm_init': use_fmm_init,
    }


def compare_gradient_schemes(frequency_hz=10.0, **kwargs):
    """
    Run comparison between Godunov and WENO5 gradient schemes.
    """
    print("\n" + "="*80)
    print("COMPARISON: Godunov vs WENO5 Gradient Schemes")
    print("="*80)

    results = {}

    # Test 1: Godunov (baseline)
    print("\n" + "-"*80)
    print("TEST 1: GODUNOV gradient scheme (baseline)")
    print("-"*80)
    results['godunov'] = test_ftf_linear_flame_single_improved(
        frequency_hz=frequency_hz,
        gradient_scheme='godunov',
        use_fmm_init=True,
        verbose=True,
        **kwargs
    )

    # Test 2: WENO5 (improved)
    print("\n" + "-"*80)
    print("TEST 2: WENO5 gradient scheme (improved)")
    print("-"*80)
    results['weno5'] = test_ftf_linear_flame_single_improved(
        frequency_hz=frequency_hz,
        gradient_scheme='weno5',
        use_fmm_init=True,
        verbose=True,
        **kwargs
    )

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Frequency: {frequency_hz:.2f} Hz\n")
    print(f"{'Scheme':<15} {'Gain':<12} {'Phase (deg)':<15} {'Time (s)':<12}")
    print("-"*80)
    for scheme in ['godunov', 'weno5']:
        r = results[scheme]
        print(f"{scheme.upper():<15} {r['gain']:<12.6f} {r['phase_deg']:<15.2f} {r.get('elapsed', 0.0):<12.2f}")

    gain_diff = abs(results['weno5']['gain'] - results['godunov']['gain'])
    phase_diff = abs(results['weno5']['phase_deg'] - results['godunov']['phase_deg'])

    print("\n" + "="*80)
    print(f"Gain difference:  {gain_diff:.6f} ({gain_diff/results['godunov']['gain']*100:.2f}%)")
    print(f"Phase difference: {phase_diff:.2f}°")
    print("="*80)

    return results


if __name__ == '__main__':
    # CLI parsing
    frequency = 10.0
    scheme = 'rk3'  # Use RK3 for better accuracy
    gradient_scheme = 'weno5'  # Default to WENO5
    use_fmm_init = True
    use_reinit = True
    reinit_interval = 100  # Less frequent than original (was 50)
    steps_per_period = 20
    drop_cycles = 50
    measure_cycles = 10
    contour_times = [10.0, 10.01, 10.02, 10.03, 10.04, 10.05]
    save_restart = True
    restart_path = None
    notes = "improved FTF checkpoint"
    compare_mode = False

    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2', 'rk3']:
            scheme = arg.lower()
        elif arg.lower() in ['godunov', 'weno5']:
            gradient_scheme = arg.lower()
        elif arg == 'no_fmm':
            use_fmm_init = False
        elif arg == 'no_reinit':
            use_reinit = False
        elif arg == 'compare':
            compare_mode = True
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
        elif arg.lower() == 'no_save':
            save_restart = False
        elif arg.startswith('ckpt='):
            restart_path = arg.split('=',1)[1]
        elif arg.startswith('notes='):
            notes = arg.split('=',1)[1]

    if compare_mode:
        print(f"\nRunning COMPARISON mode: Godunov vs WENO5, f={frequency} Hz\n")
        compare_gradient_schemes(
            frequency_hz=frequency,
            time_scheme=scheme,
            use_reinit=use_reinit,
            reinit_interval=reinit_interval,
            steps_per_period=steps_per_period,
            drop_cycles=drop_cycles,
            measure_cycles=measure_cycles,
            contour_times=contour_times,
            save_restart=save_restart,
            notes=notes,
        )
    else:
        print(f"\nRunning IMPROVED FTF: f={frequency} Hz, scheme={scheme}, gradient={gradient_scheme}, FMM={use_fmm_init}\n")
        test_ftf_linear_flame_single_improved(
            frequency_hz=frequency,
            time_scheme=scheme,
            gradient_scheme=gradient_scheme,
            use_fmm_init=use_fmm_init,
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
