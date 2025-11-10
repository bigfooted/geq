import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
"""
Multi-frequency Flame Transfer Function (FTF) test for the non-homogeneous linear flame.
Loops over multiple forcing frequencies and computes gain/phase between inlet velocity
(reference mid-band velocity at y=0) and flame surface length (|Gamma| of G=0 contour).

This version efficiently runs multiple frequencies using the memory-optimized callback approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import csv
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


def run_single_frequency(
    frequency_hz,
    time_scheme='rk3',
    use_reinit=False,
    steps_per_period=20,
    drop_cycles=1,
    measure_cycles=10,
    verbose=True,
    A=0.05,
    K=0.1,
    save_contours=False,
    save_time_series=False,
):
    """Run single frequency simulation - optimized for batch processing."""
    # Domain/grid
    nx, ny = 201, 201
    Lx, Ly = 0.12, 0.25
    S_L = 0.4

    # Velocity regions
    x1, x2 = 0.05*Lx, 0.95*Lx
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

    # Compute actual CFL for reporting
    cfl_actual = (u_max + S_L) * dt / dx

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running f={frequency_hz:.1f} Hz (ω={omega:.1f} rad/s)")
        print(f"{'='*80}")
        print(f"  dt={dt:.6f} s, CFL={cfl_actual:.3f}, steps={n_steps}, t_final={t_final:.2f}s")
        print(f"  drop_cycles={drop_cycles}, measure_cycles={measure_cycles}")

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
    spatial_skip_N = 5

    def compute_flame_length_callback(G, t, step):
        """Callback to compute flame length on-the-fly"""
        length = compute_contour_length(G, solver.X, solver.Y, iso_value=0.0, N=spatial_skip_N)
        return length

    # Determine save_interval for good temporal resolution
    target_snapshots = 1500
    save_interval = max(1, int(np.ceil(n_steps / target_snapshots)))
    expected_calls = n_steps // save_interval + 1

    # Optional: save snapshots for contour plots
    contour_times = None
    if save_contours:
        t_start = t_final - measure_cycles * T
        contour_times = [t_start + k * T/5 for k in range(6)]

    if verbose:
        print(f"  Solving with on-the-fly flame length computation...")

    start = time.time()
    snapshots, t_hist, callback_results = solver.solve(
        G0, t_final, dt,
        save_interval=save_interval,
        time_scheme=time_scheme,
        reinit_interval=(50 if use_reinit else 0),
        reinit_method=('fast_marching' if use_reinit else 'none'),
        reinit_local=True,
        velocity_updater=velocity_updater,
        callback=compute_flame_length_callback,
        snapshot_times=contour_times,
    )
    elapsed = time.time() - start

    # Extract flame lengths and time from callback results
    flame_lengths = callback_results
    t_arr = np.array(t_hist)

    if verbose:
        print(f"  Simulation completed in {elapsed:.2f}s ({elapsed/n_steps*1000:.2f} ms/step)")

    # Signals
    u_ref = u_y_mid * (1.0 + A * np.sin(omega * (t_arr - K * 0.0)))

    # Compute FTF
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

    if verbose:
        print(f"  FTF result: gain={gain:.6f}, phase={phase:.2f}° ({phase_rad:.4f} rad)")

    # Optional: save time series plot
    if save_time_series:
        try:
            fig_ts, ax_l = plt.subplots(1, 1, figsize=(9, 4))
            ax_r = ax_l.twinx()
            ln1 = ax_l.plot(t_arr, flame_lengths, 'k-', linewidth=1.5, label='|Γ|')
            ln2 = ax_r.plot(t_arr, u_ref, 'b--', linewidth=1.5, label='u_ref')
            t_start_win = t_arr[0] + drop_cycles * T
            vline = ax_l.axvline(x=t_start_win, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label='window start')
            ax_l.set_xlabel('Time (s)')
            ax_l.set_ylabel('Flame length |Γ|')
            ax_r.set_ylabel('Velocity u_y (m/s)')
            ax_l.grid(True, alpha=0.3)
            lines = ln1 + ln2 + [vline]
            labels = [l.get_label() for l in lines]
            ax_l.legend(lines, labels, loc='best')
            ax_l.set_title(f'f={frequency_hz:.1f} Hz: gain={gain:.4f}, phase={phase:.1f}°')
            fig_ts.tight_layout()
            fig_ts.savefig(f'ftf_multi_time_f{frequency_hz:.1f}Hz.png', dpi=300, bbox_inches='tight')
            plt.close(fig_ts)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not save time series plot: {e}")

    # Optional: save contour plots
    if save_contours and snapshots is not None and len(snapshots) > 0:
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
                    ax.set_title(f't={t_actual:.3f}s')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
            plt.tight_layout()
            figc.savefig(f'ftf_multi_contours_f{frequency_hz:.1f}Hz.png', dpi=300, bbox_inches='tight')
            plt.close(figc)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not save contour plots: {e}")

    return {
        'frequency_hz': frequency_hz,
        'omega': omega,
        'gain': gain,
        'phase_deg': phase,
        'phase_rad': phase_rad,
        'dt': dt,
        'steps': n_steps,
        'elapsed_time': elapsed,
        'cfl': cfl_actual,
    }


def test_ftf_linear_flame_multiple(
    frequency_list=[5, 10, 15, 20, 25, 50, 100],
    time_scheme='rk3',
    use_reinit=False,
    steps_per_period=20,
    drop_cycles=1,
    measure_cycles=10,
    A=0.05,
    K=0.1,
    save_contours=False,
    save_time_series=False,
    output_csv='ftf_multiple_results.csv',
):
    """
    Run FTF computation for multiple frequencies.

    Parameters:
    -----------
    frequency_list : list
        List of frequencies to test (Hz)
    output_csv : str
        Output CSV file for results
    save_contours : bool
        Save contour plots for each frequency
    save_time_series : bool
        Save time series plots for each frequency
    """

    print(f"\n{'='*80}")
    print(f"MULTI-FREQUENCY FTF TEST")
    print(f"{'='*80}")
    print(f"Frequencies: {frequency_list}")
    print(f"Time scheme: {time_scheme}")
    print(f"Reinitialization: {use_reinit}")
    print(f"Steps per period: {steps_per_period}")
    print(f"Drop cycles: {drop_cycles}, Measure cycles: {measure_cycles}")
    print(f"A={A}, K={K}")
    print(f"{'='*80}\n")

    results = []
    total_start = time.time()

    for freq in frequency_list:
        result = run_single_frequency(
            frequency_hz=freq,
            time_scheme=time_scheme,
            use_reinit=use_reinit,
            steps_per_period=steps_per_period,
            drop_cycles=drop_cycles,
            measure_cycles=measure_cycles,
            verbose=True,
            A=A,
            K=K,
            save_contours=save_contours,
            save_time_series=save_time_series,
        )
        results.append(result)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"ALL SIMULATIONS COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
    print(f"Average time per frequency: {total_elapsed/len(frequency_list):.2f}s")
    print(f"\nResults Summary:")
    print(f"{'Freq (Hz)':>10} {'Omega':>10} {'Gain':>12} {'Phase (°)':>12} {'Phase (rad)':>13} {'Time (s)':>10}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['frequency_hz']:>10.1f} {r['omega']:>10.2f} {r['gain']:>12.6f} {r['phase_deg']:>12.2f} {r['phase_rad']:>13.4f} {r['elapsed_time']:>10.2f}")

    # Save results to CSV
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['frequency_hz', 'omega', 'gain', 'phase_deg', 'phase_rad', 'dt', 'steps', 'cfl', 'elapsed_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to: {output_csv}")
    except Exception as e:
        print(f"\nWarning: Could not save CSV: {e}")

    # Create summary plot: gain and phase vs frequency
    try:
        frequencies = [r['frequency_hz'] for r in results]
        gains = [r['gain'] for r in results]
        phases = [r['phase_deg'] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Gain plot
        ax1.semilogx(frequencies, gains, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('FTF Gain')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.set_title(f'Flame Transfer Function: {len(frequency_list)} frequencies')

        # Phase plot
        ax2.semilogx(frequencies, phases, 's-', linewidth=2, markersize=8, color='C1')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (°)')
        ax2.grid(True, alpha=0.3, which='both')

        fig.tight_layout()
        fig.savefig('ftf_multiple_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Summary plot saved to: ftf_multiple_summary.png")
    except Exception as e:
        print(f"Warning: Could not create summary plot: {e}")

    return results


if __name__ == '__main__':
    # Default parameters
    frequency_list = [5, 10, 15, 20, 25, 50, 100]
    scheme = 'rk3'
    use_reinit = False
    steps_per_period = 20
    drop_cycles = 1
    measure_cycles = 10
    A = 0.05
    K = 0.1
    save_contours = False
    save_time_series = False
    output_csv = 'ftf_multiple_results.csv'

    # CLI parsing
    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2', 'rk3']:
            scheme = arg.lower()
        elif arg == 'use_reinit':
            use_reinit = True
        elif arg == 'save_contours':
            save_contours = True
        elif arg == 'save_time_series':
            save_time_series = True
        elif arg.startswith('frequencies='):
            try:
                vals = arg.split('=', 1)[1]
                frequency_list = [float(v) for v in vals.split(',') if v.strip()]
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
        elif arg.startswith('A='):
            try:
                A = float(arg.split('=')[1])
            except Exception:
                pass
        elif arg.startswith('K='):
            try:
                K = float(arg.split('=')[1])
            except Exception:
                pass
        elif arg.startswith('output='):
            output_csv = arg.split('=', 1)[1]

    print(f"\nRunning multi-frequency FTF: {len(frequency_list)} frequencies")
    print(f"Frequencies: {frequency_list}")
    print(f"Scheme: {scheme}, reinit: {use_reinit}\n")

    test_ftf_linear_flame_multiple(
        frequency_list=frequency_list,
        time_scheme=scheme,
        use_reinit=use_reinit,
        steps_per_period=steps_per_period,
        drop_cycles=drop_cycles,
        measure_cycles=measure_cycles,
        A=A,
        K=K,
        save_contours=save_contours,
        save_time_series=save_time_series,
        output_csv=output_csv,
    )
