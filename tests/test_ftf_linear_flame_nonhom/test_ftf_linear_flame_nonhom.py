import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
FTF estimation test based on the non-homogeneous linear flame case.
Uses per-frequency sweep method to compute gain and phase between inlet velocity
and flame surface area (length of G=0 contour).
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D
from contour_utils import compute_contour_length
from ftf_utils import compute_ftf, plot_ftf_bode
from reporting_utils import print_solver_overview, print_solve_start, print_performance, print_completion
import time


def initial_horizontal_flame_linear_profile(X, Y, y_flame):
    """Linear-in-y level set with G=+1 at y=0, G=0 at y_flame, G=-1 at y=Ly."""
    Ly = float(np.max(Y))
    eps = 1e-12
    y0 = float(y_flame)
    y0_eff = max(y0, eps)
    Ly_minus_y0_eff = max(Ly - y0, eps)
    G_lower = 1.0 - (Y / y0_eff)
    G_upper = - (Y - y0) / Ly_minus_y0_eff
    return np.where(Y <= y0, G_lower, G_upper)


def create_three_region_updater(X, Y, x1=0.1, x2=0.9, U_Y=0.2, A=0.05, St=12.566370614359172, K=0.1,
                                u_y_left=0.0, u_y_right=0.0):
    """
    Time-dependent middle-band velocity: u_y = U_Y*(1 + A*sin(St*(t - K*y))).
    Left and right bands are constant u_y_left/u_y_right.
    Returns updater(solver, t) -> (u_x, u_y).
    """
    Xl = X.copy(); Yl = Y.copy()
    left_mask = (Xl <= x1)
    right_mask = (Xl >= x2)
    def updater(solver, t):
        u_x = np.zeros_like(Xl)
        u_y_mid = U_Y * (1.0 + A * np.sin(St * (t - K * Yl)))
        u_y = np.where(left_mask, u_y_left, np.where(right_mask, u_y_right, u_y_mid))
        return u_x, u_y
    return updater

def test_ftf_linear_flame_nonhom(
    time_scheme='rk2', use_reinit=True, verbose=True,
    N=3, f_min_hz=50.0, f_max_hz=500.0,
    steps_per_period=20, drop_cycles=500, measure_cycles=10,
    plot_start_t: float = 10.0,
):
    """Run FTF sweeps for N frequencies between f_min_hz and f_max_hz and plot gain/phase.

    Time per frequency ≈ (drop+measure)/f, and dt chosen to achieve steps_per_period.
    This keeps steps per frequency roughly constant: (drop+measure)*steps_per_period.
    """

    # Domain and numerics (grid and speeds)
    nx, ny = 101, 101
    Lx, Ly = 1.0, 1.5
    S_L = 0.1

    # Velocity regions and forcing (A and K fixed across sweeps)
    x1, x2 = 0.1, 0.9
    u_y_left, u_y_mid, u_y_right = 0.0, 0.2, 0.0
    A = 0.10
    K = 0.1

    # Initial flame position
    y0 = 0.3

    if verbose:
        flow_desc = [
            "Three-region vertical flow with time-dependent middle band",
            f"  u_y_left={u_y_left}, u_y_mid={u_y_mid}, u_y_right={u_y_right}",
            f"  Forcing sweep: f in [{f_min_hz:.1f}, {f_max_hz:.1f}] Hz, N={N}",
            f"  steps_per_period={steps_per_period}, drop_cycles={drop_cycles}, measure_cycles={measure_cycles}",
        ]
        reinit_desc = (
            f"Reinitialization: LOCAL every 50 steps (fast_marching)"
            if use_reinit else "Reinitialization: Disabled"
        )
        print_solver_overview(
            "FTF Estimation: Linear Flame (Three-Region) — Frequency Sweep",
            nx, ny, Lx, Ly, S_L, flow_desc, time_scheme, reinit_desc
        )

    # Solver with placeholder velocities (overridden by updater each step)
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)

    # Spatially varying S_L to pin bottom sides (keep as in reference test)
    mask_bottom_side = (solver.Y < 0.001) & ((solver.X < x1) | (solver.X > x2))
    solver.S_L = np.where(mask_bottom_side, 0.0, S_L)

    # Dirichlet inlet band G=+1 at bottom
    bottom_band_height = min(solver.dy * 0.5, 0.001)
    pin_mask = (solver.Y < bottom_band_height)
    pin_values = np.zeros_like(solver.X)
    pin_values[pin_mask] = +1.0
    try:
        solver.set_pinned_region(pin_mask, pin_values)
    except AttributeError:
        pass

    # Frequency list
    freqs = np.linspace(f_min_hz, f_max_hz, N)
    gains = []
    phases = []

    total_start = time.time()

    for idx, f in enumerate(freqs):
        omega = 2*np.pi*f
        T = 1.0 / f
        dt = min(0.001, T / float(steps_per_period))
        t_final = (drop_cycles + measure_cycles) * T
        n_steps = int(np.ceil(t_final / dt))
        save_interval = 1  # save every step for clean projection

        if verbose:
            print_solve_start()
            print(f"Frequency {idx+1}/{N}: f={f:.2f} Hz, omega={omega:.3f} rad/s, T={T:.4f} s, dt={dt:.6f} s, steps≈{n_steps}")

        # Initial condition fresh for each run
        G0 = initial_horizontal_flame_linear_profile(solver.X, solver.Y, y0)

        # Time-dependent velocity updater for this frequency
        velocity_updater = create_three_region_updater(
            solver.X, solver.Y, x1=x1, x2=x2, U_Y=u_y_mid, A=A, St=omega, K=K,
            u_y_left=u_y_left, u_y_right=u_y_right
        )

        t_start = time.time()
        G_hist, t_hist = solver.solve(
            G0, t_final, dt,
            save_interval=save_interval,
            time_scheme=time_scheme,
            reinit_interval=(50 if use_reinit else 0),
            reinit_method='fast_marching',
            reinit_local=True,
            velocity_updater=velocity_updater,
        )
        elapsed = time.time() - t_start
        if verbose:
            print_performance(elapsed, int(np.ceil(t_final/dt)), len(t_hist))

        # Signals
        flame_lengths = [compute_contour_length(G, solver.X, solver.Y, iso_value=0.0) for G in G_hist]
        t_arr = np.array(t_hist)
        u_ref = u_y_mid * (1.0 + A * np.sin(omega * (t_arr - K * 0.0)))

        # FTF at this frequency
        ftf = compute_ftf(
            area=np.array(flame_lengths),
            u=u_ref,
            t=t_arr,
            frequency_hz=f,
            drop_cycles=drop_cycles,
            normalize='relative',
            detrend=True,
            window='hann',
            phase_units='deg',
        )
        gains.append(float(ftf['gain'][0]))
        phases.append(float(ftf['phase'][0]))

        # Per-frequency time plot: flame area and forcing velocity (from plot_start_t),
        # with a vertical line marking the start of the steady-state window
        try:
            T_win = T
            t_start_win = t_arr[0] + drop_cycles * T_win
            if len(t_arr) > 1:
                fig_ts, ax_l = plt.subplots(1, 1, figsize=(9, 4))
                ax_r = ax_l.twinx()
                # Start plotting from two cycles before the window start, clipped to domain start
                plot_start = max(t_arr[0], t_start_win - 2.0 * T_win)
                m_plot = t_arr >= plot_start
                ln1 = ax_l.plot(t_arr[m_plot], np.array(flame_lengths)[m_plot], 'k-', linewidth=1.5, label='Flame length |Γ|')
                ln2 = ax_r.plot(t_arr[m_plot], u_ref[m_plot], 'b--', linewidth=1.5, label='u_ref (mid, y=0)')
                # Mark window start (post-transient)
                vline = ax_l.axvline(x=t_start_win, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label='window start')
                ax_l.set_xlabel('Time (s)')
                ax_l.set_ylabel('Flame length |Γ|', color='k')
                ax_r.set_ylabel('Velocity u_y (m/s)', color='b')
                ax_l.grid(True, alpha=0.3)
                # Build combined legend
                lines = ln1 + ln2 + [vline]
                labels = [l.get_label() for l in lines]
                ax_l.legend(lines, labels, loc='best')
                # Title with FTF info
                gain_f = float(ftf['gain'][0])
                phase_f = float(ftf['phase'][0])
                ax_l.set_title(
                    f'FTF sweep at f={f:.1f} Hz (ω={omega:.1f} rad/s): '
                    f'gain={gain_f:.4f}, phase={phase_f:.1f}°, dt={dt:.6f}s'
                )
                fname = f'ftf_sweep_time_f{f:.1f}Hz.png'
                fig_ts.tight_layout()
                fig_ts.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close(fig_ts)
        except Exception:
            pass

    total_elapsed = time.time() - total_start

    gains = np.array(gains)
    phases = np.array(phases)

    # Plot FTF curves
    plot_ftf_bode(freqs, gains, phases,
                  title='FTF Bode: Linear Flame (Three-Region)',
                  savepath='ftf_bode_linear_flame_nonhom.png',
                  phase_units='deg')

    if verbose:
        print("\nFTF SWEEP SUMMARY:")
        for f, g, ph in zip(freqs, gains, phases):
            print(f"  f={f:7.2f} Hz  |H|={g:8.5f}  phase={ph:8.2f} deg")
        print(f"\nTotal sweep time: {total_elapsed:.3f} s for {N} frequencies")
        print_completion()

    return {
        'frequencies_hz': freqs,
        'gains': gains,
        'phases_deg': phases,
    }


if __name__ == "__main__":
    import sys
    scheme = 'rk2'
    use_reinit = True
    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2']:
            scheme = arg.lower()
        elif arg == 'no_reinit':
            use_reinit = False
    print(f"\nRunning FTF sweep with: scheme={scheme}, use_reinit={use_reinit}\n")
    test_ftf_linear_flame_nonhom(time_scheme=scheme, use_reinit=use_reinit, verbose=True)
