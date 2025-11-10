import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Compare four reinitialization configurations for the non-homogeneous linear flame
under a single-frequency forcing. For each configuration, we run the same setup
and compare flame length |Γ|(t) around and inside the measurement window, and
report FTF gain/phase.

Configurations:
  1) none:            No reinitialization at all (reinit_method='none')
  2) fm_transient:    Fast-marching reinit during transient only; disabled in measurement
  3) pde_transient:   PDE-based reinit during transient only; disabled in measurement
  4) fm_continuous:   Fast-marching reinit for the entire run

Notes:
  - Uses spatial_scheme='weno5' and time_scheme selectable (default 'rk2').
  - Flame length computed with contour subsampling N=5 for speed.
  - Saves an overlay plot 'compare_reinit_configs_f{f}Hz.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

from g_equation_solver_improved import GEquationSolver2D
from contour_utils import compute_contour_length
import csv
from ftf_utils import compute_ftf


def initial_horizontal_flame_linear_profile(X, Y, y_flame):
    Ly = float(np.max(Y))
    eps = 1e-12
    y0 = float(y_flame)
    y0_eff = max(y0, eps)
    Ly_minus_y0_eff = max(Ly - y0, eps)
    G_lower = 1.0 - (Y / y0_eff)
    G_upper = - (Y - y0) / Ly_minus_y0_eff
    return np.where(Y <= y0, G_lower, G_upper)


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


def run_config(config_name, solver, G0, dt, T, drop_cycles, measure_cycles,
               velocity_updater, u_y_mid, A, omega, K,
               spatial_scheme='weno5', time_scheme='rk2'):
    """Run one configuration and return (G_hist, t_arr, flame_lengths, u_ref, ftf_info).

    u_ref(t) = u_y_mid * (1 + A * sin(omega * (t - K*0))). (0 used for y=0 mid-band.)
    """

    spatial_skip_N = 5

    def compute_lengths(G_hist):
        return [compute_contour_length(G, solver.X, solver.Y, iso_value=0.0, N=spatial_skip_N) for G in G_hist]

    # Helper to perform a segment run
    def do_run(G_init, t_final, *, save_interval, reinit_interval, reinit_method, t0=0.0):
        G_hist, t_hist = solver.solve(
            G_init, t_final, dt,
            save_interval=save_interval,
            time_scheme=time_scheme,
            reinit_interval=reinit_interval,
            reinit_method=reinit_method,
            reinit_local=True,
            smooth_ic=False,
            velocity_updater=velocity_updater,
            spatial_scheme=spatial_scheme,
            t0=t0,
        )
        return G_hist, t_hist

    # Define timing
    t_transient = drop_cycles * T
    t_meas = measure_cycles * T

    if config_name == 'none':
        # Single run, no reinit, save all steps
        G_hist, t_hist = do_run(G0, t_transient + t_meas, save_interval=1,
                                reinit_interval=0, reinit_method='none', t0=0.0)

    elif config_name == 'fm_transient':
        # Transient with fast marching, no-save, then measurement with no reinit, save
        Gt, tt = do_run(G0, t_transient, save_interval=int(1e9),
                        reinit_interval=50, reinit_method='fast_marching', t0=0.0)
        Gm, tm = do_run(Gt[-1], t_transient + t_meas, save_interval=1,
                        reinit_interval=0, reinit_method='none', t0=tt[-1])
        G_hist, t_hist = Gm, tm

    elif config_name == 'pde_transient':
        Gt, tt = do_run(G0, t_transient, save_interval=int(1e9),
                        reinit_interval=50, reinit_method='pde', t0=0.0)
        Gm, tm = do_run(Gt[-1], t_transient + t_meas, save_interval=1,
                        reinit_interval=0, reinit_method='none', t0=tt[-1])
        G_hist, t_hist = Gm, tm

    elif config_name == 'fm_continuous':
        G_hist, t_hist = do_run(G0, t_transient + t_meas, save_interval=1,
                                reinit_interval=50, reinit_method='fast_marching', t0=0.0)
    elif config_name == 'upwind_rk2':
        # Behavior identical to fm_transient, but caller should pass spatial_scheme='upwind' and time_scheme='rk2'
        Gt, tt = do_run(G0, t_transient, save_interval=int(1e9),
                        reinit_interval=50, reinit_method='fast_marching', t0=0.0)
        Gm, tm = do_run(Gt[-1], t_transient + t_meas, save_interval=1,
                        reinit_interval=0, reinit_method='none', t0=tt[-1])
        G_hist, t_hist = Gm, tm
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Signals and FTF
    t_arr = np.array(t_hist)
    flame_lengths = compute_lengths(G_hist)
    u_ref = u_y_mid * (1.0 + A * np.sin(omega * (t_arr - K * 0.0)))
    ftf = compute_ftf(
        area=np.array(flame_lengths),
        u=u_ref,
        t=t_arr,
        frequency_hz=omega / (2*np.pi),
        drop_cycles=drop_cycles,
        normalize='relative',
        detrend=True,
        window='hann',
        phase_units='deg',
    )
    return G_hist, t_arr, np.array(flame_lengths), u_ref, ftf


def test_compare_reinit_configs(
    frequency_hz=50.0,
    time_scheme='rk2',
    steps_per_period=20,
    drop_cycles=200,
    measure_cycles=10,
    verbose=True,
):
    # Domain
    nx, ny = 101, 101
    Lx, Ly = 1.0, 1.5
    S_L = 0.1
    x1, x2 = 0.1, 0.9
    u_y_left, u_y_mid, u_y_right = 0.0, 0.2, 0.0
    y0 = 0.3

    omega = 2*np.pi*frequency_hz
    T = 1.0 / frequency_hz
    dt = min(0.001, T / float(steps_per_period))
    # Forcing parameters (keep consistent with updater)
    A = 0.10
    K = 0.1

    if verbose:
        print(f"Compare reinit configs at f={frequency_hz:.2f} Hz, dt={dt:.6f}, drop={drop_cycles} cycles, measure={measure_cycles} cycles")

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)
    # Pin bottom sides (keep consistent with other tests)
    mask_bottom_side = (solver.Y < 0.001) & ((solver.X < x1) | (solver.X > x2))
    solver.S_L = np.where(mask_bottom_side, 0.0, S_L)
    bottom_band_height = min(solver.dy * 0.5, 0.001)
    pin_mask = (solver.Y < bottom_band_height)
    pin_values = np.zeros_like(solver.X); pin_values[pin_mask] = +1.0
    try:
        solver.set_pinned_region(pin_mask, pin_values)
    except AttributeError:
        pass

    G0 = initial_horizontal_flame_linear_profile(solver.X, solver.Y, y0)
    velocity_updater = create_three_region_updater(
        solver.X, solver.Y, x1=x1, x2=x2, U_Y=u_y_mid, A=0.10, St=omega, K=0.1,
        u_y_left=u_y_left, u_y_right=u_y_right
    )

    configs = ['none', 'fm_transient', 'pde_transient', 'fm_continuous', 'upwind_rk2']
    colors = {
        'none': 'k', 'fm_transient': 'tab:blue', 'pde_transient': 'tab:green', 'fm_continuous': 'tab:red',
        'upwind_rk2': 'tab:purple'
    }
    labels = {
        'none': 'No reinit (WENO5)', 'fm_transient': 'FM reinit (transient only, WENO5)',
        'pde_transient': 'PDE reinit (transient only, WENO5)', 'fm_continuous': 'FM reinit (continuous, WENO5)',
        'upwind_rk2': 'FM reinit (transient only, Upwind+RK2)'
    }

    results = {}
    timings = {}
    start = time.time()
    for name in configs:
        if verbose:
            print(f"Running config: {name} ...")
        # Choose discretization per config
        if name == 'upwind_rk2':
            sp_scheme = 'upwind'
            t_scheme = 'rk2'
        else:
            sp_scheme = 'weno5'
            t_scheme = time_scheme
        t0c = time.time()
        G_hist, t_arr, lengths, u_ref, ftf = run_config(
            name, solver, G0, dt, T, drop_cycles, measure_cycles,
            velocity_updater, u_y_mid, A, omega, K,
            spatial_scheme=sp_scheme, time_scheme=t_scheme
        )
        timings[name] = time.time() - t0c
        results[name] = (G_hist, t_arr, lengths, u_ref, ftf)
    elapsed = time.time() - start
    if verbose:
        print(f"All configs completed in {elapsed:.2f} s")

    # Plot overlay around measurement window
    # Extract window start from any config's time history
    t_start_win = results['none'][1][0] + drop_cycles * T
    plot_start = max(results['none'][1][0], t_start_win - 2.0 * T)
    fig, ax_l = plt.subplots(1, 1, figsize=(10, 5))
    ax_r = ax_l.twinx()

    lines = []
    labels_list = []
    for name in configs:
        _, t_arr, lengths, u_ref, ftf = results[name]
        m_plot = t_arr >= plot_start
        ln = ax_l.plot(t_arr[m_plot], lengths[m_plot], color=colors[name], linewidth=1.6,
                       label=f"{labels[name]} |H|={float(ftf['gain'][0]):.4f}, φ={float(ftf['phase'][0]):.1f}°")
        lines += ln
        labels_list += [ln[0].get_label()]
    # Plot u_ref from any run (use 'none')
    _, t_arr0, _, u_ref0, _ = results['none']
    m_plot0 = t_arr0 >= plot_start
    ln_u = ax_r.plot(t_arr0[m_plot0], u_ref0[m_plot0], 'k--', alpha=0.4, label='u_ref (mid, y=0)')
    lines += ln_u
    labels_list += [ln_u[0].get_label()]

    vline = ax_l.axvline(x=t_start_win, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label='window start')
    lines += [vline]
    labels_list += ['window start']

    ax_l.set_xlabel('Time (s)'); ax_l.set_ylabel('Flame length |Γ|'); ax_r.set_ylabel('Velocity u_y (m/s)')
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(lines, labels_list, loc='best')
    ax_l.set_title(f'Reinit configuration comparison at f={frequency_hz:.1f} Hz (dt={dt:.6f}s)')
    fig.tight_layout()
    fig.savefig(f'compare_reinit_configs_f{frequency_hz:.1f}Hz.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Gradient statistics per configuration (over measurement window only)
    def gradient_stats(G_hist, t_arr, dx, dy):
        # Identify measurement window indices
        win_mask = t_arr >= t_start_win
        if not np.any(win_mask):
            return {
                'samples': 0, 'grad_mean_mean': np.nan, 'grad_mean_std': np.nan,
                'grad_min': np.nan, 'grad_max': np.nan
            }
        # finite difference gradient magnitude
        means = []
        all_vals = []
        h = max(dx, dy)
        # interface band threshold similar to solver (bandwidth=2)
        thresh = 2.0 * h
        for G in [G_hist[i] for i in np.where(win_mask)[0]]:
            # Central differences interior, one-sided at edges
            dGdx = np.zeros_like(G)
            dGdy = np.zeros_like(G)
            dGdx[:,1:-1] = (G[:,2:] - G[:,0:-2])/(2*dx)
            dGdx[:,0] = (G[:,1] - G[:,0])/dx
            dGdx[:,-1] = (G[:,-1] - G[:,-2])/dx
            dGdy[1:-1,:] = (G[2:,:] - G[0:-2,:])/(2*dy)
            dGdy[0,:] = (G[1,:] - G[0,:])/dy
            dGdy[-1,:] = (G[-1,:] - G[-2,:])/dy
            grad = np.sqrt(dGdx*dGdx + dGdy*dGdy)
            band_mask = np.abs(G) < thresh
            band_vals = grad[band_mask]
            if band_vals.size == 0:
                continue
            means.append(band_vals.mean())
            all_vals.append(band_vals)
        if len(means) == 0:
            return {
                'samples': 0, 'grad_mean_mean': np.nan, 'grad_mean_std': np.nan,
                'grad_min': np.nan, 'grad_max': np.nan
            }
        concatenated = np.concatenate(all_vals)
        return {
            'samples': len(means),
            'grad_mean_mean': float(np.mean(means)),
            'grad_mean_std': float(np.std(means)),
            'grad_min': float(np.min(concatenated)),
            'grad_max': float(np.max(concatenated)),
        }

    dx, dy = solver.dx, solver.dy
    summary_rows = []
    for name in configs:
        G_hist, t_arr, lengths, u_ref, ftf = results[name]
        stats = gradient_stats(G_hist, t_arr, dx, dy)
        summary_rows.append({
            'config': name,
            'description': labels[name],
            'gain': float(ftf['gain'][0]),
            'phase_deg': float(ftf['phase'][0]),
            'mean_grad_mean': stats['grad_mean_mean'],
            'mean_grad_std': stats['grad_mean_std'],
            'grad_min': stats['grad_min'],
            'grad_max': stats['grad_max'],
            'window_samples': stats['samples'],
            'elapsed_sec': float(timings.get(name, float('nan'))),
        })

    csv_path = f'compare_reinit_configs_f{frequency_hz:.1f}Hz_summary.csv'
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        if verbose:
            print(f"Saved summary CSV: {csv_path}")
    except Exception as e:
        if verbose:
            print(f"Failed to write CSV summary: {e}")

    return results


if __name__ == '__main__':
    # CLI parsing
    frequency = 50.0
    steps_per_period = 20
    drop_cycles = 200
    measure_cycles = 10
    scheme = 'rk2'
    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2', 'rk3']:
            scheme = arg.lower()
        elif arg.startswith('f='):
            try:
                frequency = float(arg.split('=', 1)[1])
            except Exception:
                pass
        elif arg.startswith('steps_per_period='):
            try:
                steps_per_period = int(arg.split('=', 1)[1])
            except Exception:
                pass
        elif arg.startswith('drop_cycles='):
            try:
                drop_cycles = int(arg.split('=', 1)[1])
            except Exception:
                pass
        elif arg.startswith('measure_cycles='):
            try:
                measure_cycles = int(arg.split('=', 1)[1])
            except Exception:
                pass
    print(f"\nComparing reinit configurations: f={frequency} Hz, scheme={scheme}\n")
    test_compare_reinit_configs(
        frequency_hz=frequency,
        time_scheme=scheme,
        steps_per_period=steps_per_period,
        drop_cycles=drop_cycles,
        measure_cycles=measure_cycles,
        verbose=True,
    )
