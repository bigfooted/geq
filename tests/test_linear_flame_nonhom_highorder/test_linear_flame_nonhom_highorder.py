import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
High-order version of the three-region non-homogeneous vertical flow linear flame test.

Differences from `test_linear_flame_nonhom.py`:
 - Uses SSP RK3 time integration (time_scheme='rk3')
 - Uses WENO5 high-order spatial discretization (spatial_scheme='weno5')
 - Maintains fast-marching local reinitialization for gradient control.
 - Reports velocity errors and signed-distance quality metrics.

You can run:
    python test_linear_flame_nonhom_highorder.py t=2.0 reinit
    python test_linear_flame_nonhom_highorder.py t=2.0 no_reinit

Optional args:
    t=FINAL_TIME   set final time
    dt=TIMESTEP    override adaptive CFL dt
    nx=GRID_X ny=GRID_Y  set grid resolution
    no_reinit      disable reinitialization
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D
from contour_utils import compute_contour_length
from reporting_utils import (
    print_solver_overview, print_solve_start, print_performance,
    print_region_stats, print_domain_info, print_completion, sub_banner
)
from checkpoint_utils import save_checkpoint, CheckpointMeta


def create_time_dependent_velocity_middle_y(X, Y, x_threshold_1, x_threshold_2,
                                            U_Y=0.2, A=0.005, St=1.0, K=0.1,
                                            u_y_left=0.0, u_y_right=0.0):
    X_local = X.copy(); Y_local = Y.copy()
    left_mask = (X_local <= x_threshold_1)
    right_mask = (X_local >= x_threshold_2)
    def updater(solver, t):
        u_x = np.zeros_like(X_local)
        u_y_mid = U_Y * (1.0 + A * np.sin(St * (t - K * Y_local)))
        u_y = np.where(left_mask, u_y_left, np.where(right_mask, u_y_right, u_y_mid))
        return u_x, u_y
    return updater


def initial_horizontal_flame(X, Y, y_flame):
    Ly = float(np.max(Y))
    eps = 1e-12
    y0 = float(y_flame)
    y0_eff = max(y0, eps)
    Ly_minus_y0_eff = max(Ly - y0, eps)
    G_lower = 1.0 - (Y / y0_eff)
    G_upper = - (Y - y0) / Ly_minus_y0_eff
    return np.where(Y <= y0, G_lower, G_upper)


def extract_flame_position(G, X, Y, x_regions):
    ny, nx = G.shape
    y_positions_all = []; x_positions_all = []
    for i in range(nx):
        for j in range(ny - 1):
            G1 = G[j, i]; G2 = G[j + 1, i]
            if G1 * G2 <= 0 and G1 != G2:
                denom = (G2 - G1)
                alpha = -G1 / (denom if abs(denom) > 1e-15 else 1e-15)
                y_cross = Y[j, i] + alpha * (Y[j + 1, i] - Y[j, i])
                x_cross = X[j, i]
                y_positions_all.append(y_cross); x_positions_all.append(x_cross)
    if not y_positions_all:
        return {'overall': np.nan}
    y_positions_all = np.array(y_positions_all)
    x_positions_all = np.array(x_positions_all)
    result = {'overall': np.mean(y_positions_all)}
    for region_name, (x_min, x_max) in x_regions.items():
        mask = (x_positions_all >= x_min) & (x_positions_all < x_max)
        result[region_name] = np.mean(y_positions_all[mask]) if np.any(mask) else np.nan
    return result


def run_highorder_test(t_final=2.0, nx=101, ny=101, use_reinit=True, dt_override=None, verbose=True):
    Lx = 0.5; Ly = 1.0; S_L = 0.1
    x_threshold_1 = 0.1 * Lx; x_threshold_2 = 0.9 * Lx
    u_y_left = 0.0; u_y_mid = 0.2; u_y_right = 0.0
    U_left = u_y_left - S_L; U_mid = u_y_mid - S_L; U_right = u_y_right - S_L
    y0 = 0.3 * Ly

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)
    velocity_updater = create_time_dependent_velocity_middle_y(
        solver.X, solver.Y, x_threshold_1, x_threshold_2,
        U_Y=u_y_mid, A=0.005, St=1.0, K=0.1,
        u_y_left=u_y_left, u_y_right=u_y_right
    )
    solver.S_L = S_L
    G_initial = initial_horizontal_flame(solver.X, solver.Y, y0)

    # Pin inlet
    bottom_band_height = min(solver.dy * 0.5, 0.001)
    pin_mask = (solver.Y < bottom_band_height)
    pin_values = np.zeros_like(solver.X); pin_values[pin_mask] = +1.0
    try:
        solver.set_pinned_region(pin_mask, pin_values)
    except AttributeError:
        pass

    x_regions = {
        'left': (0.0, x_threshold_1),
        'middle': (x_threshold_1, x_threshold_2),
        'right': (x_threshold_2, Lx)
    }

    # Adaptive dt based on convective CFL target ~0.4 if not overridden
    max_u = max(abs(u_y_left), abs(u_y_mid), abs(u_y_right))
    cfl_target = 0.4
    if dt_override is None:
        dt = cfl_target * min(solver.dx, solver.dy) / max_u
    else:
        dt = dt_override
    reinit_interval = 50 if use_reinit else 0

    if verbose:
        flow_desc = [
            "High-order test (RK3 + WENO5)",
            f"Grid: {nx}x{ny}",
            f"Adaptive dt: {dt:.6f}",
            f"CFL(conv) ≈ {max_u * dt / min(solver.dx, solver.dy):.3f}",
            f"Reinit every {reinit_interval} steps" if use_reinit else "Reinit disabled"
        ]
        print_solver_overview(
            "High-Order Linear Flame (Three-Region Flow)",
            nx, ny, Lx, Ly, S_L, flow_desc, 'rk3',
            'LOCAL fast-marching' if use_reinit else 'None'
        )
        print_solve_start()

    start = time.time()
    G_history, t_history = solver.solve(
        G_initial, t_final, dt,
        save_interval=50,
        time_scheme='rk3', spatial_scheme='weno5',
        reinit_interval=reinit_interval, reinit_method='fast_marching', reinit_local=True,
        velocity_updater=velocity_updater, smooth_ic=True
    )
    elapsed = time.time() - start

    # Save checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), f'highorder_linear_flame_ckpt_{nx}x{ny}.npz')
    meta = CheckpointMeta(nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, time_scheme='rk3',
                          reinit_interval=reinit_interval, reinit_method='fast_marching',
                          reinit_local=True, save_interval=50, notes='high-order final state')
    save_checkpoint(ckpt_path, solver, G_history[-1], t_history[-1], meta)
    print(f"Saved checkpoint: {ckpt_path}")

    numerical_mid_positions = []
    for G in G_history:
        pos = extract_flame_position(G, solver.X, solver.Y, x_regions)
        numerical_mid_positions.append(pos.get('middle', np.nan))
    numerical_mid_positions = np.array(numerical_mid_positions)
    velocity_mid = (numerical_mid_positions[-1] - y0) / t_final
    vel_error_mid = abs(velocity_mid - U_mid)

    grad_mag = solver.compute_gradient_magnitude(G_history[-1])
    band = solver.find_interface_band(G_history[-1], bandwidth=2)
    grad_mean = float(np.mean(grad_mag[band])) if np.any(band) else np.nan

    if verbose:
        regions_stats = [
            {
                'name': 'middle', 'header': f"Middle Region ({x_threshold_1:.3f} < x < {x_threshold_2:.3f})",
                'u_y': u_y_mid, 'U_expected': U_mid,
                'y0': y0, 'y_final': float(numerical_mid_positions[-1]),
                'displacement': float(numerical_mid_positions[-1] - y0),
                'velocity': float(velocity_mid), 'velocity_error': float(vel_error_mid),
            }
        ]
        print_region_stats(regions_stats)
        print_domain_info(nx, ny, Lx, Ly, solver.dx, solver.dy,
                          cfl_conv=max_u * dt / min(solver.dx, solver.dy),
                          cfl_prop=S_L * dt / min(solver.dx, solver.dy))
        print(f"Mean |∇G| near interface: {grad_mean:.3f}")
        print_performance(elapsed, int(t_final / dt), len(t_history))
        print_completion()

    return {
        'nx': nx, 'ny': ny, 'dt': dt, 't_final': t_final,
        'velocity_mid': velocity_mid, 'vel_error_mid': vel_error_mid,
        'grad_mean': grad_mean, 'elapsed': elapsed
    }


def _parse_args(argv):
    t_final = 2.0; nx = 101; ny = 101; dt_override = None; use_reinit = True
    for arg in argv[1:]:
        if arg.startswith('t='): t_final = float(arg.split('=')[1])
        elif arg.startswith('dt='): dt_override = float(arg.split('=')[1])
        elif arg.startswith('nx='): nx = int(arg.split('=')[1])
        elif arg.startswith('ny='): ny = int(arg.split('=')[1])
        elif arg == 'no_reinit': use_reinit = False
        elif arg == 'reinit': use_reinit = True
    return t_final, nx, ny, dt_override, use_reinit


def main():
    t_final, nx, ny, dt_override, use_reinit = _parse_args(sys.argv)
    print(f"Running high-order nonhom flame: t_final={t_final}, nx={nx}, ny={ny}, dt_override={dt_override}, reinit={use_reinit}")
    run_highorder_test(t_final=t_final, nx=nx, ny=ny, use_reinit=use_reinit, dt_override=dt_override, verbose=True)


if __name__ == '__main__':
    main()
