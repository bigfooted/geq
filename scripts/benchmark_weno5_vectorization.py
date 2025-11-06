import os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g_equation_solver_improved import GEquationSolver2D


def velocity_updater_three_regions(X, Y, x_threshold_1, x_threshold_2,
                                   U_Y=0.2, A=0.005, St=1.0, K=0.1,
                                   u_y_left=0.0, u_y_right=0.0):
    X_local = X.copy(); Y_local = Y.copy()
    left = (X_local <= x_threshold_1)
    right = (X_local >= x_threshold_2)
    def updater(solver, t):
        u_x = np.zeros_like(X_local)
        u_y_mid = U_Y * (1.0 + A * np.sin(St * (t - K * Y_local)))
        u_y = np.where(left, u_y_left, np.where(right, u_y_right, u_y_mid))
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


def run_benchmark(nx=151, ny=151, t_final=2.0):
    Lx = 0.5; Ly = 1.0; S_L = 0.1
    x_threshold_1 = 0.1 * Lx; x_threshold_2 = 0.9 * Lx
    u_y_left = 0.0; u_y_mid = 0.2; u_y_right = 0.0
    y0 = 0.3 * Ly

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)
    velocity_updater = velocity_updater_three_regions(
        solver.X, solver.Y, x_threshold_1, x_threshold_2,
        U_Y=u_y_mid, A=0.005, St=1.0, K=0.1,
        u_y_left=u_y_left, u_y_right=u_y_right
    )
    G_initial = initial_horizontal_flame(solver.X, solver.Y, y0)

    # CFL-based dt
    max_u = max(abs(u_y_left), abs(u_y_mid), abs(u_y_right))
    cfl_target = 0.4
    dt = cfl_target * min(solver.dx, solver.dy) / max_u

    # Common solve args
    kwargs = dict(
        G_initial=G_initial, t_final=t_final, dt=dt,
        save_interval=50, time_scheme='rk3', spatial_scheme='weno5',
        reinit_interval=50, reinit_method='fast_marching', reinit_local=True,
        velocity_updater=velocity_updater, smooth_ic=True,
    )

    # Scalar mode
    solver.set_weno5_mode('scalar')
    t0 = time.time()
    solver.solve(**kwargs)
    scalar_time = time.time() - t0

    # Vector mode
    solver.set_weno5_mode('vector')
    t1 = time.time()
    solver.solve(**kwargs)
    vector_time = time.time() - t1

    speedup = scalar_time / max(vector_time, 1e-9)
    print(f"WENO5 benchmark nx=ny={nx}: scalar={scalar_time:.3f}s, vector={vector_time:.3f}s, speedup={speedup:.2f}x")


if __name__ == '__main__':
    run_benchmark()
