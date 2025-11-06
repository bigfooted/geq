import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from g_equation_solver_improved import GEquationSolver2D
from checkpoint_utils import save_checkpoint, load_checkpoint, restart_solve, CheckpointMeta


def initial_horizontal_flame(X, Y, y_flame):
    # Signed distance: positive below (unburnt), negative above (burnt)
    return -(Y - y_flame)


def run_restart_smoke():
    # Small, fast case
    nx = ny = 81
    Lx = Ly = 1.0
    S_L = 0.1

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.2)
    y0 = 0.3
    G0 = initial_horizontal_flame(solver.X, solver.Y, y0)

    dt = 1e-3
    scheme = 'rk2'
    reinit_interval = 40

    # Continuous run to t_final
    t_final = 10
    full_G_hist, full_t_hist = solver.solve(
        G0, t_final, dt,
        save_interval=50,
        time_scheme=scheme,
        reinit_interval=reinit_interval,
        reinit_method='fast_marching',
        reinit_local=True,
    )
    G_full = full_G_hist[-1]

    # Two-part run: stop at mid, save, and restart
    t_mid = 10 / 2
    solver2 = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.2)
    G_hist1, t_hist1 = solver2.solve(
        G0, t_mid, dt,
        save_interval=50,
        time_scheme=scheme,
        reinit_interval=reinit_interval,
        reinit_method='fast_marching',
        reinit_local=True,
    )
    G_mid = G_hist1[-1]

    ckpt_path = os.path.join(os.path.dirname(__file__), 'restart_checkpoint.npz')
    meta = CheckpointMeta(nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, time_scheme=scheme,
                          reinit_interval=reinit_interval, reinit_method='fast_marching',
                          reinit_local=True, save_interval=50, notes='smoke test')
    save_checkpoint(ckpt_path, solver2, G_mid, t_hist1[-1], meta)

    # Resume to t_final
    G_hist2, t_hist2 = restart_solve(
        ckpt_path,
        t_final=t_final,
        dt=dt,
        save_interval=50,
        time_scheme=scheme,
        reinit_interval=reinit_interval,
        reinit_method='fast_marching',
        reinit_local=True,
        smooth_ic=False,
        velocity_updater=None,
    )
    G_restart = G_hist2[-1]

    # Compare final states
    diff = np.linalg.norm(G_restart - G_full) / (np.linalg.norm(G_full) + 1e-12)
    print(f"Relative L2 difference (restart vs continuous) = {diff:.3e}")

    # Accept small numerical differences due to floating point and reinit alignment
    assert diff < 1e-10, f"Restart mismatch too large: {diff}"

    # Cleanup
    try:
        os.remove(ckpt_path)
    except OSError:
        pass


if __name__ == "__main__":
    run_restart_smoke()
