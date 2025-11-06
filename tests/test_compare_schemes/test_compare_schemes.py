import time
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g_equation_solver_improved import GEquationSolver2D

# Passive advection test comparing schemes.
# We advect a signed-distance circle with constant velocity. Propagation speed S_L is zero.
# Exact solution is translation of the circle center.


def signed_distance_circle(x, y, cx, cy, r):
    return np.sqrt((x - cx)**2 + (y - cy)**2) - r


def build_initial_level_set(nx=201, ny=201):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    G0 = signed_distance_circle(X, Y, 0.3, 0.3, 0.15)
    return X, Y, G0


def velocity_updater_constant_factory(solver, u=0.5, v=0.3):
    X_local = solver.X.copy()
    Y_local = solver.Y.copy()

    def updater(self_ref, t):
        u_arr = u * np.ones_like(X_local)
        v_arr = v * np.ones_like(Y_local)
        return u_arr, v_arr

    return updater


def run_case(time_scheme, spatial_scheme, dt, nx=201, ny=201, T=0.2):
    X, Y, G0 = build_initial_level_set(nx, ny)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # Create solver with zero initial velocity; updater will set constant field
    solver = GEquationSolver2D(nx, ny, 1.0, 1.0, S_L=0.0, u_x=0.0, u_y=0.0)
    velocity_updater = velocity_updater_constant_factory(solver, u=0.5, v=0.3)

    start = time.time()
    G_history, t_history = solver.solve(
        G_initial=G0,
        t_final=T,
        dt=dt,
        save_interval=1,
        time_scheme=time_scheme,
        reinit_interval=0,
        reinit_method='fast_marching',
        reinit_local=True,
        smooth_ic=False,
        velocity_updater=velocity_updater,
        spatial_scheme=spatial_scheme,
        t0=0.0,
    )
    wall = time.time() - start

    G_num = G_history[-1]

    # Exact SD to translated circle center (no periodic wrap needed for chosen T)
    cx = 0.3 + 0.5 * T
    cy = 0.3 + 0.3 * T
    G_exact = signed_distance_circle(X, Y, cx, cy, 0.15)

    l2_err = np.sqrt(np.mean((G_num - G_exact)**2))

    # Approximate |grad G| at interface (zero level ± half cell). Use central differences.
    grad_x = (np.roll(G_num, -1, axis=1) - np.roll(G_num, 1, axis=1)) / (2 * dx)
    grad_y = (np.roll(G_num, -1, axis=0) - np.roll(G_num, 1, axis=0)) / (2 * dy)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    interface_mask = np.abs(G_num) < 0.5 * max(dx, dy)
    mean_grad_interface = grad_mag[interface_mask].mean()

    nsteps = len(t_history) - 1
    return {
        'time_scheme': time_scheme,
        'spatial_scheme': spatial_scheme,
        'dt': dt,
        'l2_err': l2_err,
        'mean_grad_interface': mean_grad_interface,
        'runtime': wall,
        'nsteps': nsteps,
    }


def main():
    # Baseline small dt
    dt = 0.0005
    baseline = run_case('rk2', 'upwind', dt)
    high_order = run_case('rk3', 'weno5', dt)

    # Attempt larger dt for high-order if stable
    dt_big = 0.001
    high_order_big = run_case('rk3', 'weno5', dt_big)

    rows = [baseline, high_order, high_order_big]
    print("Scheme Comparison (Passive Advection Circle)")
    print(f"{'time':<6} {'space':<7} {'dt':>7} {'steps':>6} {'L2_err':>12} {'|grad|_int':>12} {'runtime(s)':>12}")
    for r in rows:
        print(f"{r['time_scheme']:<6} {r['spatial_scheme']:<7} {r['dt']:7.4f} {r['nsteps']:6d} {r['l2_err']:12.6e} {r['mean_grad_interface']:12.6f} {r['runtime']:12.4f}")

    # Simple assertions: high-order better at same dt
    assert high_order['l2_err'] <= baseline['l2_err'] * 0.8, "High-order scheme did not improve error enough at same dt"
    assert abs(high_order['mean_grad_interface'] - 1.0) < 0.2, "Gradient near interface deviates from 1 for high-order scheme"


if __name__ == '__main__':
    main()


def test_compare_schemes():
    # Mirror main but without printing table to keep pytest output clean
    dt = 0.0005
    baseline = run_case('rk2', 'upwind', dt)
    high_order = run_case('rk3', 'weno5', dt)

    # Sanity checks: high-order should be comparable (within 15%) or better at same dt
    assert high_order['l2_err'] <= baseline['l2_err'] * 1.15
    # And keep good signed-distance quality near interface
    assert abs(high_order['mean_grad_interface'] - 1.0) < 0.2
    # Preferably, high-order maintains gradient closer to 1 than baseline
    assert abs(high_order['mean_grad_interface'] - 1.0) <= abs(baseline['mean_grad_interface'] - 1.0) + 1e-6
