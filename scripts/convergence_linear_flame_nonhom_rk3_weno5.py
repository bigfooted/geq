import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g_equation_solver_improved import GEquationSolver2D


def create_velocity_updater(X, Y, x_threshold_1, x_threshold_2,
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


def run_case(nx, ny, t_final=2.0, use_reinit=True, time_scheme='rk3', spatial_scheme='weno5'):
    Lx = 0.5; Ly = 1.0; S_L = 0.1
    x_threshold_1 = 0.1 * Lx; x_threshold_2 = 0.9 * Lx
    u_y_left = 0.0; u_y_mid = 0.2; u_y_right = 0.0
    U_left = u_y_left - S_L; U_mid = u_y_mid - S_L; U_right = u_y_right - S_L
    y0 = 0.3 * Ly

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)
    velocity_updater = create_velocity_updater(
        solver.X, solver.Y, x_threshold_1, x_threshold_2,
        U_Y=u_y_mid, A=0.005, St=1.0, K=0.1,
        u_y_left=u_y_left, u_y_right=u_y_right
    )
    solver.S_L = S_L

    G_initial = initial_horizontal_flame(solver.X, solver.Y, y0)

    bottom_band_height = min(solver.dy * 0.5, 0.001)
    pin_mask = (solver.Y < bottom_band_height)
    pin_values = np.zeros_like(solver.X); pin_values[pin_mask] = +1.0
    try:
        solver.set_pinned_region(pin_mask, pin_values)
    except AttributeError:
        pass

    max_u = max(abs(u_y_left), abs(u_y_mid), abs(u_y_right))
    cfl_target = 0.4
    dt = cfl_target * min(solver.dx, solver.dy) / max_u
    reinit_interval = 50 if use_reinit else 0

    start = time.time()
    G_history, t_history = solver.solve(
        G_initial, t_final, dt,
        save_interval=50,
        time_scheme=time_scheme, spatial_scheme=spatial_scheme,
        reinit_interval=reinit_interval, reinit_method='fast_marching', reinit_local=True,
        velocity_updater=velocity_updater, smooth_ic=True
    )
    elapsed = time.time() - start

    x_regions = {
        'left': (0.0, x_threshold_1),
        'middle': (x_threshold_1, x_threshold_2),
        'right': (x_threshold_2, Lx)
    }

    pos = extract_flame_position(G_history[-1], solver.X, solver.Y, x_regions)
    vel_mid = (pos['middle'] - y0) / t_final
    vel_err_mid = abs(vel_mid - U_mid)

    grad_mag = solver.compute_gradient_magnitude(G_history[-1])
    band = solver.find_interface_band(G_history[-1], bandwidth=2)
    grad_mean = float(np.mean(grad_mag[band])) if np.any(band) else np.nan

    return {
        'nx': nx, 'ny': ny, 'dx': solver.dx, 'dy': solver.dy,
        'dt': dt, 'nsteps': len(t_history)-1, 'elapsed': elapsed,
        'vel_mid': vel_mid, 'vel_err_mid': vel_err_mid, 'grad_mean': grad_mean,
        'scheme_time': time_scheme, 'scheme_space': spatial_scheme,
    }


def main():
    resolutions = [51, 101, 151, 201]
    results_hi = []
    results_base = []
    for n in resolutions:
        print(f"Running convergence case (high-order) nx=ny={n}...")
        res_hi = run_case(n, n, t_final=2.0, use_reinit=True, time_scheme='rk3', spatial_scheme='weno5')
        results_hi.append(res_hi)
        print(f"  -> HO: dx={res_hi['dx']:.5f}, dt={res_hi['dt']:.6f}, steps={res_hi['nsteps']}, vel_err_mid={res_hi['vel_err_mid']:.3e}, |grad|_int={res_hi['grad_mean']:.3f}")

        print(f"Running convergence case (baseline) nx=ny={n}...")
        res_base = run_case(n, n, t_final=2.0, use_reinit=True, time_scheme='rk2', spatial_scheme='upwind')
        results_base.append(res_base)
        print(f"  -> BL: dx={res_base['dx']:.5f}, dt={res_base['dt']:.6f}, steps={res_base['nsteps']}, vel_err_mid={res_base['vel_err_mid']:.3e}, |grad|_int={res_base['grad_mean']:.3f}")

    # Save CSV
    outdir = os.path.join(ROOT, 'results')
    os.makedirs(outdir, exist_ok=True)
    csv_path_hi = os.path.join(outdir, 'convergence_rk3_weno5_nonhom.csv')
    with open(csv_path_hi, 'w') as f:
        f.write('nx,dx,dt,nsteps,vel_err_mid,grad_mean,elapsed\n')
        for r in results_hi:
            f.write(f"{r['nx']},{r['dx']:.8f},{r['dt']:.8f},{r['nsteps']},{r['vel_err_mid']:.6e},{r['grad_mean']:.6f},{r['elapsed']:.4f}\n")
    print(f"Saved: {csv_path_hi}")

    csv_path_bl = os.path.join(outdir, 'convergence_rk2_upwind_nonhom.csv')
    with open(csv_path_bl, 'w') as f:
        f.write('nx,dx,dt,nsteps,vel_err_mid,grad_mean,elapsed\n')
        for r in results_base:
            f.write(f"{r['nx']},{r['dx']:.8f},{r['dt']:.8f},{r['nsteps']},{r['vel_err_mid']:.6e},{r['grad_mean']:.6f},{r['elapsed']:.4f}\n")
    print(f"Saved: {csv_path_bl}")

    # Plots
    dxs_hi = np.array([r['dx'] for r in results_hi])
    vel_errs_hi = np.array([r['vel_err_mid'] for r in results_hi])
    grad_dev_hi = np.array([abs(r['grad_mean'] - 1.0) for r in results_hi])
    dxs_bl = np.array([r['dx'] for r in results_base])
    vel_errs_bl = np.array([r['vel_err_mid'] for r in results_base])
    grad_dev_bl = np.array([abs(r['grad_mean'] - 1.0) for r in results_base])

    plt.figure(figsize=(6,4))
    plt.loglog(dxs_hi, vel_errs_hi, 'o-', label='HO: |U_mid - U_exp| (RK3+WENO5)')
    plt.loglog(dxs_bl, vel_errs_bl, 's-', label='BL: |U_mid - U_exp| (RK2+Upwind)')
    # reference slope ~ O(dx^2) line (visual guide) based on HO finest point
    c = vel_errs_hi[-1] / (dxs_hi[-1]**2 + 1e-16)
    plt.loglog(dxs_hi, c*dxs_hi**2, '--', label='O(dx^2) ref')
    plt.gca().invert_xaxis()
    plt.xlabel('dx'); plt.ylabel('Velocity error (middle region)')
    plt.grid(True, which='both', alpha=0.3); plt.legend()
    fig1 = os.path.join(outdir, 'convergence_velocity_error_compare.png')
    plt.tight_layout(); plt.savefig(fig1, dpi=200)
    print(f"Saved: {fig1}")

    plt.figure(figsize=(6,4))
    plt.loglog(dxs_hi, grad_dev_hi, 'o-', color='tab:orange', label='HO: |mean(|∇G|_int) - 1|')
    plt.loglog(dxs_bl, grad_dev_bl, 's-', color='tab:green', label='BL: |mean(|∇G|_int) - 1|')
    plt.gca().invert_xaxis()
    plt.xlabel('dx'); plt.ylabel('Gradient deviation at interface')
    plt.grid(True, which='both', alpha=0.3); plt.legend()
    fig2 = os.path.join(outdir, 'convergence_grad_deviation_compare.png')
    plt.tight_layout(); plt.savefig(fig2, dpi=200)
    print(f"Saved: {fig2}")


if __name__ == '__main__':
    main()
