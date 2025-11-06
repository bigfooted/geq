import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Restart test for 2D G-equation: linear flame with non-homogeneous vertical flow (three regions).
This test loads the checkpoint produced by tests/test_linear_flame_nonhom/test_linear_flame_nonhom.py
and resumes the simulation from the saved time to a later t_final.
"""

import numpy as np
import matplotlib.pyplot as plt
from checkpoint_utils import restart_solve, load_checkpoint
from g_equation_solver_improved import GEquationSolver2D
from reporting_utils import print_solve_start, print_performance, print_region_stats, print_domain_info
from tests.test_linear_flame_nonhom.test_linear_flame_nonhom import (
    create_time_dependent_velocity_middle_y,
    extract_flame_position,
)
from contour_utils import compute_contour_length


def test_restart_linear_flame_nonhom(t_extend=2.0, time_scheme='rk2', use_reinit=True, verbose=True,
                                     write_figures=True, show_figures=False):
    # Locate checkpoint from the original test
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'test_linear_flame_nonhom')
    ckpt_path = os.path.abspath(os.path.join(base_dir, 'linear_flame_nonhom_ckpt.npz'))
    assert os.path.exists(ckpt_path), (
        f"Checkpoint not found at {ckpt_path}. Run test_linear_flame_nonhom.py first to generate it."
    )

    # Load checkpoint to extract solver/grid info and time
    solver0, G0, t0, meta = load_checkpoint(ckpt_path)

    # Use same run configuration as meta when possible
    nx, ny, Lx, Ly = meta.nx, meta.ny, meta.Lx, meta.Ly
    dt = meta.dt if (meta.dt is not None) else 0.0005
    save_interval = meta.save_interval if (meta.save_interval is not None) else 50
    reinit_interval = meta.reinit_interval if (meta.reinit_interval is not None) else (50 if use_reinit else 0)
    reinit_method = meta.reinit_method if (meta.reinit_method is not None) else 'fast_marching'
    reinit_local = meta.reinit_local if (meta.reinit_local is not None) else True

    # Flow parameters used in the original test
    x_threshold_1 = 0.1*Lx
    x_threshold_2 = 0.9*Lx
    u_y_left = 0.0
    u_y_mid = 0.2
    u_y_right = 0.0
    S_L = float(np.mean(solver0.S_L)) if np.isscalar(solver0.S_L) else 0.1

    # Create the same time-dependent velocity updater
    velocity_updater = create_time_dependent_velocity_middle_y(
        solver0.X, solver0.Y,
        x_threshold_1=x_threshold_1, x_threshold_2=x_threshold_2,
        U_Y=u_y_mid, A=0.005, St=1.0, K=0.1,
        u_y_left=u_y_left, u_y_right=u_y_right
    )

    if verbose:
        print_solve_start()

    start = time.time()
    G_hist, t_hist = restart_solve(
        ckpt_path,
        t_final=t0 + t_extend,
        dt=dt,
        save_interval=save_interval,
        time_scheme=time_scheme,
        reinit_interval=reinit_interval,
        reinit_method=reinit_method,
        reinit_local=reinit_local,
        smooth_ic=False,  # don't re-smooth on restart
        velocity_updater=velocity_updater,
    )
    elapsed = time.time() - start

    # Basic sanity checks on time continuity
    assert len(t_hist) >= 2
    assert np.isclose(t_hist[0], t0)
    assert t_hist[-1] >= t0 + t_extend - 1e-12

    # Compute region-wise flame positions at end
    x_regions = {
        'left': (0.0, x_threshold_1),
        'middle': (x_threshold_1, x_threshold_2),
        'right': (x_threshold_2, Lx),
    }
    positions = extract_flame_position(G_hist[-1], solver0.X, solver0.Y, x_regions)

    # Expected net velocities
    U_left = u_y_left - S_L
    U_mid = u_y_mid - S_L
    U_right = u_y_right - S_L

    # Report quick stats
    regions_to_print = [
        {
            'name': 'left', 'header': f"Left Region (x ≤ {x_threshold_1})",
            'u_y': u_y_left, 'U_expected': U_left,
            'y0': None, 'y_final': float(positions.get('left', np.nan)),
            'displacement': np.nan,
            'velocity': np.nan,
            'velocity_error': np.nan,
        },
        {
            'name': 'middle', 'header': f"Middle Region ({x_threshold_1} < x < {x_threshold_2})",
            'u_y': u_y_mid, 'U_expected': U_mid,
            'y0': None, 'y_final': float(positions.get('middle', np.nan)),
            'displacement': np.nan,
            'velocity': np.nan,
            'velocity_error': np.nan,
        },
        {
            'name': 'right', 'header': f"Right Region (x ≥ {x_threshold_2})",
            'u_y': u_y_right, 'U_expected': U_right,
            'y0': None, 'y_final': float(positions.get('right', np.nan)),
            'displacement': np.nan,
            'velocity': np.nan,
            'velocity_error': np.nan,
        },
    ]
    if verbose:
        print_region_stats(regions_to_print)
        print_domain_info(nx, ny, Lx, Ly, solver0.dx, solver0.dy)
        n_steps = int((t_hist[-1]-t_hist[0]) / dt)
        print_performance(elapsed, n_steps, len(t_hist))

    # Write figures for the restart segment
    if write_figures:
        # Compute positions per region over the restart segment
        x_regions = {
            'left': (0.0, x_threshold_1),
            'middle': (x_threshold_1, x_threshold_2),
            'right': (x_threshold_2, Lx),
        }
        pos_left, pos_mid, pos_right, pos_overall = [], [], [], []
        for G in G_hist:
            pos = extract_flame_position(G, solver0.X, solver0.Y, x_regions)
            pos_overall.append(pos['overall'])
            pos_left.append(pos.get('left', np.nan))
            pos_mid.append(pos.get('middle', np.nan))
            pos_right.append(pos.get('right', np.nan))

    # Contour snapshots during restart
        n_snapshots = min(6, len(G_hist))
        indices = np.linspace(0, len(G_hist) - 1, n_snapshots, dtype=int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for k, idx in enumerate(indices):
            ax = axes[k]
            Gk = G_hist[idx]
            tk = t_hist[idx]
            c = ax.contourf(solver0.X, solver0.Y, Gk, levels=20, cmap='RdBu_r')
            plt.colorbar(c, ax=ax)
            ax.contour(solver0.X, solver0.Y, Gk, levels=[0], colors='black', linewidths=2)
            ax.axvline(x=x_threshold_1, color='magenta', linestyle=':', linewidth=2)
            ax.axvline(x=x_threshold_2, color='orange', linestyle=':', linewidth=2)
            ax.set_title(f't = {tk:.3f}s')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname_cont = f'restart_linear_flame_nonhom_contours_{time_scheme}_t{t_hist[0]:.2f}_to_{t_hist[-1]:.2f}.png'
        plt.savefig(fname_cont, dpi=300, bbox_inches='tight')

        # Velocity field snapshots (u_y) with G=0 overlay
        figv, axesv = plt.subplots(2, 3, figsize=(15, 10))
        axesv = axesv.flatten()
        left_mask = (solver0.X <= x_threshold_1)
        right_mask = (solver0.X >= x_threshold_2)
        for k, idx in enumerate(indices):
            ax = axesv[k]
            tk = t_hist[idx]
            # Rebuild the velocity field at time tk (same formula as original test)
            u_y_mid_field = u_y_mid * (1.0 + 0.005 * np.sin(1.0 * (tk - 0.1 * solver0.Y)))
            u_y_field = np.where(left_mask, u_y_left,
                                 np.where(right_mask, u_y_right, u_y_mid_field))
            cv = ax.contourf(solver0.X, solver0.Y, u_y_field, levels=20, cmap='viridis')
            plt.colorbar(cv, ax=ax, label='u_y')
            # Overlay the zero contour
            Gk = G_hist[idx]
            ax.contour(solver0.X, solver0.Y, Gk, levels=[0], colors='white', linewidths=2)
            ax.axvline(x=x_threshold_1, color='magenta', linestyle=':', linewidth=2)
            ax.axvline(x=x_threshold_2, color='orange', linestyle=':', linewidth=2)
            ax.set_title(f'u_y at t = {tk:.3f}s')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname_vel = f'restart_linear_flame_nonhom_velocity_{time_scheme}_t{t_hist[0]:.2f}_to_{t_hist[-1]:.2f}.png'
        plt.savefig(fname_vel, dpi=300, bbox_inches='tight')

        # Positions vs time during restart
        fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(t_hist, pos_left, 'b-o', linewidth=2, markersize=3, label=f'Left (x≤{x_threshold_1:.3f})')
        ax.plot(t_hist, pos_mid, 'g-^', linewidth=2, markersize=3, label=f'Middle ({x_threshold_1:.3f} < x < {x_threshold_2:.3f})')
        ax.plot(t_hist, pos_right, 'r-s', linewidth=2, markersize=3, label=f'Right (x≥{x_threshold_2:.3f})')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Flame position y (G=0)'); ax.grid(True, alpha=0.3)
        ax.legend(); ax.set_title(f'Restart positions ({time_scheme.upper()})')
        plt.tight_layout()
        fname_pos = f'restart_linear_flame_nonhom_position_{time_scheme}_t{t_hist[0]:.2f}_to_{t_hist[-1]:.2f}.png'
        plt.savefig(fname_pos, dpi=300, bbox_inches='tight')

        # Flame length vs time during restart
        flame_lengths = [compute_contour_length(Gk, solver0.X, solver0.Y, iso_value=0.0) for Gk in G_hist]
        fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
        ax3.plot(t_hist, flame_lengths, 'k-', linewidth=2)
        ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Flame Length |Γ|'); ax3.grid(True, alpha=0.3)
        ax3.set_title('Flame Length during Restart')
        plt.tight_layout()
        fname_len = f'restart_linear_flame_nonhom_length_{time_scheme}_t{t_hist[0]:.2f}_to_{t_hist[-1]:.2f}.png'
        plt.savefig(fname_len, dpi=300, bbox_inches='tight')

        if verbose:
            print(f"Saved figures: {fname_cont}, {fname_vel}, {fname_pos}, {fname_len}")
        if show_figures:
            plt.show()

    return G_hist, t_hist


if __name__ == "__main__":
    # Allow optional CLI arg 'show' to display figures
    import sys
    show = any(arg.lower() == 'show' for arg in sys.argv[1:])
    test_restart_linear_flame_nonhom(t_extend=2.0, time_scheme='rk2', use_reinit=True,
                                     verbose=True, write_figures=True, show_figures=show)
