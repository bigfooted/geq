import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Test case for 2D G-equation solver: horizontal linear flame with non-homogeneous vertical flow.
Initial condition: horizontal line separating burnt (G=-1, above) and unburnt (G=+1, below) regions.
Flow: non-homogeneous vertical velocity with three regions:
      - u_y = 0.0  for x <= 0.1  (left region, no flow)
      - u_y = 0.2  for 0.1 < x < 0.9 (middle region, upward flow)
      - u_y = 0.0  for x >= 0.9  (right region, no flow)
      - u_x = 0.0 everywhere (no horizontal flow)
Burning velocity: S_L=0.1

Expected net flame velocities U = u_y - S_L:
- Left region (x <= 0.1): U_left  = -0.1 (downward, flame recedes)
- Middle region (0.1 < x < 0.9): U_mid =  0.1 (upward, flame advances)
- Right region (x >= 0.9): U_right = -0.1 (downward, flame recedes)
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D
from plotting_utils import plot_surface_3d
from contour_utils import get_contour_statistics, compute_contour_length
from reporting_utils import (
    print_solver_overview, print_initial_flame_verification, print_G_field_stats,
    print_velocity_field_verification, print_solve_start, print_performance,
    print_region_stats, print_domain_info, print_completion, sub_banner
)
import time
from checkpoint_utils import save_checkpoint, CheckpointMeta

def create_velocity_field_three_regions(X, Y, u_y_left=0.0, u_y_middle=0.2, u_y_right=0.0,
                                        x_threshold_1=0.1, x_threshold_2=0.9):
    """
    Create non-homogeneous velocity field with three regions.

    Parameters:
    -----------
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    u_y_left : float
        Vertical velocity for x <= x_threshold_1
    u_y_middle : float
        Vertical velocity for x_threshold_1 < x < x_threshold_2
    u_y_right : float
        Vertical velocity for x >= x_threshold_2
    x_threshold_1 : float
        First X-coordinate threshold (default: 0.1)
    x_threshold_2 : float
        Second X-coordinate threshold (default: 0.9)

    Returns:
    --------
    u_x : ndarray
        Horizontal velocity field (all zeros)
    u_y : ndarray
        Vertical velocity field with three regions
    """
    u_x = np.zeros_like(X)
    u_y = np.where(X <= x_threshold_1, u_y_left,
                   np.where(X >= x_threshold_2, u_y_right, u_y_middle))
    return u_x, u_y

# Two-region creator kept for reference but unused in this test.
def create_velocity_field(X, Y, u_y_left=0.0, u_y_right=0.2, x_threshold=0.1):
    u_x = np.zeros_like(X)
    u_y = np.where(X < x_threshold, u_y_left, u_y_right)
    return u_x, u_y

def create_time_dependent_velocity_middle_y(X, Y,
                                            x_threshold_1=0.1, x_threshold_2=0.9,
                                            U_Y=0.2, A=0.01, St=1.0, K=0.1,
                                            u_y_left=0.0, u_y_right=0.0):
    """
    Create a time-dependent updater for u_y in the middle region that depends only on y and t.

    u_y(t, y) = U_Y * (1 + A * sin(St * (t - K * y))) in the middle region (x_threshold_1 < x < x_threshold_2),
    and piecewise-constant u_y_left/u_y_right in the side regions.

    Returns a callable: updater(solver, t) -> (u_x, u_y) arrays
    """
    X_local = X.copy()
    Y_local = Y.copy()

    left_mask = (X_local <= x_threshold_1)
    right_mask = (X_local >= x_threshold_2)
    mid_mask = ~(left_mask | right_mask)

    def updater(solver, t):
        u_x = np.zeros_like(X_local)
        # Time-dependent middle-band velocity depends only on y and t
        u_y_mid = U_Y * (1.0 + A * np.sin(St * (t - K * Y_local)))
        # Assemble full field
        u_y = np.where(left_mask, u_y_left, np.where(right_mask, u_y_right, u_y_mid))
        return u_x, u_y

    return updater

def initial_horizontal_flame(X, Y, y_flame):
    """
    Create initial level set function for a horizontal flame line with linear profile in y.

    Requirements:
    - G varies linearly in y from G=+1 at y=0 to G=0 at y=y_flame, then to G=-1 at y=Ly.
    - G > 0: unburnt region (below the line)
    - G < 0: burnt region (above the line)
    - G = 0: flame surface (at y = y_flame)
    """
    # Determine domain height Ly from meshgrid
    Ly = float(np.max(Y))

    # Guard against degenerate cases (y_flame at boundaries)
    eps = 1e-12
    y0 = float(y_flame)
    y0_eff = max(y0, eps)
    Ly_minus_y0_eff = max(Ly - y0, eps)

    # Piecewise linear profile:
    #   For 0 <= y <= y0:    G = 1 - y / y0
    #   For y0 < y <= Ly:    G = - (y - y0) / (Ly - y0)
    G_lower = 1.0 - (Y / y0_eff)
    G_upper = - (Y - y0) / Ly_minus_y0_eff

    G = np.where(Y <= y0, G_lower, G_upper)
    return G

def extract_flame_position(G, X, Y, x_regions=None):
    """
    Extract the average y-position of the flame (G=0 contour) in different x-regions.
    """
    ny, nx = G.shape

    # Find zero crossings in vertical direction
    y_positions_all = []
    x_positions_all = []

    for i in range(nx):
        for j in range(ny - 1):
            G1 = G[j, i]
            G2 = G[j + 1, i]

            # Check for zero crossing
            if G1 * G2 <= 0 and G1 != G2:
                denom = (G2 - G1)
                alpha = -G1 / (denom if abs(denom) > 1e-15 else 1e-15)
                y_cross = Y[j, i] + alpha * (Y[j + 1, i] - Y[j, i])
                x_cross = X[j, i]
                y_positions_all.append(y_cross)
                x_positions_all.append(x_cross)

    if len(y_positions_all) == 0:
        return {'overall': np.nan}

    y_positions_all = np.array(y_positions_all)
    x_positions_all = np.array(x_positions_all)

    result = {'overall': np.mean(y_positions_all)}

    # Compute positions for specific regions
    if x_regions is not None:
        for region_name, (x_min, x_max) in x_regions.items():
            mask = (x_positions_all >= x_min) & (x_positions_all < x_max)
            result[region_name] = np.mean(y_positions_all[mask]) if np.any(mask) else np.nan

    return result

def create_plots(solver, G_history, t_history, y0, time_scheme, t_final, use_reinit,
                 numerical_positions_left, numerical_positions_mid, numerical_positions_right,
                 x_threshold_1, x_threshold_2, S_L, u_y_left, u_y_mid, u_y_right,
                 flame_lengths=None,
                 U_Y=None, A=None, St=None, K=None, y_ref_for_velocity=0.0):
    """
    Create all visualization plots for the non-homogeneous linear flame test (three regions).
    """
    print("\nCreating visualizations...")

    suffix = f"_reinit" if use_reinit else "_no_reinit"

    # Plot 1: Contour plots at selected times
    n_snapshots = min(6, len(G_history))
    indices = np.linspace(0, len(G_history) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        G = G_history[ax_idx]
        t = t_history[ax_idx]

        # Plot level set field
        c = ax.contourf(solver.X, solver.Y, G, levels=20, cmap='RdBu_r')
        plt.colorbar(c, ax=ax)

        # Plot flame surface (G=0)
        ax.contour(solver.X, solver.Y, G, levels=[0], colors='black', linewidths=2)

        # Mark the velocity transition lines
        ax.axvline(x=x_threshold_1, color='magenta', linestyle=':', linewidth=2, label=f'x={x_threshold_1}')
        ax.axvline(x=x_threshold_2, color='orange', linestyle=':', linewidth=2, label=f'x={x_threshold_2}')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {t:.3f}s')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'linear_flame_nonhom_contours_{time_scheme}_t{t_final}{suffix}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: linear_flame_nonhom_contours_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 1b: Velocity field snapshots (u_y) with G=0 contour overlay
    if U_Y is not None and A is not None and St is not None and K is not None:
        fig_v, axes_v = plt.subplots(2, 3, figsize=(15, 10))
        axes_v = axes_v.flatten()

        left_mask = (solver.X <= x_threshold_1)
        right_mask = (solver.X >= x_threshold_2)
        mid_mask = ~(left_mask | right_mask)

        for idx, ax_idx in enumerate(indices):
            ax = axes_v[idx]
            t = t_history[ax_idx]
            # Build u_y field at time t
            u_y_mid_field = U_Y * (1.0 + A * np.sin(St * (t - K * solver.Y)))
            u_y_field = np.where(left_mask, u_y_left,
                                 np.where(right_mask, u_y_right, u_y_mid_field))

            cv = ax.contourf(solver.X, solver.Y, u_y_field, levels=20, cmap='viridis')
            plt.colorbar(cv, ax=ax, label='u_y')
            # Overlay G=0 contour for same snapshot
            G = G_history[ax_idx]
            ax.contour(solver.X, solver.Y, G, levels=[0], colors='white', linewidths=2)

            # Mark region lines
            ax.axvline(x=x_threshold_1, color='magenta', linestyle=':', linewidth=2)
            ax.axvline(x=x_threshold_2, color='orange', linestyle=':', linewidth=2)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'u_y at t = {t:.3f}s')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'linear_flame_nonhom_velocity_snapshots_{time_scheme}_t{t_final}{suffix}.png',
                    dpi=300, bbox_inches='tight')
        print(f"Saved: linear_flame_nonhom_velocity_snapshots_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 2: Flame position vs time for all regions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    U_left = u_y_left - S_L
    U_mid = u_y_mid - S_L
    U_right = u_y_right - S_L

    # 2a: Position evolution in three regions
    ax1.plot(t_history, numerical_positions_left, 'b-', linewidth=2, marker='o',
             markersize=4, markevery=max(1, len(t_history)//20),
             label=f'Left (x ≤ {x_threshold_1}): U = {U_left:+.3f}')
    ax1.plot(t_history, numerical_positions_mid, 'g-', linewidth=2, marker='^',
             markersize=4, markevery=max(1, len(t_history)//20),
             label=f'Middle ({x_threshold_1} < x < {x_threshold_2}): U = {U_mid:+.3f}')
    ax1.plot(t_history, numerical_positions_right, 'r-', linewidth=2, marker='s',
             markersize=4, markevery=max(1, len(t_history)//20),
             label=f'Right (x ≥ {x_threshold_2}): U = {U_right:+.3f}')
    ax1.axhline(y=y0, color='gray', linestyle='--', alpha=0.5, label='Initial position')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Flame Position (y)', fontsize=12)
    ax1.set_title(f'Flame Position vs Time in Three Regions\n{time_scheme.upper()}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2b: Relative displacement from initial position
    displacement_left = np.array(numerical_positions_left) - y0
    displacement_mid = np.array(numerical_positions_mid) - y0
    displacement_right = np.array(numerical_positions_right) - y0

    ax2.plot(t_history, displacement_left, 'b-', linewidth=2, label=f'Left (x ≤ {x_threshold_1})')
    ax2.plot(t_history, displacement_mid, 'g-', linewidth=2, label=f'Middle ({x_threshold_1} < x < {x_threshold_2})')
    ax2.plot(t_history, displacement_right, 'r-', linewidth=2, label=f'Right (x ≥ {x_threshold_2})')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Displacement from Initial Position', fontsize=12)
    ax2.set_title('Flame Displacement', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'linear_flame_nonhom_position_{time_scheme}_t{t_final}{suffix}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: linear_flame_nonhom_position_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 3: Flame length vs time (total length of G=0 isocontour)
    if flame_lengths is None:
        # Compute if not precomputed
        flame_lengths = [compute_contour_length(G, solver.X, solver.Y, iso_value=0.0) for G in G_history]

    fig, ax_len = plt.subplots(1, 1, figsize=(7, 5))
    ax_len.plot(t_history, flame_lengths, 'k-', linewidth=2)
    ax_len.set_xlabel('Time (s)', fontsize=12)
    ax_len.set_ylabel('Flame Length (|Γ|)', fontsize=12)
    ax_len.set_title('Flame Length vs Time', fontsize=13, fontweight='bold')
    ax_len.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'linear_flame_nonhom_length_{time_scheme}_t{t_final}{suffix}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: linear_flame_nonhom_length_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 3b: Flame length (left axis) + velocity in time (right axis)
    if U_Y is not None and A is not None and St is not None and K is not None:
        t_arr = np.array(t_history)
        # Velocity evaluated at reference y (default inlet y=0)
        vel_series = U_Y * (1.0 + A * np.sin(St * (t_arr - K * y_ref_for_velocity)))

        fig, ax_left = plt.subplots(1, 1, figsize=(8, 5))
        ax_right = ax_left.twinx()
        ln1 = ax_left.plot(t_arr, flame_lengths, 'k-', linewidth=2, label='Flame length')
        ln2 = ax_right.plot(t_arr, vel_series, 'b--', linewidth=2, label='Velocity (ref)')
        ax_left.set_xlabel('Time (s)', fontsize=12)
        ax_left.set_ylabel('Flame Length (|Γ|)', color='k', fontsize=12)
        ax_right.set_ylabel('Velocity u_y (m/s)', color='b', fontsize=12)
        ax_left.grid(True, alpha=0.3)
        # Build a combined legend
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='best')
        plt.tight_layout()
        plt.savefig(f'linear_flame_nonhom_length_velocity_{time_scheme}_t{t_final}{suffix}.png',
                    dpi=300, bbox_inches='tight')
        print(f"Saved: linear_flame_nonhom_length_velocity_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 4: 3D surface plot at final time
    G_final = G_history[-1]
    plot_surface_3d(solver, G_final, t_final, with_flow=True,
                   filename=f'linear_flame_nonhom_surface_{time_scheme}_t{t_final}{suffix}.png')

    plt.show()

    print_completion()

def test_linear_flame_nonhom(t_final=2.0, time_scheme='rk2', use_reinit=True, verbose=True):
    """
    Test the G-equation solver with a horizontal linear flame in three-region vertical flow.
    """
    # Parameters
    nx = 101  # Number of grid points in x
    ny = 101  # Number of grid points in y
    Lx = 0.5  # Small domain length in x (kept as requested)
    Ly = 1.0  # Small domain length in y (kept as requested)
    S_L = 0.1  # Laminar flame speed

    # Flow velocity parameters (three regions)
    x_threshold_1 = 0.1*Lx
    x_threshold_2 = 0.9*Lx
    u_y_left = 0.0
    u_y_mid = 0.2
    u_y_right = 0.0

    # Expected net flame velocities in each region
    U_left = u_y_left - S_L    # -0.1 (downward)
    U_mid = u_y_mid - S_L      # 0.1 (upward)
    U_right = u_y_right - S_L  # -0.1 (downward)

    # Initial flame position
    # Start away from the boundaries to avoid compression near y=0/1
    y0 = 0.3*Ly

    # Time parameters
    # Smaller dt to maintain low CFL on a small grid
    dt = 0.0005
    save_interval = 50  # Save solution every 50 steps

    # Reinitialization parameters
    # Use robust fast-marching reinitialization at a moderate frequency
    reinit_interval = 50 if use_reinit else 0
    reinit_method = 'fast_marching'
    reinit_local = True

    if verbose:
        flow_desc = [
            "Flow velocity (three regions):",
            f"  Region 1: x ≤ {x_threshold_1}:         u_y = {u_y_left}",
            f"  Region 2: {x_threshold_1} < x < {x_threshold_2}: u_y = {u_y_mid}",
            f"  Region 3: x ≥ {x_threshold_2}:         u_y = {u_y_right}",
            f"  u_x = 0.0 everywhere",
            "Expected net flame velocities U = u_y - S_L:",
            f"  U_left  = {U_left:+.3f} (downward)",
            f"  U_mid   = {U_mid:+.3f} (upward)",
            f"  U_right = {U_right:+.3f} (downward)",
            f"Initial flame position y0 = {y0}",
            f"Time step dt = {dt}",
            f"Final time t_final = {t_final}",
            f"Save interval: every {save_interval} steps",
        ]
        reinit_desc = (
            f"Reinitialization: LOCAL every {reinit_interval} steps ({reinit_method})"
            if use_reinit else "Reinitialization: Disabled"
        )
        print_solver_overview(
            "2D G-Equation Solver: Linear Flame Test (Three-Region Vertical Flow)",
            nx, ny, Lx, Ly, S_L, flow_desc, time_scheme, reinit_desc
        )

    # Create solver with scalar velocities first (will be overridden with arrays)
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)

    # Create time-dependent velocity updater for the middle region (depends on y and t)
    velocity_updater = create_time_dependent_velocity_middle_y(
        solver.X, solver.Y,
        x_threshold_1=x_threshold_1, x_threshold_2=x_threshold_2,
        U_Y=u_y_mid, A=0.005, St=1.0, K=0.1,
        u_y_left=u_y_left, u_y_right=u_y_right
    )

    # Keep S_L uniform to avoid sharp speed discontinuities that steepen G
    solver.S_L = S_L

    # Create initial condition
    G_initial = initial_horizontal_flame(solver.X, solver.Y, y0)

    # Enforce Dirichlet inlet boundary: keep G = +1 over the entire inlet (bottom) band
    # Inlet band thickness: y < min(dy/2, 0.001) across all x
    bottom_band_height = min(solver.dy * 0.5, 0.001)
    pin_mask = (solver.Y < bottom_band_height)
    pin_values = np.zeros_like(solver.X)
    pin_values[pin_mask] = +1.0
    try:
        solver.set_pinned_region(pin_mask, pin_values)
    except AttributeError:
        # Older solver without pinning hook: skip silently
        pass

    # Define regions for analysis
    x_regions = {
        'left': (0.0, x_threshold_1),
        'middle': (x_threshold_1, x_threshold_2),
        'right': (x_threshold_2, Lx)
    }

    # Verify initial flame position
    # initial_positions = extract_flame_position(G_initial, solver.X, solver.Y, x_regions)

    # if verbose:
    #     # Choose sample x positions for verifying u_y in each region
    #     x_left_sample = 0.5 * x_threshold_1
    #     x_mid_sample = 0.5 * (x_threshold_1 + x_threshold_2)
    #     x_right_sample = 0.5 * (x_threshold_2 + Lx)
    #     ix_left = int((x_left_sample / Lx) * (nx - 1))
    #     ix_mid = int((x_mid_sample / Lx) * (nx - 1))
    #     ix_right = int((x_right_sample / Lx) * (nx - 1))

    #     print_initial_flame_verification(y0, initial_positions, regions=['left', 'middle', 'right'])
    #     print_G_field_stats(
    #         G_initial,
    #         bottom_val=G_initial[0, nx//2],
    #         top_val=G_initial[-1, nx//2],
    #         bottom_label='y=0', top_label='y=Ly'
    #     )
    #     print_velocity_field_verification([
    #         {'label': f'u_y at x≈{x_left_sample:.3f} (left)', 'value': float(solver.u_y[ny//2, ix_left]), 'expected': u_y_left},
    #         {'label': f'u_y at x≈{x_mid_sample:.3f} (middle)', 'value': float(solver.u_y[ny//2, ix_mid]), 'expected': u_y_mid},
    #         {'label': f'u_y at x≈{x_right_sample:.3f} (right)', 'value': float(solver.u_y[ny//2, ix_right]), 'expected': u_y_right},
    #     ])

    # Solve
    if verbose:
        print_solve_start()

    start_time = time.time()
    G_history, t_history = solver.solve(
        G_initial, t_final, dt,
        save_interval=save_interval,
        time_scheme=time_scheme,
        reinit_interval=reinit_interval,
        reinit_method=reinit_method,
        reinit_local=reinit_local,
        velocity_updater=velocity_updater,
        smooth_ic=True
    )
    elapsed_time = time.time() - start_time

    # Save checkpoint of the final state for restart tests
    ckpt_path = os.path.join(os.path.dirname(__file__), 'linear_flame_nonhom_ckpt.npz')
    meta = CheckpointMeta(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, time_scheme=time_scheme,
        reinit_interval=reinit_interval, reinit_method=reinit_method,
        reinit_local=reinit_local, save_interval=save_interval,
        notes='final state checkpoint for restart'
    )
    save_checkpoint(ckpt_path, solver, G_history[-1], t_history[-1], meta)
    print(f"Saved checkpoint to: {ckpt_path} (t={t_history[-1]:.6f})")

    n_total_steps = int(t_final / dt)
    if verbose:
        print_performance(elapsed_time, n_total_steps, len(t_history))

    # Extract numerical flame position at each time step
    if verbose:
        sub_banner("Extracting flame positions in different regions...")

    numerical_positions_left = []
    numerical_positions_mid = []
    numerical_positions_right = []
    numerical_positions_overall = []

    extraction_start = time.time()

    for G in G_history:
        positions = extract_flame_position(G, solver.X, solver.Y, x_regions)
        numerical_positions_overall.append(positions['overall'])
        numerical_positions_left.append(positions.get('left', np.nan))
        numerical_positions_mid.append(positions.get('middle', np.nan))
        numerical_positions_right.append(positions.get('right', np.nan))

    extraction_time = time.time() - extraction_start

    if verbose:
        print_performance(elapsed_time, n_total_steps, len(t_history), extraction_time=extraction_time)

    # Convert to arrays
    numerical_positions_left = np.array(numerical_positions_left)
    numerical_positions_mid = np.array(numerical_positions_mid)
    numerical_positions_right = np.array(numerical_positions_right)
    t_array = np.array(t_history)

    # Compute velocities from numerical data
    if t_final > 0:
        velocity_left = (numerical_positions_left[-1] - y0) / t_final
        velocity_mid = (numerical_positions_mid[-1] - y0) / t_final
        velocity_right = (numerical_positions_right[-1] - y0) / t_final
    else:
        velocity_left = velocity_mid = velocity_right = 0.0

    # Print statistics
    # print("\n" + "="*80)
    # print("SIMULATION STATISTICS")
    # print("="*80)

    print("\n" + "="*80)
    print("SIMULATION STATISTICS")
    print("="*80)

    regions_to_print = [
        {
            'name': 'left', 'header': f"Left Region (x ≤ {x_threshold_1})",
            'u_y': u_y_left, 'U_expected': U_left,
            'y0': y0, 'y_final': float(numerical_positions_left[-1]),
            'displacement': float(numerical_positions_left[-1] - y0),
            'velocity': float(velocity_left),
            'velocity_error': float(abs(velocity_left - U_left)),
        },
        {
            'name': 'middle', 'header': f"Middle Region ({x_threshold_1} < x < {x_threshold_2})",
            'u_y': u_y_mid, 'U_expected': U_mid,
            'y0': y0, 'y_final': float(numerical_positions_mid[-1]),
            'displacement': float(numerical_positions_mid[-1] - y0),
            'velocity': float(velocity_mid),
            'velocity_error': float(abs(velocity_mid - U_mid)),
        },
        {
            'name': 'right', 'header': f"Right Region (x ≥ {x_threshold_2})",
            'u_y': u_y_right, 'U_expected': U_right,
            'y0': y0, 'y_final': float(numerical_positions_right[-1]),
            'displacement': float(numerical_positions_right[-1] - y0),
            'velocity': float(velocity_right),
            'velocity_error': float(abs(velocity_right - U_right)),
        },
    ]
    print_region_stats(regions_to_print)

    max_u = max(abs(u_y_left), abs(u_y_mid), abs(u_y_right))
    print_domain_info(nx, ny, Lx, Ly, solver.dx, solver.dy,
                      cfl_conv=max_u * dt / min(solver.dx, solver.dy),
                      cfl_prop=S_L * dt / min(solver.dx, solver.dy))

    print("\n" + "="*80)

    # Visualization (only if verbose)
    if verbose:
        # Precompute flame lengths for speed in plotting
        flame_lengths = [
            compute_contour_length(G, solver.X, solver.Y, iso_value=0.0)
            for G in G_history
        ]
        create_plots(
            solver, G_history, t_history, y0, time_scheme, t_final, use_reinit,
            numerical_positions_left, numerical_positions_mid, numerical_positions_right,
            x_threshold_1, x_threshold_2, S_L, u_y_left, u_y_mid, u_y_right,
            flame_lengths=flame_lengths,
            U_Y=u_y_mid, A=0.005, St=1.0, K=0.1, y_ref_for_velocity=0.0
        )

    return numerical_positions_left, numerical_positions_mid, numerical_positions_right, elapsed_time

if __name__ == "__main__":
    import sys

    # Default parameters
    scheme = 'rk2'
    t_final = 10.0
    use_reinit = True

    # Parse command-line arguments
    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2']:
            scheme = arg.lower()
        elif arg.startswith('t=') or arg.startswith('time='):
            t_final = float(arg.split('=')[1])
        elif arg == 'no_reinit':
            use_reinit = False

    print(f"\nRunning with: t_final={t_final}, scheme={scheme}, use_reinit={use_reinit}\n")

    test_linear_flame_nonhom(t_final=t_final, time_scheme=scheme, use_reinit=use_reinit, verbose=True)
