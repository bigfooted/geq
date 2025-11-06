import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Test case for 2D G-equation solver: horizontal linear flame with non-homogeneous vertical flow.
Initial condition: horizontal line separating burnt (G=-1, above) and unburnt (G=+1, below) regions.
Flow: non-homogeneous vertical velocity:
      - u_y = 0.0 for x < 0.1 (left region, no flow)
      - u_y = 0.2 for x >= 0.1 (right region, upward flow)
      - u_x = 0.0 everywhere (no horizontal flow)
Burning velocity: S_L=0.1

This tests flame-flow interaction with spatially varying velocity, where the flame
will move at different rates in different regions:
- Left region (x < 0.1): U = 0.0 - 0.1 = -0.1 (downward, flame recedes)
- Right region (x >= 0.1): U = 0.2 - 0.1 = 0.1 (upward, flame advances)
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

def create_velocity_field(X, Y, u_y_left=0.0, u_y_right=0.2, x_threshold=0.1):
    """
    Create non-homogeneous velocity field.

    Parameters:
    -----------
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    u_y_left : float
        Vertical velocity for x < x_threshold
    u_y_right : float
        Vertical velocity for x >= x_threshold
    x_threshold : float
        X-coordinate threshold for velocity change

    Returns:
    --------
    u_x : ndarray
        Horizontal velocity field (all zeros)
    u_y : ndarray
        Vertical velocity field
    """
    u_x = np.zeros_like(X)
    u_y = np.where(X < x_threshold, u_y_left, u_y_right)
    return u_x, u_y

def initial_horizontal_flame(X, Y, y_flame):
    """
    Create initial level set function for a horizontal flame line.

    G > 0: unburnt region (below the line)
    G < 0: burnt region (above the line)
    G = 0: flame surface (at y = y_flame)

    Parameters:
    -----------
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    y_flame : float
        Initial y-position of the flame

    Returns:
    --------
    G : ndarray
        Initial level set function
    """
    # Signed distance: positive below (unburnt), negative above (burnt)
    G = -(Y - y_flame)
    return G

def extract_flame_position(G, X, Y, x_regions=None):
    """
    Extract the average y-position of the flame (G=0 contour) in different x-regions.

    Parameters:
    -----------
    G : ndarray
        Level set function
    X : ndarray
        X coordinates
    Y : ndarray
        Y coordinates
    x_regions : dict, optional
        Dictionary with region names as keys and (x_min, x_max) tuples as values

    Returns:
    --------
    positions : dict
        Dictionary with 'overall' mean position and positions for each region
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
                # Linear interpolation
                alpha = -G1 / (G2 - G1)
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
            if np.any(mask):
                result[region_name] = np.mean(y_positions_all[mask])
            else:
                result[region_name] = np.nan

    return result

def create_plots(solver, G_history, t_history, y0, time_scheme, t_final, use_reinit,
                 numerical_positions_left, numerical_positions_right, x_threshold):
    """
    Create all visualization plots for the non-homogeneous linear flame test.

    Parameters:
    -----------
    solver : GEquationSolver2D
        The solver instance
    G_history : list
        History of G fields
    t_history : list
        History of time values
    y0 : float
        Initial flame position
    time_scheme : str
        Time discretization scheme
    t_final : float
        Final simulation time
    use_reinit : bool
        Whether reinitialization was used
    numerical_positions_left : array
        Numerical flame positions over time in left region
    numerical_positions_right : array
        Numerical flame positions over time in right region
    x_threshold : float
        X-coordinate threshold separating regions
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

        # Mark the velocity transition line
        ax.axvline(x=x_threshold, color='magenta', linestyle=':', linewidth=2,
                   label=f'Velocity transition x={x_threshold}')

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

    # Plot 2: Flame position vs time for both regions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 2a: Position evolution in both regions
    ax1.plot(t_history, numerical_positions_left, 'b-', linewidth=2, marker='o',
             markersize=4, markevery=max(1, len(t_history)//20),
             label=f'Left (x < {x_threshold}): U = -0.1')
    ax1.plot(t_history, numerical_positions_right, 'r-', linewidth=2, marker='s',
             markersize=4, markevery=max(1, len(t_history)//20),
             label=f'Right (x ≥ {x_threshold}): U = 0.1')
    ax1.axhline(y=y0, color='gray', linestyle='--', alpha=0.5, label='Initial position')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Flame Position (y)', fontsize=12)
    ax1.set_title(f'Flame Position vs Time in Different Regions\n{time_scheme.upper()}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2b: Relative displacement from initial position
    displacement_left = np.array(numerical_positions_left) - y0
    displacement_right = np.array(numerical_positions_right) - y0

    ax2.plot(t_history, displacement_left, 'b-', linewidth=2, label=f'Left (x < {x_threshold})')
    ax2.plot(t_history, displacement_right, 'r-', linewidth=2, label=f'Right (x ≥ {x_threshold})')
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

    # Plot 3: 3D surface plot at final time
    G_final = G_history[-1]
    plot_surface_3d(solver, G_final, t_final, with_flow=True,
                   filename=f'linear_flame_nonhom_surface_{time_scheme}_t{t_final}{suffix}.png')

    plt.show()

    print_completion()

def test_linear_flame_nonhom(t_final=2.0, time_scheme='rk2', use_reinit=True, verbose=True):
    """
    Test the G-equation solver with a horizontal linear flame in non-homogeneous vertical flow.

    Parameters:
    -----------
    t_final : float
        Final simulation time (default: 2.0)
    time_scheme : str
        Time discretization scheme: 'euler' or 'rk2'
    use_reinit : bool
        Use local reinitialization every 50 steps (default: True)
    verbose : bool
        Print detailed output (default: True)

    Returns:
    --------
    positions_left : array
        Flame positions in left region over time
    positions_right : array
        Flame positions in right region over time
    elapsed_time : float
        Computation time in seconds
    """

    # Parameters
    nx = 101  # Number of grid points in x
    ny = 101  # Number of grid points in y
    Lx = 1.0  # Domain length in x
    Ly = 1.0  # Domain length in y
    S_L = 0.1  # Laminar flame speed

    # Flow velocity parameters
    u_y_left = 0.0   # Vertical velocity for x < 0.1
    u_y_right = 0.2  # Vertical velocity for x >= 0.1
    x_threshold = 0.1  # Threshold x-coordinate

    # Expected net flame velocities in each region
    U_left = u_y_left - S_L   # -0.1 (downward)
    U_right = u_y_right - S_L  # 0.1 (upward)

    # Initial flame position
    y0 = 0.3  # Start at y = 0.3

    # Time parameters
    dt = 0.001  # Time step (CFL condition)
    save_interval = 50  # Save solution every 50 steps

    # Reinitialization parameters
    reinit_interval = 50 if use_reinit else 0
    reinit_method = 'fast_marching'
    reinit_local = True

    if verbose:
        flow_desc = [
            "Flow velocity (non-homogeneous):",
            f"  u_y = {u_y_left} for x < {x_threshold} (left region)",
            f"  u_y = {u_y_right} for x >= {x_threshold} (right region)",
            f"  u_x = 0.0 everywhere",
            "Expected net flame velocities:",
            f"  U_left = {U_left} (downward)",
            f"  U_right = {U_right} (upward)",
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
            "2D G-Equation Solver: Linear Flame Test (Non-Homogeneous Vertical Flow)",
            nx, ny, Lx, Ly, S_L, flow_desc, time_scheme, reinit_desc
        )

    # Create solver with scalar velocities first (will be overridden with arrays)
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)

    # Create non-homogeneous velocity field
    u_x, u_y = create_velocity_field(solver.X, solver.Y, u_y_left, u_y_right, x_threshold)

    # Override the velocity fields with numpy arrays
    # The solver should handle these as spatially varying fields
    solver.u_x = u_x
    solver.u_y = u_y

    # Create initial condition
    G_initial = initial_horizontal_flame(solver.X, solver.Y, y0)

    # Define regions for analysis
    x_regions = {
        'left': (0.0, x_threshold),
        'right': (x_threshold, Lx)
    }

    # Verify initial flame position
    initial_positions = extract_flame_position(G_initial, solver.X, solver.Y, x_regions)

    if verbose:
        print_initial_flame_verification(y0, initial_positions, regions=['left', 'right'])
        print_G_field_stats(
            G_initial,
            bottom_val=G_initial[0, nx//2],
            top_val=G_initial[-1, nx//2],
            bottom_label='y=0', top_label='y=Ly'
        )
        print_velocity_field_verification([
            {'label': 'u_y at x=0.05 (left)', 'value': float(solver.u_y[ny//2, 5]), 'expected': u_y_left},
            {'label': 'u_y at x=0.5 (right)', 'value': float(solver.u_y[ny//2, nx//2]), 'expected': u_y_right},
        ])

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
        reinit_local=reinit_local
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    n_total_steps = int(t_final / dt)
    if verbose:
        print_performance(elapsed_time, n_total_steps, len(t_history))

    # Extract numerical flame position at each time step
    if verbose:
        sub_banner("Extracting flame positions in different regions...")

    numerical_positions_left = []
    numerical_positions_right = []
    numerical_positions_overall = []

    extraction_start = time.time()

    for G in G_history:
        positions = extract_flame_position(G, solver.X, solver.Y, x_regions)
        numerical_positions_overall.append(positions['overall'])
        numerical_positions_left.append(positions.get('left', np.nan))
        numerical_positions_right.append(positions.get('right', np.nan))

    extraction_time = time.time() - extraction_start

    if verbose:
        print_performance(elapsed_time, n_total_steps, len(t_history), extraction_time=extraction_time)

    # Convert to arrays
    numerical_positions_left = np.array(numerical_positions_left)
    numerical_positions_right = np.array(numerical_positions_right)
    t_array = np.array(t_history)

    # Compute velocities from numerical data
    if t_final > 0:
        velocity_left = (numerical_positions_left[-1] - y0) / t_final
        velocity_right = (numerical_positions_right[-1] - y0) / t_final
    else:
        velocity_left = 0.0
        velocity_right = 0.0

    # Print statistics
    print("\n" + "="*80)
    print("SIMULATION STATISTICS")
    print("="*80)

    print("\n" + "="*80)
    print("SIMULATION STATISTICS")
    print("="*80)

    print_performance(elapsed_time, n_total_steps, len(t_history), extraction_time=extraction_time)

    regions_to_print = [
        {
            'name': 'left', 'header': f"Left Region (x < {x_threshold})",
            'u_y': u_y_left, 'U_expected': U_left,
            'y0': y0, 'y_final': float(numerical_positions_left[-1]),
            'displacement': float(numerical_positions_left[-1] - y0),
            'velocity': float(velocity_left),
            'velocity_error': float(abs(velocity_left - U_left)),
        },
        {
            'name': 'right', 'header': f"Right Region (x ≥ {x_threshold})",
            'u_y': u_y_right, 'U_expected': U_right,
            'y0': y0, 'y_final': float(numerical_positions_right[-1]),
            'displacement': float(numerical_positions_right[-1] - y0),
            'velocity': float(velocity_right),
            'velocity_error': float(abs(velocity_right - U_right)),
        },
    ]
    print_region_stats(regions_to_print)

    print_domain_info(
        nx, ny, Lx, Ly, solver.dx, solver.dy,
        cfl_conv=abs(u_y_right) * dt / min(solver.dx, solver.dy),
        cfl_prop=S_L * dt / min(solver.dx, solver.dy)
    )

    print("\n" + "="*80)

    # Visualization (only if verbose)
    if verbose:
        create_plots(solver, G_history, t_history, y0, time_scheme, t_final, use_reinit,
                     numerical_positions_left, numerical_positions_right, x_threshold)

    return numerical_positions_left, numerical_positions_right, elapsed_time

if __name__ == "__main__":
    import sys

    # Default parameters
    scheme = 'rk2'
    t_final = 2.0
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
