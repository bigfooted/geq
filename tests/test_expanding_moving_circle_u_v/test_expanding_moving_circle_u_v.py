import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Test case for 2D G-equation solver: expanding circle in uniform flow.
This tests the case with constant velocity: u = (0.0, 0.1)
Updated to use local reinitialization by default and compute flame surface area.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import (GEquationSolver2D, initial_solution,
                                compute_circle_radius, compute_circle_center,
                                analytical_radius)  # removed analytical_center
from plotting_utils import (plot_contours_comparison, plot_radius_and_center_combined,
                            plot_trajectory, plot_surface_3d)
from contour_utils import compute_contour_length
import time


def analytical_surface_area(t, R0, S_L):
    """
    Analytical flame surface area for a circle.
    Surface area (2D) = circumference = 2π R(t)

    Parameters:
    -----------
    t : float or ndarray
        Time
    R0 : float
        Initial radius
    S_L : float
        Laminar flame speed

    Returns:
    --------
    area : float or ndarray
        Flame surface area at time t
    """
    R = R0 + S_L * t
    return 2.0 * np.pi * R


def _extract_zero_level_points(G, X, Y, eps=1e-12):
    """
    Extract approximate points on the zero level set using edge interpolation.
    Returns an (N, 2) array of (x, y) points.
    """
    ny, nx = G.shape
    pts = []

    # Helper to interpolate along an edge (p1 -> p2) with values g1, g2
    def interp(p1x, p1y, p2x, p2y, g1, g2):
        denom = (g1 - g2)
        if abs(denom) < eps:
            t = 0.5  # nearly equal; pick midpoint
        else:
            t = g1 / denom
        t = np.clip(t, 0.0, 1.0)
        x = p1x + t * (p2x - p1x)
        y = p1y + t * (p2y - p1y)
        return x, y

    # Horizontal edges
    for j in range(ny):
        for i in range(nx - 1):
            g1 = G[j, i]
            g2 = G[j, i + 1]
            if g1 == 0.0 and g2 == 0.0:
                # whole edge lies on zero: add endpoints
                pts.append((X[j, i],     Y[j, i]))
                pts.append((X[j, i + 1], Y[j, i + 1]))
            elif g1 * g2 < 0.0:
                x, y = interp(X[j, i], Y[j, i], X[j, i + 1], Y[j, i + 1], g1, g2)
                pts.append((x, y))

    # Vertical edges
    for j in range(ny - 1):
        for i in range(nx):
            g1 = G[j, i]
            g2 = G[j + 1, i]
            if g1 == 0.0 and g2 == 0.0:
                pts.append((X[j, i],     Y[j, i]))
                pts.append((X[j + 1, i], Y[j + 1, i]))
            elif g1 * g2 < 0.0:
                x, y = interp(X[j, i], Y[j, i], X[j + 1, i], Y[j + 1, i], g1, g2)
                pts.append((x, y))

    if not pts:
        return np.empty((0, 2))
    return np.array(pts)


def compute_center_from_contour(G, X, Y):
    """
    Compute the center as the centroid of the zero level set points.
    Falls back to the legacy center computation if too few points are found.
    """
    pts = _extract_zero_level_points(G, X, Y)
    if pts.shape[0] < 8:
        # Fallback to legacy method if available
        try:
            from g_equation_solver_improved import compute_circle_center as _legacy_center
            return _legacy_center(G, X, Y, X[0,1] - X[0,0], Y[1,0] - Y[0,0])
        except Exception:
            # As a last resort, use domain centroid
            return float(np.mean(X)), float(np.mean(Y))
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    return cx, cy


def test_expanding_moving_circle(t_final=1.5, time_scheme='euler', use_reinit=True, verbose=True):
    """
    Test the G-equation solver with an expanding circle in uniform flow.
    Compares numerical solution with analytical solution.

    Parameters:
    -----------
    t_final : float
        Final simulation time (default: 1.5)
    time_scheme : str
        Time discretization scheme: 'euler' or 'rk2'
    use_reinit : bool
        Use local reinitialization every 50 steps (default: True)
    verbose : bool
        Print detailed output (default: True)

    Returns:
    --------
    radius_error : ndarray
        Absolute error in radius over time
    x_center_error : ndarray
        Absolute error in x-center over time
    y_center_error : ndarray
        Absolute error in y-center over time
    elapsed_time : float
        Computation time in seconds
    """

    # Parameters
    nx = 101  # Number of grid points in x
    ny = 101  # Number of grid points in y
    Lx = 2.0  # Domain length in x
    Ly = 2.0  # Domain length in y
    S_L = 0.2  # Laminar flame speed

    # Flow velocity (constant uniform flow)
    u_x = 0.0
    u_y = 0.1

    # Circle parameters
    x_center_0 = 1.0  # Initial center x-coordinate
    y_center_0 = 1.0  # Initial center y-coordinate
    R0 = 0.3  # Initial radius

    # Time parameters
    dt = 0.001  # Time step (CFL condition)
    save_interval = 50  # Save solution every 50 steps for speed

    # Reinitialization parameters
    reinit_interval = 50 if use_reinit else 0
    reinit_method = 'fast_marching'
    reinit_local = True  # Use local (narrow-band) reinitialization

    if verbose:
        print("="*60)
        print(f"2D G-Equation Solver: Expanding Moving Circle Test")
        print(f"Time Scheme: {time_scheme.upper()}")
        if use_reinit:
            print(f"Reinitialization: LOCAL every {reinit_interval} steps ({reinit_method})")
        else:
            print("Reinitialization: Disabled")
        print("="*60)
        print(f"Grid: {nx} x {ny}")
        print(f"Domain: [{0}, {Lx}] x [{0}, {Ly}]")
        print(f"Laminar flame speed S_L = {S_L}")
        print(f"Flow velocity u = ({u_x}, {u_y})")
        print(f"Initial center = ({x_center_0}, {y_center_0})")
        print(f"Initial radius R0 = {R0}")
        print(f"Time step dt = {dt}")
        print(f"Final time t_final = {t_final}")
        print(f"Save interval: every {save_interval} steps")
        print("="*60)

    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)

    # Create initial condition
    G_initial = initial_solution(solver.X, solver.Y, x_center_0, y_center_0, R0)

    # Verify initial radius, center, and surface area
    initial_radius = compute_circle_radius(G_initial, solver.X, solver.Y,
                                          x_center_0, y_center_0, solver.dx, solver.dy)
    initial_x_center, initial_y_center = compute_circle_center(G_initial, solver.X, solver.Y,
                                                               solver.dx, solver.dy)
    initial_area = compute_contour_length(G_initial, solver.X, solver.Y, iso_value=0.0)
    analytical_initial_area = analytical_surface_area(0.0, R0, S_L)

    if verbose:
        print(f"\nInitial verification:")
        print(f"  Expected R0 = {R0:.6f}, Computed R0 = {initial_radius:.6f}")
        print(f"  Expected center = ({x_center_0:.6f}, {y_center_0:.6f})")
        print(f"  Computed center = ({initial_x_center:.6f}, {initial_y_center:.6f})")
        print(f"\nInitial flame surface area:")
        print(f"  Analytical = {analytical_initial_area:.6f}")
        print(f"  Computed   = {initial_area:.6f}")
        print(f"  Error      = {abs(initial_area - analytical_initial_area):.6f}")

    # Solve
    if verbose:
        print("\nSolving G-equation...")
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
        print(f"Completed {n_total_steps} time steps in {elapsed_time:.2f} seconds.")
        print(f"Saved {len(t_history)} snapshots.")
        print(f"Average time per step: {elapsed_time/n_total_steps*1000:.3f} ms")

    # Extract numerical radius, center, and surface area at each time step
    if verbose:
        print("\nExtracting flame radius, center, and surface area...")
    numerical_radii = []
    numerical_x_centers = []
    numerical_y_centers = []
    numerical_areas = []

    for G in G_history:
        # Compute center from zero-contour
        x_c, y_c = compute_center_from_contour(G, solver.X, solver.Y)
        numerical_x_centers.append(x_c)
        numerical_y_centers.append(y_c)

        # Compute radius using computed center
        radius = compute_circle_radius(G, solver.X, solver.Y, x_c, y_c,
                                      solver.dx, solver.dy)
        numerical_radii.append(radius)

        # Compute surface area
        area = compute_contour_length(G, solver.X, solver.Y, iso_value=0.0)
        numerical_areas.append(area)

    # Compute analytical solution
    t_array = np.array(t_history)
    analytical_radii = analytical_radius(t_array, R0, S_L)
    # Replace previous analytical_center(...) with explicit advection by u
    analytical_x_centers = x_center_0 + u_x * t_array
    analytical_y_centers = y_center_0 + u_y * t_array
    analytical_areas = analytical_surface_area(t_array, R0, S_L)

    # Compute errors
    radius_error = np.abs(np.array(numerical_radii) - analytical_radii)
    x_center_error = np.abs(np.array(numerical_x_centers) - analytical_x_centers)
    y_center_error = np.abs(np.array(numerical_y_centers) - analytical_y_centers)
    area_error = np.abs(np.array(numerical_areas) - analytical_areas)
    relative_area_error = area_error / analytical_areas * 100

    if verbose:
        print(f"\nRadius errors:")
        print(f"  Maximum absolute error: {np.max(radius_error):.6f}")
        print(f"  Mean absolute error: {np.mean(radius_error):.6f}")

        print(f"\nCenter position errors:")
        print(f"  Maximum x-error: {np.max(x_center_error):.6f}")
        print(f"  Maximum y-error: {np.max(y_center_error):.6f}")
        print(f"  Mean x-error: {np.mean(x_center_error):.6f}")
        print(f"  Mean y-error: {np.mean(y_center_error):.6f}")

        print(f"\nFlame surface area errors:")
        print(f"  Maximum absolute error: {np.max(area_error):.6f}")
        print(f"  Maximum relative error: {np.max(relative_area_error):.2f}%")
        print(f"  Mean absolute error: {np.mean(area_error):.6f}")
        print(f"  Mean relative error: {np.mean(relative_area_error):.2f}%")

    # Visualization (only if verbose)
    if verbose:
        print("\nCreating visualizations...")

        suffix = f"_reinit" if use_reinit else "_no_reinit"

        # Plot 1: Contour plots at selected times
        plot_contours_comparison(solver, G_history, t_history, R0, S_L,
                                x_center_0=x_center_0, y_center_0=y_center_0,
                                u_x=u_x, u_y=u_y,
                                numerical_centers=(numerical_x_centers, numerical_y_centers),
                                filename=f'contour_plots_moving_{time_scheme}_t{t_final}{suffix}.png')

        # Plot 2: Radius and center position vs time (combined)
        plot_radius_and_center_combined(t_history, numerical_radii, analytical_radii,
                                       numerical_x_centers, numerical_y_centers,
                                       analytical_x_centers, analytical_y_centers,
                                       filename=f'radius_center_comparison_moving_{time_scheme}_t{t_final}{suffix}.png')

        # Plot 3: Surface area comparison
        fig_area, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Surface area comparison
        ax1.plot(t_history, numerical_areas, 'b-', linewidth=2, label='Numerical')
        ax1.plot(t_history, analytical_areas, 'r--', linewidth=2, label='Analytical')
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Flame Surface Area', fontsize=12)
        ax1.set_title(f'Flame Surface Area vs Time (With Flow)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Area error plot
        ax2.plot(t_history, area_error, 'k-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Absolute Error', fontsize=12)
        ax2.set_title('Error in Flame Surface Area', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'surface_area_comparison_moving_{time_scheme}_t{t_final}{suffix}.png',
                   dpi=300, bbox_inches='tight')
        print(f"Saved: surface_area_comparison_moving_{time_scheme}_t{t_final}{suffix}.png")

        # Plot 4: Trajectory in x-y plane
        plot_trajectory(numerical_x_centers, numerical_y_centers,
                       analytical_x_centers, analytical_y_centers,
                       x_center_0, y_center_0,
                       filename=f'trajectory_moving_{time_scheme}_t{t_final}{suffix}.png')

        # Plot 5: 3D surface plot at final time
        G_final = G_history[-1]
        plot_surface_3d(solver, G_final, t_final, with_flow=True,
                       filename=f'surface_plot_moving_{time_scheme}_t{t_final}{suffix}.png')

        plt.show()

        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)

    return radius_error, x_center_error, y_center_error, elapsed_time


if __name__ == "__main__":
    import sys

    # Default parameters
    scheme = 'euler'
    t_final = 1.5
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

    test_expanding_moving_circle(t_final=t_final, time_scheme=scheme, use_reinit=use_reinit, verbose=True)
