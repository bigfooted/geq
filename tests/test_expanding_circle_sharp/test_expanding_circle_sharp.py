import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Test case for 2D G-equation solver: expanding circle with sharp initial condition.
This tests the case with zero velocity: u = (0, 0)
Initial condition: G = -1 inside circle, G = +1 outside circle (discontinuous)
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver import (GEquationSolver2D, compute_circle_radius,
                                analytical_radius)
from plotting_utils import (plot_contours_comparison, plot_radius_comparison,
                            plot_surface_3d)
import time


def initial_solution_sharp(X, Y, x_center, y_center, radius):
    """
    Create sharp (discontinuous) initial level set function for a circle.

    The level set is defined as:
    G(x,y,t=0) = -1  if distance < R_0 (inside)
    G(x,y,t=0) = +1  if distance >= R_0 (outside)

    This creates a discontinuous jump at the flame surface.

    Parameters:
    -----------
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    x_center : float
        X coordinate of circle center
    y_center : float
        Y coordinate of circle center
    radius : float
        Initial radius of circle

    Returns:
    --------
    G : ndarray
        Initial level set function (sharp/discontinuous)
    """
    # Compute distance from center
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)

    # Create sharp interface: -1 inside, +1 outside
    G = np.ones_like(X)
    G[distance < radius] = -1.0

    return G


def test_expanding_circle_sharp(time_scheme='euler'):
    """
    Test the G-equation solver with a sharp initial condition (no flow).
    Initial condition has discontinuous jump: G=-1 inside, G=+1 outside.
    Compares numerical solution with analytical solution.

    Parameters:
    -----------
    time_scheme : str
        Time discretization scheme: 'euler' or 'rk2'
    """

    # Parameters
    nx = 101  # Number of grid points in x
    ny = 101  # Number of grid points in y
    Lx = 2.0  # Domain length in x
    Ly = 2.0  # Domain length in y
    S_L = 0.2  # Laminar flame speed

    # Flow velocity (zero for this test)
    u_x = 0.0
    u_y = 0.0

    # Circle parameters
    x_center = 1.0  # Center x-coordinate
    y_center = 1.0  # Center y-coordinate
    R0 = 0.3  # Initial radius

    # Time parameters
    t_final = 1.5
    dt = 0.001  # Time step (CFL condition)
    save_interval = 50  # Save solution every 50 steps for speed

    print("="*60)
    print(f"2D G-Equation Solver: Expanding Circle Test (Sharp IC)")
    print(f"Time Scheme: {time_scheme.upper()}")
    print("="*60)
    print(f"Grid: {nx} x {ny}")
    print(f"Domain: [{0}, {Lx}] x [{0}, {Ly}]")
    print(f"Laminar flame speed S_L = {S_L}")
    print(f"Flow velocity u = ({u_x}, {u_y})")
    print(f"Initial radius R0 = {R0}")
    print(f"Initial condition: G = -1 inside, G = +1 outside (SHARP)")
    print(f"Time step dt = {dt}")
    print(f"Final time t_final = {t_final}")
    print(f"Save interval: every {save_interval} steps")
    print("="*60)

    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)

    # Create sharp initial condition
    G_initial = initial_solution_sharp(solver.X, solver.Y, x_center, y_center, R0)

    print(f"\nInitial condition properties:")
    print(f"  G min = {G_initial.min():.6f}")
    print(f"  G max = {G_initial.max():.6f}")
    print(f"  G range = {G_initial.max() - G_initial.min():.6f}")
    print(f"  Points inside (G=-1): {np.sum(G_initial < 0)}")
    print(f"  Points outside (G=+1): {np.sum(G_initial > 0)}")

    # Verify initial radius
    initial_radius = compute_circle_radius(G_initial, solver.X, solver.Y,
                                          x_center, y_center, solver.dx, solver.dy)
    print(f"\nInitial radius verification:")
    print(f"  Expected R0 = {R0:.6f}")
    print(f"  Computed R0 = {initial_radius:.6f}")
    print(f"  Error = {abs(initial_radius - R0):.6f}")

    # Solve
    print("\nSolving G-equation...")
    start_time = time.time()
    G_history, t_history = solver.solve(G_initial, t_final, dt,
                                       save_interval=save_interval,
                                       time_scheme=time_scheme)
    end_time = time.time()
    elapsed_time = end_time - start_time

    n_total_steps = int(t_final / dt)
    print(f"Completed {n_total_steps} time steps in {elapsed_time:.2f} seconds.")
    print(f"Saved {len(t_history)} snapshots.")
    print(f"Average time per step: {elapsed_time/n_total_steps*1000:.3f} ms")

    # Check how the solution evolves
    print(f"\nSolution evolution:")
    for i, (G, t) in enumerate(zip(G_history[::len(G_history)//5], t_history[::len(t_history)//5])):
        print(f"  t = {t:.3f}: G_min = {G.min():.3f}, G_max = {G.max():.3f}, G_range = {G.max()-G.min():.3f}")

    # Extract numerical radius at each time step
    print("\nExtracting flame radius...")
    numerical_radii = []
    for G in G_history:
        radius = compute_circle_radius(G, solver.X, solver.Y, x_center, y_center,
                                      solver.dx, solver.dy)
        numerical_radii.append(radius)

    # Compute analytical solution
    t_array = np.array(t_history)
    analytical_radii = analytical_radius(t_array, R0, S_L)

    # Compute error
    error = np.abs(np.array(numerical_radii) - analytical_radii)
    relative_error = error / analytical_radii * 100

    print(f"\nMaximum absolute error: {np.max(error):.6f}")
    print(f"Maximum relative error: {np.max(relative_error):.2f}%")
    print(f"Mean absolute error: {np.mean(error):.6f}")
    print(f"Mean relative error: {np.mean(relative_error):.2f}%")

    # Visualization
    print("\nCreating visualizations...")

    # Plot 1: Contour plots at selected times
    plot_contours_comparison(solver, G_history, t_history, R0, S_L,
                            x_center_0=x_center, y_center_0=y_center,
                            u_x=u_x, u_y=u_y,
                            filename=f'contour_plots_sharp_{time_scheme}.png')

    # Plot 2: Radius vs time comparison
    plot_radius_comparison(t_history, numerical_radii, analytical_radii,
                          with_flow=False,
                          filename=f'radius_comparison_sharp_{time_scheme}.png')

    # Plot 3: 3D surface plot at final time
    G_final = G_history[-1]
    plot_surface_3d(solver, G_final, t_final, with_flow=False,
                   filename=f'surface_plot_sharp_{time_scheme}.png')

    # Plot 4: Additional diagnostic - G profile along x-axis through center
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Initial profile
    j_center = ny // 2
    ax1.plot(solver.x, G_initial[j_center, :], 'b-', linewidth=2, label='Initial (t=0)')
    ax1.axvline(x_center - R0, color='r', linestyle='--', linewidth=1, label=f'Expected boundary')
    ax1.axvline(x_center + R0, color='r', linestyle='--', linewidth=1)
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('G', fontsize=12)
    ax1.set_title('G Profile at y=1.0 (Initial - Sharp)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Final profile
    ax2.plot(solver.x, G_final[j_center, :], 'b-', linewidth=2, label=f'Final (t={t_final})')
    R_final = analytical_radius(t_final, R0, S_L)
    ax2.axvline(x_center - R_final, color='r', linestyle='--', linewidth=1, label=f'Expected boundary')
    ax2.axvline(x_center + R_final, color='r', linestyle='--', linewidth=1)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('G', fontsize=12)
    ax2.set_title(f'G Profile at y=1.0 (Final - Smoothed)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'profile_comparison_sharp_{time_scheme}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: profile_comparison_sharp_{time_scheme}.png")

    # Plot 5: Evolution of G at a fixed point
    fig5, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Pick a point just outside the initial flame (should transition from +1 to negative as flame passes)
    i_probe = int(nx * 0.55)  # x slightly greater than center
    j_probe = ny // 2  # y at center
    x_probe = solver.x[i_probe]
    y_probe = solver.y[j_probe]

    G_at_probe = [G[j_probe, i_probe] for G in G_history]

    # Time when flame should reach this point analytically
    distance_probe = np.sqrt((x_probe - x_center)**2 + (y_probe - y_center)**2)
    if distance_probe > R0:
        t_arrival = (distance_probe - R0) / S_L
    else:
        t_arrival = 0.0

    ax.plot(t_history, G_at_probe, 'b-', linewidth=2, label=f'G at probe ({x_probe:.3f}, {y_probe:.3f})')
    ax.axvline(t_arrival, color='r', linestyle='--', linewidth=2, label=f'Expected arrival time')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('G', fontsize=12)
    ax.set_title(f'Evolution of G at Fixed Point (Distance from center = {distance_probe:.3f})',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'temporal_evolution_sharp_{time_scheme}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: temporal_evolution_sharp_{time_scheme}.png")

    plt.show()

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("\nNotes:")
    print("  - Sharp initial condition creates numerical challenges")
    print("  - Solution naturally smooths out over time (numerical diffusion)")
    print("  - Zero level set (G=0) tracks the flame front")
    print("  - Compare with smooth initial condition for accuracy assessment")
    print("="*60)

    return error, elapsed_time


if __name__ == "__main__":
    import sys

    # Check if time scheme is provided as command line argument
    if len(sys.argv) > 1:
        scheme = sys.argv[1].lower()
        if scheme not in ['euler', 'rk2']:
            print("Usage: python test_expanding_circle_sharp.py [euler|rk2]")
            print("Defaulting to 'euler'")
            scheme = 'euler'
    else:
        scheme = 'euler'

    test_expanding_circle_sharp(time_scheme=scheme)
