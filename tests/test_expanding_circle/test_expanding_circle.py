import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Test case for 2D G-equation solver: expanding circle without flow.
This tests the case with zero velocity: u = (0, 0)
Supports command-line argument for simulation time.
Updated to use local reinitialization by default, marching squares for surface area,
and comprehensive statistics reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
import time  # FIX: missing import
from g_equation_solver_improved import (GEquationSolver2D, initial_solution,
                                compute_circle_radius, analytical_radius)
from plotting_utils import (plot_contours_comparison, plot_radius_comparison,
                            plot_surface_3d)
from contour_utils import get_contour_statistics, compute_contour_length

# Centralized verbose visualizations
def create_verbose_visualizations(
    solver,
    G_history,
    t_history,
    R0,
    S_L,
    x_center,
    y_center,
    u_x,
    u_y,
    time_scheme,
    t_final,
    use_reinit,
    numerical_radii,
    analytical_radii,
    numerical_areas,
    analytical_areas,
    radius_error,
    area_error,
    relative_radius_error,
    relative_area_error,
    analytical_initial_area,
    show_plots=True,  # NEW
):
    """
    Generate plots when verbose=True.
    """
    print("\nCreating visualizations...")
    suffix = "_reinit" if use_reinit else "_no_reinit"

    # Contour comparison
    plot_contours_comparison(
        solver,
        G_history,
        t_history,
        R0,
        S_L,
        x_center_0=x_center,
        y_center_0=y_center,
        u_x=u_x,
        u_y=u_y,
        filename=f'contour_plots_stationary_{time_scheme}_t{t_final}{suffix}.png'
    )

    # Radius vs time
    plot_radius_comparison(
        t_history,
        numerical_radii,
        analytical_radii,
        with_flow=False,
        filename=f'radius_comparison_stationary_{time_scheme}_t{t_final}{suffix}.png'
    )

    # Flame surface area vs time
    fig_area, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 3a: Surface area evolution
    ax1.plot(t_history, numerical_areas, 'b-', linewidth=2, marker='o',
             markersize=4, markevery=max(1, len(t_history)//20), label='Numerical')
    ax1.plot(t_history, analytical_areas, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Flame Surface Area', fontsize=12)
    ax1.set_title(f'Flame Surface Area vs Time (No Flow)\n{time_scheme.upper()}, '
                  f'Reinit: {use_reinit}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 3b: Area error evolution
    ax2.plot(t_history, area_error, 'k-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error in Flame Surface Area', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    # 3c: Relative error evolution
    ax3.plot(t_history, relative_area_error, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Relative Error (%)', fontsize=12)
    ax3.set_title('Relative Error in Flame Surface Area', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 3d: Both radius and area on same plot (normalized)
    ax4.plot(t_history, np.array(numerical_radii) / R0, 'b-', linewidth=2,
             label='Radius / R₀')
    ax4.plot(t_history, np.array(numerical_areas) / analytical_initial_area, 'r-',
             linewidth=2, label='Area / Area₀')
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Normalized Value', fontsize=12)
    ax4.set_title('Normalized Growth (Radius and Area)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'flame_surface_area_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: flame_surface_area_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 4: Combined error analysis
    fig_err, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 4a: Absolute errors comparison
    ax1.semilogy(t_history, radius_error, 'b-', linewidth=2, marker='o',
                 markersize=4, markevery=max(1, len(t_history)//20), label='Radius Error')
    ax1.semilogy(t_history, area_error, 'r-', linewidth=2, marker='s',
                 markersize=4, markevery=max(1, len(t_history)//20), label='Surface Area Error')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Absolute Error (log scale)', fontsize=12)
    ax1.set_title('Error Evolution Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')

    # 4b: Error statistics over time windows
    window_size = max(1, len(t_history) // 10)
    n_windows = len(t_history) // window_size

    window_times = []
    window_radius_rms = []
    window_area_rms = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(t_history))

        window_times.append(np.mean(t_history[start_idx:end_idx]))
        window_radius_rms.append(np.sqrt(np.mean(radius_error[start_idx:end_idx]**2)))
        window_area_rms.append(np.sqrt(np.mean(area_error[start_idx:end_idx]**2)))

    ax2.plot(window_times, window_radius_rms, 'b-', linewidth=2, marker='o',
             markersize=6, label='Radius RMS Error')
    ax2.plot(window_times, window_area_rms, 'r-', linewidth=2, marker='s',
             markersize=6, label='Surface Area RMS Error')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('RMS Error (windowed)', fontsize=12)
    ax2.set_title('RMS Error in Time Windows', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'error_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: error_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 5: 3D surface plot at final time
    G_final = G_history[-1]
    plot_surface_3d(solver, G_final, t_final, with_flow=False,
                    filename=f'surface_plot_stationary_{time_scheme}_t{t_final}{suffix}.png')

    # NEW: show figures on screen (if a GUI backend is available)
    if show_plots:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: could not display figures: {e}. Figures were saved to disk.")
        finally:
            plt.close('all')


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


def create_plots(solver, G_history, t_history, R0, S_L, x_center, y_center,
                 u_x, u_y, time_scheme, t_final, use_reinit,
                 numerical_radii, numerical_areas, analytical_radii, analytical_areas,
                 radius_error, area_error, relative_area_error, analytical_initial_area):
    """
    Create all visualization plots for the expanding circle test.

    Parameters:
    -----------
    solver : GEquationSolver2D
        The solver instance
    G_history : list
        History of G fields
    t_history : list
        History of time values
    R0 : float
        Initial radius
    S_L : float
        Laminar flame speed
    x_center : float
        Circle center x-coordinate
    y_center : float
        Circle center y-coordinate
    u_x : float
        Flow velocity x-component
    u_y : float
        Flow velocity y-component
    time_scheme : str
        Time discretization scheme
    t_final : float
        Final simulation time
    use_reinit : bool
        Whether reinitialization was used
    numerical_radii : array
        Numerical radii over time
    numerical_areas : array
        Numerical surface areas over time
    analytical_radii : array
        Analytical radii over time
    analytical_areas : array
        Analytical surface areas over time
    radius_error : array
        Radius errors over time
    area_error : array
        Area errors over time
    relative_area_error : array
        Relative area errors over time
    analytical_initial_area : float
        Analytical initial surface area
    """
    print("\nCreating visualizations...")

    suffix = f"_reinit" if use_reinit else "_no_reinit"

    # Plot 1: Contour plots at selected times
    plot_contours_comparison(solver, G_history, t_history, R0, S_L,
                            x_center_0=x_center, y_center_0=y_center,
                            u_x=u_x, u_y=u_y,
                            filename=f'contour_plots_stationary_{time_scheme}_t{t_final}{suffix}.png')

    # Plot 2: Radius vs time comparison
    plot_radius_comparison(t_history, numerical_radii, analytical_radii,
                          with_flow=False,
                          filename=f'radius_comparison_stationary_{time_scheme}_t{t_final}{suffix}.png')

    # Plot 3: Flame surface area vs time (NEW)
    fig_area, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 3a: Surface area evolution
    ax1.plot(t_history, numerical_areas, 'b-', linewidth=2, marker='o',
            markersize=4, markevery=max(1, len(t_history)//20), label='Numerical')
    ax1.plot(t_history, analytical_areas, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Flame Surface Area', fontsize=12)
    ax1.set_title(f'Flame Surface Area vs Time (No Flow)\n{time_scheme.upper()}, '
                 f'Reinit: {use_reinit}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 3b: Area error evolution
    ax2.plot(t_history, area_error, 'k-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error in Flame Surface Area', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    # 3c: Relative error evolution
    ax3.plot(t_history, relative_area_error, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Relative Error (%)', fontsize=12)
    ax3.set_title('Relative Error in Flame Surface Area', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 3d: Both radius and area on same plot (normalized)
    ax4.plot(t_history, np.array(numerical_radii) / R0, 'b-', linewidth=2,
            label='Radius / R₀')
    ax4.plot(t_history, np.array(numerical_areas) / analytical_initial_area, 'r-',
            linewidth=2, label='Area / Area₀')
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Normalized Value', fontsize=12)
    ax4.set_title('Normalized Growth (Radius and Area)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'flame_surface_area_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png',
               dpi=300, bbox_inches='tight')
    print(f"Saved: flame_surface_area_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 4: Combined error analysis
    fig_err, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 4a: Absolute errors comparison
    ax1.semilogy(t_history, radius_error, 'b-', linewidth=2, marker='o',
                markersize=4, markevery=max(1, len(t_history)//20), label='Radius Error')
    ax1.semilogy(t_history, area_error, 'r-', linewidth=2, marker='s',
                markersize=4, markevery=max(1, len(t_history)//20), label='Surface Area Error')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Absolute Error (log scale)', fontsize=12)
    ax1.set_title('Error Evolution Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')

    # 4b: Error statistics over time windows
    window_size = max(1, len(t_history) // 10)
    n_windows = len(t_history) // window_size

    window_times = []
    window_radius_rms = []
    window_area_rms = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(t_history))

        window_times.append(np.mean(t_history[start_idx:end_idx]))
        window_radius_rms.append(np.sqrt(np.mean(radius_error[start_idx:end_idx]**2)))
        window_area_rms.append(np.sqrt(np.mean(area_error[start_idx:end_idx]**2)))

    ax2.plot(window_times, window_radius_rms, 'b-', linewidth=2, marker='o',
            markersize=6, label='Radius RMS Error')
    ax2.plot(window_times, window_area_rms, 'r-', linewidth=2, marker='s',
            markersize=6, label='Surface Area RMS Error')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('RMS Error (windowed)', fontsize=12)
    ax2.set_title('RMS Error in Time Windows', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'error_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png',
               dpi=300, bbox_inches='tight')
    print(f"Saved: error_analysis_stationary_{time_scheme}_t{t_final}{suffix}.png")

    # Plot 5: 3D surface plot at final time
    G_final = G_history[-1]
    plot_surface_3d(solver, G_final, t_final, with_flow=False,
                   filename=f'surface_plot_stationary_{time_scheme}_t{t_final}{suffix}.png')

    plt.show()

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80 + "\n")

def test_expanding_circle(t_final=1.5, time_scheme='euler', use_reinit=True, verbose=True):
    """
    Test the G-equation solver with a pure expanding circle (no flow).
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
    error : ndarray
        Absolute error in radius over time
    elapsed_time : float
        Computation time in seconds
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
    dt = 0.001  # Time step (CFL condition)
    save_interval = 50  # Save solution every 50 steps for speed

    # Reinitialization parameters
    reinit_interval = 50 if use_reinit else 0
    reinit_method = 'fast_marching'
    reinit_local = True  # Use local (narrow-band) reinitialization

    if verbose:
        print("="*80)
        print(f"2D G-Equation Solver: Expanding Circle Test (No Flow)")
        print(f"Time Scheme: {time_scheme.upper()}")
        if use_reinit:
            print(f"Reinitialization: LOCAL every {reinit_interval} steps ({reinit_method})")
        else:
            print("Reinitialization: Disabled")
        print("="*80)
        print(f"Grid: {nx} x {ny}")
        print(f"Domain: [{0}, {Lx}] x [{0}, {Ly}]")
        print(f"Laminar flame speed S_L = {S_L}")
        print(f"Flow velocity u = ({u_x}, {u_y})")
        print(f"Initial radius R0 = {R0}")
        print(f"Time step dt = {dt}")
        print(f"Final time t_final = {t_final}")
        print(f"Save interval: every {save_interval} steps")
        print("="*80)

    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)

    # Create initial condition
    G_initial = initial_solution(solver.X, solver.Y, x_center, y_center, R0)

    # Verify initial radius and surface area using marching squares
    initial_radius = compute_circle_radius(G_initial, solver.X, solver.Y,
                                          x_center, y_center, solver.dx, solver.dy)
    initial_stats = get_contour_statistics(G_initial, solver.X, solver.Y, iso_value=0.0)
    initial_area = initial_stats['total_length']
    analytical_initial_area = analytical_surface_area(0.0, R0, S_L)

    if verbose:
        print(f"\nInitial radius verification:")
        print(f"  Expected R0 = {R0:.6f}")
        print(f"  Computed R0 = {initial_radius:.6f}")
        print(f"  Error = {abs(initial_radius - R0):.6f}")
        print(f"\nInitial flame surface area (Marching Squares):")
        print(f"  Analytical = {analytical_initial_area:.8f}")
        print(f"  Computed   = {initial_area:.8f}")
        print(f"  Error      = {abs(initial_area - analytical_initial_area):.8f}")
        print(f"  Rel. Error = {abs(initial_area - analytical_initial_area)/analytical_initial_area*100:.4f}%")
        print(f"\nInitial contour statistics:")
        print(f"  Number of contours: {initial_stats['num_contours']}")
        print(f"  Closed contours: {initial_stats['num_closed']}")
        print(f"  Open contours: {initial_stats['num_open']}")

    # Solve
    if verbose:
        print("\n" + "="*80)
        print("Solving G-equation...")
        print("="*80)
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
        print(f"\nCompleted {n_total_steps} time steps in {elapsed_time:.2f} seconds.")
        print(f"Saved {len(t_history)} snapshots.")
        print(f"Average time per step: {elapsed_time/n_total_steps*1000:.4f} ms")
        print(f"Average time per saved snapshot: {elapsed_time/len(t_history):.4f} s")

    # Extract numerical radius and surface area at each time step
    if verbose:
        print("\n" + "="*80)
        print("Extracting flame radius and surface area (Marching Squares)...")
        print("="*80)

    numerical_radii = []
    numerical_areas = []
    contour_stats = []

    extraction_start = time.time()
    for idx, G in enumerate(G_history):
        radius = compute_circle_radius(G, solver.X, solver.Y, x_center, y_center,
                                      solver.dx, solver.dy)
        numerical_radii.append(radius)

        # Use marching squares for accurate contour length
        area = compute_contour_length(G, solver.X, solver.Y, iso_value=0.0)
        numerical_areas.append(area)

        # Get detailed statistics for selected snapshots
        if idx % max(1, len(G_history) // 5) == 0:
            stats = get_contour_statistics(G, solver.X, solver.Y, iso_value=0.0)
            contour_stats.append((t_history[idx], stats))

    extraction_time = time.time() - extraction_start

    if verbose:
        print(f"Contour extraction completed in {extraction_time:.2f} seconds.")
        print(f"Average time per extraction: {extraction_time/len(G_history)*1000:.2f} ms")

    # Compute analytical solution
    t_array = np.array(t_history)
    analytical_radii = analytical_radius(t_array, R0, S_L)
    analytical_areas = analytical_surface_area(t_array, R0, S_L)

    # Compute errors
    radius_error = np.abs(np.array(numerical_radii) - analytical_radii)
    relative_radius_error = radius_error / analytical_radii * 100

    area_error = np.abs(np.array(numerical_areas) - analytical_areas)
    relative_area_error = area_error / analytical_areas * 100

    # Verbose visualizations
    if verbose:
        create_verbose_visualizations(
            solver=solver,
            G_history=G_history,
            t_history=t_history,
            R0=R0,
            S_L=S_L,
            x_center=x_center,
            y_center=y_center,
            u_x=u_x,
            u_y=u_y,
            time_scheme=time_scheme,
            t_final=t_final,
            use_reinit=use_reinit,
            numerical_radii=numerical_radii,
            analytical_radii=analytical_radii,
            numerical_areas=numerical_areas,
            analytical_areas=analytical_areas,
            radius_error=radius_error,
            area_error=area_error,
            relative_radius_error=relative_radius_error,
            relative_area_error=relative_area_error,
            analytical_initial_area=analytical_initial_area,
            show_plots=True  # NEW
        )

    # Print statistics
    print("\n" + "="*80)
    print("SIMULATION STATISTICS")
    print("="*80)

    print(f"\nComputation Performance:")
    print(f"  Total simulation time: {elapsed_time:.4f} seconds")
    print(f"  Time steps computed: {n_total_steps}")
    print(f"  Snapshots saved: {len(t_history)}")
    print(f"  Time per step: {elapsed_time/n_total_steps*1000:.4f} ms")
    print(f"  Contour extraction time: {extraction_time:.4f} seconds")
    print(f"  Time per extraction: {extraction_time/len(G_history)*1000:.2f} ms")
    print(f"  Total elapsed time: {elapsed_time + extraction_time:.4f} seconds")

    print(f"\nRadius Accuracy:")
    print(f"  Maximum absolute error: {np.max(radius_error):.8f}")
    print(f"  Maximum relative error: {np.max(relative_radius_error):.4f}%")
    print(f"  Mean absolute error: {np.mean(radius_error):.8f}")
    print(f"  Mean relative error: {np.mean(relative_radius_error):.4f}%")
    print(f"  RMS error: {np.sqrt(np.mean(radius_error**2)):.8f}")
    print(f"  Final radius error: {radius_error[-1]:.8f} ({relative_radius_error[-1]:.4f}%)")

    print(f"\nFlame Surface Area Accuracy (Marching Squares):")
    print(f"  Maximum absolute error: {np.max(area_error):.8f}")
    print(f"  Maximum relative error: {np.max(relative_area_error):.4f}%")
    print(f"  Mean absolute error: {np.mean(area_error):.8f}")
    print(f"  Mean relative error: {np.mean(relative_area_error):.4f}%")
    print(f"  RMS error: {np.sqrt(np.mean(area_error**2)):.8f}")
    print(f"  Final area error: {area_error[-1]:.8f} ({relative_area_error[-1]:.4f}%)")

    print(f"\nContour Topology Statistics:")
    for t, stats in contour_stats:
        print(f"  t = {t:.3f}s:")
        print(f"    Contours: {stats['num_contours']} "
              f"(closed: {stats['num_closed']}, open: {stats['num_open']})")
        print(f"    Total length: {stats['total_length']:.8f}")

    print(f"\nFinal State (t = {t_final}s):")
    print(f"  Analytical radius: {analytical_radii[-1]:.8f}")
    print(f"  Numerical radius: {numerical_radii[-1]:.8f}")
    print(f"  Analytical surface area: {analytical_areas[-1]:.8f}")
    print(f"  Numerical surface area: {numerical_areas[-1]:.8f}")

    print(f"\nDomain Information:")
    print(f"  Grid resolution: {nx} × {ny} = {nx*ny} points")
    print(f"  Domain size: [{0}, {Lx}] × [{0}, {Ly}]")
    print(f"  Grid spacing: dx = {solver.dx:.6f}, dy = {solver.dy:.6f}")
    print(f"  CFL number: {S_L * dt / min(solver.dx, solver.dy):.4f}")

    print("\n" + "="*80)

    # Visualization (only if verbose)
    if verbose:
        create_plots(solver, G_history, t_history, R0, S_L, x_center, y_center,
                     u_x, u_y, time_scheme, t_final, use_reinit,
                     numerical_radii, numerical_areas, analytical_radii, analytical_areas,
                     radius_error, area_error, relative_area_error, analytical_initial_area)

    return radius_error, elapsed_time


if __name__ == "__main__":
    import sys

    # Default parameters
    scheme = 'rk2'
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

test_expanding_circle(t_final=t_final, time_scheme=scheme, use_reinit=use_reinit, verbose=True)