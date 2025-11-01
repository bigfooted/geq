"""
Test case for 2D G-equation solver: expanding circle with sharp initial condition.
IMPROVED VERSION with reinitialization and smoothing options.
Supports command-line argument for simulation time.
Updated to use latest function signatures with reinit_local parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import (GEquationSolver2D, compute_circle_radius, 
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


def test_expanding_circle_sharp_improved(t_final=1.5, time_scheme='euler', 
                                        reinit_interval=0, reinit_method='fast_marching',
                                        reinit_local=True, smooth_ic=False, 
                                        method_label='', verbose=True):
    """
    Test the G-equation solver with a sharp initial condition (no flow).
    IMPROVED VERSION with optional reinitialization and IC smoothing.
    
    Parameters:
    -----------
    t_final : float
        Final simulation time (default: 1.5)
    time_scheme : str
        Time discretization scheme: 'euler' or 'rk2'
    reinit_interval : int
        Reinitialize every reinit_interval steps (0 = disabled)
    reinit_method : str
        Reinitialization method: 'pde' or 'fast_marching' (default: 'fast_marching')
    reinit_local : bool
        Use local (narrow-band) reinitialization (default: True)
    smooth_ic : bool
        Apply smoothing to initial condition
    method_label : str
        Label for output files
    verbose : bool
        Print detailed output (default: True)
        
    Returns:
    --------
    error : ndarray
        Absolute error in radius over time
    elapsed_time : float
        Computation time in seconds
    numerical_radii : list
        Computed radii at each saved time step
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
    
    if verbose:
        print("="*60)
        print(f"2D G-Equation Solver: Sharp IC Test (IMPROVED)")
        print(f"Time Scheme: {time_scheme.upper()}")
        if reinit_interval > 0:
            reinit_type = "LOCAL" if reinit_local else "GLOBAL"
            print(f"Reinitialization: Every {reinit_interval} steps ({reinit_method}, {reinit_type})")
        else:
            print("Reinitialization: Disabled")
        print(f"IC Smoothing: {'Enabled' if smooth_ic else 'Disabled'}")
        print("="*60)
        print(f"Grid: {nx} x {ny}")
        print(f"Domain: [{0}, {Lx}] x [{0}, {Ly}]")
        print(f"Laminar flame speed S_L = {S_L}")
        print(f"Flow velocity u = ({u_x}, {u_y})")
        print(f"Initial radius R0 = {R0}")
        print(f"Time step dt = {dt}")
        print(f"Final time t_final = {t_final}")
        print(f"Save interval: every {save_interval} steps")
        print("="*60)
    
    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)
    
    # Create sharp initial condition
    G_initial = initial_solution_sharp(solver.X, solver.Y, x_center, y_center, R0)
    
    if verbose:
        print(f"\nInitial condition properties (before processing):")
        print(f"  G min = {G_initial.min():.6f}")
        print(f"  G max = {G_initial.max():.6f}")
        print(f"  G range = {G_initial.max() - G_initial.min():.6f}")
    
    # Verify initial radius
    initial_radius = compute_circle_radius(G_initial, solver.X, solver.Y, 
                                          x_center, y_center, solver.dx, solver.dy)
    if verbose:
        print(f"\nInitial radius verification:")
        print(f"  Expected R0 = {R0:.6f}")
        print(f"  Computed R0 = {initial_radius:.6f}")
        print(f"  Error = {abs(initial_radius - R0):.6f}")
    
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
        reinit_local=reinit_local,
        smooth_ic=smooth_ic
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    n_total_steps = int(t_final / dt)
    if verbose:
        print(f"Completed {n_total_steps} time steps in {elapsed_time:.2f} seconds.")
        print(f"Saved {len(t_history)} snapshots.")
        print(f"Average time per step: {elapsed_time/n_total_steps*1000:.3f} ms")
    
    # Extract numerical radius at each time step
    if verbose:
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
    
    if verbose:
        print(f"\nMaximum absolute error: {np.max(error):.6f}")
        print(f"Maximum relative error: {np.max(relative_error):.2f}%")
        print(f"Mean absolute error: {np.mean(error):.6f}")
        print(f"Mean relative error: {np.mean(relative_error):.2f}%")
    
    # Visualization (only if verbose)
    if verbose:
        print("\nCreating visualizations...")
        
        suffix = f"_sharp_improved_{method_label}_t{t_final}" if method_label else f"_sharp_improved_t{t_final}"
        
        # Plot 1: Contour plots at selected times
        plot_contours_comparison(solver, G_history, t_history, R0, S_L,
                                x_center_0=x_center, y_center_0=y_center,
                                u_x=u_x, u_y=u_y,
                                filename=f'contour_plots{suffix}.png')
        
        # Plot 2: Radius vs time comparison
        plot_radius_comparison(t_history, numerical_radii, analytical_radii,
                              with_flow=False,
                              filename=f'radius_comparison{suffix}.png')
        
        # Plot 3: 3D surface plot at final time
        G_final = G_history[-1]
        plot_surface_3d(solver, G_final, t_final, with_flow=False,
                       filename=f'surface_plot{suffix}.png')
        
        print(f"Saved plots with suffix: {suffix}")
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
    
    return error, elapsed_time, numerical_radii


def parse_arguments(args):
    """
    Parse command-line arguments.
    
    Parameters:
    -----------
    args : list
        Command-line arguments (typically sys.argv[1:])
        
    Returns:
    --------
    dict : Dictionary with parsed parameters
    """
    params = {
        't_final': 1.5,
        'scheme': 'rk2',
        'reinit': 0,
        'reinit_method': 'fast_marching',
        'reinit_local': True,
        'smooth': False
    }
    
    for arg in args:
        if arg.lower() in ['euler', 'rk2']:
            params['scheme'] = arg.lower()
        elif arg.startswith('reinit='):
            params['reinit'] = int(arg.split('=')[1])
        elif arg.startswith('method='):
            method = arg.split('=')[1].lower()
            if method in ['pde', 'fast_marching']:
                params['reinit_method'] = method
        elif arg.startswith('t=') or arg.startswith('time='):
            params['t_final'] = float(arg.split('=')[1])
        elif arg == 'smooth':
            params['smooth'] = True
        elif arg == 'global':
            params['reinit_local'] = False
        elif arg == 'local':
            params['reinit_local'] = True
    
    return params


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    params = parse_arguments(sys.argv[1:])
    
    # Build method label
    label = params['scheme']
    if params['reinit'] > 0:
        reinit_type = 'local' if params['reinit_local'] else 'global'
        label += f"_reinit{params['reinit']}_{params['reinit_method']}_{reinit_type}"
    if params['smooth']:
        label += "_smooth"
    
    print(f"\nRunning with:")
    print(f"  t_final = {params['t_final']}")
    print(f"  scheme = {params['scheme']}")
    print(f"  reinit_interval = {params['reinit']}")
    print(f"  reinit_method = {params['reinit_method']}")
    print(f"  reinit_local = {params['reinit_local']}")
    print(f"  smooth_ic = {params['smooth']}\n")
    
    test_expanding_circle_sharp_improved(
        t_final=params['t_final'],
        time_scheme=params['scheme'], 
        reinit_interval=params['reinit'],
        reinit_method=params['reinit_method'],
        reinit_local=params['reinit_local'],
        smooth_ic=params['smooth'],
        method_label=label,
        verbose=True
    )