"""
Test case for 2D G-equation solver: flame stabilization with vertical flow.
Initial condition: horizontal line separating burnt (G=-1, above) and unburnt (G=1, below) regions.
Flow: vertical inlet velocity u_y=0.8, horizontal velocity u_x=0.
Burning velocity: S_L=0.4
Expected flame velocity: U = u_y - S_L = 0.4 (upward)

This tests flame-flow interaction where the flame propagates downward (negative S_L direction)
but is carried upward by the flow, resulting in a net upward motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D
from plotting_utils import plot_contours_comparison, plot_surface_3d
from contour_utils import get_contour_statistics, compute_contour_length
import time

def initial_horizontal_line(X, Y, y_flame):
    """
    Create initial level set function for a horizontal line.
    
    G < 0: burnt region (above the line)
    G > 0: unburnt region (below the line)
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
    # Signed distance: negative above, positive below
    G = Y - y_flame
    return G

def analytical_flame_position(t, y0, U):
    """
    Analytical solution for flame position.
    
    The flame moves with velocity U = u_y - S_L
    y_flame(t) = y0 + U * t
    
    Parameters:
    -----------
    t : float or ndarray
        Time
    y0 : float
        Initial flame position
    U : float
        Net flame velocity (u_y - S_L)
        
    Returns:
    --------
    y : float or ndarray
        Flame position at time t
    """
    return y0 + U * t

def extract_flame_position(G, X, Y):
    """
    Extract the average y-position of the flame (G=0 contour).
    
    Parameters:
    -----------
    G : ndarray
        Level set function
    X : ndarray
        X coordinates
    Y : ndarray
        Y coordinates
        
    Returns:
    --------
    y_mean : float
        Mean y-position of the flame
    """
    ny, nx = G.shape
    
    # Find zero crossings in vertical direction
    y_positions = []
    
    for i in range(nx):
        for j in range(ny - 1):
            G1 = G[j, i]
            G2 = G[j + 1, i]
            
            # Check for zero crossing
            if G1 * G2 <= 0 and G1 != G2:
                # Linear interpolation
                alpha = -G1 / (G2 - G1)
                y_cross = Y[j, i] + alpha * (Y[j + 1, i] - Y[j, i])
                y_positions.append(y_cross)
    
    if len(y_positions) > 0:
        return np.mean(y_positions)
    else:
        return np.nan

def create_plots(solver, G_history, t_history, y0, U, time_scheme, t_final, use_reinit,
                 numerical_positions, analytical_positions, position_error):
    """
    Create all visualization plots for the flame stabilization test.
    
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
    U : float
        Net flame velocity
    time_scheme : str
        Time discretization scheme
    t_final : float
        Final simulation time
    use_reinit : bool
        Whether reinitialization was used
    numerical_positions : array
        Numerical flame positions over time
    analytical_positions : array
        Analytical flame positions over time
    position_error : array
        Position errors over time
    """
    print("\nCreating visualizations...")
    
    suffix = f"_reinit" if use_reinit else "_no_reinit"
    
    # Plot 1: Contour plots at selected times (custom for this test)
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
        
        # Plot analytical flame position
        y_analytical = analytical_flame_position(t, y0, U)
        ax.axhline(y=y_analytical, color='green', linestyle='--', linewidth=2, 
                   label=f'Analytical: y={{y_analytical:.3f}}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {{t:.3f}}s')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'flame_stabilization_contours_{{time_scheme}}_t{{t_final}}{{suffix}}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: flame_stabilization_contours_{{time_scheme}}_t{{t_final}}{{suffix}}.png")
    
    # Plot 2: Flame position vs time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2a: Position comparison
    ax1.plot(t_history, numerical_positions, 'b-', linewidth=2, marker='o',
             markersize=4, markevery=max(1, len(t_history)//20), label='Numerical')
    ax1.plot(t_history, analytical_positions, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Flame Position (y)', fontsize=12)
    ax1.set_title(f'Flame Position vs Time\n{{time_scheme.upper()}}, U = u_y - S_L = {{U:.2f}}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2b: Position error
    ax2.plot(t_history, position_error, 'k-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error in Flame Position', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(f'flame_stabilization_position_{{time_scheme}}_t{{t_final}}{{suffix}}.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: flame_stabilization_position_{{time_scheme}}_t{{t_final}}{{suffix}}.png")
    
    # Plot 3: 3D surface plot at final time
    G_final = G_history[-1]
    plot_surface_3d(solver, G_final, t_final, with_flow=True,
                   filename=f'flame_stabilization_surface_{{time_scheme}}_t{{t_final}}{{suffix}}.png')
    
    plt.show()
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80 + "\n")

def test_flame_stabilization(t_final=2.0, time_scheme='rk2', use_reinit=True, verbose=True):
    """
    Test the G-equation solver with flame stabilization in vertical flow.
    
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
    error : ndarray
        Absolute error in flame position over time
    elapsed_time : float
        Computation time in seconds
    """
    
    # Parameters
    nx = 101  # Number of grid points in x
    ny = 101  # Number of grid points in y
    Lx = 1.0  # Domain length in x
    Ly = 1.0  # Domain length in y
    S_L = 0.4  # Laminar flame speed (downward propagation)
    
    # Flow velocity
    u_x = 0.0  # No horizontal flow
    u_y = 0.8  # Vertical inlet velocity (upward)
    
    # Net flame velocity
    U = u_y - S_L  # Expected: 0.4 (upward)
    
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
        print("="*80)
        print(f"2D G-Equation Solver: Flame Stabilization Test (Vertical Flow)")
        print(f"Time Scheme: {{time_scheme.upper()}}")
        if use_reinit:
            print(f"Reinitialization: LOCAL every {{reinit_interval}} steps ({{reinit_method}})")
        else:
            print("Reinitialization: Disabled")
        print("="*80)
        print(f"Grid: {{nx}} x {{ny}}")
        print(f"Domain: [0, {{Lx}}] x [0, {{Ly}}]")
        print(f"Laminar flame speed S_L = {{S_L}}")
        print(f"Flow velocity u = ({{u_x}}, {{u_y}})")
        print(f"Net flame velocity U = u_y - S_L = {{U}}")
        print(f"Initial flame position y0 = {{y0}}")
        print(f"Time step dt = {{dt}}")
        print(f"Final time t_final = {{t_final}}")
        print(f"Save interval: every {{save_interval}} steps")
        print("="*80)
    
    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)
    
    # Create initial condition
    G_initial = initial_horizontal_line(solver.X, solver.Y, y0)
    
    # Verify initial flame position
    initial_position = extract_flame_position(G_initial, solver.X, solver.Y)
    
    if verbose:
        print(f"\nInitial flame position verification:")
        print(f"  Expected y0 = {{y0:.6f}}")
        print(f"  Computed y0 = {{initial_position:.6f}}")
        print(f"  Error = {{abs(initial_position - y0):.6f}}")
    
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
        print(f"\nCompleted {{n_total_steps}} time steps in {{elapsed_time:.2f}} seconds.")
        print(f"Saved {{len(t_history)}} snapshots.")
        print(f"Average time per step: {{elapsed_time/n_total_steps*1000:.4f}} ms")
    
    # Extract numerical flame position at each time step
    if verbose:
        print("\n" + "="*80)
        print("Extracting flame position...")
        print("="*80)
    
    numerical_positions = []
    extraction_start = time.time()
    
    for G in G_history:
        y_flame = extract_flame_position(G, solver.X, solver.Y)
        numerical_positions.append(y_flame)
    
    extraction_time = time.time() - extraction_start
    
    if verbose:
        print(f"Flame position extraction completed in {{extraction_time:.2f}} seconds.")
    
    # Compute analytical solution
    t_array = np.array(t_history)
    analytical_positions = analytical_flame_position(t_array, y0, U)
    
    # Compute errors
    position_error = np.abs(np.array(numerical_positions) - analytical_positions)
    relative_position_error = position_error / (analytical_positions - y0 + 1e-10) * 100
    
    # Print statistics
    print("\n" + "="*80)
    print("SIMULATION STATISTICS")
    print("="*80)
    
    print(f"\nComputation Performance:")
    print(f"  Total simulation time: {{elapsed_time:.4f}} seconds")
    print(f"  Time steps computed: {{n_total_steps}}")
    print(f"  Snapshots saved: {{len(t_history)}}")
    print(f"  Time per step: {{elapsed_time/n_total_steps*1000:.4f}} ms")
    print(f"  Extraction time: {{extraction_time:.4f}} seconds")
    print(f"  Total elapsed time: {{elapsed_time + extraction_time:.4f}} seconds")
    
    print(f"\nFlame Position Accuracy:")
    print(f"  Maximum absolute error: {{np.max(position_error):.8f}}")
    print(f"  Mean absolute error: {{np.mean(position_error):.8f}}")
    print(f"  RMS error: {{np.sqrt(np.mean(position_error**2)):.8f}}")
    print(f"  Final position error: {{position_error[-1]:.8f}}")
    
    print(f"\nFinal State (t = {{t_final}}s):")
    print(f"  Analytical position: {{analytical_positions[-1]:.8f}}")
    print(f"  Numerical position: {{numerical_positions[-1]:.8f}}")
    print(f"  Distance traveled: {{analytical_positions[-1] - y0:.8f}}")
    print(f"  Expected velocity U = {{U:.4f}}")
    print(f"  Computed velocity = {{(numerical_positions[-1] - y0) / t_final:.4f}}")
    
    print(f"\nDomain Information:")
    print(f"  Grid resolution: {{nx}} × {{ny}} = {{nx*ny}} points")
    print(f"  Domain size: [0, {{Lx}}] × [0, {{Ly}}]")
    print(f"  Grid spacing: dx = {{solver.dx:.6f}}, dy = {{solver.dy:.6f}}")
    print(f"  CFL number (convection): {{max(abs(u_x), abs(u_y)) * dt / min(solver.dx, solver.dy):.4f}}")
    print(f"  CFL number (propagation): {{S_L * dt / min(solver.dx, solver.dy):.4f}}")
    
    print("\n" + "="*80)
    
    # Visualization (only if verbose)
    if verbose:
        create_plots(solver, G_history, t_history, y0, U, time_scheme, t_final, use_reinit,
                     numerical_positions, analytical_positions, position_error)
    
    return position_error, elapsed_time


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
    
    print(f"\nRunning with: t_final={{t_final}}, scheme={{scheme}}, use_reinit={{use_reinit}}\n")
    
    test_flame_stabilization(t_final=t_final, time_scheme=scheme, use_reinit=use_reinit, verbose=True)