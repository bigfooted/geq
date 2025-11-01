"""
Plotting utilities for 2D G-equation solver visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_contours_comparison(solver, G_history, t_history, R0, S_L, 
                             x_center_0=None, y_center_0=None, u_x=0.0, u_y=0.0,
                             numerical_centers=None, filename='contour_plots.png'):
    """
    Create contour plots comparing numerical and analytical solutions at selected times.
    
    Parameters:
    -----------
    solver : GEquationSolver2D
        The solver instance
    G_history : list
        List of G fields at different times
    t_history : list
        List of time values
    R0 : float
        Initial radius
    S_L : float
        Laminar flame speed
    x_center_0 : float, optional
        Initial x-center (for moving flames)
    y_center_0 : float, optional
        Initial y-center (for moving flames)
    u_x : float, optional
        Flow velocity in x-direction
    u_y : float, optional
        Flow velocity in y-direction
    numerical_centers : tuple of lists, optional
        (x_centers, y_centers) for moving flames
    filename : str
        Output filename
    """
    from g_equation_solver import analytical_radius, analytical_center
    
    # Default center for stationary flames
    if x_center_0 is None:
        x_center_0 = solver.Lx / 2.0
    if y_center_0 is None:
        y_center_0 = solver.Ly / 2.0
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Select 6 time instances
    n_plots = 6
    indices = np.linspace(0, len(G_history)-1, n_plots, dtype=int)
    
    # Determine global min/max for consistent color scale
    G_min = min([G.min() for G in G_history])
    G_max = max([G.max() for G in G_history])
    levels = np.linspace(G_min, G_max, 51)
    
    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        G = G_history[ax_idx]
        t = t_history[ax_idx]
        
        # Contour plot with full domain coverage
        contourf = ax.contourf(solver.X, solver.Y, G, levels=levels, 
                              cmap='RdBu_r', extend='both')
        
        # Highlight zero level set (flame surface) - Numerical solution
        ax.contour(solver.X, solver.Y, G, levels=[0], colors='black', 
                  linewidths=2.5, linestyles='solid', label='Numerical (G=0)')
        
        # Analytical circle
        theta = np.linspace(0, 2*np.pi, 100)
        R_analytical = analytical_radius(t, R0, S_L)
        x_c_analytical, y_c_analytical = analytical_center(t, x_center_0, y_center_0, u_x, u_y)
        x_circle = x_c_analytical + R_analytical * np.cos(theta)
        y_circle = y_c_analytical + R_analytical * np.sin(theta)
        ax.plot(x_circle, y_circle, 'g--', linewidth=2, label='Analytical')
        
        # Mark centers for moving flames
        if numerical_centers is not None:
            numerical_x_centers, numerical_y_centers = numerical_centers
            ax.plot(numerical_x_centers[ax_idx], numerical_y_centers[ax_idx], 
                   'bo', markersize=6, label='Numerical center')
            ax.plot(x_c_analytical, y_c_analytical, 'g^', markersize=6, 
                   label='Analytical center')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {t:.3f} s')
        ax.set_aspect('equal')
        ax.set_xlim([0, solver.Lx])
        ax.set_ylim([0, solver.Ly])
        
        if numerical_centers is not None:
            ax.legend(loc='upper left', fontsize=7)
        else:
            ax.legend(loc='upper right', fontsize=9)
        
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(contourf, ax=ax, label='G')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_radius_comparison(t_history, numerical_radii, analytical_radii, 
                           with_flow=False, filename='radius_comparison.png'):
    """
    Plot radius vs time comparison and error.
    
    Parameters:
    -----------
    t_history : list or array
        Time values
    numerical_radii : list or array
        Numerical radius values
    analytical_radii : list or array
        Analytical radius values
    with_flow : bool
        Whether this is for a moving flame case
    filename : str
        Output filename
    """
    error = np.abs(np.array(numerical_radii) - np.array(analytical_radii))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Radius comparison
    ax1.plot(t_history, numerical_radii, 'b-', linewidth=2, label='Numerical')
    ax1.plot(t_history, analytical_radii, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Radius', fontsize=12)
    
    if with_flow:
        ax1.set_title('Flame Radius vs Time (With Flow)', fontsize=14, fontweight='bold')
    else:
        ax1.set_title('Flame Radius vs Time (No Flow)', fontsize=14, fontweight='bold')
    
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Error plot
    ax2.plot(t_history, error, 'k-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error in Radius', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_center_comparison(t_history, numerical_x_centers, numerical_y_centers,
                           analytical_x_centers, analytical_y_centers,
                           filename='center_comparison.png'):
    """
    Plot center position vs time comparison.
    
    Parameters:
    -----------
    t_history : list or array
        Time values
    numerical_x_centers : list or array
        Numerical x-center values
    numerical_y_centers : list or array
        Numerical y-center values
    analytical_x_centers : list or array
        Analytical x-center values
    analytical_y_centers : list or array
        Analytical y-center values
    filename : str
        Output filename
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # X-center comparison
    ax1.plot(t_history, numerical_x_centers, 'b-', linewidth=2, label='Numerical')
    ax1.plot(t_history, analytical_x_centers, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('X-Center', fontsize=12)
    ax1.set_title('Flame Center X-Position vs Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Y-center comparison
    ax2.plot(t_history, numerical_y_centers, 'b-', linewidth=2, label='Numerical')
    ax2.plot(t_history, analytical_y_centers, 'r--', linewidth=2, label='Analytical')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Y-Center', fontsize=12)
    ax2.set_title('Flame Center Y-Position vs Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_trajectory(numerical_x_centers, numerical_y_centers,
                   analytical_x_centers, analytical_y_centers,
                   x_center_0, y_center_0, filename='trajectory.png'):
    """
    Plot flame center trajectory in x-y plane.
    
    Parameters:
    -----------
    numerical_x_centers : list or array
        Numerical x-center values
    numerical_y_centers : list or array
        Numerical y-center values
    analytical_x_centers : list or array
        Analytical x-center values
    analytical_y_centers : list or array
        Analytical y-center values
    x_center_0 : float
        Initial x-center
    y_center_0 : float
        Initial y-center
    filename : str
        Output filename
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.plot(numerical_x_centers, numerical_y_centers, 'b-', linewidth=2, 
           marker='o', markersize=4, label='Numerical')
    ax.plot(analytical_x_centers, analytical_y_centers, 'r--', linewidth=2, 
           marker='s', markersize=4, label='Analytical')
    ax.plot(x_center_0, y_center_0, 'go', markersize=10, label='Initial position')
    
    ax.set_xlabel('X-Center', fontsize=12)
    ax.set_ylabel('Y-Center', fontsize=12)
    ax.set_title('Flame Center Trajectory', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_surface_3d(solver, G_final, t_final, with_flow=False, 
                   filename='surface_plot.png'):
    """
    Create 3D surface plot of the level set function.
    
    Parameters:
    -----------
    solver : GEquationSolver2D
        The solver instance
    G_final : ndarray
        Final G field
    t_final : float
        Final time
    with_flow : bool
        Whether this is for a moving flame case
    filename : str
        Output filename
    """
    fig = plt.figure(figsize=(12, 9))
    ax3d = fig.add_subplot(111, projection='3d')
    
    surf = ax3d.plot_surface(solver.X, solver.Y, G_final, cmap='viridis', 
                             edgecolor='none', alpha=0.9)
    
    ax3d.set_xlabel('x', fontsize=12)
    ax3d.set_ylabel('y', fontsize=12)
    ax3d.set_zlabel('G', fontsize=12)
    
    if with_flow:
        ax3d.set_title(f'Level Set Function G at t = {t_final} s (With Flow)', 
                       fontsize=14, fontweight='bold')
    else:
        ax3d.set_title(f'Level Set Function G at t = {t_final} s (No Flow)', 
                       fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5, label='G')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_radius_and_center_combined(t_history, numerical_radii, analytical_radii,
                                   numerical_x_centers, numerical_y_centers,
                                   analytical_x_centers, analytical_y_centers,
                                   filename='radius_center_comparison.png'):
    """
    Combined plot showing radius and center positions vs time.
    
    Parameters:
    -----------
    t_history : list or array
        Time values
    numerical_radii : list or array
        Numerical radius values
    analytical_radii : list or array
        Analytical radius values
    numerical_x_centers : list or array
        Numerical x-center values
    numerical_y_centers : list or array
        Numerical y-center values
    analytical_x_centers : list or array
        Analytical x-center values
    analytical_y_centers : list or array
        Analytical y-center values
    filename : str
        Output filename
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Radius comparison
    ax1.plot(t_history, numerical_radii, 'b-', linewidth=2, label='Numerical')
    ax1.plot(t_history, analytical_radii, 'r--', linewidth=2, label='Analytical')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Radius', fontsize=12)
    ax1.set_title('Flame Radius vs Time (With Flow)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # X-center comparison
    ax2.plot(t_history, numerical_x_centers, 'b-', linewidth=2, label='Numerical')
    ax2.plot(t_history, analytical_x_centers, 'r--', linewidth=2, label='Analytical')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('X-Center', fontsize=12)
    ax2.set_title('Flame Center X-Position vs Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Y-center comparison
    ax3.plot(t_history, numerical_y_centers, 'b-', linewidth=2, label='Numerical')
    ax3.plot(t_history, analytical_y_centers, 'r--', linewidth=2, label='Analytical')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Y-Center', fontsize=12)
    ax3.set_title('Flame Center Y-Position vs Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")