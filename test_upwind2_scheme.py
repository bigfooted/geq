#!/usr/bin/env python3
"""
Test script to verify second-order upwind spatial discretization scheme.
Compares accuracy of upwind, upwind2, and weno5 schemes on expanding circle test.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D

def run_expanding_circle_test(spatial_scheme='upwind', nx=101, t_final=0.1):
    """
    Run expanding circle test with specified spatial scheme.

    Parameters:
    -----------
    spatial_scheme : str
        'upwind', 'upwind2', or 'weno5'
    nx : int
        Grid resolution
    t_final : float
        Final time

    Returns:
    --------
    error : float
        L2 error between numerical and analytical solution
    """
    # Domain
    ny = nx
    Lx, Ly = 0.2, 0.2

    # Initial flame position
    x0, y0 = Lx/2, Ly/2
    R0 = 0.02  # Initial radius

    # Flame speed
    S_L = 0.4

    # Zero velocity field
    u_x = np.zeros((ny, nx))
    u_y = np.zeros((ny, nx))

    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x, u_y)

    # Initial condition: circle
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    G0 = np.sqrt((X - x0)**2 + (Y - y0)**2) - R0

    # Time stepping
    dt = 0.0001

    # Solve
    G_hist, t_hist = solver.solve(
        G0, t_final, dt,
        save_interval=int(t_final/dt),
        time_scheme='rk3',
        spatial_scheme=spatial_scheme,
        reinit_interval=0,
        reinit_method='none'
    )

    # Final solution
    G_final = G_hist[-1]

    # Analytical solution: expanding circle
    R_final = R0 + S_L * t_final
    G_analytical = np.sqrt((X - x0)**2 + (Y - y0)**2) - R_final

    # Compute L2 error
    error = np.sqrt(np.mean((G_final - G_analytical)**2))

    return error, G_final, G_analytical

def main():
    """Compare accuracy of different spatial schemes."""

    print("Testing spatial discretization schemes on expanding circle problem")
    print("=" * 70)

    # Test parameters
    resolutions = [51, 101, 201]
    schemes = ['upwind', 'upwind2', 'weno5']
    t_final = 0.05

    # Store errors
    errors = {scheme: [] for scheme in schemes}

    # Run tests
    for nx in resolutions:
        print(f"\nResolution: {nx}x{nx}")
        print("-" * 40)

        for scheme in schemes:
            print(f"  Testing {scheme}...", end=' ', flush=True)
            error, _, _ = run_expanding_circle_test(scheme, nx, t_final)
            errors[scheme].append(error)
            print(f"L2 error = {error:.6e}")

    # Compute convergence rates
    print("\n" + "=" * 70)
    print("Convergence rates (log2 of error reduction):")
    print("-" * 40)

    for scheme in schemes:
        print(f"\n{scheme}:")
        for i in range(len(resolutions) - 1):
            if errors[scheme][i] > 0 and errors[scheme][i+1] > 0:
                rate = np.log2(errors[scheme][i] / errors[scheme][i+1])
                print(f"  {resolutions[i]} -> {resolutions[i+1]}: rate = {rate:.2f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error vs resolution
    dx_values = [0.2/nx for nx in resolutions]
    for scheme in schemes:
        ax1.loglog(dx_values, errors[scheme], 'o-', label=scheme, linewidth=2, markersize=8)

    # Add reference lines
    dx_ref = np.array(dx_values)
    ax1.loglog(dx_ref, 0.01 * dx_ref, 'k--', alpha=0.5, label='O(Δx)')
    ax1.loglog(dx_ref, 0.001 * dx_ref**2, 'k:', alpha=0.5, label='O(Δx²)')

    ax1.set_xlabel('Grid spacing Δx', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Spatial Scheme Accuracy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Visual comparison at finest resolution
    nx = resolutions[-1]
    for i, scheme in enumerate(schemes):
        error, G_final, G_analytical = run_expanding_circle_test(scheme, nx, t_final)

        # Plot difference from analytical
        diff = G_final - G_analytical

        if i == 0:
            im = ax2.imshow(diff, cmap='RdBu_r', vmin=-diff.max(), vmax=diff.max())
            ax2.set_title(f'{scheme} (error={error:.2e})', fontsize=12)
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, label='G - G_analytical')

    plt.tight_layout()
    plt.savefig('/home/nijso/Downloads/GEQ/spatial_scheme_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n\nPlot saved to: spatial_scheme_comparison.png")
    plt.show()

if __name__ == '__main__':
    main()
