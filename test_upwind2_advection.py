#!/usr/bin/env python3
"""
Test script to verify second-order upwind spatial discretization scheme.
Tests pure advection (translation) problem where spatial discretization matters.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D

def run_advection_test(spatial_scheme='upwind', nx=101, t_final=0.2):
    """
    Run advection test with specified spatial scheme.
    Pure translation of Gaussian bump with constant velocity.

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
    Lx, Ly = 1.0, 1.0

    # Constant velocity field (diagonal flow)
    u_mag = 1.0
    angle = np.pi / 4  # 45 degrees
    u_x = u_mag * np.cos(angle) * np.ones((ny, nx))
    u_y = u_mag * np.sin(angle) * np.ones((ny, nx))

    # Zero flame speed (pure advection)
    S_L = 0.0

    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x, u_y)

    # Initial condition: Gaussian bump centered at (0.3, 0.3)
    x0, y0 = 0.3, 0.3
    sigma = 0.1

    x = solver.x
    y = solver.y
    X, Y = np.meshgrid(x, y)

    G0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    # Time stepping - CFL-limited
    dx = Lx / (nx - 1)
    cfl = 0.3
    dt = cfl * dx / u_mag

    # Solve
    G_hist, t_hist = solver.solve(
        G0, t_final, dt,
        save_interval=max(1, int(t_final/(10*dt))),
        time_scheme='rk3',
        spatial_scheme=spatial_scheme,
        reinit_interval=0,
        reinit_method='none'
    )

    # Final solution
    G_final = G_hist[-1]

    # Analytical solution: translated Gaussian
    x_final = x0 + u_mag * np.cos(angle) * t_final
    y_final = y0 + u_mag * np.sin(angle) * t_final

    # Handle periodic boundary (if translation goes outside domain)
    x_final = x_final % Lx
    y_final = y_final % Ly

    G_analytical = np.exp(-((X - x_final)**2 + (Y - y_final)**2) / (2 * sigma**2))

    # Compute L2 error
    error = np.sqrt(np.mean((G_final - G_analytical)**2))

    # Also compute L-infinity error
    error_inf = np.max(np.abs(G_final - G_analytical))

    return error, error_inf, G_final, G_analytical

def main():
    """Compare accuracy of different spatial schemes on pure advection."""

    print("Testing spatial discretization schemes on pure advection problem")
    print("(Gaussian bump translating with constant velocity)")
    print("=" * 75)

    # Test parameters
    resolutions = [51, 101, 201, 401]
    schemes = ['upwind', 'upwind2', 'weno5']
    t_final = 0.2

    # Store errors
    errors_l2 = {scheme: [] for scheme in schemes}
    errors_inf = {scheme: [] for scheme in schemes}

    # Run tests
    for nx in resolutions:
        print(f"\nResolution: {nx}x{nx}")
        print("-" * 50)

        for scheme in schemes:
            print(f"  {scheme:8s}...", end=' ', flush=True)
            error_l2, error_inf, _, _ = run_advection_test(scheme, nx, t_final)
            errors_l2[scheme].append(error_l2)
            errors_inf[scheme].append(error_inf)
            print(f"L2 = {error_l2:.6e}, L∞ = {error_inf:.6e}")

    # Compute convergence rates
    print("\n" + "=" * 75)
    print("Convergence rates (L2 error):")
    print("-" * 50)

    for scheme in schemes:
        print(f"\n{scheme}:")
        rates = []
        for i in range(len(resolutions) - 1):
            if errors_l2[scheme][i] > 0 and errors_l2[scheme][i+1] > 0:
                rate = np.log2(errors_l2[scheme][i] / errors_l2[scheme][i+1])
                rates.append(rate)
                print(f"  {resolutions[i]:3d} -> {resolutions[i+1]:3d}: rate = {rate:.2f}")
        if rates:
            avg_rate = np.mean(rates[1:]) if len(rates) > 1 else rates[0]
            print(f"  Average order of accuracy: {avg_rate:.2f}")

    # Plot results
    fig = plt.figure(figsize=(15, 5))

    # L2 Error vs resolution
    ax1 = plt.subplot(1, 3, 1)
    dx_values = [1.0/(nx-1) for nx in resolutions]

    markers = {'upwind': 'o', 'upwind2': 's', 'weno5': '^'}
    colors = {'upwind': 'C0', 'upwind2': 'C1', 'weno5': 'C2'}

    for scheme in schemes:
        ax1.loglog(dx_values, errors_l2[scheme],
                   marker=markers[scheme], color=colors[scheme],
                   label=scheme, linewidth=2, markersize=8)

    # Add reference lines
    dx_ref = np.array(dx_values[:3])
    ax1.loglog(dx_ref, 0.5 * dx_ref, 'k--', alpha=0.4, linewidth=1, label='O(Δx)')
    ax1.loglog(dx_ref, 0.05 * dx_ref**2, 'k:', alpha=0.4, linewidth=1, label='O(Δx²)')

    ax1.set_xlabel('Grid spacing Δx', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Spatial Scheme Accuracy (L2)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # L-infinity Error vs resolution
    ax2 = plt.subplot(1, 3, 2)
    for scheme in schemes:
        ax2.loglog(dx_values, errors_inf[scheme],
                   marker=markers[scheme], color=colors[scheme],
                   label=scheme, linewidth=2, markersize=8)

    ax2.loglog(dx_ref, 0.5 * dx_ref, 'k--', alpha=0.4, linewidth=1, label='O(Δx)')
    ax2.loglog(dx_ref, 0.05 * dx_ref**2, 'k:', alpha=0.4, linewidth=1, label='O(Δx²)')

    ax2.set_xlabel('Grid spacing Δx', fontsize=12)
    ax2.set_ylabel('L∞ Error', fontsize=12)
    ax2.set_title('Spatial Scheme Accuracy (L∞)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Visual comparison at medium resolution
    ax3 = plt.subplot(1, 3, 3)
    nx = 101

    scheme_labels = []
    scheme_errors = []

    for scheme in schemes:
        error_l2, error_inf, G_final, G_analytical = run_advection_test(scheme, nx, t_final)
        scheme_labels.append(scheme)
        scheme_errors.append(error_l2)

    # Bar plot of errors
    x_pos = np.arange(len(schemes))
    bars = ax3.bar(x_pos, scheme_errors, color=[colors[s] for s in schemes], alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(schemes)
    ax3.set_ylabel('L2 Error', fontsize=12)
    ax3.set_title(f'Error Comparison at {nx}×{nx}', fontsize=13, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, err in zip(bars, scheme_errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height*1.2,
                f'{err:.2e}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/nijso/Downloads/GEQ/spatial_scheme_advection_test.png', dpi=150, bbox_inches='tight')
    print(f"\n\nPlot saved to: spatial_scheme_advection_test.png")
    plt.show()

if __name__ == '__main__':
    main()
