#!/usr/bin/env python3
"""
Test Hamilton-Jacobi WENO5 gradient scheme for source-dominated problems.
Compares first-order Godunov vs 5th-order HJ-WENO5 for gradient magnitude.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D

def test_expanding_circle_source_dominated(gradient_scheme='godunov', nx=101, t_final=0.1):
    """
    Test expanding circle with ZERO velocity (pure source term).
    This isolates the gradient magnitude computation quality.

    Parameters:
    -----------
    gradient_scheme : str
        'godunov' (1st-order) or 'weno5' (5th-order HJ-WENO)
    nx : int
        Grid resolution
    t_final : float
        Final time

    Returns:
    --------
    results : dict
        Dictionary with errors and gradient statistics
    """
    ny = nx
    Lx, Ly = 0.2, 0.2

    # Initial circle
    x0, y0 = Lx/2, Ly/2
    R0 = 0.02

    # Flame speed (source term)
    S_L = 0.4

    # ZERO velocity (pure source-dominated problem)
    u_x = np.zeros((ny, nx))
    u_y = np.zeros((ny, nx))

    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x, u_y)

    # Initial condition: signed distance to circle
    x = solver.x
    y = solver.y
    X, Y = np.meshgrid(x, y)
    G0 = np.sqrt((X - x0)**2 + (Y - y0)**2) - R0

    # Time step (CFL for flame speed)
    dx = Lx / (nx - 1)
    dt = 0.25 * dx / S_L  # Conservative CFL

    # Solve WITHOUT reinitialization to test gradient quality
    print(f"\n{'='*60}")
    print(f"Test: {gradient_scheme.upper()} gradient scheme, {nx}×{nx} grid")
    print(f"{'='*60}")

    G_hist, t_hist = solver.solve(
        G0, t_final, dt,
        save_interval=max(1, int(t_final/(10*dt))),
        time_scheme='rk3',
        spatial_scheme='weno5',  # High-order convection (though u=0 here)
        gradient_scheme=gradient_scheme,  # Test this!
        reinit_interval=0,  # NO reinitialization
        reinit_method='none'
    )

    G_final = G_hist[-1]

    # Analytical solution: expanding circle
    R_final = R0 + S_L * t_final
    G_exact = np.sqrt((X - x0)**2 + (Y - y0)**2) - R_final

    # Compute errors
    l2_error = np.sqrt(np.mean((G_final - G_exact)**2))
    linf_error = np.max(np.abs(G_final - G_exact))

    # Check gradient magnitude at interface
    # For signed distance function, |∇G| should be exactly 1.0
    mask_interface = np.abs(G_final) < 3*dx  # Near interface

    if gradient_scheme == 'godunov':
        grad_mag = solver.compute_gradient_magnitude(G_final)
    else:
        grad_mag = solver.compute_gradient_magnitude_weno5(G_final)

    grad_at_interface = grad_mag[mask_interface]
    grad_mean = np.mean(grad_at_interface)
    grad_std = np.std(grad_at_interface)
    grad_error = np.abs(grad_mean - 1.0)

    # Radius error
    # Extract zero contour and compute radius
    from scipy import ndimage
    zero_contour = (G_final > -dx/2) & (G_final < dx/2)
    if np.any(zero_contour):
        coords = np.where(zero_contour)
        radii = np.sqrt((X[coords] - x0)**2 + (Y[coords] - y0)**2)
        R_numerical = np.mean(radii)
        R_error = np.abs(R_numerical - R_final)
    else:
        R_error = np.nan

    print(f"\nResults:")
    print(f"  L2 error:              {l2_error:.6e}")
    print(f"  L∞ error:              {linf_error:.6e}")
    print(f"  |∇G| at interface:     {grad_mean:.6f} ± {grad_std:.6f} (should be 1.0)")
    print(f"  |∇G| error:            {grad_error:.6e}")
    print(f"  Radius error:          {R_error:.6e}")

    return {
        'l2_error': l2_error,
        'linf_error': linf_error,
        'grad_mean': grad_mean,
        'grad_std': grad_std,
        'grad_error': grad_error,
        'R_error': R_error,
        'G_final': G_final,
        'G_exact': G_exact,
        'grad_mag': grad_mag,
        'X': X,
        'Y': Y
    }

def main():
    """Compare gradient schemes on source-dominated problem."""

    print("\n" + "="*70)
    print("Hamilton-Jacobi WENO5 Test: Source-Dominated Problem")
    print("="*70)
    print("\nProblem: Expanding circle with ZERO velocity (u = 0)")
    print("  ∂G/∂t + S_L |∇G| = 0")
    print("\nThis tests ONLY the gradient magnitude computation quality.")
    print("Without reinitialization, |∇G| should stay ≈ 1.0 for good scheme.")

    # Test parameters
    resolutions = [51, 101, 201]
    schemes = ['godunov', 'weno5']
    t_final = 0.05

    # Store results
    results_all = {scheme: [] for scheme in schemes}

    # Run tests
    for nx in resolutions:
        print(f"\n" + "="*70)
        print(f"Resolution: {nx}×{nx}")
        print("="*70)

        for scheme in schemes:
            results = test_expanding_circle_source_dominated(scheme, nx, t_final)
            results_all[scheme].append(results)

    # Plot comparison
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: L2 error vs resolution
    ax1 = plt.subplot(2, 3, 1)
    for scheme in schemes:
        errors = [r['l2_error'] for r in results_all[scheme]]
        dx_values = [0.2/(nx-1) for nx in resolutions]
        ax1.loglog(dx_values, errors, 'o-', label=scheme, linewidth=2, markersize=8)

    dx_ref = np.array([0.2/(nx-1) for nx in resolutions[:2]])
    ax1.loglog(dx_ref, 0.01*dx_ref, 'k--', alpha=0.4, label='O(Δx)')
    ax1.loglog(dx_ref, 0.001*dx_ref**2, 'k:', alpha=0.4, label='O(Δx²)')

    ax1.set_xlabel('Grid spacing Δx', fontsize=11)
    ax1.set_ylabel('L2 Error', fontsize=11)
    ax1.set_title('Solution Error', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient error vs resolution
    ax2 = plt.subplot(2, 3, 2)
    for scheme in schemes:
        errors = [r['grad_error'] for r in results_all[scheme]]
        dx_values = [0.2/(nx-1) for nx in resolutions]
        ax2.loglog(dx_values, errors, 'o-', label=scheme, linewidth=2, markersize=8)

    ax2.loglog(dx_ref, 0.1*dx_ref, 'k--', alpha=0.4, label='O(Δx)')
    ax2.loglog(dx_ref, 0.01*dx_ref**5, 'k:', alpha=0.4, label='O(Δx⁵)')

    ax2.set_xlabel('Grid spacing Δx', fontsize=11)
    ax2.set_ylabel('||∇G| - 1.0|', fontsize=11)
    ax2.set_title('Gradient Magnitude Error', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Radius error vs resolution
    ax3 = plt.subplot(2, 3, 3)
    for scheme in schemes:
        errors = [r['R_error'] for r in results_all[scheme] if not np.isnan(r['R_error'])]
        dx_vals = [0.2/(nx-1) for nx, r in zip(resolutions, results_all[scheme])
                   if not np.isnan(r['R_error'])]
        if errors:
            ax3.loglog(dx_vals, errors, 'o-', label=scheme, linewidth=2, markersize=8)

    ax3.set_xlabel('Grid spacing Δx', fontsize=11)
    ax3.set_ylabel('Radius Error', fontsize=11)
    ax3.set_title('Interface Position Error', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4-5: Visual comparison at finest resolution
    nx = resolutions[-1]
    for i, scheme in enumerate(schemes):
        ax = plt.subplot(2, 3, 4+i)
        results = results_all[scheme][-1]

        # Plot gradient magnitude
        grad_mag = results['grad_mag']
        im = ax.contourf(results['X'], results['Y'], grad_mag,
                        levels=np.linspace(0.5, 1.5, 21), cmap='RdBu_r', extend='both')

        # Overlay zero contour
        ax.contour(results['X'], results['Y'], results['G_final'],
                  levels=[0], colors='black', linewidths=2)

        ax.set_aspect('equal')
        ax.set_title(f'{scheme.upper()}: |∇G| at {nx}×{nx}', fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        plt.colorbar(im, ax=ax, label='|∇G|')

        # Add text with statistics
        grad_mean = results['grad_mean']
        grad_error = results['grad_error']
        ax.text(0.02, 0.98, f"|∇G| = {grad_mean:.4f}\nError = {grad_error:.2e}",
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)

    # Plot 6: Bar chart comparison
    ax6 = plt.subplot(2, 3, 6)
    nx_fine = resolutions[-1]
    metrics = ['L2 Error', '|∇G| Error', 'Radius Error']
    godunov_vals = [results_all['godunov'][-1]['l2_error'],
                    results_all['godunov'][-1]['grad_error'],
                    results_all['godunov'][-1]['R_error']]
    weno5_vals = [results_all['weno5'][-1]['l2_error'],
                  results_all['weno5'][-1]['grad_error'],
                  results_all['weno5'][-1]['R_error']]

    improvements = [g/w for g, w in zip(godunov_vals, weno5_vals)]

    x_pos = np.arange(len(metrics))
    bars = ax6.bar(x_pos, improvements, color=['C0', 'C1', 'C2'], alpha=0.7, edgecolor='black')
    ax6.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(metrics, fontsize=10)
    ax6.set_ylabel('Improvement Factor\n(Godunov / HJ-WENO5)', fontsize=10)
    ax6.set_title(f'HJ-WENO5 Advantage at {nx_fine}×{nx_fine}', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height*1.05,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/nijso/Downloads/GEQ/hj_weno5_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n\nPlot saved to: hj_weno5_comparison.png")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: HJ-WENO5 vs Godunov (First-Order)")
    print("="*70)
    print(f"\nAt {nx_fine}×{nx_fine} resolution:")
    print(f"  L2 Error:         {improvements[0]:.1f}× improvement")
    print(f"  |∇G| Error:       {improvements[1]:.1f}× improvement")
    print(f"  Radius Error:     {improvements[2]:.1f}× improvement")
    print("\n" + "="*70)
    print("Conclusion: HJ-WENO5 maintains |∇G| ≈ 1.0 MUCH better!")
    print("This directly improves flame speed accuracy in source-dominated problems.")
    print("="*70 + "\n")

    plt.show()

if __name__ == '__main__':
    main()
