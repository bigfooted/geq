#!/usr/bin/env python3
"""
Diagnose why expanding ellipse shows only marginal improvement with higher-order gradient scheme.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D

def main():
    # Setup
    nx, ny = 121, 121
    Lx, Ly = 1.0, 1.0
    S_L = 0.4
    u_x = 0.0
    u_y = 0.0

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x, u_y)

    x = solver.x
    y = solver.y
    X, Y = np.meshgrid(x, y)

    # Initial condition: ellipse with semi-axes a=0.15, b=0.08
    a, b = 0.15, 0.08
    G0 = np.sqrt(((X-0.5)/a)**2 + ((Y-0.5)/b)**2) * min(a, b) - 1.0 * min(a, b)

    # Time parameters
    t_final = 0.3
    dx = Lx / (nx - 1)
    dt = 0.2 * dx / (S_L + 1e-10)

    print(f"Grid: {nx}×{ny}")
    print(f"dx = {dx:.6f}")
    print(f"dt = {dt:.6f}")
    print(f"t_final = {t_final}")
    print(f"Ellipse semi-axes: a={a}, b={b}, aspect ratio={a/b:.2f}")

    # ========================================================================
    # Analyze initial condition
    # ========================================================================
    print("\n" + "="*70)
    print("INITIAL CONDITION ANALYSIS")
    print("="*70)

    # Check gradient at different locations on ellipse
    grad_x_0 = np.gradient(G0, dx, axis=1)
    grad_y_0 = np.gradient(G0, dx, axis=0)
    grad_mag_0 = np.sqrt(grad_x_0**2 + grad_y_0**2)

    # Find points on ellipse interface (G ≈ 0)
    mask_interface = np.abs(G0) < 2*dx

    print(f"\nInitial gradient on interface:")
    grad_at_interface = grad_mag_0[mask_interface]
    print(f"  |∇G| mean: {np.mean(grad_at_interface):.6f}")
    print(f"  |∇G| std:  {np.std(grad_at_interface):.6f}")
    print(f"  |∇G| min:  {np.min(grad_at_interface):.6f}")
    print(f"  |∇G| max:  {np.max(grad_at_interface):.6f}")

    # Check specific points
    center_i, center_j = ny//2, nx//2

    # Major axis endpoints (left/right at y=0.5)
    right_j = center_j + int(a / dx)
    left_j = center_j - int(a / dx)

    # Minor axis endpoints (top/bottom at x=0.5)
    top_i = center_i - int(b / dx)
    bottom_i = center_i + int(b / dx)

    print(f"\nGradient at specific points (initial):")
    print(f"  Right (major axis):  |∇G| = {grad_mag_0[center_i, right_j]:.6f}")
    print(f"  Left (major axis):   |∇G| = {grad_mag_0[center_i, left_j]:.6f}")
    print(f"  Top (minor axis):    |∇G| = {grad_mag_0[top_i, center_j]:.6f}")
    print(f"  Bottom (minor axis): |∇G| = {grad_mag_0[bottom_i, center_j]:.6f}")

    # Check curvature variation
    # For ellipse: κ = 1/(a*b) * sqrt((b²sin²θ + a²cos²θ)³)
    # Max curvature at minor axis (θ=π/2): κ_max = a/b²
    # Min curvature at major axis (θ=0): κ_min = b/a²
    kappa_max = a / (b**2)
    kappa_min = b / (a**2)
    print(f"\nCurvature variation:")
    print(f"  κ_max (at minor axis): {kappa_max:.3f}")
    print(f"  κ_min (at major axis): {kappa_min:.3f}")
    print(f"  Ratio: {kappa_max/kappa_min:.1f}× variation")

    # ========================================================================
    # Run with both gradient schemes
    # ========================================================================

    results = {}

    for scheme in ['godunov', 'weno5']:
        print("\n" + "="*70)
        print(f"RUNNING WITH {scheme.upper()} GRADIENT SCHEME")
        print("="*70)

        G_hist, t_hist = solver.solve(
            G0.copy(), t_final, dt,
            save_interval=max(1, int(t_final/(10*dt))),
            time_scheme='rk3',
            spatial_scheme='weno5',
            gradient_scheme=scheme,
            reinit_interval=0,
            reinit_method='none'
        )

        G_final = G_hist[-1]

        # Compute gradient
        if scheme == 'godunov':
            grad_mag = solver.compute_gradient_magnitude(G_final)
        else:
            grad_mag = solver.compute_gradient_magnitude_weno5(G_final)

        # Analyze
        mask_interface = np.abs(G_final) < 3*dx
        grad_at_interface = grad_mag[mask_interface]

        print(f"\nFinal time t = {t_hist[-1]:.4f}")
        print(f"  |∇G| mean:  {np.mean(grad_at_interface):.6f}")
        print(f"  |∇G| std:   {np.std(grad_at_interface):.6f}")
        print(f"  |∇G| min:   {np.min(grad_at_interface):.6f}")
        print(f"  |∇G| max:   {np.max(grad_at_interface):.6f}")
        print(f"  |∇G| drift: {abs(np.mean(grad_at_interface) - 1.0):.6e}")

        # Check gradients at specific locations
        print(f"\nGradient at specific points (final):")
        print(f"  Right (major axis):  |∇G| = {grad_mag[center_i, right_j]:.6f}")
        print(f"  Left (major axis):   |∇G| = {grad_mag[center_i, left_j]:.6f}")
        print(f"  Top (minor axis):    |∇G| = {grad_mag[top_i, center_j]:.6f}")
        print(f"  Bottom (minor axis): |∇G| = {grad_mag[bottom_i, center_j]:.6f}")

        # Check if gradient is uniform around ellipse
        grad_variation = np.std(grad_at_interface) / np.mean(grad_at_interface)
        print(f"\nGradient non-uniformity (CV): {grad_variation:.3f}")

        grad_drift = abs(np.mean(grad_at_interface) - 1.0)

        results[scheme] = {
            'G_final': G_final,
            'grad_mag': grad_mag,
            'grad_mean': np.mean(grad_at_interface),
            'grad_std': np.std(grad_at_interface),
            'grad_drift': grad_drift,
            'grad_variation': grad_variation
        }

    # ========================================================================
    # Compare with perfect circle
    # ========================================================================

    print("\n" + "="*70)
    print("COMPARISON WITH CIRCLE (for reference)")
    print("="*70)

    # Run a circle with same area as ellipse
    area_ellipse = np.pi * a * b
    r_circle = np.sqrt(area_ellipse / np.pi)

    G0_circle = np.sqrt((X-0.5)**2 + (Y-0.5)**2) - r_circle

    print(f"\nCircle radius (same area): r = {r_circle:.4f}")

    results_circle = {}

    for scheme in ['godunov', 'weno5']:
        G_hist, t_hist = solver.solve(
            G0_circle.copy(), t_final, dt,
            save_interval=max(1, int(t_final/(10*dt))),
            time_scheme='rk3',
            spatial_scheme='weno5',
            gradient_scheme=scheme,
            reinit_interval=0,
            reinit_method='none'
        )

        G_final = G_hist[-1]

        if scheme == 'godunov':
            grad_mag = solver.compute_gradient_magnitude(G_final)
        else:
            grad_mag = solver.compute_gradient_magnitude_weno5(G_final)

        mask_interface = np.abs(G_final) < 3*dx
        grad_at_interface = grad_mag[mask_interface]

        drift = abs(np.mean(grad_at_interface) - 1.0)

        results_circle[scheme] = {
            'grad_mean': np.mean(grad_at_interface),
            'grad_drift': drift
        }

        print(f"{scheme.upper()}: drift = {drift:.6e}")

    circle_improvement = results_circle['godunov']['grad_drift'] / results_circle['weno5']['grad_drift']
    ellipse_improvement = results['godunov']['grad_drift'] / results['weno5']['grad_drift']

    print(f"\nCircle improvement:  {circle_improvement:.1f}×")
    print(f"Ellipse improvement: {ellipse_improvement:.1f}×")
    print(f"Ratio: Ellipse is {circle_improvement/ellipse_improvement:.1f}× less effective than circle")

    # ========================================================================
    # Detailed visualization
    # ========================================================================

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))

    # Row 1: Initial condition
    ax = axes[0, 0]
    im = ax.contourf(X, Y, G0, levels=20, cmap='coolwarm')
    ax.contour(X, Y, G0, levels=[0], colors='black', linewidths=2)
    ax.set_title("Initial G (Ellipse)", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.contourf(X, Y, grad_mag_0, levels=20, cmap='viridis')
    ax.contour(X, Y, G0, levels=[0], colors='black', linewidths=2)
    ax.set_title("Initial |∇G|", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    # Show gradient variation around ellipse boundary
    angles = np.linspace(0, 2*np.pi, 100)
    x_ellipse = 0.5 + a * np.cos(angles)
    y_ellipse = 0.5 + b * np.sin(angles)

    # Interpolate gradient magnitude along ellipse
    from scipy.interpolate import griddata
    points = np.column_stack((X.ravel(), Y.ravel()))
    grad_on_ellipse = griddata(points, grad_mag_0.ravel(),
                               np.column_stack((x_ellipse, y_ellipse)),
                               method='linear')

    ax.plot(angles * 180/np.pi, grad_on_ellipse, 'b-', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('|∇G|')
    ax.set_title('|∇G| around ellipse (initial)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 360])

    # Row 2: Godunov final
    scheme = 'godunov'
    r = results[scheme]

    ax = axes[1, 0]
    im = ax.contourf(X, Y, r['G_final'], levels=20, cmap='coolwarm')
    ax.contour(X, Y, r['G_final'], levels=[0], colors='black', linewidths=2)
    ax.set_title(f"Godunov: Final G", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.contourf(X, Y, r['grad_mag'], levels=np.linspace(0.5, 1.5, 21),
                     cmap='RdBu_r', extend='both')
    ax.contour(X, Y, r['G_final'], levels=[0], colors='black', linewidths=2)
    ax.set_title(f"Godunov: |∇G| (mean={r['grad_mean']:.3f})", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    grad_on_ellipse_god = griddata(points, r['grad_mag'].ravel(),
                                   np.column_stack((x_ellipse, y_ellipse)),
                                   method='linear')
    ax.plot(angles * 180/np.pi, grad_on_ellipse_god, 'b-', linewidth=2, label='Godunov')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('|∇G|')
    ax.set_title(f'Godunov: |∇G| around ellipse\n(CV={r["grad_variation"]:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 360])
    ax.set_ylim([0.5, 1.5])

    # Row 3: WENO5 final
    scheme = 'weno5'
    r = results[scheme]

    ax = axes[2, 0]
    im = ax.contourf(X, Y, r['G_final'], levels=20, cmap='coolwarm')
    ax.contour(X, Y, r['G_final'], levels=[0], colors='black', linewidths=2)
    ax.set_title(f"WENO5: Final G", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[2, 1]
    im = ax.contourf(X, Y, r['grad_mag'], levels=np.linspace(0.5, 1.5, 21),
                     cmap='RdBu_r', extend='both')
    ax.contour(X, Y, r['G_final'], levels=[0], colors='black', linewidths=2)
    ax.set_title(f"WENO5: |∇G| (mean={r['grad_mean']:.3f})", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[2, 2]
    grad_on_ellipse_weno = griddata(points, r['grad_mag'].ravel(),
                                    np.column_stack((x_ellipse, y_ellipse)),
                                    method='linear')
    ax.plot(angles * 180/np.pi, grad_on_ellipse_weno, 'r-', linewidth=2, label='WENO5')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('|∇G|')
    ax.set_title(f'WENO5: |∇G| around ellipse\n(CV={r["grad_variation"]:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 360])
    ax.set_ylim([0.5, 1.5])

    plt.tight_layout()
    plt.savefig('diagnose_ellipse.png', dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: diagnose_ellipse.png")

if __name__ == '__main__':
    main()
