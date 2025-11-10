#!/usr/bin/env python3
"""
Diagnose why merging circles test fails with higher-order gradient scheme.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D

def main():
    # Setup
    nx, ny = 121, 121
    Lx, Ly = 1.0, 1.0
    S_L = 0.5
    u_x = 0.0
    u_y = 0.0

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x, u_y)

    x = solver.x
    y = solver.y
    X, Y = np.meshgrid(x, y)

    # Initial condition: two circles
    d1 = np.sqrt((X-0.35)**2 + (Y-0.5)**2) - 0.12
    d2 = np.sqrt((X-0.65)**2 + (Y-0.5)**2) - 0.12
    G0 = np.minimum(d1, d2)

    # Time parameters
    t_final = 0.25
    dx = Lx / (nx - 1)
    dt = 0.2 * dx / (S_L + 1e-10)

    print(f"Grid: {nx}×{ny}")
    print(f"dx = {dx:.6f}")
    print(f"dt = {dt:.6f}")
    print(f"t_final = {t_final}")
    print(f"Number of steps: {int(t_final/dt)}")

    # ========================================================================
    # Analyze initial condition
    # ========================================================================
    print("\n" + "="*70)
    print("INITIAL CONDITION ANALYSIS")
    print("="*70)

    # Check gradient at the saddle point between circles
    grad_x_0 = np.gradient(G0, dx, axis=1)
    grad_y_0 = np.gradient(G0, dx, axis=0)
    grad_mag_0 = np.sqrt(grad_x_0**2 + grad_y_0**2)

    # Find the saddle point (where circles will merge)
    saddle_i, saddle_j = ny//2, nx//2

    print(f"\nAt saddle point (x=0.5, y=0.5):")
    print(f"  G = {G0[saddle_i, saddle_j]:.6f}")
    print(f"  |∇G| = {grad_mag_0[saddle_i, saddle_j]:.6f}")
    print(f"  ∂G/∂x = {grad_x_0[saddle_i, saddle_j]:.6f}")
    print(f"  ∂G/∂y = {grad_y_0[saddle_i, saddle_j]:.6f}")

    # Check gradient quality across interface
    mask_interface = np.abs(G0) < 3*dx
    grad_at_interface = grad_mag_0[mask_interface]
    print(f"\nAt interface (|G| < 3Δx):")
    print(f"  |∇G| mean: {np.mean(grad_at_interface):.6f}")
    print(f"  |∇G| std:  {np.std(grad_at_interface):.6f}")
    print(f"  |∇G| min:  {np.min(grad_at_interface):.6f}")
    print(f"  |∇G| max:  {np.max(grad_at_interface):.6f}")

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

        # Check for merger - did circles merge?
        # If merged, there should be a single connected contour
        G_at_saddle = G_final[saddle_i, saddle_j]
        print(f"\n  G at saddle point: {G_at_saddle:.6f}")
        if G_at_saddle > 0:
            print(f"  → Circles have MERGED (saddle is now inside)")
        else:
            print(f"  → Circles still SEPARATE")

        results[scheme] = {
            'G_final': G_final,
            'grad_mag': grad_mag,
            'grad_mean': np.mean(grad_at_interface),
            'grad_std': np.std(grad_at_interface),
            'merged': G_at_saddle > 0
        }

    # ========================================================================
    # Detailed visualization
    # ========================================================================

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))

    # Row 1: Initial condition
    ax = axes[0, 0]
    im = ax.contourf(X, Y, G0, levels=20, cmap='coolwarm')
    ax.contour(X, Y, G0, levels=[0], colors='black', linewidths=2)
    ax.plot(0.5, 0.5, 'r*', markersize=15, label='Saddle')
    ax.set_title("Initial G", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    ax.legend()

    ax = axes[0, 1]
    im = ax.contourf(X, Y, grad_mag_0, levels=20, cmap='viridis')
    ax.contour(X, Y, G0, levels=[0], colors='black', linewidths=2)
    ax.set_title("Initial |∇G|", fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    # Show gradient along line y=0.5 (through saddle)
    line_j = ny//2
    ax.plot(x, grad_mag_0[line_j, :], 'b-', linewidth=2, label='|∇G|')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    ax.axvline(0.5, color='r', linestyle=':', alpha=0.5, label='Saddle')
    ax.set_xlabel('x')
    ax.set_ylabel('|∇G|')
    ax.set_title('|∇G| along y=0.5 (initial)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Godunov final
    scheme = 'godunov'
    r = results[scheme]

    ax = axes[1, 0]
    im = ax.contourf(X, Y, r['G_final'], levels=20, cmap='coolwarm')
    ax.contour(X, Y, r['G_final'], levels=[0], colors='black', linewidths=2)
    ax.plot(0.5, 0.5, 'r*', markersize=15)
    ax.set_title(f"Godunov: Final G\n(merged={r['merged']})", fontweight='bold')
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
    ax.plot(x, r['grad_mag'][line_j, :], 'b-', linewidth=2, label='|∇G|')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    ax.axvline(0.5, color='r', linestyle=':', alpha=0.5, label='Saddle')
    ax.set_xlabel('x')
    ax.set_ylabel('|∇G|')
    ax.set_title('Godunov: |∇G| along y=0.5', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 2])

    # Row 3: WENO5 final
    scheme = 'weno5'
    r = results[scheme]

    ax = axes[2, 0]
    im = ax.contourf(X, Y, r['G_final'], levels=20, cmap='coolwarm')
    ax.contour(X, Y, r['G_final'], levels=[0], colors='black', linewidths=2)
    ax.plot(0.5, 0.5, 'r*', markersize=15)
    ax.set_title(f"WENO5: Final G\n(merged={r['merged']})", fontweight='bold')
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
    ax.plot(x, r['grad_mag'][line_j, :], 'b-', linewidth=2, label='|∇G|')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    ax.axvline(0.5, color='r', linestyle=':', alpha=0.5, label='Saddle')
    ax.set_xlabel('x')
    ax.set_ylabel('|∇G|')
    ax.set_title('WENO5: |∇G| along y=0.5', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 2])

    plt.tight_layout()
    plt.savefig('diagnose_merging_circles.png', dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: diagnose_merging_circles.png")

    # ========================================================================
    # Gradient comparison at key locations
    # ========================================================================

    print("\n" + "="*70)
    print("GRADIENT COMPARISON AT KEY LOCATIONS")
    print("="*70)

    locations = [
        ("Saddle point (x=0.5, y=0.5)", saddle_i, saddle_j),
        ("Left circle (x=0.35, y=0.5)", ny//2, int(0.35*nx)),
        ("Right circle (x=0.65, y=0.5)", ny//2, int(0.65*nx)),
    ]

    for loc_name, i, j in locations:
        print(f"\n{loc_name}:")
        print(f"  Initial |∇G|: {grad_mag_0[i, j]:.6f}")
        print(f"  Godunov:      {results['godunov']['grad_mag'][i, j]:.6f}")
        print(f"  WENO5:        {results['weno5']['grad_mag'][i, j]:.6f}")

if __name__ == '__main__':
    main()
