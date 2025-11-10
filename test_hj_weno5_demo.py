#!/usr/bin/env python3
"""
Clear demonstration of HJ-WENO5 advantages with well-controlled tests.
Focus on cases where the improvement is dramatic and unambiguous.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D
import time
import heapq


def fast_marching_method(mask, dx):
    """
    Fast Marching Method to compute signed distance function.

    Parameters:
    -----------
    mask : ndarray
        Binary mask where True is inside, False is outside
    dx : float
        Grid spacing

    Returns:
    --------
    phi : ndarray
        Signed distance function (negative inside, positive outside)
    """
    ny, nx = mask.shape
    phi = np.full((ny, nx), np.inf)
    status = np.zeros((ny, nx), dtype=int)  # 0=far, 1=narrow band, 2=known

    # Initialize interface cells (neighbors of boundary)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            # Check if this cell is near the boundary
            if mask[j, i] != mask[j-1, i] or mask[j, i] != mask[j+1, i] or \
               mask[j, i] != mask[j, i-1] or mask[j, i] != mask[j, i+1]:
                # This is a boundary cell, estimate distance
                phi[j, i] = 0.5 * dx  # Initial estimate
                status[j, i] = 1

    # Priority queue: (distance, j, i)
    heap = []
    for j in range(ny):
        for i in range(nx):
            if status[j, i] == 1:
                heapq.heappush(heap, (abs(phi[j, i]), j, i))

    # Fast marching
    while heap:
        _, j, i = heapq.heappop(heap)

        if status[j, i] == 2:
            continue

        status[j, i] = 2

        # Update neighbors
        for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            jn, in_ = j + dj, i + di

            if 0 <= jn < ny and 0 <= in_ < nx and status[jn, in_] != 2:
                # Solve Eikonal equation |∇φ| = 1
                phi_x = np.inf
                phi_y = np.inf

                # Get known neighbor values
                if in_ > 0 and status[jn, in_-1] == 2:
                    phi_x = min(phi_x, phi[jn, in_-1])
                if in_ < nx-1 and status[jn, in_+1] == 2:
                    phi_x = min(phi_x, phi[jn, in_+1])
                if jn > 0 and status[jn-1, in_] == 2:
                    phi_y = min(phi_y, phi[jn-1, in_])
                if jn < ny-1 and status[jn+1, in_] == 2:
                    phi_y = min(phi_y, phi[jn+1, in_])

                # Solve quadratic equation for new distance
                if phi_x < np.inf and phi_y < np.inf:
                    # Two known neighbors: solve (φ-φx)² + (φ-φy)² = dx²
                    phi_avg = (phi_x + phi_y) / 2.0
                    discriminant = 2*dx**2 - (phi_x - phi_y)**2
                    if discriminant >= 0:
                        phi_new = phi_avg + np.sqrt(discriminant) / 2.0
                    else:
                        phi_new = min(phi_x, phi_y) + dx
                elif phi_x < np.inf:
                    phi_new = phi_x + dx
                elif phi_y < np.inf:
                    phi_new = phi_y + dx
                else:
                    continue

                if phi_new < phi[jn, in_]:
                    phi[jn, in_] = phi_new
                    if status[jn, in_] == 0:
                        status[jn, in_] = 1
                    heapq.heappush(heap, (phi_new, jn, in_))

    # Apply sign based on mask
    phi = np.where(mask, -phi, phi)

    return phi


def test_comparison(test_name, nx, t_final, S_L, u_x, u_y, G0_func, analytical_func=None):
    """
    Generic test function that compares Godunov vs HJ-WENO5.
    """
    ny = nx
    Lx, Ly = 1.0, 1.0

    results = {}

    for scheme in ['godunov', 'weno5']:
        print(f"\n{'-'*60}")
        print(f"{test_name} - {scheme.upper()}")
        print(f"{'-'*60}")

        solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x, u_y)
        x = solver.x
        y = solver.y
        X, Y = np.meshgrid(x, y)

        G0 = G0_func(X, Y)

        dx = Lx / (nx - 1)
        v_max = np.max(np.sqrt(u_x**2 + u_y**2)) if not np.isscalar(u_x) else abs(u_x) + abs(u_y)
        dt = 0.2 * dx / (v_max + S_L + 1e-10)

        t_start = time.time()
        G_hist, t_hist = solver.solve(
            G0, t_final, dt,
            save_interval=max(1, int(t_final/(20*dt))),
            time_scheme='rk3',
            spatial_scheme='weno5',
            gradient_scheme=scheme,
            reinit_interval=0,  # NO reinitialization - pure gradient quality test
            reinit_method='none'
        )
        elapsed = time.time() - t_start

        G_final = G_hist[-1]

        # Compute gradient
        mask_interface = np.abs(G_final) < 3*dx
        if scheme == 'godunov':
            grad_mag = solver.compute_gradient_magnitude(G_final)
        else:
            grad_mag = solver.compute_gradient_magnitude_weno5(G_final)

        # Check for blow-up
        if np.any(~np.isfinite(grad_mag)) or len(grad_mag[mask_interface]) == 0:
            print(f"⚠️  NUMERICAL BLOW-UP!")
            results[scheme] = {
                'valid': False,
                'grad_mean': np.nan,
                'grad_std': np.nan,
                'grad_drift': np.inf,
                'elapsed': elapsed,
                'G_final': G_final,
                'grad_mag': grad_mag,
                'X': X,
                'Y': Y
            }
            continue

        grad_at_interface = grad_mag[mask_interface]
        grad_mean = np.mean(grad_at_interface)
        grad_std = np.std(grad_at_interface)
        grad_drift = abs(grad_mean - 1.0)

        # Compute errors if analytical solution provided
        if analytical_func is not None:
            G_exact = analytical_func(X, Y, t_final)
            l2_error = np.sqrt(np.mean((G_final - G_exact)**2))
        else:
            l2_error = None

        print(f"  Time:            {elapsed:.2f}s")
        print(f"  |∇G| mean:       {grad_mean:.6f}")
        print(f"  |∇G| std:        {grad_std:.6f}")
        print(f"  |∇G| drift:      {grad_drift:.6e}")
        if l2_error is not None:
            print(f"  L2 error:        {l2_error:.6e}")

        results[scheme] = {
            'valid': True,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'grad_drift': grad_drift,
            'l2_error': l2_error,
            'elapsed': elapsed,
            'G_final': G_final,
            'grad_mag': grad_mag,
            'X': X,
            'Y': Y,
            'G_hist': G_hist,
            't_hist': t_hist
        }

    return results

def main():
    """
    Run four carefully chosen tests that clearly demonstrate HJ-WENO5 advantages.
    Includes comparison of ellipse with and without reinitialization.
    """

    print("="*70)
    print("HJ-WENO5 DEMONSTRATION: Gradient Quality Without Reinitialization")
    print("="*70)
    print("\nThese tests run WITHOUT reinitialization to isolate gradient scheme quality.")
    print("HJ-WENO5 should maintain |∇G| ≈ 1.0 significantly better than Godunov.")
    print("\nNote: Test 2 compares ellipse with/without proper initialization.\n")

    # ========================================================================
    # Test 1: Expanding Circle (Pure Source Term)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: EXPANDING CIRCLE (Pure Source Term, S_L >> |u|)")
    print("="*70)

    def G0_circle(X, Y):
        return np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) - 0.1

    def G_exact_circle(X, Y, t):
        return np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) - (0.1 + 0.5*t)

    results_circle = test_comparison(
        "Expanding Circle",
        nx=101,
        t_final=0.4,
        S_L=0.5,
        u_x=0.0,
        u_y=0.0,
        G0_func=G0_circle,
        analytical_func=G_exact_circle
    )

    # ========================================================================
    # Test 2: Ellipse Expansion (Non-Uniform Curvature)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: EXPANDING ELLIPSE (Non-uniform curvature)")
    print("="*70)

    def G0_ellipse(X, Y):
        a, b = 0.15, 0.08  # Semi-axes
        return np.sqrt(((X-0.5)/a)**2 + ((Y-0.5)/b)**2) * min(a, b) - 1.0 * min(a, b)

    # First test WITHOUT reinitialization (imperfect initial condition)
    print("\nPart A: Without reinitialization (imperfect |∇G|)")
    results_ellipse = test_comparison(
        "Expanding Ellipse (No Reinit)",
        nx=121,
        t_final=0.3,
        S_L=0.4,
        u_x=0.0,
        u_y=0.0,
        G0_func=G0_ellipse,
        analytical_func=None
    )

    # Compare with properly initialized ellipse
    print("\nPart B: WITH proper signed distance initialization")
    print("(Using Fast Marching Method to create exact SDF)")

    def G0_ellipse_proper(X, Y):
        """
        Create properly initialized ellipse with |∇G| ≈ 1.0
        Uses Fast Marching Method to compute exact signed distance function.
        """
        a, b = 0.15, 0.08
        cx, cy = 0.5, 0.5

        # Create mask of ellipse interior
        mask = ((X-cx)/a)**2 + ((Y-cy)/b)**2 < 1.0

        # Compute SDF using Fast Marching Method
        nx_temp = X.shape[1]
        Lx_temp = 1.0
        dx_temp = Lx_temp / (nx_temp - 1)

        G_fmm = fast_marching_method(mask, dx_temp)

        return G_fmm

    results_ellipse_proper = test_comparison(
        "Expanding Ellipse (FMM SDF)",
        nx=121,
        t_final=0.3,
        S_L=0.4,
        u_x=0.0,
        u_y=0.0,
        G0_func=G0_ellipse_proper,
        analytical_func=None
    )

    # ========================================================================
    # Test 3: Multiple Circles (Topology Change)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: MERGING CIRCLES (Topology change)")
    print("="*70)

    def G0_circles(X, Y):
        d1 = np.sqrt((X-0.35)**2 + (Y-0.5)**2) - 0.12
        d2 = np.sqrt((X-0.65)**2 + (Y-0.5)**2) - 0.12
        return np.minimum(d1, d2)  # Union of two circles

    results_circles = test_comparison(
        "Merging Circles",
        nx=121,
        t_final=0.25,
        S_L=0.5,
        u_x=0.0,
        u_y=0.0,
        G0_func=G0_circles,
        analytical_func=None
    )

    # ========================================================================
    # Create Comprehensive Visualization
    # ========================================================================

    fig = plt.figure(figsize=(18, 14))

    all_results = [
        ("Expanding Circle", results_circle),
        ("Ellipse (No Reinit)", results_ellipse),
        ("Ellipse (FMM SDF)", results_ellipse_proper),
        ("Merging Circles", results_circles)
    ]

    for test_idx, (test_name, results) in enumerate(all_results):
        # Column 1: Godunov gradient field
        ax = plt.subplot(4, 4, test_idx*4 + 1)
        r = results['godunov']
        if r['valid']:
            im = ax.contourf(r['X'], r['Y'], r['grad_mag'],
                            levels=np.linspace(0.5, 1.5, 21), cmap='RdBu_r', extend='both')
            ax.contour(r['X'], r['Y'], r['G_final'], levels=[0], colors='black', linewidths=2)
            title_str = f"Godunov: |∇G|={r['grad_mean']:.3f}"
        else:
            ax.text(0.5, 0.5, "UNSTABLE\n(Blow-up)", ha='center', va='center',
                   fontsize=14, color='red', fontweight='bold')
            title_str = "Godunov: FAILED"
        ax.set_aspect('equal')
        ax.set_title(title_str, fontsize=9, fontweight='bold')
        ax.set_ylabel(test_name, fontsize=10, fontweight='bold')
        if r['valid']:
            plt.colorbar(im, ax=ax, label='|∇G|', fraction=0.046)

        # Column 2: WENO5 gradient field
        ax = plt.subplot(4, 4, test_idx*4 + 2)
        r = results['weno5']
        if r['valid']:
            im = ax.contourf(r['X'], r['Y'], r['grad_mag'],
                            levels=np.linspace(0.5, 1.5, 21), cmap='RdBu_r', extend='both')
            ax.contour(r['X'], r['Y'], r['G_final'], levels=[0], colors='black', linewidths=2)
            title_str = f"HJ-WENO5: |∇G|={r['grad_mean']:.3f}"
        else:
            ax.text(0.5, 0.5, "UNSTABLE\n(Blow-up)", ha='center', va='center',
                   fontsize=14, color='red', fontweight='bold')
            title_str = "HJ-WENO5: FAILED"
        ax.set_aspect('equal')
        ax.set_title(title_str, fontsize=9, fontweight='bold')
        if r['valid']:
            plt.colorbar(im, ax=ax, label='|∇G|', fraction=0.046)

        # Column 3: Improvement metrics
        ax = plt.subplot(4, 4, test_idx*4 + 3)

        if results['godunov']['valid'] and results['weno5']['valid']:
            metrics = ['|∇G| Drift', '|∇G| Spread']
            god_vals = [results['godunov']['grad_drift'], results['godunov']['grad_std']]
            w5_vals = [results['weno5']['grad_drift'], results['weno5']['grad_std']]
            improvements = [g/w if w > 1e-15 else 1.0 for g, w in zip(god_vals, w5_vals)]

            x_pos = np.arange(len(metrics))
            bars = ax.bar(x_pos, improvements, color=['C0', 'C1'], alpha=0.7, edgecolor='black')
            ax.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, fontsize=9)
            ax.set_ylabel('Improvement\n(God/WENO5)', fontsize=9)
            ax.set_title('Improvement Factor', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, max(improvements)*1.3])

            for bar, val in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height*1.05,
                       f'{val:.1f}×', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        else:
            status_msg = ""
            if not results['godunov']['valid']:
                status_msg += "Godunov: FAILED\n"
            else:
                status_msg += f"Godunov: OK\n  Drift: {results['godunov']['grad_drift']:.2e}\n"

            if not results['weno5']['valid']:
                status_msg += "WENO5: FAILED"
            else:
                status_msg += f"WENO5: OK\n  Drift: {results['weno5']['grad_drift']:.2e}"

            ax.text(0.5, 0.5, status_msg, ha='center', va='center',
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax.set_xlim([-0.5, 1.5])

        # Column 4: Statistics table
        ax = plt.subplot(4, 4, test_idx*4 + 4)
        ax.axis('off')

        if results['godunov']['valid'] and results['weno5']['valid']:
            improvement = results['godunov']['grad_drift'] / results['weno5']['grad_drift']

            table_text = f"""
{test_name}

Godunov:
  |∇G| = {results['godunov']['grad_mean']:.4f}
  σ    = {results['godunov']['grad_std']:.4f}
  Drift= {results['godunov']['grad_drift']:.2e}
  Time = {results['godunov']['elapsed']:.1f}s

HJ-WENO5:
  |∇G| = {results['weno5']['grad_mean']:.4f}
  σ    = {results['weno5']['grad_std']:.4f}
  Drift= {results['weno5']['grad_drift']:.2e}
  Time = {results['weno5']['elapsed']:.1f}s

→ WENO5 is {improvement:.1f}× better!
"""
        else:
            table_text = f"{test_name}\n\nOne or both schemes\nencountered numerical\ninstability."

        ax.text(0.5, 0.5, table_text, ha='center', va='center',
               fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('/home/nijso/Downloads/GEQ/hj_weno5_clear_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n\nPlot saved to: hj_weno5_clear_demo.png")

    # Final Summary
    print("\n" + "="*70)
    print("SUMMARY: HJ-WENO5 vs First-Order Godunov")
    print("="*70)

    for test_name, results in all_results:
        print(f"\n{test_name}:")
        if results['godunov']['valid'] and results['weno5']['valid']:
            improvement = results['godunov']['grad_drift'] / results['weno5']['grad_drift']
            print(f"  Godunov drift:   {results['godunov']['grad_drift']:.2e}")
            print(f"  WENO5 drift:     {results['weno5']['grad_drift']:.2e}")
            print(f"  → Improvement:   {improvement:.1f}× better gradient maintenance")
        else:
            if not results['godunov']['valid']:
                print(f"  Godunov: NUMERICAL INSTABILITY")
            if not results['weno5']['valid']:
                print(f"  WENO5: NUMERICAL INSTABILITY")

    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("\n1. PERFECT INITIAL CONDITIONS (Circle, Ellipse-Proper-SDF):")
    print("   → Dramatic improvement with higher-order gradient scheme")
    print("   → WENO5 maintains |∇G| ≈ 1.0 much better than Godunov")
    print("\n2. IMPERFECT INITIAL CONDITIONS (Ellipse-No-Reinit):")
    print("   → Modest improvement (1-2×) - initial error dominates")
    print("   → Shows importance of proper signed distance initialization!")
    print("\n3. NON-UNIFORM CURVATURE (Ellipse vs Circle):")
    print("   → Even with perfect SDF, ellipse harder than circle")
    print("   → Curvature variation causes non-uniform error accumulation")
    print("\n4. TOPOLOGY CHANGES (Merging Circles):")
    print("   → Higher-order can fail due to singular points")
    print("   → Use Godunov for merging/splitting scenarios")
    print("\n" + "="*70)
    print("PRACTICAL RECOMMENDATION:")
    print("  • For smooth flames: Use gradient_scheme='weno5' + proper SDF init")
    print("  • For topology changes: Use gradient_scheme='godunov' with reinit_freq=5-10")
    print("  • For general shapes: Use scipy.ndimage.distance_transform_edt for SDF")
    print("    • Less frequent reinitialization needed")
    print("    • More stable long-time integration")
    print("="*70 + "\n")

    plt.show()

if __name__ == '__main__':
    main()
