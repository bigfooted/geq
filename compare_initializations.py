#!/usr/bin/env python3
"""
Compare different SDF initialization methods: Bad approximation, EDT, FMM, and Reinitialization PDE.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D
from scipy.ndimage import distance_transform_edt
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

def main():
    nx, ny = 121, 121
    Lx, Ly = 1.0, 1.0
    S_L = 0.4

    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0)
    x, y = solver.x, solver.y
    X, Y = np.meshgrid(x, y)
    dx = Lx / (nx - 1)

    a, b = 0.15, 0.08
    cx, cy = 0.5, 0.5

    # Create FOUR initial conditions
    # 1. Bad approximation (current)
    G_bad = np.sqrt(((X-cx)/a)**2 + ((Y-cy)/b)**2) * min(a, b) - 1.0 * min(a, b)

    # 2. EDT proper SDF
    mask = ((X-cx)/a)**2 + ((Y-cy)/b)**2 < 1.0
    dist_in = distance_transform_edt(mask, sampling=[dx, dx])
    dist_out = distance_transform_edt(~mask, sampling=[dx, dx])
    G_edt = np.where(mask, -dist_in, dist_out)

    # 3. Fast Marching Method
    print("Computing Fast Marching Method SDF...")
    G_fmm = fast_marching_method(mask, dx)

    # 4. Reinitialization PDE (for comparison)
    G_reinit = solver.reinitialize_pde(G_bad, 0.1*dx, n_steps=30)

    print("="*70)
    print("INITIAL CONDITION ANALYSIS - FOUR METHODS")
    print("="*70)

    # Check gradient quality
    for name, G in [("Bad Approx", G_bad),
                     ("EDT SDF", G_edt),
                     ("Fast Marching", G_fmm),
                     ("Reinitialized", G_reinit)]:
        grad_x = np.gradient(G, dx, axis=1)
        grad_y = np.gradient(G, dx, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        mask_interface = np.abs(G) < 3*dx
        grad_int = grad_mag[mask_interface]

        print(f"\n{name}:")
        print(f"  |∇G| mean: {np.mean(grad_int):.6f}")
        print(f"  |∇G| std:  {np.std(grad_int):.6f}")
        print(f"  |∇G| min:  {np.min(grad_int):.6f}")
        print(f"  |∇G| max:  {np.max(grad_int):.6f}")
        print(f"  Drift:     {abs(np.mean(grad_int) - 1.0):.6e}")

    # Visualize initial conditions
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))

    for idx, (name, G) in enumerate([("Bad Approx", G_bad),
                                      ("EDT SDF", G_edt),
                                      ("Fast Marching", G_fmm),
                                      ("Reinitialized", G_reinit)]):
        # Column 1: Level set
        ax = axes[idx, 0]
        im = ax.contourf(X, Y, G, levels=20, cmap='coolwarm')
        ax.contour(X, Y, G, levels=[0], colors='black', linewidths=2)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nLevel Set G', fontweight='bold')
        plt.colorbar(im, ax=ax)

        # Column 2: Gradient magnitude
        grad_x = np.gradient(G, dx, axis=1)
        grad_y = np.gradient(G, dx, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        ax = axes[idx, 1]
        im = ax.contourf(X, Y, grad_mag, levels=np.linspace(0.5, 1.5, 21),
                        cmap='RdBu_r', extend='both')
        ax.contour(X, Y, G, levels=[0], colors='black', linewidths=2)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n|∇G| (mean={np.mean(grad_mag):.3f})', fontweight='bold')
        plt.colorbar(im, ax=ax)

        # Column 3: Cross-section
        ax = axes[idx, 2]
        line_j = ny//2
        ax.plot(x, grad_mag[line_j, :], 'b-', linewidth=2, label='|∇G|')
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlabel('x')
        ax.set_ylabel('|∇G|')
        ax.set_title(f'{name}\n|∇G| along y=0.5', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.5])

    plt.tight_layout()
    plt.savefig('compare_ellipse_initializations.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: compare_ellipse_initializations.png")

    # Now test propagation with both schemes
    print("\n" + "="*70)
    print("PROPAGATION TEST")
    print("="*70)

    t_final = 0.3
    dt = 0.2 * dx / (S_L + 1e-10)

    for init_name, G0 in [("Bad Approx", G_bad),
                           ("EDT SDF", G_edt),
                           ("Fast Marching", G_fmm),
                           ("Reinitialized", G_reinit)]:
        print(f"\n{'-'*70}")
        print(f"Initial condition: {init_name}")
        print(f"{'-'*70}")

        for scheme in ['godunov', 'weno5']:
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

            if scheme == 'godunov':
                grad_mag = solver.compute_gradient_magnitude(G_final)
            else:
                grad_mag = solver.compute_gradient_magnitude_weno5(G_final)

            mask_interface = np.abs(G_final) < 3*dx
            grad_int = grad_mag[mask_interface]

            drift = abs(np.mean(grad_int) - 1.0)
            print(f"  {scheme.upper():8s}: drift = {drift:.6e}, std = {np.std(grad_int):.6f}")

if __name__ == '__main__':
    main()
