import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Test case for a laminar premixed flame attached to a solid wall.
The flame is stabilized at 45 degrees where S_L = u_y.
The flame is anchored at the corner (0,0) and this constraint is enforced.

Configuration:
- Wall at x = 0 (left boundary)
- Inlet at y = 0 (bottom boundary)
- Flame anchored at origin (0, 0) - FIXED
- Initial: line from (0,0) to (1.0, 0.2), angle ≈ 11.31°
- Final steady state: 45° angle, height y = x up to domain edge
- S_L = u_y = 0.4 (ensures 45-degree stabilization)
- u_x = 0 (no horizontal flow)
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import GEquationSolver2D
from contour_utils import get_contour_statistics, compute_contour_length
import time as time_module


def compute_flame_height_analytical(x, S_L, u_y):
    """
    Analytical solution for 45-degree flame.

    For a steady flame at 45 degrees:
    tan(45°) = 1 = S_L / u_y

    The flame front is a straight line: y = x

    Parameters:
    -----------
    x : float or array
        Horizontal position
    S_L : float
        Laminar flame speed
    u_y : float
        Vertical inlet velocity

    Returns:
    --------
    y : float or array
        Flame height at position x
    """
    # For 45 degrees: y = x
    return x


def create_initial_flame_profile_from_line(X, Y, x0, y0, x1, y1, thickness=0.05):
    """
    Create initial flame profile as a straight line from (x0,y0) to (x1,y1).

    Sign convention:
    - Above the line:  G = -1 (negative)
    - Below the line:  G = +1 (positive)
    G = 0 on the line (flame surface).

    Note:
    - This initializer uses a hard sign (no smoothing).
    - The 'thickness' parameter is kept for API compatibility but is not used.
    - Not limited to the segment [anchor -> tip]; applies to the infinite line.
    """
    dx = x1 - x0
    dy = y1 - y0
    line_length = np.sqrt(dx**2 + dy**2)
    if line_length == 0:
        raise ValueError("Line length is zero. Choose distinct (x1, y1).")

    # Unit tangent and normal to the line
    tx = dx / line_length
    ty = dy / line_length
    nx = -ty
    ny = tx

    # Signed distance to the infinite line through (x0,y0)-(x1,y1)
    distance = (X - x0) * nx + (Y - y0) * ny

    # Hard sign initialization:
    # Above the line (distance > 0): G = -1
    # Below the line (distance < 0): G = +1
    # Exactly on the line: G = 0
    G = np.where(distance > 0.0, -1.0, 1.0)
    G = np.where(np.isclose(distance, 0.0), 0.0, G)

    return G


def enforce_corner_anchor(G, X, Y, x_anchor, y_anchor, anchor_radius):
    """
    Enforce that the flame remains anchored at the corner.

    Sets G = 0 (flame surface) in a small region around the anchor point.
    This acts as a Dirichlet boundary condition.

    Parameters:
    -----------
    G : ndarray
        Level set function
    X, Y : ndarray
        Coordinate meshgrids
    x_anchor, y_anchor : float
        Anchor point coordinates
    anchor_radius : float
        Radius around anchor point to fix

    Returns:
    --------
    G : ndarray
        Modified level set function with enforced anchor
    """
    ny, nx = G.shape

    # Find cells near anchor point
    for j in range(ny):
        for i in range(nx):
            x, y = X[j, i], Y[j, i]
            dist = np.sqrt((x - x_anchor)**2 + (y - y_anchor)**2)

            if dist < anchor_radius:
                # At the anchor: enforce flame surface (G = 0)
                G[j, i] = 0.0

    return G


class GEquationSolverWithAnchor(GEquationSolver2D):
    """
    Extended solver that enforces flame anchoring at a specified point.
    """

    def __init__(self, nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0,
                 anchor_point=(0.0, 0.0), anchor_radius=0.02):
        """
        Initialize solver with anchor point enforcement.

        Parameters:
        -----------
        anchor_point : tuple
            (x, y) coordinates of anchor point
        anchor_radius : float
            Radius around anchor to enforce boundary condition
        """
        super().__init__(nx, ny, Lx, Ly, S_L, u_x, u_y)
        self.anchor_point = anchor_point
        self.anchor_radius = anchor_radius

    def solve(self, G_initial, t_final, dt, save_interval=None, time_scheme='euler',
              reinit_interval=0, reinit_method='fast_marching', reinit_local=True,
              smooth_ic=False):
        """
        Solve with anchor point enforcement after each time step.
        """
        if time_scheme not in ['euler', 'rk2']:
            raise ValueError("time_scheme must be 'euler' or 'rk2'")

        if reinit_method not in ['pde', 'fast_marching']:
            raise ValueError("reinit_method must be 'pde' or 'fast_marching'")

        self.G = G_initial.copy()

        # Apply initial smoothing if requested
        if smooth_ic:
            print("Applying initial condition smoothing...")
            self.G = self.smooth_initial_condition(self.G, bandwidth=4)

        # Enforce anchor on initial condition
        self.G = enforce_corner_anchor(self.G, self.X, self.Y,
                                       self.anchor_point[0], self.anchor_point[1],
                                       self.anchor_radius)

        # Storage for history
        G_history = [self.G.copy()]
        t_history = [0.0]

        t = 0.0
        step = 0

        # Determine save interval
        if save_interval is None:
            save_interval = 1

        print(f"Using {time_scheme.upper()} time discretization scheme")
        if reinit_interval > 0:
            reinit_type = "LOCAL (narrow-band)" if reinit_local else "GLOBAL"
            print(f"Reinitialization enabled: every {reinit_interval} steps using '{reinit_method}' method ({reinit_type})")
        print(f"Anchor point enforcement: ({self.anchor_point[0]:.3f}, {self.anchor_point[1]:.3f}) "
              f"with radius {self.anchor_radius:.4f}")

        while t < t_final:
            # Adjust last time step
            if t + dt > t_final:
                dt = t_final - t

            # Store G before time step (for reinitialization)
            G_before = self.G.copy()

            # Time integration
            if time_scheme == 'euler':
                rhs = self.compute_rhs(self.G)
                self.G = self.G + dt * rhs

            elif time_scheme == 'rk2':
                k1 = self.compute_rhs(self.G)
                G_temp = self.G + dt * k1
                k2 = self.compute_rhs(G_temp)
                self.G = self.G + dt * 0.5 * (k1 + k2)

            # ENFORCE ANCHOR POINT after time step
            self.G = enforce_corner_anchor(self.G, self.X, self.Y,
                                          self.anchor_point[0], self.anchor_point[1],
                                          self.anchor_radius)

            t += dt
            step += 1

            # Reinitialize if requested
            if reinit_interval > 0 and step % reinit_interval == 0:
                if reinit_method == 'pde':
                    self.G = self.reinitialize_pde(G_before, n_steps=10, bandwidth=5,
                                                   use_local=reinit_local)
                elif reinit_method == 'fast_marching':
                    self.G = self.reinitialize_fast_marching(self.G, bandwidth=5,
                                                             use_local=reinit_local)

                # Re-enforce anchor after reinitialization
                self.G = enforce_corner_anchor(self.G, self.X, self.Y,
                                              self.anchor_point[0], self.anchor_point[1],
                                              self.anchor_radius)

            # Store solution at specified intervals
            if step % save_interval == 0:
                G_history.append(self.G.copy())
                t_history.append(t)

            if step % 100 == 0:
                grad_mag = self.compute_gradient_magnitude(self.G)
                grad_mag_interface = grad_mag[self.find_interface_band(self.G, bandwidth=2)]
                if len(grad_mag_interface) > 0:
                    avg_grad = np.mean(grad_mag_interface)
                else:
                    avg_grad = 0.0
                print(f"Step {step}, t = {t:.4f}, |∇G|_interface ≈ {avg_grad:.3f}")

        # Ensure final time is saved
        if t_history[-1] < t_final:
            G_history.append(self.G.copy())
            t_history.append(t)

        return G_history, t_history


def test_flame_wall_attachment(t_final=10.0, time_scheme='rk2', use_reinit=True):
    """
    Test case for flame attached to wall at 45 degrees with anchoring.
    Initial condition: line from (0,0) to (1.0, 0.2)
    Final steady state: 45° angle (y = x)

    Parameters:
    -----------
    t_final : float
        Final simulation time (default: 10.0)
    time_scheme : str
        Time discretization scheme: 'euler' or 'rk2'
    use_reinit : bool
        Use local reinitialization (default: True)
    """

    print("\n" + "="*80)
    print("2D G-EQUATION: FLAME WALL ATTACHMENT TEST (WITH CORNER ANCHORING)")
    print("="*80)

    # Physical parameters
    S_L = 0.4          # Laminar flame speed [m/s]
    u_x = 0.0          # No horizontal flow
    u_y = 0.4          # Vertical inlet velocity [m/s]

    # Initial flame line: from (x0,y0) to (x1,y1)
    x0, y0 = 0.0, 0.0   # Anchor point (corner)
    x1, y1 = 1.0, 0.2   # Initial flame tip

    # Calculate initial flame properties
    initial_flame_length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    initial_flame_angle = np.degrees(np.arctan2(y1-y0, x1-x0))

    # Final steady state properties
    final_flame_angle = 45.0     # Degrees (S_L = u_y)

    # Flame parameters
    x_anchor = x0
    y_anchor = y0
    anchor_radius = 0.03         # Radius to enforce anchoring

    # Domain sizing
    Lx = 1.0           # Domain width [m]
    final_flame_height = Lx  # For 45 degrees at steady state: y = x
    Ly = 2.0 * final_flame_height  # Domain height (twice flame height)

    # Grid parameters
    nx = 151           # Grid points in x
    ny = 301           # Grid points in y (finer in y due to larger domain)

    # Time parameters
    CFL = 0.4
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = CFL * min(dx, dy) / max(S_L, u_y)
    save_interval = int(0.1 / dt)  # Save every 0.1 time units

    # Reinitialization parameters
    reinit_interval = 50 if use_reinit else 0
    reinit_method = 'fast_marching'
    reinit_local = True

    print(f"\nPhysical Parameters:")
    print(f"  Laminar flame speed S_L = {S_L:.3f} m/s")
    print(f"  Vertical inlet velocity u_y = {u_y:.3f} m/s")
    print(f"  Horizontal velocity u_x = {u_x:.3f} m/s")
    print(f"  S_L / u_y = {S_L / u_y:.3f} (should be ~1 for 45°)")

    print(f"\nInitial Flame Configuration:")
    print(f"  Flame line: from ({x0:.3f}, {y0:.3f}) to ({x1:.3f}, {y1:.3f})")
    print(f"  Initial flame length: {initial_flame_length:.6f} m")
    print(f"  Initial flame angle: {initial_flame_angle:.2f}°")
    print(f"  Initial flame tip height: {y1:.3f} m")
    print(f"  Above line: BURNT (G > 0)")
    print(f"  Below line: UNBURNT (G < 0)")

    print(f"\nExpected Final (Steady) State:")
    print(f"  Flame angle: {final_flame_angle:.2f}°")
    print(f"  Flame equation: y = x")
    print(f"  Expected final flame length: {np.sqrt(2) * Lx:.6f} m")

    print(f"\nDomain Configuration:")
    print(f"  Width Lx = {Lx:.3f} m")
    print(f"  Height Ly = {Ly:.3f} m")
    print(f"  Grid: {nx} × {ny} = {nx*ny} points")
    print(f"  Grid spacing: dx = {dx:.6f} m, dy = {dy:.6f} m")

    print(f"\nAnchor Point Configuration:")
    print(f"  Anchor point: ({x_anchor:.3f}, {y_anchor:.3f})")
    print(f"  Anchor radius: {anchor_radius:.4f} m ({anchor_radius/dx:.1f} × dx)")
    print(f"  Boundary condition: G = 0 enforced at anchor")

    print(f"\nNumerical Parameters:")
    print(f"  Time scheme: {time_scheme.upper()}")
    print(f"  Time step dt = {dt:.6f} s")
    print(f"  CFL number = {CFL:.3f}")
    print(f"  Final time t_final = {t_final:.3f} s")
    print(f"  Save interval: every {save_interval} steps (~0.1s)")
    if use_reinit:
        print(f"  Reinitialization: LOCAL every {reinit_interval} steps ({reinit_method})")
    else:
        print(f"  Reinitialization: Disabled")

    print("\n" + "="*80)
    print("Initializing solver with anchor enforcement...")
    print("="*80)

    # Create solver with anchoring
    solver = GEquationSolverWithAnchor(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y,
                                       anchor_point=(x_anchor, y_anchor),
                                       anchor_radius=anchor_radius)

    # Create initial flame profile: line from (0,0) to (1.0, 0.2)
    print(f"\nCreating initial flame profile:")
    print(f"  Flame line from ({x0:.3f}, {y0:.3f}) to ({x1:.3f}, {y1:.3f})")
    print(f"  Transition thickness: {min(dx, dy) * 2:.6f} m")

    G_initial = create_initial_flame_profile_from_line(
        solver.X, solver.Y,
        x0, y0, x1, y1,
        thickness=min(dx, dy) * 2
    )

    # Enforce anchor on initial condition
    G_initial = enforce_corner_anchor(G_initial, solver.X, solver.Y,
                                      x_anchor, y_anchor, anchor_radius)

    # Get initial statistics
    initial_stats = get_contour_statistics(G_initial, solver.X, solver.Y, iso_value=0.0)

    print(f"\nInitial flame statistics:")
    print(f"  Number of contours: {initial_stats['num_contours']}")
    print(f"  Closed contours: {initial_stats['num_closed']}")
    print(f"  Open contours: {initial_stats['num_open']}")
    print(f"  Total flame length: {initial_stats['total_length']:.6f} m")
    print(f"  Expected initial length: {initial_flame_length:.6f} m")
    print(f"  Difference: {abs(initial_stats['total_length'] - initial_flame_length):.6f} m")

    # Solve
    print("\n" + "="*80)
    print("Solving G-equation with anchor enforcement...")
    print("="*80)

    start_time = time_module.time()
    G_history, t_history = solver.solve(
        G_initial, t_final, dt,
        save_interval=save_interval,
        time_scheme=time_scheme,
        reinit_interval=reinit_interval,
        reinit_method=reinit_method,
        reinit_local=reinit_local
    )
    elapsed_time = time_module.time() - start_time

    n_total_steps = int(t_final / dt)
    print(f"\nSolver completed:")
    print(f"  Total time steps: {n_total_steps}")
    print(f"  Saved snapshots: {len(t_history)}")
    print(f"  Computation time: {elapsed_time:.2f} s")
    print(f"  Time per step: {elapsed_time/n_total_steps*1000:.3f} ms")

    # Extract flame characteristics over time
    print("\n" + "="*80)
    print("Analyzing flame characteristics...")
    print("="*80)

    flame_lengths = []
    flame_angles = []
    flame_heights = []
    anchor_distances = []

    for idx, (G, t) in enumerate(zip(G_history, t_history)):
        # Compute flame length
        length = compute_contour_length(G, solver.X, solver.Y, iso_value=0.0)
        flame_lengths.append(length)

        # Extract contour points
        stats = get_contour_statistics(G, solver.X, solver.Y, iso_value=0.0)

        if stats['num_contours'] > 0 and len(stats['contours'][0]['points']) > 2:
            points = np.array(stats['contours'][0]['points'])
            xs, ys = points[:, 0], points[:, 1]

            # Find maximum height
            max_height = np.max(ys)
            flame_heights.append(max_height)

            # Check minimum distance from anchor
            distances = np.sqrt((xs - x_anchor)**2 + (ys - y_anchor)**2)
            min_dist = np.min(distances)
            anchor_distances.append(min_dist)

            # Estimate angle from linear fit
            # Use region from 20% to 80% of current flame height
            mask = (ys > 0.2 * max_height) & (ys < 0.8 * max_height) & (xs > 0.05)
            if np.sum(mask) > 10:
                coeffs = np.polyfit(xs[mask], ys[mask], 1)
                angle_rad = np.arctan(coeffs[0])
                angle_deg = np.degrees(angle_rad)
                flame_angles.append(angle_deg)
            else:
                flame_angles.append(np.nan)
        else:
            flame_heights.append(0.0)
            flame_angles.append(np.nan)
            anchor_distances.append(np.nan)

        # Print progress
        if idx % max(1, len(G_history) // 10) == 0:
            print(f"  t = {t:.3f}s: length = {length:.6f} m, "
                  f"height = {flame_heights[-1]:.6f} m, "
                  f"angle = {flame_angles[-1]:.2f}°")

    # Statistics
    print("\n" + "="*80)
    print("SIMULATION STATISTICS")
    print("="*80)

    expected_final_length = np.sqrt(2) * Lx

    print(f"\nFlame Length Evolution:")
    print(f"  Initial length: {flame_lengths[0]:.6f} m (expected: {initial_flame_length:.6f} m)")
    print(f"  Final length: {flame_lengths[-1]:.6f} m (expected: {expected_final_length:.6f} m)")
    print(f"  Length growth: {(flame_lengths[-1] / flame_lengths[0] - 1) * 100:.1f}%")

    print(f"\nFlame Height Evolution:")
    print(f"  Initial height: {flame_heights[0]:.6f} m (expected: {y1:.6f} m)")
    print(f"  Final height: {flame_heights[-1]:.6f} m (expected: {final_flame_height:.6f} m)")
    print(f"  Height growth: {(flame_heights[-1] / flame_heights[0] - 1) * 100:.1f}%")

    # Angle statistics
    valid_angles = [a for a in flame_angles if not np.isnan(a)]
    if valid_angles:
        print(f"\nFlame Angle Evolution:")
        print(f"  Initial angle: {valid_angles[0]:.2f}° (expected: {initial_flame_angle:.2f}°)")
        print(f"  Final angle: {valid_angles[-1]:.2f}° (expected: {final_flame_angle:.2f}°)")
        print(f"  Mean angle: {np.mean(valid_angles):.2f}°")
        print(f"  Angle change: {valid_angles[-1] - valid_angles[0]:.2f}°")

    # Anchor verification
    valid_anchor_dists = [d for d in anchor_distances if not np.isnan(d)]
    if valid_anchor_dists:
        print(f"\nAnchor Point Verification:")
        print(f"  Min distance: {np.min(valid_anchor_dists):.6f} m")
        print(f"  Max distance: {np.max(valid_anchor_dists):.6f} m")
        print(f"  Mean distance: {np.mean(valid_anchor_dists):.6f} m")
        if np.max(valid_anchor_dists) < anchor_radius * 1.5:
            print(f"  Status: ✓ FLAME STAYS ANCHORED")
        else:
            print(f"  Status: ✗ FLAME MAY HAVE DETACHED")

    print(f"\nComputational Performance:")
    print(f"  Total elapsed time: {elapsed_time:.2f} s")
    print(f"  Simulated physical time: {t_final:.2f} s")
    print(f"  Speed-up ratio: {t_final / elapsed_time:.2f}x")

    # Check steady state
    steady_start = int(0.8 * len(flame_lengths))
    if steady_start > 0:
        steady_lengths = flame_lengths[steady_start:]
        steady_variation = np.std(steady_lengths) / np.mean(steady_lengths) * 100
        print(f"\nSteady State Analysis (last 20%):")
        print(f"  Mean flame length: {np.mean(steady_lengths):.6f} m")
        print(f"  Coefficient of variation: {steady_variation:.3f}%")
        if steady_variation < 1.0:
            print(f"  Status: ✓ STEADY STATE REACHED")
        else:
            print(f"  Status: ! STILL EVOLVING")

    print("\n" + "="*80)

    # Visualization
    print("\nCreating visualizations...")

    # Figure 1: Flame evolution snapshots
    fig1 = plt.figure(figsize=(18, 12))

    n_snapshots = 6
    snapshot_indices = np.linspace(0, len(G_history) - 1, n_snapshots, dtype=int)

    for idx, snap_idx in enumerate(snapshot_indices):
        ax = plt.subplot(2, 3, idx + 1)

        G = G_history[snap_idx]
        t = t_history[snap_idx]

        # Plot level set
        levels = np.linspace(-1, 1, 21)
        contourf = ax.contourf(solver.X, solver.Y, G, levels=levels,
                              cmap='RdBu_r', extend='both')

        # Highlight flame surface
        ax.contour(solver.X, solver.Y, G, levels=[0], colors='black',
                  linewidths=2.5, label='Flame')

        # Plot analytical steady state (45°)
        x_analytical = np.linspace(0, Lx, 100)
        y_analytical = compute_flame_height_analytical(x_analytical, S_L, u_y)
        ax.plot(x_analytical, y_analytical, 'g--', linewidth=2,
               label='Final (45°)')

        # Plot initial condition line
        if idx == 0:
            ax.plot([x0, x1], [y0, y1], 'c--', linewidth=2,
                   label=f'Initial ({initial_flame_angle:.1f}°)')

        # Mark boundaries
        ax.axvline(x=0, color='brown', linewidth=4, label='Wall' if idx==0 else '', alpha=0.7)
        ax.axhline(y=0, color='blue', linewidth=2, alpha=0.5, label='Inlet' if idx==0 else '')

        # Mark anchor
        ax.plot(x_anchor, y_anchor, 'ro', markersize=10, markeredgecolor='white',
               markeredgewidth=2, label='Anchor' if idx==0 else '', zorder=10)

        circle = plt.Circle((x_anchor, y_anchor), anchor_radius,
                           color='red', fill=False, linestyle='--', linewidth=1.5)
        ax.add_patch(circle)

        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_title(f't = {t:.3f} s\nLength = {flame_lengths[snap_idx]:.4f} m, '
                    f'Angle ≈ {flame_angles[snap_idx]:.1f}°',
                    fontsize=11, fontweight='bold')
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, 1.5])
        ax.set_aspect('equal')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.colorbar(contourf, ax=ax, label='G')

    plt.tight_layout()
    plt.savefig(f'flame_wall_attachment_evolution_t{t_final}.png',
               dpi=300, bbox_inches='tight')
    print(f"Saved: flame_wall_attachment_evolution_t{t_final}.png")

    # Figure 2: Time series analysis
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 2a: Flame length vs time
    ax1.plot(t_history, flame_lengths, 'b-', linewidth=2)
    ax1.axhline(y=expected_final_length, color='r', linestyle='--', linewidth=2,
               label=f'Expected final: {expected_final_length:.3f} m')
    ax1.axhline(y=initial_flame_length, color='c', linestyle=':', linewidth=2,
               label=f'Initial: {initial_flame_length:.3f} m')
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Flame Length [m]', fontsize=11)
    ax1.set_title('Flame Length Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2b: Flame height vs time
    ax2.plot(t_history, flame_heights, 'g-', linewidth=2)
    ax2.axhline(y=final_flame_height, color='r', linestyle='--', linewidth=2,
               label=f'Final: {final_flame_height:.3f} m')
    ax2.axhline(y=y1, color='c', linestyle=':', linewidth=2,
               label=f'Initial: {y1:.3f} m')
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('Maximum Flame Height [m]', fontsize=11)
    ax2.set_title('Flame Height Evolution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 2c: Flame angle vs time
    t_valid = [t_history[i] for i, a in enumerate(flame_angles) if not np.isnan(a)]
    ax3.plot(t_valid, valid_angles, 'r-', linewidth=2)
    ax3.axhline(y=final_flame_angle, color='k', linestyle='--', linewidth=2,
               label=f'Final: {final_flame_angle:.1f}°')
    ax3.axhline(y=initial_flame_angle, color='c', linestyle=':', linewidth=2,
               label=f'Initial: {initial_flame_angle:.1f}°')
    ax3.set_xlabel('Time [s]', fontsize=11)
    ax3.set_ylabel('Flame Angle [degrees]', fontsize=11)
    ax3.set_title('Flame Angle Evolution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([initial_flame_angle - 5, final_flame_angle + 10])

    # 2d: Distance from anchor
    t_anchor = [t_history[i] for i, d in enumerate(anchor_distances) if not np.isnan(d)]
    valid_dists = [d for d in anchor_distances if not np.isnan(d)]
    ax4.plot(t_anchor, valid_dists, 'm-', linewidth=2)
    ax4.axhline(y=anchor_radius, color='r', linestyle='--', linewidth=2,
               label=f'Anchor radius: {anchor_radius:.4f} m')
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.set_ylabel('Min Distance from Anchor [m]', fontsize=11)
    ax4.set_title('Flame Anchoring Verification', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'flame_wall_attachment_analysis_t{t_final}.png',
               dpi=300, bbox_inches='tight')
    print(f"Saved: flame_wall_attachment_analysis_t{t_final}.png")

    # Figure 3: Final comparison
    fig3, ax = plt.subplots(1, 1, figsize=(10, 10))

    G_final = G_history[-1]

    # Plot level set
    levels = np.linspace(-1, 1, 21)
    contourf = ax.contourf(solver.X, solver.Y, G_final, levels=levels,
                          cmap='RdBu_r', extend='both', alpha=0.7)

    # Plot numerical flame
    ax.contour(solver.X, solver.Y, G_final, levels=[0], colors='black',
              linewidths=3, label='Numerical Flame')

    # Plot analytical
    x_analytical = np.linspace(0, Lx, 100)
    y_analytical = compute_flame_height_analytical(x_analytical, S_L, u_y)
    ax.plot(x_analytical, y_analytical, 'g--', linewidth=3, label='Analytical (45°)')

    # Plot initial line
    ax.plot([x0, x1], [y0, y1], 'c--', linewidth=3,
           label=f'Initial line ({initial_flame_angle:.1f}°)')

    # Boundaries
    ax.axvline(x=0, color='brown', linewidth=5, label='Wall', alpha=0.7)
    ax.axhline(y=0, color='blue', linewidth=3, alpha=0.5, label='Inlet')

    # Anchor
    ax.plot(x_anchor, y_anchor, 'ro', markersize=15, markeredgecolor='white',
           markeredgewidth=3, label='Anchor Point', zorder=10)
    circle = plt.Circle((x_anchor, y_anchor), anchor_radius,
                       color='red', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)

    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title(f'Final Flame Profile (t = {t_final:.2f}s)\n'
                f'Initial: line from (0,0) to (1.0,0.2), {initial_flame_angle:.1f}°\n'
                f'Final: Stabilized at {final_flame_angle:.1f}°',
                fontsize=13, fontweight='bold')
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, 1.5])
    ax.set_aspect('equal')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(contourf, ax=ax, label='G', shrink=0.8)

    plt.tight_layout()
    plt.savefig(f'flame_wall_attachment_final_t{t_final}.png',
               dpi=300, bbox_inches='tight')
    print(f"Saved: flame_wall_attachment_final_t{t_final}.png")

    plt.show()

    print("\n" + "="*80)
    print("Simulation completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    # Default parameters
    t_final = 1.0
    scheme = 'rk2'
    use_reinit = True

    # Parse command-line arguments
    for arg in sys.argv[1:]:
        if arg.lower() in ['euler', 'rk2']:
            scheme = arg.lower()
        elif arg.startswith('t=') or arg.startswith('time='):
            t_final = float(arg.split('=')[1])
        elif arg == 'no_reinit':
            use_reinit = False

    print(f"\nRunning flame wall attachment test:")
    print(f"  Initial: line from (0,0) to (1.0, 0.2)")
    print(f"  Above line: BURNT, Below line: UNBURNT")
    print(f"  Final: 45° angle (steady state)")
    print(f"  t_final = {t_final}")
    print(f"  scheme = {scheme}")
    print(f"  use_reinit = {use_reinit}\n")

    test_flame_wall_attachment(t_final=t_final, time_scheme=scheme,
                               use_reinit=use_reinit)
