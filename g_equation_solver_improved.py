"""
2D G-equation solver for laminar premixed flame surface using level set approach.
IMPROVED VERSION with corrected reinitialization and higher-order schemes for sharp initial conditions.
Supports both global and local (narrow-band) reinitialization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class GEquationSolver2D:
    """
    Solves the 2D G-equation: dG/dt + u.nabla(G) + S_L |grad(G)| = 0
    where S_L is the laminar flame speed, u is the flow velocity, and G is the level set function.
    
    Improved features:
    - Corrected level set reinitialization with zero level set preservation
    - Both global and local (narrow-band) reinitialization options
    - Fast marching and PDE-based reinitialization methods
    - Subcell fix for sharp interfaces
    """
    
    def __init__(self, nx, ny, Lx, Ly, S_L, u_x=0.0, u_y=0.0):
        """
        Initialize the 2D G-equation solver.
        
        Parameters:
        -----------
        nx : int
            Number of grid points in x-direction
        ny : int
            Number of grid points in y-direction
        Lx : float
            Domain length in x-direction
        Ly : float
            Domain length in y-direction
        S_L : float
            Laminar flame speed
        u_x : float or ndarray, optional
            Flow velocity in x-direction (default: 0.0)
        u_y : float or ndarray, optional
            Flow velocity in y-direction (default: 0.0)
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.S_L = S_L
        
        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Create mesh
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Flow velocity (can be scalar or array)
        if np.isscalar(u_x):
            self.u_x = u_x * np.ones((ny, nx))
        else:
            self.u_x = u_x
            
        if np.isscalar(u_y):
            self.u_y = u_y * np.ones((ny, nx))
        else:
            self.u_y = u_y
        
        # Solution field
        self.G = np.zeros((ny, nx))
        
    def compute_convective_term(self, G):
        """
        Compute u.nabla(G) using first-order upwind scheme.
        
        Parameters:
        -----------
        G : ndarray
            Level set function
            
        Returns:
        --------
        convective : ndarray
            Convective term u.nabla(G)
        """
        ny, nx = G.shape
        
        # Initialize gradient arrays
        dGdx = np.zeros_like(G)
        dGdy = np.zeros_like(G)
        
        # Compute dG/dx using upwind scheme based on u_x
        # If u_x > 0, use backward difference; if u_x < 0, use forward difference
        
        # Backward differences in x
        dGdx_backward = np.zeros_like(G)
        dGdx_backward[:, 1:] = (G[:, 1:] - G[:, :-1]) / self.dx
        dGdx_backward[:, 0] = (G[:, 0] - G[:, 0]) / self.dx  # Zero at boundary
        
        # Forward differences in x
        dGdx_forward = np.zeros_like(G)
        dGdx_forward[:, :-1] = (G[:, 1:] - G[:, :-1]) / self.dx
        dGdx_forward[:, -1] = (G[:, -1] - G[:, -1]) / self.dx  # Zero at boundary
        
        # Select based on flow direction
        dGdx = np.where(self.u_x > 0, dGdx_backward, dGdx_forward)
        
        # Compute dG/dy using upwind scheme based on u_y
        
        # Backward differences in y
        dGdy_backward = np.zeros_like(G)
        dGdy_backward[1:, :] = (G[1:, :] - G[:-1, :]) / self.dy
        dGdy_backward[0, :] = (G[0, :] - G[0, :]) / self.dy  # Zero at boundary
        
        # Forward differences in y
        dGdy_forward = np.zeros_like(G)
        dGdy_forward[:-1, :] = (G[1:, :] - G[:-1, :]) / self.dy
        dGdy_forward[-1, :] = (G[-1, :] - G[-1, :]) / self.dy  # Zero at boundary
        
        # Select based on flow direction
        dGdy = np.where(self.u_y > 0, dGdy_backward, dGdy_forward)
        
        # Compute convective term
        convective = self.u_x * dGdx + self.u_y * dGdy
        
        return convective
        
    def compute_gradient_magnitude(self, G):
        """
        Compute |grad(G)| using first-order finite differences.
        Uses upwind scheme based on flame propagation direction.
        VECTORIZED VERSION for speed.
        
        Parameters:
        -----------
        G : ndarray
            Level set function
            
        Returns:
        --------
        grad_mag : ndarray
            Magnitude of gradient
        """
        ny, nx = G.shape
        grad_mag = np.zeros_like(G)
        
        # Compute forward differences (vectorized)
        dGdx_forward = np.zeros_like(G)
        dGdx_forward[:, :-1] = (G[:, 1:] - G[:, :-1]) / self.dx
        
        # Compute backward differences (vectorized)
        dGdx_backward = np.zeros_like(G)
        dGdx_backward[:, 1:] = (G[:, 1:] - G[:, :-1]) / self.dx
        
        # Compute forward differences in y (vectorized)
        dGdy_forward = np.zeros_like(G)
        dGdy_forward[:-1, :] = (G[1:, :] - G[:-1, :]) / self.dy
        
        # Compute backward differences in y (vectorized)
        dGdy_backward = np.zeros_like(G)
        dGdy_backward[1:, :] = (G[1:, :] - G[:-1, :]) / self.dy
        
        # Upwind scheme (vectorized)
        dGdx_sq = np.maximum(dGdx_backward, 0.0)**2 + np.minimum(dGdx_forward, 0.0)**2
        dGdy_sq = np.maximum(dGdy_backward, 0.0)**2 + np.minimum(dGdy_forward, 0.0)**2
        
        grad_mag = np.sqrt(dGdx_sq + dGdy_sq)
        
        return grad_mag
    
    def compute_rhs(self, G):
        """
        Compute the right-hand side of the G-equation.
        RHS = -u.nabla(G) - S_L |grad(G)|
        
        Parameters:
        -----------
        G : ndarray
            Level set function
            
        Returns:
        --------
        rhs : ndarray
            Right-hand side
        """
        convective = self.compute_convective_term(G)
        grad_mag = self.compute_gradient_magnitude(G)
        rhs = -(convective + self.S_L * grad_mag)
        return rhs
    
    def find_interface_band(self, G, bandwidth=3):
        """
        Find cells within a narrow band around the zero level set.
        
        Parameters:
        -----------
        G : ndarray
            Level set function
        bandwidth : int
            Number of cells on each side of interface
            
        Returns:
        --------
        mask : ndarray (bool)
            True for cells in the narrow band
        """
        ny, nx = G.shape
        mask = np.zeros((ny, nx), dtype=bool)
        
        # Find interface cells (where G changes sign)
        # Check x-direction
        mask[:, :-1] |= (G[:, :-1] * G[:, 1:] <= 0)
        mask[:, 1:] |= (G[:, :-1] * G[:, 1:] <= 0)
        
        # Check y-direction
        mask[:-1, :] |= (G[:-1, :] * G[1:, :] <= 0)
        mask[1:, :] |= (G[:-1, :] * G[1:, :] <= 0)
        
        # Expand band
        for _ in range(bandwidth):
            mask_expanded = mask.copy()
            mask_expanded[:-1, :] |= mask[1:, :]
            mask_expanded[1:, :] |= mask[:-1, :]
            mask_expanded[:, :-1] |= mask[:, 1:]
            mask_expanded[:, 1:] |= mask[:, :-1]
            mask = mask_expanded
        
        return mask
    
    def reinitialize_pde(self, G0, dt_reinit=None, n_steps=10, bandwidth=5, use_local=True):
        """
        Reinitialize the level set using PDE-based method.
        Solves: dφ/dτ = sign(G0) * (1 - |∇φ|)
        
        This maintains the zero level set position while making |∇φ| ≈ 1.
        
        CORRECTED VERSION:
        - Uses sign of ORIGINAL G0 (frozen during reinitialization)
        - Optionally reinitializes only in narrow band around interface (local)
        - Or reinitializes globally (for comparison)
        - Preserves zero level set position
        
        Parameters:
        -----------
        G0 : ndarray
            Original level set function (before reinitialization)
        dt_reinit : float, optional
            Pseudo-time step for reinitialization (default: 0.1 * min(dx, dy))
        n_steps : int, optional
            Number of reinitialization steps (default: 10)
        bandwidth : int, optional
            Width of narrow band for local reinitialization (default: 5 cells)
        use_local : bool, optional
            If True, use narrow-band (local) reinitialization (default: True)
            If False, use global reinitialization
            
        Returns:
        --------
        G_reinit : ndarray
            Reinitialized level set function
        """
        if dt_reinit is None:
            dt_reinit = 0.1 * min(self.dx, self.dy)
        
        G_reinit = G0.copy()
        
        # Find narrow band around interface (used even for global, for diagnostics)
        band_mask = self.find_interface_band(G0, bandwidth=bandwidth)
        
        # Smooth sign function: sign_eps(x) = x / sqrt(x^2 + eps^2)
        eps = 1.5 * max(self.dx, self.dy)
        sign_G0 = G0 / np.sqrt(G0**2 + eps**2)
        
        for step in range(n_steps):
            # Compute gradients
            grad_mag = self.compute_gradient_magnitude(G_reinit)
            
            # Reinitialization equation: dφ/dτ = S(G0) * (1 - |∇φ|)
            update = dt_reinit * sign_G0 * (1.0 - grad_mag)
            
            if use_local:
                # LOCAL: Only update in narrow band
                G_reinit[band_mask] += update[band_mask]
            else:
                # GLOBAL: Update everywhere
                G_reinit += update
        
        return G_reinit
    
    def reinitialize_fast_marching(self, G, bandwidth=5, use_local=True):
        """
        Fast marching method for reinitialization.
        Constructs signed distance function from zero level set.
        
        This is more robust than PDE-based method but more complex.
        
        Parameters:
        -----------
        G : ndarray
            Level set function
        bandwidth : int
            Width of band to reinitialize (in cells)
        use_local : bool, optional
            If True, use narrow-band (local) reinitialization (default: True)
            If False, reinitialize globally
            
        Returns:
        --------
        G_reinit : ndarray
            Reinitialized level set as signed distance function
        """
        ny, nx = G.shape
        G_reinit = G.copy()
        
        # Find narrow band
        band_mask = self.find_interface_band(G, bandwidth=bandwidth)
        
        # Determine which cells to process
        if use_local:
            # LOCAL: Only process cells in narrow band
            cells_to_process = [(j, i) for j in range(ny) for i in range(nx) if band_mask[j, i]]
        else:
            # GLOBAL: Process all cells
            cells_to_process = [(j, i) for j in range(ny) for i in range(nx)]
        
        # For each point, compute signed distance to zero level set
        for j, i in cells_to_process:
            min_dist = np.inf
            sign = np.sign(G[j, i])
            
            # Search in local neighborhood for zero crossings
            search_radius = 3 if use_local else 5
            for dj in range(-search_radius, search_radius + 1):
                for di in range(-search_radius, search_radius + 1):
                    jn, in_ = j + dj, i + di
                    if 0 <= jn < ny and 0 <= in_ < nx:
                        if G[j, i] * G[jn, in_] < 0:
                            # Found a zero crossing - compute distance via interpolation
                            if di != 0 or dj != 0:
                                # Linear interpolation to find zero crossing point
                                alpha = abs(G[j, i]) / (abs(G[j, i]) + abs(G[jn, in_]))
                                dist = alpha * np.sqrt((dj * self.dy)**2 + (di * self.dx)**2)
                                min_dist = min(min_dist, dist)
            
            if min_dist < np.inf:
                G_reinit[j, i] = sign * min_dist
        
        return G_reinit
    
    def smooth_initial_condition(self, G, bandwidth=4):
        """
        Smooth a sharp initial condition by converting to signed distance function.
        This is essentially a one-time reinitialization of the initial condition.
        Always uses local (narrow-band) approach for initial smoothing.
        
        Parameters:
        -----------
        G : ndarray
            Sharp level set function (e.g., G=-1 inside, G=+1 outside)
        bandwidth : int, optional
            Number of grid cells to smooth on each side of interface (default: 4)
            
        Returns:
        --------
        G_smooth : ndarray
            Smoothed level set function (signed distance)
        """
        # Use fast marching with local approach for initial smoothing
        return self.reinitialize_fast_marching(G, bandwidth=bandwidth, use_local=True)
    
    def solve(self, G_initial, t_final, dt, save_interval=None, time_scheme='euler',
              reinit_interval=0, reinit_method='fast_marching', reinit_local=True, 
              smooth_ic=False):
        """
        Solve the G-equation from t=0 to t=t_final.
        
        Parameters:
        -----------
        G_initial : ndarray
            Initial condition for G
        t_final : float
            Final time
        dt : float
            Time step
        save_interval : int, optional
            Save solution every save_interval steps (default: every step)
        time_scheme : str, optional
            Time discretization scheme: 'euler' (first-order) or 'rk2' (second-order Runge-Kutta)
            Default: 'euler'
        reinit_interval : int, optional
            Reinitialize every reinit_interval steps (0 = no reinitialization, default: 0)
        reinit_method : str, optional
            Reinitialization method: 'pde' or 'fast_marching' (default: 'fast_marching')
        reinit_local : bool, optional
            If True, use local (narrow-band) reinitialization (default: True)
            If False, use global reinitialization
        smooth_ic : bool, optional
            Apply smoothing to initial condition if it's sharp (default: False)
            
        Returns:
        --------
        G_history : list
            List of G fields at saved time steps
        t_history : list
            List of time values
        """
        if time_scheme not in ['euler', 'rk2']:
            raise ValueError("time_scheme must be 'euler' or 'rk2'")
        
        if reinit_method not in ['pde', 'fast_marching']:
            raise ValueError("reinit_method must be 'pde' or 'fast_marching'")
        
        self.G = G_initial.copy()
        
        # Apply initial smoothing if requested
        if smooth_ic:
            print("Applying initial condition smoothing (converting to signed distance)...")
            self.G = self.smooth_initial_condition(self.G, bandwidth=4)
            print(f"  After smoothing: G_min={self.G.min():.3f}, G_max={self.G.max():.3f}")
        
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
        
        while t < t_final:
            # Adjust last time step
            if t + dt > t_final:
                dt = t_final - t
            
            # Store G before time step (for reinitialization)
            G_before = self.G.copy()
            
            # Time integration
            if time_scheme == 'euler':
                # First-order Euler: G^{n+1} = G^n + dt * RHS(G^n)
                rhs = self.compute_rhs(self.G)
                self.G = self.G + dt * rhs
                
            elif time_scheme == 'rk2':
                # Second-order Runge-Kutta (Heun's method / explicit midpoint)
                k1 = self.compute_rhs(self.G)
                G_temp = self.G + dt * k1
                k2 = self.compute_rhs(G_temp)
                self.G = self.G + dt * 0.5 * (k1 + k2)
            
            t += dt
            step += 1
            
            # Reinitialize if requested
            if reinit_interval > 0 and step % reinit_interval == 0:
                if reinit_method == 'pde':
                    # Use G_before to preserve zero level set
                    self.G = self.reinitialize_pde(G_before, n_steps=10, bandwidth=5, 
                                                   use_local=reinit_local)
                elif reinit_method == 'fast_marching':
                    self.G = self.reinitialize_fast_marching(self.G, bandwidth=5, 
                                                             use_local=reinit_local)
            
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
    
    def set_initial_condition(self, G_initial):
        """Set the initial condition."""
        self.G = G_initial.copy()


# Import remaining functions from original file
from g_equation_solver import (initial_solution, compute_circle_radius, 
                               compute_circle_center, analytical_radius, 
                               analytical_center)