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
        # WENO5 mode (vectorized by default). Can be toggled for benchmarking.
        self.use_vectorized_weno5 = True

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

    # ----------------- WENO5 helpers for high-order advection -----------------
    def _weno5_reconstruct_positive(self, v_m2, v_m1, v_0, v_p1, v_p2, eps=1e-6):
        # Left-biased reconstruction at i+1/2 (positive flux)
        p0 = (2.0/6.0)*v_m2 - (7.0/6.0)*v_m1 + (11.0/6.0)*v_0
        p1 = (-1.0/6.0)*v_m1 + (5.0/6.0)*v_0 + (2.0/6.0)*v_p1
        p2 = (2.0/6.0)*v_0 + (5.0/6.0)*v_p1 - (1.0/6.0)*v_p2
        beta0 = (13.0/12.0)*(v_m2 - 2*v_m1 + v_0)**2 + 0.25*(v_m2 - 4*v_m1 + 3*v_0)**2
        beta1 = (13.0/12.0)*(v_m1 - 2*v_0 + v_p1)**2 + 0.25*(v_m1 - v_p1)**2
        beta2 = (13.0/12.0)*(v_0 - 2*v_p1 + v_p2)**2 + 0.25*(3*v_0 - 4*v_p1 + v_p2)**2
        d0, d1, d2 = 0.1, 0.6, 0.3
        a0 = d0 / (eps + beta0)**2
        a1 = d1 / (eps + beta1)**2
        a2 = d2 / (eps + beta2)**2
        w0 = a0 / (a0 + a1 + a2)
        w1 = a1 / (a0 + a1 + a2)
        w2 = a2 / (a0 + a1 + a2)
        return w0*p0 + w1*p1 + w2*p2

    def _weno5_reconstruct_negative(self, v_m2, v_m1, v_0, v_p1, v_p2, eps=1e-6):
        # Right-biased reconstruction at i-1/2 (negative flux) mirrored
        # Reconstruct value at i-1/2 from the right (indices shifted)
        p0 = (2.0/6.0)*v_p2 - (7.0/6.0)*v_p1 + (11.0/6.0)*v_0
        p1 = (-1.0/6.0)*v_p1 + (5.0/6.0)*v_0 + (2.0/6.0)*v_m1
        p2 = (2.0/6.0)*v_0 + (5.0/6.0)*v_m1 - (1.0/6.0)*v_m2
        beta0 = (13.0/12.0)*(v_p2 - 2*v_p1 + v_0)**2 + 0.25*(v_p2 - 4*v_p1 + 3*v_0)**2
        beta1 = (13.0/12.0)*(v_p1 - 2*v_0 + v_m1)**2 + 0.25*(v_p1 - v_m1)**2
        beta2 = (13.0/12.0)*(v_0 - 2*v_m1 + v_m2)**2 + 0.25*(3*v_0 - 4*v_m1 + v_m2)**2
        d0, d1, d2 = 0.1, 0.6, 0.3
        a0 = d0 / (eps + beta0)**2
        a1 = d1 / (eps + beta1)**2
        a2 = d2 / (eps + beta2)**2
        w0 = a0 / (a0 + a1 + a2)
        w1 = a1 / (a0 + a1 + a2)
        w2 = a2 / (a0 + a1 + a2)
        return w0*p0 + w1*p1 + w2*p2

    def _weno5_flux_1d(self, v, u, axis, dx):
        # v: 1D array of G values along a line, u: 1D velocities on same line
        n = v.shape[0]
        # face fluxes length n-1
        F_face = np.zeros(n-1, dtype=float)
        # approximate face velocity by average
        u_face = 0.5 * (u[:-1] + u[1:])
        # iterate interior faces with sufficient stencil: i from 2..n-3 (face index)
        for i in range(n-1):
            # boundaries: fallback to upwind first order
            if i < 2 or i > n-4:
                F_face[i] = (u_face[i] * (v[i] if u_face[i] >= 0 else v[i+1]))
                continue
            if u_face[i] >= 0:
                # reconstruct G at i+1/2 from left
                G_face = self._weno5_reconstruct_positive(v[i-2], v[i-1], v[i], v[i+1], v[i+2])
                F_face[i] = u_face[i] * G_face
            else:
                # reconstruct from right for negative velocity
                G_face = self._weno5_reconstruct_negative(v[i-1], v[i], v[i+1], v[i+2], v[i+3])
                F_face[i] = u_face[i] * G_face
        # conservative derivative
        dvdt = np.zeros_like(v)
        # interior cells i=1..n-2 have both faces
        dvdt[1:-1] = (F_face[1:] - F_face[:-1]) / dx
        # boundaries fallback to first-order upwind derivative
        # left boundary i=0
        dvdt[0] = ((u[0] * v[0]) - (u[0] * v[0])) / dx
        # right boundary i=n-1
        dvdt[-1] = ((u[-1] * v[-1]) - (u[-1] * v[-1])) / dx
        return dvdt

    def _weno5_flux_1d_vec_lastdim(self, V, U, dx):
        """
        Vectorized WENO5 flux along the last dimension.
        Computes conservative derivative d(u*G)/dx for each row in V (shape: [m, n]).
        Boundary faces fall back to first-order upwind.
        Returns array with same shape as V.
        """
        # V, U: shape (m, n)
        m, n = V.shape
        if n < 6:
            # too short for WENO5 stencils; fallback to upwind conservative form
            u_face = 0.5 * (U[..., :-1] + U[..., 1:])
            F_face = u_face * np.where(u_face >= 0.0, V[..., :-1], V[..., 1:])
            dV = np.zeros_like(V)
            dV[..., 1:-1] = (F_face[..., 1:] - F_face[..., :-1]) / dx
            return dV

        # Face velocities
        u_face = 0.5 * (U[..., :-1] + U[..., 1:])  # (m, n-1)

        # Initialize with first-order upwind as default (boundaries)
        F_face = u_face * np.where(u_face >= 0.0, V[..., :-1], V[..., 1:])

        # Interior faces indices f in [2 .. n-4]
        # Build stencil slices for positive reconstruction (left-biased at i+1/2)
        # Interior face indices replicate scalar loop condition (i in [2 .. n-5])
        # Adjust slices so all stencil arrays have consistent length = n-6 faces.
        vm2 = V[..., 0:n-6]   # i-2
        vm1 = V[..., 1:n-5]   # i-1
        v0  = V[..., 2:n-4]   # i
        vp1 = V[..., 3:n-3]   # i+1
        vp2 = V[..., 4:n-2]   # i+2
        ufi = u_face[..., 2:n-4]  # face velocity at i+1/2 for i in [2..n-5]

        # Smoothness indicators and polynomials (vectorized)
        eps = 1e-6
        # Positive reconstruction weights
        beta0 = (13.0/12.0) * (vm2 - 2*vm1 + v0)**2 + 0.25 * (vm2 - 4*vm1 + 3*v0)**2
        beta1 = (13.0/12.0) * (vm1 - 2*v0 + vp1)**2 + 0.25 * (vm1 - vp1)**2
        beta2 = (13.0/12.0) * (v0  - 2*vp1 + vp2)**2 + 0.25 * (3*v0 - 4*vp1 + vp2)**2
        d0, d1, d2 = 0.1, 0.6, 0.3
        a0 = d0 / (eps + beta0)**2
        a1 = d1 / (eps + beta1)**2
        a2 = d2 / (eps + beta2)**2
        wsum = a0 + a1 + a2
        w0 = a0 / wsum
        w1 = a1 / wsum
        w2 = a2 / wsum
        p0 = (2.0/6.0)*vm2 - (7.0/6.0)*vm1 + (11.0/6.0)*v0
        p1 = (-1.0/6.0)*vm1 + (5.0/6.0)*v0 + (2.0/6.0)*vp1
        p2 = (2.0/6.0)*v0 + (5.0/6.0)*vp1 - (1.0/6.0)*vp2
        G_pos = w0*p0 + w1*p1 + w2*p2  # (m, n-5)

        # Negative reconstruction (right-biased at i+1/2)
        # Stencil variables around face index f (cells f,f+1):
        v_im1 = V[..., 1:n-5]   # i-1
        v_i   = V[..., 2:n-4]   # i
        v_ip1 = V[..., 3:n-3]   # i+1
        v_ip2 = V[..., 4:n-2]   # i+2
        v_ip3 = V[..., 5:n-1]   # i+3
        beta0n = (13.0/12.0) * (v_ip1 - 2*v_ip2 + v_ip3)**2 + 0.25 * (v_ip1 - 4*v_ip2 + 3*v_ip3)**2
        beta1n = (13.0/12.0) * (v_i   - 2*v_ip1 + v_ip2)**2 + 0.25 * (v_i - v_ip2)**2
        beta2n = (13.0/12.0) * (v_im1 - 2*v_i   + v_ip1)**2 + 0.25 * (3*v_im1 - 4*v_i + v_ip1)**2
        a0n = d0 / (eps + beta0n)**2
        a1n = d1 / (eps + beta1n)**2
        a2n = d2 / (eps + beta2n)**2
        wsum_n = a0n + a1n + a2n
        w0n = a0n / wsum_n
        w1n = a1n / wsum_n
        w2n = a2n / wsum_n
        q0 = (2.0/6.0)*v_ip1 - (7.0/6.0)*v_ip2 + (11.0/6.0)*v_ip3
        q1 = (-1.0/6.0)*v_i   + (5.0/6.0)*v_ip1 + (2.0/6.0)*v_ip2
        q2 = (2.0/6.0)*v_im1 + (5.0/6.0)*v_i   - (1.0/6.0)*v_ip1
        G_neg = w0n*q0 + w1n*q1 + w2n*q2  # (m, n-5)

        # Select by sign of face velocity on interior faces [2..n-4]
        mask_pos = (ufi >= 0.0)
        F_int = np.where(mask_pos, ufi * G_pos, ufi * G_neg)
        F_face[..., 2:n-4] = F_int

        # Conservative derivative
        dV = np.zeros_like(V)
        dV[..., 1:-1] = (F_face[..., 1:] - F_face[..., :-1]) / dx
        return dV

    def compute_convective_term_weno5(self, G):
        """
        Compute u·∇G using 5th-order WENO reconstruction in each coordinate.
        Conservative flux-difference form per direction with local face velocities.
        """
        if self.use_vectorized_weno5:
            # Vectorized along rows (x) and columns (y)
            conv_x = self._weno5_flux_1d_vec_lastdim(G, self.u_x, self.dx)
            Gy = np.swapaxes(G, 0, 1)
            Uy = np.swapaxes(self.u_y, 0, 1)
            dFdy_T = self._weno5_flux_1d_vec_lastdim(Gy, Uy, self.dy)
            conv_y = np.swapaxes(dFdy_T, 0, 1)
        else:
            # Scalar loop version (original) for benchmarking
            ny, nx = G.shape
            conv_x = np.zeros_like(G)
            for j in range(ny):
                v = G[j, :]
                u = self.u_x[j, :]
                conv_x[j, :] = self._weno5_flux_1d(v, u, axis=1, dx=self.dx)
            conv_y = np.zeros_like(G)
            for i in range(nx):
                v = G[:, i]
                u = self.u_y[:, i]
                conv_y[:, i] = self._weno5_flux_1d(v, u, axis=0, dx=self.dy)
        # convective term is d(uG)/dx + d(vG)/dy which equals u dG/dx + v dG/dy if u,v const.
        # With variable u,v this is the conservative discretization of ∇·(G u).
        # To match original form, we accept conservative discretization (more robust).
        return conv_x + conv_y

    def set_weno5_mode(self, mode: str):
        """Set WENO5 implementation mode.

        Parameters
        ----------
        mode : {'vector','scalar'}
            'vector' uses fully vectorized implementation (default).
            'scalar' uses original Python loops (slow) for benchmarking comparison.
        """
        if mode not in ('vector','scalar'):
            raise ValueError("mode must be 'vector' or 'scalar'")
        self.use_vectorized_weno5 = (mode == 'vector')

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

    def compute_rhs(self, G, spatial_scheme: str = 'upwind'):
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
        if spatial_scheme == 'weno5':
            convective = self.compute_convective_term_weno5(G)
        else:
            convective = self.compute_convective_term(G)
        grad_mag = self.compute_gradient_magnitude(G)
        rhs = -(convective + self.S_L * grad_mag)
        return rhs

    # --- Optional pinning utilities to enforce boundary conditions ---
    def set_pinned_region(self, mask, values):
        """
        Define a region where G will be enforced each step.
        mask: boolean ndarray shape (ny, nx)
        values: ndarray or scalar; if ndarray, must be (ny, nx). Only mask cells are used.
        """
        if mask.shape != (self.ny, self.nx):
            raise ValueError(f"Pinned mask must have shape {(self.ny, self.nx)}, got {mask.shape}")
        self._pinned_mask = mask.astype(bool)
        if np.isscalar(values):
            self._pinned_values = np.full((self.ny, self.nx), float(values))
        else:
            if values.shape != (self.ny, self.nx):
                raise ValueError(f"Pinned values must have shape {(self.ny, self.nx)}, got {values.shape}")
            self._pinned_values = values.copy()

    def _apply_pinning(self):
        mask = getattr(self, '_pinned_mask', None)
        if mask is not None:
            self.G[mask] = self._pinned_values[mask]

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
              smooth_ic=False, velocity_updater=None, t0: float = 0.0, spatial_scheme: str = 'upwind'):
        """
        Solve the G-equation from t=t0 to t=t_final.

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
        if time_scheme not in ['euler', 'rk2', 'rk3']:
            raise ValueError("time_scheme must be 'euler', 'rk2' or 'rk3'")
        if spatial_scheme not in ['upwind', 'weno5']:
            raise ValueError("spatial_scheme must be 'upwind' or 'weno5'")

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
        t_history = [float(t0)]

        t = float(t0)
        step = 0

        # Determine save interval
        if save_interval is None:
            save_interval = 1

        print(f"Using {time_scheme.upper()} time discretization scheme (spatial: {spatial_scheme.upper()})")
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
                # Update velocity at current time t if callback is provided
                if velocity_updater is not None:
                    updated = velocity_updater(self, t)
                    if updated is not None:
                        ux, uy = updated
                        self.u_x, self.u_y = ux, uy
                # First-order Euler: G^{n+1} = G^n + dt * RHS(G^n)
                rhs = self.compute_rhs(self.G, spatial_scheme=spatial_scheme)
                self.G = self.G + dt * rhs
                # Enforce pinned boundary (if any)
                self._apply_pinning()
                # Optional: advance velocity to t+dt for next step
                if velocity_updater is not None:
                    updated = velocity_updater(self, t + dt)
                    if updated is not None:
                        ux, uy = updated
                        self.u_x, self.u_y = ux, uy

            elif time_scheme == 'rk2':
                # Stage 1: update velocity at time t
                if velocity_updater is not None:
                    updated = velocity_updater(self, t)
                    if updated is not None:
                        ux, uy = updated
                        self.u_x, self.u_y = ux, uy
                # Second-order Runge-Kutta (Heun's method / explicit midpoint)
                k1 = self.compute_rhs(self.G, spatial_scheme=spatial_scheme)
                G_temp = self.G + dt * k1
                # Stage 2: update velocity at time t + dt
                if velocity_updater is not None:
                    updated = velocity_updater(self, t + dt)
                    if updated is not None:
                        ux, uy = updated
                        self.u_x, self.u_y = ux, uy
                k2 = self.compute_rhs(G_temp, spatial_scheme=spatial_scheme)
                self.G = self.G + dt * 0.5 * (k1 + k2)
                # Enforce pinned boundary (if any)
                self._apply_pinning()

            elif time_scheme == 'rk3':
                # SSP RK3 (Shu-Osher)
                # Stage 1 @ t
                if velocity_updater is not None:
                    updated = velocity_updater(self, t)
                    if updated is not None:
                        ux, uy = updated
                        self.u_x, self.u_y = ux, uy
                k1 = self.compute_rhs(self.G, spatial_scheme=spatial_scheme)
                G1 = self.G + dt * k1
                # Stage 2 @ t+dt
                if velocity_updater is not None:
                    updated = velocity_updater(self, t + dt)
                    if updated is not None:
                        ux, uy = updated
                        self.u_x, self.u_y = ux, uy
                k2 = self.compute_rhs(G1, spatial_scheme=spatial_scheme)
                G2 = 0.75 * self.G + 0.25 * (G1 + dt * k2)
                # Stage 3 @ t+dt
                if velocity_updater is not None:
                    updated = velocity_updater(self, t + dt)
                    if updated is not None:
                        ux, uy = updated
                        self.u_x, self.u_y = ux, uy
                k3 = self.compute_rhs(G2, spatial_scheme=spatial_scheme)
                self.G = (1.0/3.0) * self.G + (2.0/3.0) * (G2 + dt * k3)
                self._apply_pinning()

            t += dt
            step += 1

            # Reinitialize if requested
            if reinit_interval > 0 and step % reinit_interval == 0:
                if reinit_method == 'pde':
                    # Use G_before to preserve zero level set
                    self.G = self.reinitialize_pde(
                        G_before,
                        dt_reinit=0.3 * min(self.dx, self.dy),
                        n_steps=30,
                        bandwidth=8,
                        use_local=reinit_local,
                    )
                elif reinit_method == 'fast_marching':
                    self.G = self.reinitialize_fast_marching(self.G, bandwidth=5,
                                                             use_local=reinit_local)
                # Re-apply pinning after reinitialization to preserve boundary
                self._apply_pinning()

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

                # Optional auto-reinitialization if gradient drifts too far from 1
                if reinit_interval == 0 and (avg_grad > 1.3 or avg_grad < 0.8):
                    if reinit_method == 'pde':
                        self.G = self.reinitialize_pde(
                            self.G,
                            dt_reinit=0.3 * min(self.dx, self.dy),
                            n_steps=30,
                            bandwidth=8,
                            use_local=reinit_local,
                        )
                    elif reinit_method == 'fast_marching':
                        self.G = self.reinitialize_fast_marching(self.G, bandwidth=5, use_local=reinit_local)
                    self._apply_pinning()

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