"""
2D G-equation solver for laminar premixed flame surface using level set approach.
Uses first-order finite difference for spatial discretization.
Time discretization: first-order Euler or second-order Runge-Kutta (RK2).
Includes convective term u.nabla(G).
Optimized version with vectorized operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class GEquationSolver2D:
    """
    Solves the 2D G-equation: dG/dt + u.nabla(G) + S_L |grad(G)| = 0
    where S_L is the laminar flame speed, u is the flow velocity, and G is the level set function.
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
    
    def solve(self, G_initial, t_final, dt, save_interval=None, time_scheme='euler'):
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
            
        Returns:
        --------
        G_history : list
            List of G fields at saved time steps
        t_history : list
            List of time values
        """
        if time_scheme not in ['euler', 'rk2']:
            raise ValueError("time_scheme must be 'euler' or 'rk2'")
        
        self.G = G_initial.copy()
        
        # Storage for history
        G_history = [self.G.copy()]
        t_history = [0.0]
        
        t = 0.0
        step = 0
        
        # Determine save interval
        if save_interval is None:
            save_interval = 1
        
        print(f"Using {time_scheme.upper()} time discretization scheme")
        
        while t < t_final:
            # Adjust last time step
            if t + dt > t_final:
                dt = t_final - t
            
            # Time integration
            if time_scheme == 'euler':
                # First-order Euler: G^{n+1} = G^n + dt * RHS(G^n)
                rhs = self.compute_rhs(self.G)
                self.G = self.G + dt * rhs
                
            elif time_scheme == 'rk2':
                # Second-order Runge-Kutta (Heun's method / explicit midpoint)
                # k1 = RHS(G^n)
                # k2 = RHS(G^n + dt * k1)
                # G^{n+1} = G^n + dt/2 * (k1 + k2)
                
                k1 = self.compute_rhs(self.G)
                G_temp = self.G + dt * k1
                k2 = self.compute_rhs(G_temp)
                self.G = self.G + dt * 0.5 * (k1 + k2)
            
            t += dt
            step += 1
            
            # Store solution at specified intervals
            if step % save_interval == 0:
                G_history.append(self.G.copy())
                t_history.append(t)
            
            if step % 100 == 0:
                print(f"Step {step}, t = {t:.4f}")
        
        # Ensure final time is saved
        if t_history[-1] < t_final:
            G_history.append(self.G.copy())
            t_history.append(t)
        
        return G_history, t_history
    
    def set_initial_condition(self, G_initial):
        """Set the initial condition."""
        self.G = G_initial.copy()


def initial_solution(X, Y, x_center, y_center, radius):
    """
    Create initial level set function for a circle.
    
    The level set is defined as:
    G(x,y,t=0) = sqrt((x-x_c)^2 + (y-y_c)^2) - R_0
    
    where (x_c, y_c) is the center and R_0 is the initial radius.
    The zero level set (G=0) represents the flame surface.
    
    Parameters:
    -----------
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    x_center : float
        X coordinate of circle center
    y_center : float
        Y coordinate of circle center
    radius : float
        Initial radius of circle
        
    Returns:
    --------
    G : ndarray
        Initial level set function
    """
    G = np.sqrt((X - x_center)**2 + (Y - y_center)**2) - radius
    return G


def compute_circle_radius(G, X, Y, x_center, y_center, dx, dy):
    """
    Compute the radius of the circle by interpolating the zero level set.
    Uses linear interpolation to find where G crosses zero.
    
    Parameters:
    -----------
    G : ndarray
        Level set function
    X : ndarray
        X coordinates
    Y : ndarray
        Y coordinates
    x_center : float
        X coordinate of circle center
    y_center : float
        Y coordinate of circle center
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
        
    Returns:
    --------
    radius : float
        Estimated radius
    """
    ny, nx = G.shape
    
    # Collect zero-crossing points using linear interpolation
    zero_crossing_radii = []
    
    # Check horizontal edges (between i and i+1)
    for j in range(ny):
        for i in range(nx-1):
            G1 = G[j, i]
            G2 = G[j, i+1]
            
            # Check if zero crossing occurs
            if G1 * G2 <= 0 and G1 != G2:
                # Linear interpolation to find exact crossing point
                alpha = -G1 / (G2 - G1)
                x_cross = X[j, i] + alpha * dx
                y_cross = Y[j, i]
                
                # Compute radius
                r = np.sqrt((x_cross - x_center)**2 + (y_cross - y_center)**2)
                zero_crossing_radii.append(r)
    
    # Check vertical edges (between j and j+1)
    for j in range(ny-1):
        for i in range(nx):
            G1 = G[j, i]
            G2 = G[j+1, i]
            
            # Check if zero crossing occurs
            if G1 * G2 <= 0 and G1 != G2:
                # Linear interpolation to find exact crossing point
                alpha = -G1 / (G2 - G1)
                x_cross = X[j, i]
                y_cross = Y[j, i] + alpha * dy
                
                # Compute radius
                r = np.sqrt((x_cross - x_center)**2 + (y_cross - y_center)**2)
                zero_crossing_radii.append(r)
    
    # Return mean radius from all zero crossings
    if len(zero_crossing_radii) > 0:
        radius = np.mean(zero_crossing_radii)
    else:
        # Fallback: shouldn't happen for well-defined circular level sets
        radius = 0.0
        print("Warning: No zero crossings found!")
    
    return radius


def compute_circle_center(G, X, Y, dx, dy):
    """
    Compute the center of the circle by averaging zero level set positions.
    
    Parameters:
    -----------
    G : ndarray
        Level set function
    X : ndarray
        X coordinates
    Y : ndarray
        Y coordinates
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
        
    Returns:
    --------
    x_center : float
        X coordinate of circle center
    y_center : float
        Y coordinate of circle center
    """
    ny, nx = G.shape
    
    # Collect zero-crossing points
    x_crossings = []
    y_crossings = []
    
    # Check horizontal edges
    for j in range(ny):
        for i in range(nx-1):
            G1 = G[j, i]
            G2 = G[j, i+1]
            
            if G1 * G2 <= 0 and G1 != G2:
                alpha = -G1 / (G2 - G1)
                x_cross = X[j, i] + alpha * dx
                y_cross = Y[j, i]
                x_crossings.append(x_cross)
                y_crossings.append(y_cross)
    
    # Check vertical edges
    for j in range(ny-1):
        for i in range(nx):
            G1 = G[j, i]
            G2 = G[j+1, i]
            
            if G1 * G2 <= 0 and G1 != G2:
                alpha = -G1 / (G2 - G1)
                x_cross = X[j, i]
                y_cross = Y[j, i] + alpha * dy
                x_crossings.append(x_cross)
                y_crossings.append(y_cross)
    
    if len(x_crossings) > 0:
        x_center = np.mean(x_crossings)
        y_center = np.mean(y_crossings)
    else:
        x_center = 0.0
        y_center = 0.0
    
    return x_center, y_center


def analytical_radius(t, R0, S_L):
    """
    Analytical solution for expanding circle radius.
    
    For a circular flame expanding at constant speed S_L:
    R(t) = R_0 + S_L * t
    
    Parameters:
    -----------
    t : float or ndarray
        Time
    R0 : float
        Initial radius
    S_L : float
        Laminar flame speed
        
    Returns:
    --------
    R : float or ndarray
        Radius at time t
    """
    return R0 + S_L * t


def analytical_center(t, x0, y0, u_x, u_y):
    """
    Analytical solution for moving circle center in uniform flow.
    
    Parameters:
    -----------
    t : float or ndarray
        Time
    x0 : float
        Initial x-coordinate of center
    y0 : float
        Initial y-coordinate of center
    u_x : float
        Flow velocity in x-direction
    u_y : float
        Flow velocity in y-direction
        
    Returns:
    --------
    x_center : float or ndarray
        X coordinate of center at time t
    y_center : float or ndarray
        Y coordinate of center at time t
    """
    x_center = x0 + u_x * t
    y_center = y0 + u_y * t
    return x_center, y_center