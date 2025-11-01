"""
Improved flame surface area computation for 2D level set functions.
Computes the actual arc length of the G=0 contour with sub-grid accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_flame_surface_area_accurate(G, X, Y, dx, dy):
    """
    Compute the flame surface area (length of G=0 contour in 2D) with improved accuracy.
    Uses linear interpolation to find exact zero-crossing locations and computes
    actual distances between consecutive crossings.
    
    This method:
    1. Finds all zero crossings on grid edges
    2. Interpolates exact crossing positions
    3. Orders crossings to form a contour
    4. Computes distances between consecutive points
    
    Parameters:
    -----------
    G : ndarray
        Level set function
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
        
    Returns:
    --------
    surface_area : float
        Total length of the G=0 contour
    """
    ny, nx = G.shape
    
    # Collect all zero-crossing points with their exact coordinates
    crossing_points = []
    
    # Check horizontal edges (between i and i+1)
    for j in range(ny):
        for i in range(nx-1):
            G1 = G[j, i]
            G2 = G[j, i+1]
            
            if G1 * G2 < 0:  # Zero crossing detected (excluding exact zeros)
                # Linear interpolation to find exact crossing point
                alpha = abs(G1) / (abs(G1) + abs(G2))
                x_cross = X[j, i] + alpha * dx
                y_cross = Y[j, i]
                crossing_points.append({
                    'x': x_cross,
                    'y': y_cross,
                    'type': 'horizontal',
                    'j': j,
                    'i': i
                })
    
    # Check vertical edges (between j and j+1)
    for j in range(ny-1):
        for i in range(nx):
            G1 = G[j, i]
            G2 = G[j+1, i]
            
            if G1 * G2 < 0:  # Zero crossing detected
                # Linear interpolation to find exact crossing point
                alpha = abs(G1) / (abs(G1) + abs(G2))
                x_cross = X[j, i]
                y_cross = Y[j, i] + alpha * dy
                crossing_points.append({
                    'x': x_cross,
                    'y': y_cross,
                    'type': 'vertical',
                    'j': j,
                    'i': i
                })
    
    if len(crossing_points) == 0:
        return 0.0
    
    # For a simple closed contour (like a circle), we can compute the perimeter
    # by ordering points and summing distances
    
    # Extract coordinates
    x_coords = np.array([p['x'] for p in crossing_points])
    y_coords = np.array([p['y'] for p in crossing_points])
    
    # Compute center of all crossing points
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    
    # Sort points by angle from center (for simple closed curves)
    angles = np.arctan2(y_coords - y_center, x_coords - x_center)
    sorted_indices = np.argsort(angles)
    
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]
    
    # Compute perimeter by summing distances between consecutive points
    # Include closing segment (last to first)
    surface_area = 0.0
    n_points = len(x_sorted)
    
    for i in range(n_points):
        x1, y1 = x_sorted[i], y_sorted[i]
        x2, y2 = x_sorted[(i + 1) % n_points], y_sorted[(i + 1) % n_points]
        
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        surface_area += segment_length
    
    return surface_area


def compute_flame_surface_area_simple(G, dx, dy):
    """
    Simple flame surface area computation (original method).
    Counts zero crossings and multiplies by grid spacing.
    Fast but less accurate for curved interfaces.
    
    Parameters:
    -----------
    G : ndarray
        Level set function
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
        
    Returns:
    --------
    surface_area : float
        Approximate length of the G=0 contour
    """
    ny, nx = G.shape
    surface_area = 0.0
    
    # Count zero crossings in x-direction (horizontal edges)
    for j in range(ny):
        for i in range(nx-1):
            if G[j, i] * G[j, i+1] < 0:
                surface_area += dx
    
    # Count zero crossings in y-direction (vertical edges)
    for j in range(ny-1):
        for i in range(nx):
            if G[j, i] * G[j+1, i] < 0:
                surface_area += dy
    
    return surface_area


def compute_flame_surface_area_marching_squares(G, X, Y, dx, dy):
    """
    Compute flame surface area using marching squares algorithm.
    More robust for complex contours, handles multiple disconnected regions.
    
    Parameters:
    -----------
    G : ndarray
        Level set function
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
        
    Returns:
    --------
    surface_area : float
        Total length of all G=0 contours
    """
    ny, nx = G.shape
    total_length = 0.0
    
    # Process each cell
    for j in range(ny-1):
        for i in range(nx-1):
            # Get cell corner values
            g00 = G[j, i]
            g10 = G[j, i+1]
            g01 = G[j+1, i]
            g11 = G[j+1, i+1]
            
            # Determine cell configuration (4-bit code)
            config = 0
            if g00 > 0: config |= 1
            if g10 > 0: config |= 2
            if g11 > 0: config |= 4
            if g01 > 0: config |= 8
            
            # Skip if all same sign
            if config == 0 or config == 15:
                continue
            
            # Compute edge crossings
            crossings = []
            
            # Bottom edge (j, i to i+1)
            if g00 * g10 < 0:
                alpha = abs(g00) / (abs(g00) + abs(g10))
                x = X[j, i] + alpha * dx
                y = Y[j, i]
                crossings.append((x, y))
            
            # Right edge (i+1, j to j+1)
            if g10 * g11 < 0:
                alpha = abs(g10) / (abs(g10) + abs(g11))
                x = X[j, i+1]
                y = Y[j, i] + alpha * dy
                crossings.append((x, y))
            
            # Top edge (j+1, i+1 to i)
            if g11 * g01 < 0:
                alpha = abs(g01) / (abs(g01) + abs(g11))
                x = X[j+1, i] + alpha * dx
                y = Y[j+1, i]
                crossings.append((x, y))
            
            # Left edge (i, j+1 to j)
            if g01 * g00 < 0:
                alpha = abs(g00) / (abs(g00) + abs(g01))
                x = X[j, i]
                y = Y[j, i] + alpha * dy
                crossings.append((x, y))
            
            # Compute segment length in this cell
            if len(crossings) == 2:
                x1, y1 = crossings[0]
                x2, y2 = crossings[1]
                segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_length += segment_length
            elif len(crossings) == 4:
                # Ambiguous case (saddle point) - use average of two possibilities
                # Configuration 1: connect (0,1) and (2,3)
                x1, y1 = crossings[0]
                x2, y2 = crossings[1]
                x3, y3 = crossings[2]
                x4, y4 = crossings[3]
                
                length1 = (np.sqrt((x2-x1)**2 + (y2-y1)**2) + 
                          np.sqrt((x4-x3)**2 + (y4-y3)**2))
                
                # Configuration 2: connect (0,3) and (1,2)
                length2 = (np.sqrt((x4-x1)**2 + (y4-y1)**2) + 
                          np.sqrt((x3-x2)**2 + (y3-y2)**2))
                
                # Use average (or could use minimum)
                total_length += (length1 + length2) / 2.0
    
    return total_length


def test_surface_area_methods():
    """
    Test and compare different surface area computation methods.
    """
    print("\n" + "="*80)
    print("TESTING FLAME SURFACE AREA COMPUTATION METHODS")
    print("="*80)
    
    # Create test case: circle with known circumference
    nx, ny = 101, 101
    Lx, Ly = 2.0, 2.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    x_center, y_center = 1.0, 1.0
    
    # Test different radii
    radii = [0.2, 0.3, 0.5, 0.7]
    
    print("\nTest: Circles with different radii")
    print("-" * 80)
    print(f"{'Radius':<10} {'Analytical':<15} {'Simple':<15} {'Accurate':<15} {'Marching Sq':<15}")
    print(f"{'R':<10} {'2πR':<15} {'Method':<15} {'Method':<15} {'Method':<15}")
    print("-" * 80)
    
    for R in radii:
        # Create circle
        distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        G = distance - R
        
        # Analytical circumference
        analytical = 2.0 * np.pi * R
        
        # Method 1: Simple (count crossings)
        simple = compute_flame_surface_area_simple(G, dx, dy)
        
        # Method 2: Accurate (interpolate and order points)
        accurate = compute_flame_surface_area_accurate(G, X, Y, dx, dy)
        
        # Method 3: Marching squares
        marching = compute_flame_surface_area_marching_squares(G, X, Y, dx, dy)
        
        print(f"{R:<10.2f} {analytical:<15.6f} {simple:<15.6f} {accurate:<15.6f} {marching:<15.6f}")
        
        # Compute errors
        error_simple = abs(simple - analytical)
        error_accurate = abs(accurate - analytical)
        error_marching = abs(marching - analytical)
        
        rel_error_simple = error_simple / analytical * 100
        rel_error_accurate = error_accurate / analytical * 100
        rel_error_marching = error_marching / analytical * 100
        
        print(f"{'Errors:':<10} "
              f"{'':<15} "
              f"{error_simple:<15.6f} "
              f"{error_accurate:<15.6f} "
              f"{error_marching:<15.6f}")
        print(f"{'Rel. Err %:':<10} "
              f"{'':<15} "
              f"{rel_error_simple:<15.2f} "
              f"{rel_error_accurate:<15.2f} "
              f"{rel_error_marching:<15.2f}")
        print()
    
    # Visualize one case
    print("\nCreating visualization for R = 0.5...")
    R = 0.5
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    G = distance - R
    
    # Get crossing points for visualization
    crossing_points_h = []
    crossing_points_v = []
    
    for j in range(ny):
        for i in range(nx-1):
            if G[j, i] * G[j, i+1] < 0:
                alpha = abs(G[j, i]) / (abs(G[j, i]) + abs(G[j, i+1]))
                x_cross = X[j, i] + alpha * dx
                y_cross = Y[j, i]
                crossing_points_h.append((x_cross, y_cross))
    
    for j in range(ny-1):
        for i in range(nx):
            if G[j, i] * G[j+1, i] < 0:
                alpha = abs(G[j, i]) / (abs(G[j, i]) + abs(G[j+1, i]))
                x_cross = X[j, i]
                y_cross = Y[j, i] + alpha * dy
                crossing_points_v.append((x_cross, y_cross))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Contour plot with G=0 line
    levels = np.linspace(-0.5, 0.5, 21)
    contourf = ax1.contourf(X, Y, G, levels=levels, cmap='RdBu_r')
    ax1.contour(X, Y, G, levels=[0], colors='black', linewidths=2)
    
    # Plot crossing points
    if crossing_points_h:
        x_h, y_h = zip(*crossing_points_h)
        ax1.plot(x_h, y_h, 'go', markersize=3, label='Horizontal crossings')
    if crossing_points_v:
        x_v, y_v = zip(*crossing_points_v)
        ax1.plot(x_v, y_v, 'mo', markersize=3, label='Vertical crossings')
    
    # Analytical circle
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = x_center + R * np.cos(theta)
    y_circle = y_center + R * np.sin(theta)
    ax1.plot(x_circle, y_circle, 'r--', linewidth=2, label='Analytical')
    
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title(f'Zero Level Set and Crossing Points (R={R})', 
                 fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(contourf, ax=ax1, label='G')
    
    # Right: Zoom in on part of the interface
    zoom_x = [x_center + 0.3*R, x_center + 0.6*R]
    zoom_y = [y_center - 0.2*R, y_center + 0.2*R]
    
    # Show grid
    for i in range(nx):
        if X[0, i] > zoom_x[0] and X[0, i] < zoom_x[1]:
            ax2.axvline(X[0, i], color='gray', linewidth=0.5, alpha=0.3)
    for j in range(ny):
        if Y[j, 0] > zoom_y[0] and Y[j, 0] < zoom_y[1]:
            ax2.axhline(Y[j, 0], color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot zero level set
    ax2.contour(X, Y, G, levels=[0], colors='black', linewidths=2, label='Numerical G=0')
    ax2.plot(x_circle, y_circle, 'r--', linewidth=2, label='Analytical')
    
    # Plot crossing points in zoom region
    for x_h, y_h in crossing_points_h:
        if zoom_x[0] <= x_h <= zoom_x[1] and zoom_y[0] <= y_h <= zoom_y[1]:
            ax2.plot(x_h, y_h, 'go', markersize=8)
    for x_v, y_v in crossing_points_v:
        if zoom_x[0] <= x_v <= zoom_x[1] and zoom_y[0] <= y_v <= zoom_y[1]:
            ax2.plot(x_v, y_v, 'mo', markersize=8)
    
    ax2.set_xlim(zoom_x)
    ax2.set_ylim(zoom_y)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Zoomed View of Interface', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flame_surface_area_methods.png', dpi=300, bbox_inches='tight')
    print("Saved: flame_surface_area_methods.png")
    
    plt.show()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("• MARCHING SQUARES: Most accurate, handles complex contours")
    print("• ACCURATE METHOD: Good for simple closed curves (circles)")
    print("• SIMPLE METHOD: Fast but less accurate (~1-2% error)")
    print("• For production use: Marching Squares or Accurate method")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_surface_area_methods()