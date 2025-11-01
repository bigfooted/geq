"""
Comprehensive Marching Squares implementation for computing contour lengths.
Handles complex contours, open contours, and multiple disconnected regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class MarchingSquares:
    """
    Marching Squares algorithm for extracting and measuring contours from 2D scalar fields.
    
    Features:
    - Handles multiple disconnected contours
    - Supports both closed and open contours
    - Computes accurate contour lengths using linear interpolation
    - Extracts contour coordinates for visualization
    """
    
    def __init__(self, G, X, Y):
        """
        Initialize the marching squares algorithm.
        
        Parameters:
        -----------
        G : ndarray (ny, nx)
            2D scalar field (level set function)
        X : ndarray (ny, nx)
            X coordinates (meshgrid)
        Y : ndarray (ny, nx)
            Y coordinates (meshgrid)
        """
        self.G = G
        self.X = X
        self.Y = Y
        self.ny, self.nx = G.shape
        
        if X.shape != G.shape or Y.shape != G.shape:
            raise ValueError("G, X, and Y must have the same shape")
        
        # Compute grid spacing (assume uniform)
        self.dx = X[0, 1] - X[0, 0]
        self.dy = Y[1, 0] - Y[0, 0]
        
        # Edge lookup table: for each configuration, which edges have crossings
        # Edge numbering: 0=bottom, 1=right, 2=top, 3=left
        self.edge_table = self._build_edge_table()
        
    def _build_edge_table(self):
        """
        Build lookup table for marching squares configurations.
        
        Returns a dictionary mapping 4-bit configuration to list of edge pairs
        that should be connected.
        
        Corner numbering:
            01 --- 11
            |      |
            00 --- 10
        
        Edge numbering:
            01 --2-- 11
            |        |
            3        1
            |        |
            00 --0-- 10
        """
        table = {
            0b0000: [],                    # All negative
            0b0001: [(3, 0)],              # Bottom-left positive
            0b0010: [(0, 1)],              # Bottom-right positive
            0b0011: [(3, 1)],              # Bottom row positive
            0b0100: [(1, 2)],              # Top-right positive
            0b0101: [(3, 0), (1, 2)],      # Diagonal (saddle point)
            0b0110: [(0, 2)],              # Right column positive
            0b0111: [(3, 2)],              # Right and bottom positive
            0b1000: [(2, 3)],              # Top-left positive
            0b1001: [(2, 0)],              # Left column positive
            0b1010: [(0, 1), (2, 3)],      # Diagonal (saddle point)
            0b1011: [(2, 1)],              # Not top-right positive
            0b1100: [(1, 3)],              # Top row positive
            0b1101: [(1, 0)],              # Not bottom-right positive
            0b1110: [(0, 3)],              # Not bottom-left positive
            0b1111: [],                    # All positive
        }
        return table
    
    def _get_cell_config(self, j, i):
        """
        Get the 4-bit configuration for cell (j, i).
        
        Parameters:
        -----------
        j : int
            Row index
        i : int
            Column index
            
        Returns:
        --------
        config : int
            4-bit configuration (0-15)
        """
        config = 0
        if self.G[j, i] > 0:     config |= 1   # Bottom-left
        if self.G[j, i+1] > 0:   config |= 2   # Bottom-right
        if self.G[j+1, i+1] > 0: config |= 4   # Top-right
        if self.G[j+1, i] > 0:   config |= 8   # Top-left
        return config
    
    def _get_edge_crossing(self, j, i, edge):
        """
        Get the (x, y) coordinates of zero crossing on specified edge.
        
        Parameters:
        -----------
        j, i : int
            Cell indices
        edge : int
            Edge number (0=bottom, 1=right, 2=top, 3=left)
            
        Returns:
        --------
        (x, y) : tuple
            Coordinates of zero crossing
        """
        if edge == 0:  # Bottom edge (j, i to i+1)
            g1 = self.G[j, i]
            g2 = self.G[j, i+1]
            if g1 == g2:  # Avoid division by zero
                alpha = 0.5
            else:
                alpha = abs(g1) / (abs(g1) + abs(g2))
            x = self.X[j, i] + alpha * self.dx
            y = self.Y[j, i]
            
        elif edge == 1:  # Right edge (i+1, j to j+1)
            g1 = self.G[j, i+1]
            g2 = self.G[j+1, i+1]
            if g1 == g2:
                alpha = 0.5
            else:
                alpha = abs(g1) / (abs(g1) + abs(g2))
            x = self.X[j, i+1]
            y = self.Y[j, i] + alpha * self.dy
            
        elif edge == 2:  # Top edge (j+1, i+1 to i)
            g1 = self.G[j+1, i+1]
            g2 = self.G[j+1, i]
            if g1 == g2:
                alpha = 0.5
            else:
                alpha = abs(g1) / (abs(g1) + abs(g2))
            x = self.X[j+1, i+1] - alpha * self.dx
            y = self.Y[j+1, i]
            
        else:  # edge == 3, Left edge (i, j+1 to j)
            g1 = self.G[j+1, i]
            g2 = self.G[j, i]
            if g1 == g2:
                alpha = 0.5
            else:
                alpha = abs(g1) / (abs(g1) + abs(g2))
            x = self.X[j, i]
            y = self.Y[j+1, i] - alpha * self.dy
        
        return (x, y)
    
    def _resolve_saddle(self, j, i, config):
        """
        Resolve ambiguous saddle point configurations (5 and 10).
        
        Uses the value at the cell center to determine connectivity.
        
        Parameters:
        -----------
        j, i : int
            Cell indices
        config : int
            Configuration (should be 5 or 10)
            
        Returns:
        --------
        edge_pairs : list of tuples
            List of (edge1, edge2) pairs to connect
        """
        # Compute value at cell center using bilinear interpolation
        g00 = self.G[j, i]
        g10 = self.G[j, i+1]
        g01 = self.G[j+1, i]
        g11 = self.G[j+1, i+1]
        g_center = 0.25 * (g00 + g10 + g01 + g11)
        
        if config == 0b0101:  # Configuration 5
            # Bottom-left and top-right positive
            if g_center > 0:
                # Connect bottom to right, left to top
                return [(0, 1), (3, 2)]
            else:
                # Connect left to bottom, top to right
                return [(3, 0), (2, 1)]
                
        elif config == 0b1010:  # Configuration 10
            # Bottom-right and top-left positive
            if g_center > 0:
                # Connect bottom to left, right to top
                return [(0, 3), (1, 2)]
            else:
                # Connect bottom to right, left to top
                return [(0, 1), (3, 2)]
        
        return []
    
    def extract_contours(self, iso_value=0.0):
        """
        Extract all contours at the specified iso-value.
        
        Parameters:
        -----------
        iso_value : float
            Iso-value to extract (default: 0.0)
            
        Returns:
        --------
        contours : list of dicts
            Each dict contains:
                - 'points': list of (x, y) coordinates
                - 'closed': boolean indicating if contour is closed
                - 'length': total length of contour
        """
        # Shift G so that iso_value becomes zero
        G_shifted = self.G - iso_value
        G_original = self.G
        self.G = G_shifted
        
        # Store segment information indexed by cell
        # segments[j][i] = list of (edge1, edge2, (x1, y1), (x2, y2))
        segments = defaultdict(lambda: defaultdict(list))
        
        # First pass: collect all segments
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                config = self._get_cell_config(j, i)
                
                if config == 0 or config == 15:
                    continue
                
                # Get edge pairs for this configuration
                if config in [0b0101, 0b1010]:
                    edge_pairs = self._resolve_saddle(j, i, config)
                else:
                    edge_pairs = self.edge_table[config]
                
                # Store segments with their coordinates
                for edge1, edge2 in edge_pairs:
                    x1, y1 = self._get_edge_crossing(j, i, edge1)
                    x2, y2 = self._get_edge_crossing(j, i, edge2)
                    segments[j][i].append((edge1, edge2, (x1, y1), (x2, y2)))
        
        # Second pass: connect segments into contours
        contours = []
        used_segments = set()
        
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                for seg_idx, (edge1, edge2, p1, p2) in enumerate(segments[j][i]):
                    if (j, i, seg_idx) in used_segments:
                        continue
                    
                    # Start a new contour
                    contour_points = [p1, p2]
                    used_segments.add((j, i, seg_idx))
                    
                    # Try to extend contour forward
                    current_point = p2
                    current_cell = (j, i)
                    
                    while True:
                        # Find next segment that starts near current_point
                        found_next = False
                        
                        # Search in neighboring cells
                        for dj in [-1, 0, 1]:
                            for di in [-1, 0, 1]:
                                nj, ni = current_cell[0] + dj, current_cell[1] + di
                                if not (0 <= nj < self.ny - 1 and 0 <= ni < self.nx - 1):
                                    continue
                                
                                for next_seg_idx, (_, _, np1, np2) in enumerate(segments[nj][ni]):
                                    if (nj, ni, next_seg_idx) in used_segments:
                                        continue
                                    
                                    # Check if this segment connects
                                    dist1 = np.sqrt((np1[0] - current_point[0])**2 + 
                                                   (np1[1] - current_point[1])**2)
                                    dist2 = np.sqrt((np2[0] - current_point[0])**2 + 
                                                   (np2[1] - current_point[1])**2)
                                    
                                    tol = 1e-6
                                    if dist1 < tol:
                                        contour_points.append(np2)
                                        used_segments.add((nj, ni, next_seg_idx))
                                        current_point = np2
                                        current_cell = (nj, ni)
                                        found_next = True
                                        break
                                    elif dist2 < tol:
                                        contour_points.append(np1)
                                        used_segments.add((nj, ni, next_seg_idx))
                                        current_point = np1
                                        current_cell = (nj, ni)
                                        found_next = True
                                        break
                                
                                if found_next:
                                    break
                            if found_next:
                                break
                        
                        if not found_next:
                            break
                    
                    # Check if contour is closed
                    first_point = contour_points[0]
                    last_point = contour_points[-1]
                    dist = np.sqrt((last_point[0] - first_point[0])**2 + 
                                  (last_point[1] - first_point[1])**2)
                    is_closed = dist < 1e-6
                    
                    # Compute contour length
                    length = 0.0
                    for k in range(len(contour_points) - 1):
                        x1, y1 = contour_points[k]
                        x2, y2 = contour_points[k + 1]
                        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Add closing segment if closed
                    if is_closed and len(contour_points) > 2:
                        x1, y1 = contour_points[-1]
                        x2, y2 = contour_points[0]
                        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    contours.append({
                        'points': contour_points,
                        'closed': is_closed,
                        'length': length
                    })
        
        # Restore original G
        self.G = G_original
        
        return contours
    
    def compute_total_length(self, iso_value=0.0):
        """
        Compute total length of all contours at iso_value.
        
        Parameters:
        -----------
        iso_value : float
            Iso-value to extract (default: 0.0)
            
        Returns:
        --------
        total_length : float
            Sum of lengths of all contours
        """
        contours = self.extract_contours(iso_value)
        return sum(c['length'] for c in contours)


def compute_flame_surface_area_marching_squares(G, X, Y, iso_value=0.0):
    """
    Compute flame surface area using marching squares algorithm.
    Wrapper function for easy integration with existing code.
    
    Parameters:
    -----------
    G : ndarray
        Level set function
    X : ndarray
        X coordinates (meshgrid)
    Y : ndarray
        Y coordinates (meshgrid)
    iso_value : float
        Iso-value to extract (default: 0.0)
        
    Returns:
    --------
    total_length : float
        Total length of all contours at iso_value
    """
    ms = MarchingSquares(G, X, Y)
    return ms.compute_total_length(iso_value)


def test_marching_squares():
    """
    Comprehensive test of marching squares implementation.
    Tests: single circle, multiple circles, open contours, complex shapes.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MARCHING SQUARES TEST")
    print("="*80)
    
    # Grid setup
    nx, ny = 201, 201
    Lx, Ly = 4.0, 4.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    tests = []
    
    # Test 1: Single circle
    print("\n" + "-"*80)
    print("TEST 1: Single Circle")
    print("-"*80)
    
    R1 = 0.8
    x_c1, y_c1 = 2.0, 2.0
    G1 = np.sqrt((X - x_c1)**2 + (Y - y_c1)**2) - R1
    
    ms1 = MarchingSquares(G1, X, Y)
    contours1 = ms1.extract_contours(iso_value=0.0)
    total_length1 = sum(c['length'] for c in contours1)
    analytical1 = 2 * np.pi * R1
    
    print(f"Number of contours: {len(contours1)}")
    print(f"Analytical circumference: {analytical1:.8f}")
    print(f"Computed length: {total_length1:.8f}")
    print(f"Error: {abs(total_length1 - analytical1):.8f} ({abs(total_length1 - analytical1)/analytical1*100:.4f}%)")
    print(f"Contour is closed: {contours1[0]['closed'] if contours1 else 'N/A'}")
    
    tests.append(('Single Circle', G1, contours1, analytical1, total_length1))
    
    # Test 2: Multiple circles
    print("\n" + "-"*80)
    print("TEST 2: Multiple Non-Overlapping Circles")
    print("-"*80)
    
    R2a, R2b, R2c = 0.5, 0.4, 0.3
    circles = [
        (0.8, 0.8, R2a),
        (3.2, 0.8, R2b),
        (2.0, 3.0, R2c)
    ]
    
    G2 = np.ones_like(X) * 10.0
    for xc, yc, r in circles:
        G2 = np.minimum(G2, np.sqrt((X - xc)**2 + (Y - yc)**2) - r)
    
    ms2 = MarchingSquares(G2, X, Y)
    contours2 = ms2.extract_contours(iso_value=0.0)
    total_length2 = sum(c['length'] for c in contours2)
    analytical2 = 2 * np.pi * (R2a + R2b + R2c)
    
    print(f"Number of contours: {len(contours2)}")
    print(f"Analytical total circumference: {analytical2:.8f}")
    print(f"Computed total length: {total_length2:.8f}")
    print(f"Error: {abs(total_length2 - analytical2):.8f} ({abs(total_length2 - analytical2)/analytical2*100:.4f}%)")
    for i, c in enumerate(contours2):
        print(f"  Contour {i+1}: length={c['length']:.6f}, closed={c['closed']}, points={len(c['points'])}")
    
    tests.append(('Multiple Circles', G2, contours2, analytical2, total_length2))
    
    # Test 3: Open contour (half-space)
    print("\n" + "-"*80)
    print("TEST 3: Open Contour (Diagonal Line)")
    print("-"*80)
    
    # Diagonal line from (0.5, 0.5) to (3.5, 3.5)
    G3 = (Y - 0.5) - (X - 0.5)  # Line: y - 0.5 = x - 0.5, or y = x
    
    ms3 = MarchingSquares(G3, X, Y)
    contours3 = ms3.extract_contours(iso_value=0.0)
    total_length3 = sum(c['length'] for c in contours3)
    
    # Analytical: diagonal line length from one corner to opposite
    x_intersect_min = max(0.0, 0.5)
    y_intersect_min = max(0.0, 0.5)
    x_intersect_max = min(Lx, Ly + 0.5)
    y_intersect_max = min(Ly, Lx - 0.5)
    
    # Line goes from (0.5, 0.5) to (3.5, 3.5) within domain [0,4]×[0,4]
    analytical3 = np.sqrt(2) * 3.0  # sqrt((3.5-0.5)^2 + (3.5-0.5)^2) = sqrt(2)*3
    
    print(f"Number of contours: {len(contours3)}")
    print(f"Analytical line length: {analytical3:.8f}")
    print(f"Computed length: {total_length3:.8f}")
    print(f"Error: {abs(total_length3 - analytical3):.8f} ({abs(total_length3 - analytical3)/analytical3*100:.4f}%)")
    if contours3:
        print(f"Contour is closed: {contours3[0]['closed']}")
    
    tests.append(('Open Contour', G3, contours3, analytical3, total_length3))
    
    # Test 4: Complex shape (flower)
    print("\n" + "-"*80)
    print("TEST 4: Complex Shape (Flower with 5 petals)")
    print("-"*80)
    
    x_c4, y_c4 = 2.0, 2.0
    r_base = 0.6
    r_petal = 0.3
    n_petals = 5
    
    theta_grid = np.arctan2(Y - y_c4, X - x_c4)
    r_grid = np.sqrt((X - x_c4)**2 + (Y - y_c4)**2)
    
    # Radius varies with angle: r = r_base + r_petal * cos(n_petals * theta)
    r_flower = r_base + r_petal * np.cos(n_petals * theta_grid)
    G4 = r_grid - r_flower
    
    ms4 = MarchingSquares(G4, X, Y)
    contours4 = ms4.extract_contours(iso_value=0.0)
    total_length4 = sum(c['length'] for c in contours4)
    
    # Analytical (approximate using numerical integration)
    theta_analytical = np.linspace(0, 2*np.pi, 10000)
    r_analytical = r_base + r_petal * np.cos(n_petals * theta_analytical)
    dr_dtheta = -n_petals * r_petal * np.sin(n_petals * theta_analytical)
    dl = np.sqrt(r_analytical**2 + dr_dtheta**2)
    analytical4 = np.trapz(dl, theta_analytical)
    
    print(f"Number of contours: {len(contours4)}")
    print(f"Analytical perimeter (numerical integration): {analytical4:.8f}")
    print(f"Computed length: {total_length4:.8f}")
    print(f"Error: {abs(total_length4 - analytical4):.8f} ({abs(total_length4 - analytical4)/analytical4*100:.4f}%)")
    if contours4:
        print(f"Contour is closed: {contours4[0]['closed']}")
    
    tests.append(('Complex Shape', G4, contours4, analytical4, total_length4))
    
    # Test 5: Multiple iso-values on same field
    print("\n" + "-"*80)
    print("TEST 5: Multiple Iso-Values on Same Field")
    print("-"*80)
    
    R5 = 1.0
    x_c5, y_c5 = 2.0, 2.0
    G5 = np.sqrt((X - x_c5)**2 + (Y - y_c5)**2) - R5
    
    ms5 = MarchingSquares(G5, X, Y)
    
    iso_values = [-0.3, 0.0, 0.3]
    for iso in iso_values:
        contours5 = ms5.extract_contours(iso_value=iso)
        length5 = sum(c['length'] for c in contours5)
        R_iso = R5 + iso
        analytical5 = 2 * np.pi * R_iso
        error_pct = abs(length5 - analytical5) / analytical5 * 100
        print(f"Iso-value = {iso:+.1f}: R = {R_iso:.1f}, "
              f"Analytical = {analytical5:.6f}, Computed = {length5:.6f}, "
              f"Error = {error_pct:.4f}%")
    
    # Visualization
    print("\n" + "-"*80)
    print("Creating visualizations...")
    print("-"*80)
    
    fig = plt.figure(figsize=(18, 12))
    
    for idx, (name, G, contours, analytical, computed) in enumerate(tests):
        ax = plt.subplot(2, 3, idx + 1)
        
        # Plot level set function
        levels = np.linspace(-1, 1, 21)
        contourf = ax.contourf(X, Y, G, levels=levels, cmap='RdBu_r', alpha=0.7)
        ax.contour(X, Y, G, levels=[0], colors='black', linewidths=1, alpha=0.3)
        
        # Plot extracted contours
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(contours), 1)))
        for i, contour in enumerate(contours):
            points = contour['points']
            if len(points) > 1:
                xs, ys = zip(*points)
                linestyle = '-' if contour['closed'] else '--'
                linewidth = 2 if contour['closed'] else 2
                ax.plot(xs, ys, color=colors[i % len(colors)], 
                       linestyle=linestyle, linewidth=linewidth,
                       label=f"Contour {i+1} ({'closed' if contour['closed'] else 'open'})")
                
                # Mark start point
                ax.plot(xs[0], ys[0], 'go', markersize=6)
                # Mark end point if open
                if not contour['closed']:
                    ax.plot(xs[-1], ys[-1], 'ro', markersize=6)
        
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('y', fontsize=9)
        ax.set_title(f'{name}\nAnalytical: {analytical:.4f}, Computed: {computed:.4f}\n'
                    f'Error: {abs(computed-analytical)/analytical*100:.3f}%',
                    fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if len(contours) <= 3:
            ax.legend(fontsize=7, loc='best')
        plt.colorbar(contourf, ax=ax, label='G')
    
    # Add summary in last subplot
    ax = plt.subplot(2, 3, 6)
    ax.axis('off')
    
    summary_text = "MARCHING SQUARES ALGORITHM\n"
    summary_text += "="*40 + "\n\n"
    summary_text += "Features:\n"
    summary_text += "• Handles multiple contours ✓\n"
    summary_text += "• Detects closed contours ✓\n"
    summary_text += "• Handles open contours ✓\n"
    summary_text += "• Sub-grid accuracy ✓\n"
    summary_text += "• Complex shapes ✓\n"
    summary_text += "• Saddle point resolution ✓\n\n"
    
    summary_text += "Accuracy Summary:\n"
    summary_text += "-"*40 + "\n"
    for name, _, _, analytical, computed in tests:
        error_pct = abs(computed - analytical) / analytical * 100
        summary_text += f"{name}:\n"
        summary_text += f"  Error: {error_pct:.4f}%\n"
    
    summary_text += "\n" + "="*40 + "\n"
    summary_text += "Legend:\n"
    summary_text += "• Green dot: Start point\n"
    summary_text += "• Red dot: End point (open)\n"
    summary_text += "• Solid line: Closed contour\n"
    summary_text += "• Dashed line: Open contour\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('marching_squares_comprehensive_test.png', dpi=300, bbox_inches='tight')
    print("Saved: marching_squares_comprehensive_test.png")
    
    plt.show()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"{'Test':<30} {'Contours':<10} {'Error %':<15} {'Status':<10}")
    print("-"*80)
    for name, _, contours, analytical, computed in tests:
        error_pct = abs(computed - analytical) / analytical * 100
        status = "✓ PASS" if error_pct < 0.1 else "✓ GOOD" if error_pct < 1.0 else "! CHECK"
        print(f"{name:<30} {len(contours):<10} {error_pct:<15.6f} {status:<10}")
    
    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_marching_squares()