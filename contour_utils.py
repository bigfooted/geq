"""
Utility functions for contour extraction and analysis.
Simple interface to marching squares algorithm.
"""

from marching_squares import MarchingSquares
import numpy as np


def compute_contour_length(G, X, Y, iso_value=0.0, N=1):
    """
    Compute total length of all contours at specified iso-value.

    Simple wrapper for easy use in existing code.

    Parameters:
    -----------
    G : ndarray (ny, nx)
        2D scalar field
    X : ndarray (ny, nx)
        X coordinates (meshgrid)
    Y : ndarray (ny, nx)
        Y coordinates (meshgrid)
    iso_value : float
        Iso-value to extract (default: 0.0)
    N : int, optional (default: 1)
        Spatial skip factor. If > 1, use only every Nth grid point in both x and y
        directions for an approximate (accelerated) contour length. The result will
        generally UNDER-estimate the true length as fine-scale curvature is lost.

    Returns:
    --------
    total_length : float
        Sum of lengths of all contours

    Example:
    --------
    >>> flame_surface_area = compute_contour_length(G, X, Y, iso_value=0.0)
    """
    if N is None or N < 1:
        N = 1

    if N == 1:
        ms = MarchingSquares(G, X, Y)
        return ms.compute_total_length(iso_value)
    else:
        # Subsample the field for approximate length computation
        Gs = G[::N, ::N]
        Xs = X[::N, ::N]
        Ys = Y[::N, ::N]
        ms = MarchingSquares(Gs, Xs, Ys)
        return ms.compute_total_length(iso_value)


def compute_contour_length_fast(G, X, Y, iso_value=0.0):
    """
    Fast, vectorized length-only computation of the total contour length at iso_value.

    Notes:
    - Implements a marching-squares style scheme but only accumulates segment lengths,
      without assembling explicit polylines.
    - Handles ambiguous saddle configurations (5 and 10) using the cell-center sign
      rule consistent with the marching_squares implementation.
    - Assumes uniform grid spacing defined by X, Y.

    Parameters
    ----------
    G : ndarray (ny, nx)
        Scalar field
    X, Y : ndarray (ny, nx)
        Meshgrid coordinates
    iso_value : float
        Iso-value to extract (default 0.0)

    Returns
    -------
    total_length : float
        Sum of lengths of all contour segments at the given iso-value
    """
    # Shift so that iso becomes zero
    Gs = G - iso_value

    ny, nx = Gs.shape
    if ny < 2 or nx < 2:
        return 0.0

    # Cell-corner values (shape: (ny-1, nx-1))
    g00 = Gs[:-1, :-1]
    g10 = Gs[:-1, 1:]
    g11 = Gs[1:, 1:]
    g01 = Gs[1:, :-1]

    # Uniform spacings
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    # Boolean corner signs (positive = 1, non-positive = 0)
    s00 = (g00 > 0).astype(np.uint8)
    s10 = (g10 > 0).astype(np.uint8)
    s11 = (g11 > 0).astype(np.uint8)
    s01 = (g01 > 0).astype(np.uint8)

    # 4-bit configuration per cell
    conf = (s00 | (s10 << 1) | (s11 << 2) | (s01 << 3))

    # Precompute edge crossing coordinates for all cells
    # Use |g1|/(|g1|+|g2|) with 0.5 fallback if denom==0
    def _alpha(a, b):
        denom = np.abs(a) + np.abs(b)
        with np.errstate(invalid='ignore', divide='ignore'):
            alpha = np.where(denom > 0, np.abs(a) / denom, 0.5)
        return alpha

    # Coordinates per cell
    X00 = X[:-1, :-1]; Y00 = Y[:-1, :-1]
    X10 = X[:-1, 1:];  Y10 = Y[:-1, 1:]
    X11 = X[1:, 1:];   Y11 = Y[1:, 1:]
    X01 = X[1:, :-1];  Y01 = Y[1:, :-1]

    # Edge 0 (bottom): between (00)->(10)
    a0 = _alpha(g00, g10)
    x0 = X00 + a0 * dx
    y0 = Y00

    # Edge 1 (right): between (10)->(11)
    a1 = _alpha(g10, g11)
    x1 = X10
    y1 = Y10 + a1 * dy

    # Edge 2 (top): between (11)->(01)
    a2 = _alpha(g11, g01)
    x2 = X11 - a2 * dx
    y2 = Y11

    # Edge 3 (left): between (01)->(00)
    a3 = _alpha(g01, g00)
    x3 = X01
    y3 = Y01 - a3 * dy

    # Helper to accumulate lengths for a given mask and edge pair (ea, eb)
    def seglen(mask, ea, eb):
        if not np.any(mask):
            return 0.0
        xa, ya = (x0, y0) if ea == 0 else (x1, y1) if ea == 1 else (x2, y2) if ea == 2 else (x3, y3)
        xb, yb = (x0, y0) if eb == 0 else (x1, y1) if eb == 1 else (x2, y2) if eb == 2 else (x3, y3)
        dxs = xa - xb
        dys = ya - yb
        # Compute distance and sum only where mask
        dist = np.sqrt(dxs*dxs + dys*dys)
        return float(dist[mask].sum())

    total = 0.0

    # Masks for each configuration
    m = {k: (conf == k) for k in range(16)}

    # Non-ambiguous configurations: add single or two segments as per table
    total += seglen(m[0b0001], 3, 0)
    total += seglen(m[0b0010], 0, 1)
    total += seglen(m[0b0011], 3, 1)
    total += seglen(m[0b0100], 1, 2)
    total += seglen(m[0b0110], 0, 2)
    total += seglen(m[0b0111], 3, 2)
    total += seglen(m[0b1000], 2, 3)
    total += seglen(m[0b1001], 2, 0)
    total += seglen(m[0b1011], 2, 1)
    total += seglen(m[0b1100], 1, 3)
    total += seglen(m[0b1101], 1, 0)
    total += seglen(m[0b1110], 0, 3)

    # Ambiguous: 0101 (5) and 1010 (10) — need center value rule
    if np.any(m[0b0101]):
        g_center = 0.25 * (g00 + g10 + g01 + g11)
        mc = m[0b0101]
        m_pos = mc & (g_center > 0)  # connect (0,1) and (3,2)
        m_neg = mc & ~(g_center > 0) # connect (3,0) and (2,1)
        total += seglen(m_pos, 0, 1) + seglen(m_pos, 3, 2)
        total += seglen(m_neg, 3, 0) + seglen(m_neg, 2, 1)

    if np.any(m[0b1010]):
        g_center = 0.25 * (g00 + g10 + g01 + g11)
        mc = m[0b1010]
        m_pos = mc & (g_center > 0)  # connect (0,3) and (1,2)
        m_neg = mc & ~(g_center > 0) # connect (0,1) and (3,2)
        total += seglen(m_pos, 0, 3) + seglen(m_pos, 1, 2)
        total += seglen(m_neg, 0, 1) + seglen(m_neg, 3, 2)

    # Return scalar float
    return float(total)


def extract_contours(G, X, Y, iso_value=0.0):
    """
    Extract all contours at specified iso-value.

    Parameters:
    -----------
    G : ndarray (ny, nx)
        2D scalar field
    X : ndarray (ny, nx)
        X coordinates (meshgrid)
    Y : ndarray (ny, nx)
        Y coordinates (meshgrid)
    iso_value : float
        Iso-value to extract (default: 0.0)

    Returns:
    --------
    contours : list of dicts
        Each dict contains:
            - 'points': list of (x, y) tuples
            - 'closed': bool, True if contour is closed
            - 'length': float, total contour length

    Example:
    --------
    >>> contours = extract_contours(G, X, Y, iso_value=0.0)
    >>> for i, c in enumerate(contours):
    >>>     print(f"Contour {i}: length={c['length']:.4f}, closed={c['closed']}")
    """
    ms = MarchingSquares(G, X, Y)
    return ms.extract_contours(iso_value)


def get_contour_statistics(G, X, Y, iso_value=0.0):
    """
    Get comprehensive statistics about contours.

    Parameters:
    -----------
    G : ndarray (ny, nx)
        2D scalar field
    X : ndarray (ny, nx)
        X coordinates (meshgrid)
    Y : ndarray (ny, nx)
        Y coordinates (meshgrid)
    iso_value : float
        Iso-value to extract (default: 0.0)

    Returns:
    --------
    stats : dict
        Dictionary containing:
            - 'num_contours': int
            - 'total_length': float
            - 'num_closed': int
            - 'num_open': int
            - 'contours': list of contour dicts

    Example:
    --------
    >>> stats = get_contour_statistics(G, X, Y)
    >>> print(f"Found {stats['num_contours']} contours")
    >>> print(f"Total length: {stats['total_length']:.4f}")
    """
    ms = MarchingSquares(G, X, Y)
    contours = ms.extract_contours(iso_value)

    num_closed = sum(1 for c in contours if c['closed'])
    num_open = len(contours) - num_closed
    total_length = sum(c['length'] for c in contours)

    return {
        'num_contours': len(contours),
        'total_length': total_length,
        'num_closed': num_closed,
        'num_open': num_open,
        'contours': contours
    }


# Backward compatibility alias
compute_flame_surface_area = compute_contour_length