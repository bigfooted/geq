"""
Utility functions for contour extraction and analysis.
Simple interface to marching squares algorithm.
"""

from marching_squares import MarchingSquares
import numpy as np


def compute_contour_length(G, X, Y, iso_value=0.0):
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
        
    Returns:
    --------
    total_length : float
        Sum of lengths of all contours
        
    Example:
    --------
    >>> flame_surface_area = compute_contour_length(G, X, Y, iso_value=0.0)
    """
    ms = MarchingSquares(G, X, Y)
    return ms.compute_total_length(iso_value)


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