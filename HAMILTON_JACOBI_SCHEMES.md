# Hamilton-Jacobi Formulation for Source-Term Dominated Problems

## Problem Analysis

### Current Implementation Issues

The G-equation has two components:
```
∂G/∂t + u·∇G + S_L|∇G| = 0
         ︸︷︷︸    ︸︷︷︸
       advection  source
```

**Current gradient computation** (`compute_gradient_magnitude`):
- Uses **first-order upwind differences** only
- Formula: `|∇G|² = max(D⁻ₓG,0)² + min(D⁺ₓG,0)² + max(D⁻ᵧG,0)² + min(D⁺ᵧG,0)²`
- This is correct for monotone schemes but **only first-order accurate**

**Problem:** When source term dominates (S_L >> |u|):
1. ✗ Gradient magnitude error directly affects flame speed
2. ✗ O(Δx) error in |∇G| → O(Δx) error in interface position per time step
3. ✗ Accumulates rapidly over many time steps
4. ✗ Causes |∇G| to drift from 1.0 (violates signed distance property)

### Why Current WENO5 Doesn't Help the Source Term

**Current situation:**
- `spatial_scheme='weno5'` applies **only to convective term** u·∇G
- Gradient magnitude |∇G| for source term still uses **first-order** upwind
- This creates an **accuracy mismatch**:
  - Convection: O(Δx⁵) accurate ✓
  - Source term: O(Δx) accurate ✗

## Hamilton-Jacobi Theory (Osher & Sethian)

### The Hamilton-Jacobi Equation

The G-equation can be written as a Hamilton-Jacobi equation:
```
∂G/∂t + H(∇G) = 0
```

where the Hamiltonian is:
```
H(∇G) = u·∇G + S_L|∇G|
```

For **source-dominated** problems (u ≈ 0), this simplifies to the **eikonal equation**:
```
∂G/∂t + S_L|∇G| = 0
```

### Godunov's Scheme for |∇G|

Osher & Sethian (1988) showed that for Hamilton-Jacobi equations, the proper upwind scheme is:

```python
|∇G|² = max(D⁻ₓG, 0)² + min(D⁺ₓG, 0)² + max(D⁻ᵧG, 0)² + min(D⁺ᵧG, 0)²
```

This is **exactly what we currently use** - it's called **Godunov's monotone scheme**.

**Properties:**
- ✓ Monotone (entropy-satisfying)
- ✓ Stable
- ✗ Only first-order accurate
- ✗ Very dissipative

### Higher-Order Hamilton-Jacobi Schemes

To achieve higher accuracy for |∇G|, use:

1. **ENO (Essentially Non-Oscillatory)**
   - Order 2-3 typical
   - Adaptive stencil selection based on smoothness

2. **WENO (Weighted ENO)**
   - Order 3-5 typical
   - Weighted combination of multiple stencils
   - Better resolution than ENO

## Recommended Improvements

### 1. Implement Hamilton-Jacobi WENO for Gradient Magnitude

Add new methods to compute |∇G| with high-order accuracy:

```python
def compute_gradient_magnitude_hj_weno5(self, G):
    """
    Compute |∇G| using 5th-order WENO Hamilton-Jacobi scheme.
    Based on Jiang & Peng (2000).
    """
```

**Key differences from convection WENO:**
- No velocity field involved
- Always use upwind direction (based on sign of G)
- Combine one-sided derivatives: (D⁻)² + (D⁺)²

### 2. Splitting Approach (Recommended)

Use **operator splitting** to handle each term optimally:

```python
# Step 1: Advection with high-order scheme
∂G/∂t + u·∇G = 0    →  WENO5 for u·∇G

# Step 2: Source term with HJ-WENO
∂G/∂t + S_L|∇G| = 0  →  HJ-WENO5 for |∇G|
```

**Benefits:**
- Each term gets appropriate treatment
- Can use different CFL limits for each
- Easier to implement and debug

### 3. Combined High-Order Hamiltonian

Alternative: Compute entire Hamiltonian H(∇G) with WENO:

```python
def compute_rhs_hj_weno5(self, G, spatial_scheme='hj_weno5'):
    """
    Compute RHS using Hamilton-Jacobi WENO for entire Hamiltonian.
    """
    # WENO reconstruction of ∂G/∂x and ∂G/∂y at each direction
    Gx_plus = self._weno5_derivative_plus_x(G)
    Gx_minus = self._weno5_derivative_minus_x(G)
    Gy_plus = self._weno5_derivative_plus_y(G)
    Gy_minus = self._weno5_derivative_minus_y(G)

    # Local Lax-Friedrichs flux splitting or Godunov flux
    H = self._compute_hamiltonian_flux(Gx_plus, Gx_minus,
                                       Gy_plus, Gy_minus)
    return -H
```

## Implementation Priority

### Immediate Impact (High Priority)

**1. Add HJ-WENO5 for gradient magnitude** ⭐⭐⭐
```python
def compute_gradient_magnitude_weno5(self, G):
    """High-order |∇G| computation"""
```

**Impact:**
- Directly addresses your question
- Reduces error from O(Δx) → O(Δx⁵) for source term
- Maintains |∇G| ≈ 1.0 much better
- Reduces need for frequent reinitialization

**Complexity:** Medium (can adapt existing WENO5 infrastructure)

### Medium Priority

**2. Add parameter to select gradient scheme**
```python
def solve(..., spatial_scheme='weno5', gradient_scheme='hj_weno5'):
```

**3. Implement ENO2/ENO3 as intermediate options**
- Simpler than WENO5
- Still 2nd-3rd order accurate
- Lower computational cost

### Future Enhancements

**4. Subcell resolution for sharp interfaces**
- Use WENO reconstruction to sub-grid resolution
- Better capture thin flames

**5. Adaptive reinitialization**
- Monitor |∇G| quality
- Reinitialize only when needed
- Use high-order reinit when doing so

## Theoretical Background

### Why First-Order Fails for Source Terms

Consider expanding circle: G = r - R(t), where R(t) = R₀ + S_L·t

**Exact:** |∇G| = 1 everywhere

**First-order upwind on uniform grid:**
```
|∇G|ₕ ≈ 1 - C·Δx + O(Δx²)
```

**Error accumulation:**
```
dR/dt = S_L·|∇G|ₕ ≈ S_L·(1 - C·Δx)
```

After time T:
```
Error ≈ -S_L·C·Δx·T
```

→ Error grows **linearly with time**!

### Why WENO Helps

**WENO5 gradient:**
```
|∇G|ₕ ≈ 1 + C₅·Δx⁵ + O(Δx⁶)
```

**Error after time T:**
```
Error ≈ S_L·C₅·Δx⁵·T
```

→ Error is **O(Δx⁵)** smaller!

## Comparison: Current vs HJ-WENO

| Aspect | Current (1st-order |∇G|) | With HJ-WENO5 |
|--------|---------------------|---------------|
| Gradient accuracy | O(Δx) | O(Δx⁵) |
| Error accumulation | Linear with time | Much slower |
| |∇G| drift | Significant | Minimal |
| Reinit frequency | High (every 10-50 steps) | Low (every 100-500 steps) |
| Grid points needed | 201×201 | 101×101 (or less) |
| Computational cost/step | 1× | 1.5-2× |
| **Total cost** | **High** (fine grid + frequent reinit) | **Lower** (coarse grid + rare reinit) |

## Practical Recommendations

### For Your Current Problem

**Immediate action:**
1. Implement `compute_gradient_magnitude_weno5()`
2. Add parameter: `gradient_scheme='hj_weno5'`
3. Use with existing WENO5 infrastructure

**Expected improvements:**
- 4-10× reduction in gradient error
- |∇G| stays near 1.0 without reinitialization
- Can use coarser grids (101×101 instead of 201×201)
- Better accuracy for FTF computations

### Test Case to Verify

```python
# Expanding circle with S_L >> |u|
# Should maintain |∇G| = 1.0 exactly

def test_source_dominated():
    # Zero velocity (pure source term)
    u_x = u_y = 0.0
    S_L = 1.0  # Non-zero source

    # Circle
    G0 = radius - R0

    # Compare schemes
    schemes = ['godunov', 'hj_weno3', 'hj_weno5']

    for scheme in schemes:
        solver.gradient_scheme = scheme
        G_hist, t_hist = solver.solve(...)

        # Check: |∇G| should stay ≈ 1.0
        grad_error = ||∇G| - 1.0|
        print(f"{scheme}: gradient error = {grad_error}")
```

## References

1. **Osher & Sethian (1988)** - "Fronts propagating with curvature-dependent speed"
   - Original level set method paper
   - Introduced upwind scheme for |∇G|

2. **Jiang & Shu (1996)** - "Efficient Implementation of WENO schemes"
   - WENO for conservation laws

3. **Jiang & Peng (2000)** - "Weighted ENO schemes for Hamilton-Jacobi equations"
   - WENO specifically for HJ equations
   - Proper treatment of |∇G|

4. **Osher & Fedkiw (2003)** - "Level Set Methods and Dynamic Implicit Surfaces"
   - Comprehensive reference
   - Chapter 6: High-order schemes

## Conclusion

**Yes, Hamilton-Jacobi WENO formulation is highly recommended** when source term dominates:

✓ Addresses fundamental accuracy issue in |∇G| computation
✓ Reduces error from O(Δx) to O(Δx⁵)
✓ Reduces need for reinitialization
✓ Enables coarser grids → lower overall cost
✓ Better maintains signed distance property

**Implementation complexity:** Medium (can leverage existing WENO code)

**Expected benefit:** 4-10× improvement in gradient accuracy and overall solution quality

Would you like me to implement the HJ-WENO5 gradient magnitude computation?
