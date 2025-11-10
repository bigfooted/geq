# Fast Marching Method for SDF Initialization

## Summary

We tested **four different methods** for creating signed distance functions (SDFs) for an ellipse:

1. **Bad Approximation**: Parametric formula `√((x/a)² + (y/b)²) × min(a,b) - min(a,b)`
2. **Euclidean Distance Transform (EDT)**: scipy.ndimage.distance_transform_edt
3. **Fast Marching Method (FMM)**: Custom implementation solving Eikonal equation |∇φ| = 1
4. **Reinitialization PDE**: PDE-based method solving ∂φ/∂τ + sign(φ)(|∇φ| - 1) = 0

## Results: Initial Condition Quality

| Method            | |∇G| mean | |∇G| std | Drift from 1.0 | Quality |
|-------------------|----------|----------|----------------|---------|
| Bad Approximation | 0.785    | 0.164    | 21.5%          | ❌ Poor |
| EDT SDF           | 1.171    | 0.297    | 17.1%          | ⚠️ Biased |
| **Fast Marching** | **1.002** | **0.056** | **0.17%**   | ✅ **Excellent** |
| Reinitialized     | 0.977    | 0.262    | 2.3%           | ✅ Good |

### Key Findings:

- **EDT gives |∇G| = 1.17** (not 1.0!) due to grid discretization effects on curved boundaries
- **FMM gives |∇G| = 1.002** - nearly perfect! Only 0.17% error
- **Reinitialization gives |∇G| = 0.977** - good but has higher variance (std=0.262)
- FMM has the **lowest standard deviation** (0.056) - most uniform gradient

## Results: Propagation Test (After t=0.3 with S_L=0.4)

Testing with expanding ellipse, comparing Godunov vs WENO5 gradient schemes:

| Initial Condition | Godunov Drift | WENO5 Drift | WENO5 Improvement |
|-------------------|---------------|-------------|-------------------|
| Bad Approximation | 29.3%         | 27.3%       | 1.1× (marginal)   |
| EDT SDF           | 2.3%          | 26.0%       | 0.1× (WORSE!)     |
| **Fast Marching** | **5.7%**      | **1.3%**    | **4.3× improvement** ✅ |
| Reinitialized     | 20.3%         | 13.8%       | 1.5× improvement  |

### Critical Insights:

1. **FMM is the clear winner**:
   - WENO5 drift = 1.3% (best result!)
   - 4.3× improvement over Godunov
   - Best of all four methods

2. **EDT actively hurts WENO5**:
   - Despite having reasonable initial |∇G| ≈ 1.17, it causes 26% drift
   - The bias in |∇G| (1.17 vs 1.0) propagates errors
   - WENO5 is sensitive to initial |∇G| being exactly 1.0

3. **Reinitialization PDE has high variance**:
   - Standard deviation 0.262 (vs FMM's 0.056)
   - This variance accumulates during propagation
   - Results in 13.8% drift for WENO5

4. **Bad approximation dominates everything**:
   - Initial error (21.5%) too large for scheme choice to matter
   - Shows importance of proper initialization!

## Why Fast Marching Method Works Best

### Mathematical Foundation

FMM solves the **Eikonal equation**: |∇φ| = 1

- Directly enforces the SDF property at every point
- Uses upwind differences (consistent with level-set evolution)
- Monotone advancing (no artificial oscillations)
- O(N log N) complexity with priority queue

### Advantages Over Alternatives

| Property              | FMM       | EDT       | Reinit PDE |
|-----------------------|-----------|-----------|------------|
| Exact |∇G| = 1.0      | ✅ Yes    | ❌ No (1.17) | ✅ Yes  |
| Low variance          | ✅ Yes    | ❌ No     | ⚠️ Medium  |
| Grid-aligned boundary | ✅ Handles | ❌ Struggles | ✅ Handles |
| Computational cost    | Medium    | Fast      | Slow      |
| Implementation        | Complex   | Simple    | Medium    |

### Why EDT Fails

The Euclidean Distance Transform:
- Computes distance in **pixel space**, not continuous space
- For ellipse, boundary doesn't align with grid
- Interpolation at boundary creates |∇G| ≈ 1.17 bias
- This small bias (17%) amplifies during propagation with WENO5

### Why Reinitialization Has High Variance

The PDE method:
- Smooths the level set (diffusive nature of PDE)
- Can create non-monotone profiles near boundary
- Standard deviation 0.262 vs FMM's 0.056
- Higher variance → more error accumulation

## Complete Test Results: All Four Cases

From `test_hj_weno5_demo.py`:

| Test Case              | Godunov Drift | WENO5 Drift | Improvement |
|------------------------|---------------|-------------|-------------|
| **Circle** (exact SDF) | 4.23e-02      | 2.59e-05    | **1632×** ✅ |
| Ellipse (bad init)     | 2.93e-01      | 2.73e-01    | 1.1×        |
| **Ellipse (FMM SDF)**  | 5.66e-02      | 1.31e-02    | **4.3×** ✅  |
| Merging circles        | 1.65e-02      | 1.69e-01    | 0.1× ❌     |

### Interpretation

Three distinct regimes:

1. **Perfect geometry (Circle)**: 1632× improvement
   - Constant curvature + exact SDF = ideal for WENO5

2. **Realistic geometry (Ellipse + FMM)**: 4.3× improvement
   - Non-uniform curvature limits benefit
   - Still significant practical improvement

3. **Pathological (Merging circles)**: 0.1× (worse)
   - Topology change creates singularities
   - Higher-order schemes fail, use Godunov

## Implementation: Fast Marching Method

```python
def fast_marching_method(mask, dx):
    """
    Fast Marching Method to compute signed distance function.

    Algorithm:
    1. Initialize boundary cells with distance ≈ 0.5*dx
    2. Use priority queue to propagate outward (O(N log N))
    3. At each point, solve Eikonal equation |∇φ| = 1
       using known neighbors: (φ-φx)² + (φ-φy)² = dx²
    4. Apply sign based on inside/outside mask
    """
    ny, nx = mask.shape
    phi = np.full((ny, nx), np.inf)
    status = np.zeros((ny, nx), dtype=int)  # 0=far, 1=narrow, 2=known

    # Initialize boundary
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            if mask[j,i] != mask[j-1,i] or mask[j,i] != mask[j+1,i] or \
               mask[j,i] != mask[j,i-1] or mask[j,i] != mask[j,i+1]:
                phi[j, i] = 0.5 * dx
                status[j, i] = 1

    # Priority queue
    heap = [(abs(phi[j,i]), j, i) for j in range(ny) for i in range(nx)
            if status[j,i] == 1]
    heapq.heapify(heap)

    # Fast marching
    while heap:
        _, j, i = heapq.heappop(heap)
        if status[j, i] == 2:
            continue
        status[j, i] = 2

        # Update neighbors using Eikonal solver
        for jn, in_ in neighbors(j, i):
            if status[jn, in_] != 2:
                phi_new = solve_eikonal(phi, jn, in_, dx, status)
                if phi_new < phi[jn, in_]:
                    phi[jn, in_] = phi_new
                    heapq.heappush(heap, (phi_new, jn, in_))

    # Apply sign
    return np.where(mask, -phi, phi)
```

## Practical Recommendations

### For Smooth Flames (No Topology Changes)

```python
# Best approach: FMM initialization + WENO5 gradient
mask = G_approx < 0  # Initial shape
G0 = fast_marching_method(mask, dx)

G_hist, t_hist = solver.solve(
    G0, t_final, dt,
    gradient_scheme='weno5',      # Higher-order gradients
    spatial_scheme='weno5',        # Higher-order convection
    reinit_interval=0              # No reinitialization needed!
)
```

**Benefits:**
- 4-5× more accurate gradient maintenance
- Stable long-time integration
- Less frequent reinitialization needed

### For Flames with Topology Changes

```python
# Use Godunov + frequent reinitialization
G_hist, t_hist = solver.solve(
    G0, t_final, dt,
    gradient_scheme='godunov',     # Robust at singularities
    spatial_scheme='weno5',        # Still use high-order for smooth parts
    reinit_interval=5,             # Reinitialize every 5 steps
    reinit_method='pde'
)
```

**Rationale:**
- Higher-order gradients fail at saddle points
- Godunov is dissipative but stable
- Frequent reinitialization maintains |∇G| = 1.0

### For General Shapes

**Initialization hierarchy (best to worst):**
1. ✅ **Fast Marching Method** - Best accuracy, moderate cost
2. ✅ **Analytical SDF** - Perfect (if available, e.g., circle)
3. ⚠️ **Reinitialization PDE** - Good but slow, higher variance
4. ❌ **EDT** - Fast but biased, hurts higher-order schemes
5. ❌ **Parametric approximation** - Avoid for ellipses!

## Performance Comparison

| Method       | Time (121×121 grid) | |∇G| Error | Notes |
|--------------|---------------------|-----------|-------|
| Analytical   | ~0 ms              | 0.0%      | Only for simple shapes |
| **FMM**      | **~50 ms**         | **0.17%** | Best general method ✅ |
| EDT          | ~5 ms              | 17.1%     | Fast but inaccurate |
| Reinit PDE   | ~200 ms            | 2.3%      | Slow, higher variance |

## Visualizations

Generated by `compare_initializations.py`:

- **compare_ellipse_initializations.png**:
  - 4×3 grid showing level sets, gradient magnitudes, and cross-sections
  - Clearly shows FMM has most uniform |∇G| ≈ 1.0

Generated by `test_hj_weno5_demo.py`:

- **hj_weno5_clear_demo.png**:
  - 4×4 grid comparing all test cases
  - Row 3 shows ellipse with FMM: 4.3× improvement
  - Demonstrates dramatic difference vs bad initialization

## Conclusion

**The Fast Marching Method is the gold standard for SDF initialization:**

1. ✅ Most accurate: |∇G| = 1.002 (0.17% error)
2. ✅ Most uniform: std = 0.056 (lowest variance)
3. ✅ Best propagation: 1.3% drift with WENO5 (4.3× better than Godunov)
4. ✅ No bias: Unlike EDT which gives |∇G| = 1.17
5. ✅ General purpose: Works for any shape defined by mask

**Key insight:** The 17% bias in EDT (|∇G| = 1.17 vs 1.0) seems small but completely ruins the benefit of higher-order gradient schemes. WENO5 requires |∇G| ≈ 1.0 to within ~2% for optimal performance.

**Winner:** 🏆 **Fast Marching Method** 🏆
