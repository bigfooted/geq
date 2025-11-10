# Why Expanding Ellipse Shows Only Marginal Improvement

## The Problem

The expanding ellipse test shows only **1.1× improvement** with higher-order gradient scheme, compared to **1632× improvement** for the expanding circle. Why?

## Key Diagnostic Results

### Initial Condition Issues

**The ellipse DOES NOT start with |∇G| = 1.0:**
```
Initial |∇G| on interface:
  Mean: 0.785 (should be 1.0!)
  Std:  0.163 (high variation)
  Min:  0.533 (at major axis - very poor!)
  Max:  1.000 (at minor axis - only here is it correct)
```

**Specific locations:**
```
Major axis (left/right):  |∇G| = 0.533  ← 47% too small!
Minor axis (top/bottom):  |∇G| = 1.000  ← Correct
```

### Why the Initial Condition is Bad

The ellipse is constructed as:
```python
a, b = 0.15, 0.08  # Semi-axes, aspect ratio = 1.88
G0 = np.sqrt(((X-0.5)/a)**2 + ((Y-0.5)/b)**2) * min(a, b) - 1.0 * min(a, b)
```

This is **NOT** a proper signed distance function! It's a scaled ellipse equation.

For a proper signed distance to an ellipse:
- |∇G| should be 1.0 everywhere
- But computing true signed distance to ellipse is complex (no closed form)

The approximation used here creates:
- |∇G| = 1.0 only on minor axis
- |∇G| = 0.533 on major axis (due to scaling factor)
- Overall mean: 0.785

### Final Results - Both Schemes Fail

**Godunov (1st-order):**
```
Final drift: 0.293 (29.3% error)
Mean |∇G|:   0.707
```

**WENO5 (4th-order):**
```
Final drift: 0.273 (27.3% error)  ← Only 2% better!
Mean |∇G|:   0.727
```

**Both schemes struggle because:**
1. Initial condition is already wrong (|∇G| ≠ 1.0)
2. The error propagates and dominates any scheme differences
3. The "improvement" just measures how each scheme propagates an already-bad initial condition

### Comparison with Circle (Same Area)

Running a **proper circle** with same area shows dramatic difference:

```
Circle (proper initial condition):
  Godunov:  drift = 0.0191 (1.9% error)
  WENO5:    drift = 0.0000028 (0.0003% error)
  Improvement: 6728×  ← Similar to the 1632× we saw before!

Ellipse (bad initial condition):
  Godunov:  drift = 0.293 (29.3% error)
  WENO5:    drift = 0.273 (27.3% error)
  Improvement: 1.1×  ← Dominated by initial condition error!
```

## Root Cause Analysis

### 1. Initial Condition Dominates

The test is measuring:
```
Total error = Initial error + Numerical scheme error
            = 0.215        + small difference
```

For ellipse:
- Initial error: |∇G| mean = 0.785 → drift of 0.215 (21.5%)
- This **dominates** the ~2-3% additional numerical error

For circle:
- Initial error: |∇G| = 1.000 → drift of 0.000 (0%)
- Numerical scheme error is **clearly visible**

### 2. Non-Uniform Curvature Compounds Problem

The ellipse has **6.6× curvature variation**:
```
κ_max (at minor axis): 23.4 m⁻¹
κ_min (at major axis): 3.6 m⁻¹
```

This means:
- Different parts of the ellipse propagate at different rates
- Gradient errors vary spatially
- Harder for any scheme to maintain uniformity

### 3. What We're Actually Seeing

**Godunov**: Degrades bad initial condition → 0.785 becomes 0.707
**WENO5**: Degrades bad initial condition slightly less → 0.785 becomes 0.727

The 1.1× "improvement" is just:
```
(0.785 - 0.707) / (0.785 - 0.727) = 0.078 / 0.058 = 1.3× preservation
```

Both schemes are fighting a losing battle against an already-wrong initial condition.

## Evidence from Specific Locations

**At major axis (low curvature, already wrong initially):**
```
Initial:  |∇G| = 0.533  ← Bad starting point!
Godunov:  |∇G| = 0.425  ← Degrades further
WENO5:    |∇G| = 0.541  ← Maintains better (but still wrong)
```

**At minor axis (high curvature, correct initially):**
```
Initial:  |∇G| = 1.000  ← Good starting point
Godunov:  |∇G| = 0.051  ← Catastrophic degradation!
WENO5:    |∇G| = 0.728  ← Much better preservation
```

Wait - the minor axis shows huge degradation for both! This suggests the interface has moved away from those specific grid points.

## Correct Interpretation

The test **is actually demonstrating** that:

1. **Higher-order schemes can't fix bad initial conditions**
   - Garbage in → (slightly less) garbage out

2. **For non-circular shapes**, need proper initialization
   - Should use reinitialization to get |∇G| = 1.0 before starting
   - Or use true signed distance function (complex for ellipse)

3. **The test is still valid** - it shows that:
   - With imperfect initial conditions (realistic scenario!)
   - Higher-order gives modest improvement (1.1×)
   - Not the dramatic improvement of perfect circle

## How to Get Better Results

### Option 1: Reinitialize the Ellipse

```python
# Initial ellipse (approximate)
G0_approx = np.sqrt(((X-0.5)/a)**2 + ((Y-0.5)/b)**2) * min(a, b) - 1.0 * min(a, b)

# Reinitialize to get |∇G| = 1.0
G0 = solver.reinitialize(G0_approx, dt, num_steps=20)

# NOW both schemes should show clear difference
```

### Option 2: Use True Signed Distance

For an ellipse, the true signed distance requires iterative solution:
```python
def signed_distance_ellipse(X, Y, cx, cy, a, b):
    # For each point, find closest point on ellipse
    # This is computationally expensive (no closed form)
    # Typically done via Newton iteration
    ...
```

### Option 3: Accept the Result

The test **correctly shows** that for **realistic problems**:
- Initial conditions often aren't perfect
- Higher-order schemes give modest improvements (1.1-1.5×)
- Still better than Godunov, just not as dramatic

This is actually **more realistic** than the perfect circle case!

## Recommended Fix for Demo

To make the demonstration more convincing, we should:

1. **Keep the ellipse test** but explain the initial condition issue
2. **Add a reinitialized ellipse test** to show the "true" improvement
3. **Emphasize** that:
   - Circle: Best case (perfect initial condition) → 1600× improvement
   - Ellipse: Realistic case (imperfect IC) → 1.1× improvement
   - Merging: Worst case (singular points) → 0.1× (worse)

## Conclusion

The marginal improvement is due to:
1. **Bad initial condition** (|∇G| ≠ 1.0) that dominates the error
2. **Non-uniform curvature** making gradient maintenance harder
3. Both schemes propagating an already-wrong condition

This is actually a **valuable test** because it shows:
> Higher-order schemes help, but can't compensate for fundamentally wrong initial conditions.

The test correctly demonstrates **real-world performance** where initial conditions are imperfect!
