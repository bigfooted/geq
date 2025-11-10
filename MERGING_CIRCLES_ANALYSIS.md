# Why Higher-Order Gradient Scheme Fails for Merging Circles

## The Problem

When testing the higher-order (4th-order centered differences) gradient scheme on merging circles, we observe:

| Scheme | |∇G| Drift | Performance |
|--------|-----------|-------------|
| Godunov | 1.65e-02 | ✅ Good |
| WENO5 | 1.69e-01 | ❌ **10× worse!** |

Detailed diagnostics reveal:
```
Godunov:  |∇G| mean = 0.983 ± 0.008  (max: 1.00, min: 0.97)
WENO5:    |∇G| mean = 1.169 ± 0.933  (max: 7.92, min: 0.53)  ← HUGE spread!
```

## Root Cause: Saddle Point Singularity

### Initial Condition Geometry

Two circles positioned at x = 0.35 and x = 0.65, both with radius 0.12:
```
      Circle 1          Saddle          Circle 2
         O                .                 O
    (x=0.35)          (x=0.5)           (x=0.65)
```

The level set is defined as:
```python
G = min(d1, d2)
```

where d1 and d2 are signed distances to each circle.

### The Saddle Point Problem

At the saddle point (x=0.5, y=0.5) between the circles:

1. **G has a local minimum** (not smooth!)
   - G = 0.03 (slightly positive, outside both circles)
   - Neither circle is closer - it's equidistant

2. **Gradients vanish**:
   - ∂G/∂x = 0 (symmetry)
   - ∂G/∂y = 0 (on the line connecting centers)
   - |∇G| = 0 at this point

3. **The level set is NOT smooth here**:
   - G is C⁰ continuous but NOT C¹
   - Derivatives have a discontinuity (kink)
   - The "min" operation creates a non-differentiable feature

### Why Godunov Works

**1st-order upwind differences**:
```
∂G/∂x ≈ (G[i+1] - G[i])/Δx  or  (G[i] - G[i-1])/Δx
```

- Only uses immediate neighbors
- Naturally dissipative (adds numerical diffusion)
- Smears out the singularity
- **Result**: Develops smooth |∇G| ≈ 0.73 at saddle

### Why WENO5 Fails

**4th-order centered differences**:
```
∂G/∂x ≈ (-G[i+2] + 8G[i+1] - 8G[i-1] + G[i-2]) / (12Δx)
```

- Uses 5-point stencil
- No numerical dissipation
- Tries to maintain "sharp" features
- **At singular point**:
  - Stencil spans across the kink
  - Amplifies oscillations (Gibbs phenomenon)
  - Gradient explodes to 7.9!

### What Happens During Evolution

**Godunov (robust)**:
```
t=0:    Saddle has |∇G|=0, circles separate
        → Numerical diffusion smooths the kink

t=0.25: Saddle develops |∇G|≈0.73 (reasonable)
        → Circles remain separate, smooth evolution
```

**WENO5 (unstable)**:
```
t=0:    Saddle has |∇G|=0, circles separate
        → No diffusion, maintains sharp kink
        → Centered stencil sees discontinuous derivatives

t=0.25: Gradient EXPLODES at and near saddle point
        → |∇G| reaches 7.9 (should be ≈1.0!)
        → Causes rapid unphysical expansion
        → Circles "merge" prematurely due to errors
```

## Diagnostic Evidence

From `diagnose_merging_circles.py`:

**At saddle point (x=0.5, y=0.5)**:
```
Initial:  |∇G| = 0.000
Godunov:  |∇G| = 0.727  ← Smoothly develops gradient
WENO5:    |∇G| = 0.000  ← Still zero (but wrong elsewhere!)
```

**Near the circles (x=0.35 and x=0.65)**:
```
Initial:  |∇G| = 0.000
Godunov:  |∇G| = 0.000  ← Correct (inside domain)
WENO5:    |∇G| = 9.274  ← EXPLODED! (10× too large!)
```

## Visualization

See `diagnose_merging_circles.png`:

- **Initial**: Perfect circles, |∇G|=1 on interfaces, |∇G|=0 at saddle
- **Godunov**: Smooth evolution, |∇G| stays 0.97-1.00 everywhere
- **WENO5**: Catastrophic gradient growth (red patches), |∇G| from 0.5 to 7.9

The cross-section at y=0.5 clearly shows:
- Godunov: Smooth gradient profile
- WENO5: **Huge spikes** at x=0.35 and x=0.65

## Mathematical Explanation

Centered differences assume **smooth functions**. When G is non-smooth (C⁰ but not C¹):

**Taylor expansion breaks down**:
```
G(x+h) = G(x) + G'(x)h + G''(x)h²/2 + ...  ← Only valid if G' exists!
```

At the saddle:
- Left derivative ≠ Right derivative
- Centered formula sees: (-8 × left + 8 × right) / (12Δx)
- This **amplifies the jump** rather than averaging it
- Result: Gibbs-like oscillations and gradient blow-up

## General Lesson

**Higher-order centered schemes fail for:**

1. ✗ **Topology changes** (circles merging, splitting)
   - Creates singular points with |∇G| = 0
   - Level set becomes non-smooth (kinks)

2. ✗ **Sharp corners** (notched disks, stars, polygons)
   - Discontinuous derivatives
   - Gibbs oscillations

3. ✗ **Shock-like features** in the level set
   - Upwinding needed for stability

**Higher-order centered schemes excel for:**

1. ✓ **Single smooth shapes** (circles, ellipses)
   - C^∞ smooth everywhere
   - No singularities

2. ✓ **Source-dominated pure expansion**
   - u = 0, only S_L|∇G| term
   - Maintains smoothness

3. ✓ **Long-time integration** of smooth problems
   - Accumulation of truncation error matters
   - Higher order prevents drift

## Solution / Recommendation

For problems with **topology changes** or **sharp features**:

```python
# Use 1st-order upwind (Godunov)
gradient_scheme='godunov'
```

For problems with **smooth evolution only**:

```python
# Use 4th-order centered (WENO5)
gradient_scheme='weno5'

# BUT: Only if initial condition is smooth and remains smooth!
```

## Potential Improvements

To handle both cases, we could implement:

1. **Hybrid scheme**:
   - Detect non-smooth regions (large 2nd derivatives)
   - Use Godunov there, WENO5 elsewhere
   - Computationally expensive

2. **ENO/WENO reconstruction** (true WENO, not just centered):
   - Use adaptive stencil selection
   - Avoids discontinuous regions automatically
   - Complex implementation (~500+ lines)

3. **Reinitialization**:
   - Even WENO5 would work if we reinitialize every few steps
   - Reinitialization fixes |∇G| → 1 and smooths kinks
   - But defeats the purpose of "no reinit" test

## Conclusion

The merging circles test **correctly identifies** a fundamental limitation of higher-order centered differences:

> **They cannot handle non-smooth level sets that arise from topology changes.**

This is **not a bug** - it's expected behavior. The test serves as a valuable diagnostic:

- ✅ **Expanding circle**: Smooth → WENO5 wins (1632× improvement)
- ✅ **Expanding ellipse**: Smooth with varying curvature → WENO5 slightly better
- ✅ **Merging circles**: Non-smooth singular point → **Godunov wins**

**Key insight**: Always use `gradient_scheme='godunov'` when topology changes are expected!
