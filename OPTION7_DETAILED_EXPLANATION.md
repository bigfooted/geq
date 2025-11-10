# Option 7: True Signed Distance for Ellipse - Detailed Explanation

## The Problem We're Solving

**Current approximation:**
```python
G = sqrt((x/a)² + (y/b)²) * min(a,b) - min(a,b)
```

This is **NOT** a signed distance function! It's a scaled parametric form that gives:
- |∇G| = 1.0 at minor axis (top/bottom)
- |∇G| = 0.533 at major axis (left/right) ← 47% error!
- Average |∇G| ≈ 0.785 around boundary

**What we need:**
```
True signed distance: d(x,y) = min_{p ∈ ellipse} ||p - (x,y)||
```
where the sign is negative inside, positive outside.

---

## Why It's Non-Trivial

For a **circle**: `d(x,y) = sqrt(x² + y²) - r` ✓ Simple!

For an **ellipse**: No closed-form solution! 😱

The closest point on an ellipse to an arbitrary point requires solving a **quartic equation** (4th degree polynomial).

---

## The Mathematical Challenge

Given:
- Ellipse: `(x/a)² + (y/b)² = 1`
- Point: `P = (x₀, y₀)`

Find:
- Closest point on ellipse: `Q = (x_e, y_e)`
- Signed distance: `d = ±||P - Q||`

**Problem:** The point Q satisfies:
1. Q is on the ellipse: `(x_e/a)² + (y_e/b)² = 1`
2. The vector `P - Q` is perpendicular to ellipse at Q
3. This leads to a quartic equation in the parameter λ

---

## Solution Approach: Newton's Method

### Method 1: Parametric Newton Iteration

**Parametrize ellipse:**
```
x(θ) = a·cos(θ)
y(θ) = b·sin(θ)
```

**For each point (x₀, y₀), find θ that minimizes distance:**
```python
def distance_to_ellipse_point(theta, x0, y0, a, b):
    xe = a * cos(theta)
    ye = b * sin(theta)
    return (xe - x0)² + (ye - y0)²

# Find minimum using Newton's method
theta_opt = newton_minimize(distance_to_ellipse_point, ...)
xe, ye = a*cos(theta_opt), b*sin(theta_opt)
d = sqrt((x0-xe)² + (y0-ye)²)
```

**Pros:**
- Relatively simple (1D optimization)
- Always converges if initialized reasonably
- ~5-10 iterations per point

**Cons:**
- Need good initial guess for θ
- Must handle multiple local minima (ellipse has 4 symmetry)

---

### Method 2: Lagrange Multiplier (More Robust)

**Setup optimization:**
```
Minimize: f(x_e, y_e) = (x_e - x₀)² + (y_e - y₀)²
Subject to: g(x_e, y_e) = (x_e/a)² + (y_e/b)² - 1 = 0
```

**KKT conditions:**
```
∇f = λ·∇g
2(x_e - x₀) = λ·(2x_e/a²)
2(y_e - y₀) = λ·(2y_e/b²)

→ x₀ = x_e(1 - λ/a²)
→ y₀ = y_e(1 - λ/b²)
```

**Combined with constraint:**
```
(x₀/(1 - λ/a²)·a)² + (y₀/(1 - λ/b²)·b)² = 1
```

This is a **quartic in λ**! But we can solve numerically.

**Pros:**
- More robust (doesn't have symmetry issues)
- Guaranteed to find global minimum
- Standard optimization technique

**Cons:**
- Requires solving quartic or Newton iteration in 2D
- More complex implementation

---

### Method 3: Eberly's Algorithm (State-of-the-Art)

**Reference:** David Eberly (2012), "Distance from a Point to an Ellipse"

This is the **gold standard** method used in computer graphics and computational geometry.

**Key ideas:**
1. Exploit symmetry: Work in first quadrant only
2. Rational parameterization avoids angle wrapping
3. Closed-form solution for certain regions
4. Newton iteration only when necessary

**Algorithm sketch:**
```python
def signed_distance_ellipse_eberly(x0, y0, a, b):
    # Step 1: Map to first quadrant (exploit symmetry)
    x0_abs = abs(x0)
    y0_abs = abs(y0)

    # Step 2: Check special cases
    if x0_abs == 0 or y0_abs == 0:
        # On axis - closed form solution
        return special_case_solution()

    # Step 3: Initial guess using rational parameterization
    t0 = initial_guess(x0_abs, y0_abs, a, b)

    # Step 4: Newton iteration (converges in ~3-5 iterations)
    t_opt = newton_1d(objective_function, t0, tolerance=1e-8)

    # Step 5: Compute closest point and distance
    xe, ye = compute_ellipse_point(t_opt, a, b)
    dist = sqrt((x0_abs - xe)² + (y0_abs - ye)²)

    # Step 6: Determine sign (inside/outside)
    if (x0_abs/a)² + (y0_abs/b)² < 1:
        dist = -dist  # Inside

    return dist
```

**Pros:**
- Fastest convergence (3-5 iterations)
- Numerically stable
- Well-tested in production code
- Handles edge cases cleanly

**Cons:**
- Most complex to implement (~150 lines)
- Need to understand the theory deeply

---

## Implementation Strategy

### Recommended Approach: Simplified Eberly

I'd implement a **simplified version** that's easier to understand:

```python
def signed_distance_to_ellipse(x, y, cx, cy, a, b):
    """
    Compute signed distance from point (x, y) to ellipse centered at (cx, cy)
    with semi-axes a (horizontal) and b (vertical).

    Uses Newton iteration in parametric form.
    """
    # Translate to ellipse center
    x0 = x - cx
    y0 = y - cy

    # Work in first quadrant (symmetry)
    x0_abs = np.abs(x0)
    y0_abs = np.abs(y0)
    sign_x = np.sign(x0)
    sign_y = np.sign(y0)

    # Special cases
    if x0_abs < 1e-10 and y0_abs < 1e-10:
        # At center
        return -min(a, b)

    # Initial guess for angle θ
    theta = np.arctan2(y0_abs/b, x0_abs/a)

    # Newton iteration to find closest point
    max_iter = 20
    tolerance = 1e-8

    for iteration in range(max_iter):
        # Point on ellipse
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        xe = a * cos_theta
        ye = b * sin_theta

        # Distance squared and its derivative
        dx = xe - x0_abs
        dy = ye - y0_abs
        f = dx * (-a*sin_theta) + dy * (b*cos_theta)  # d(dist²)/dθ

        if abs(f) < tolerance:
            break

        # Second derivative
        fp = dx * (-a*cos_theta) + dy * (-b*sin_theta) + \
             (-a*sin_theta)**2 + (b*cos_theta)**2

        # Newton update
        theta = theta - f / fp

        # Keep in [0, π/2]
        theta = np.clip(theta, 0, np.pi/2)

    # Compute final distance
    xe = a * np.cos(theta)
    ye = b * np.sin(theta)
    dist = np.sqrt((x0_abs - xe)**2 + (y0_abs - ye)**2)

    # Determine sign (inside/outside)
    if (x0_abs/a)**2 + (y0_abs/b)**2 < 1.0:
        dist = -dist

    return dist

# Vectorized version for grid
def initialize_ellipse_sdf(X, Y, cx, cy, a, b):
    """
    Initialize signed distance function for ellipse on grid.
    """
    G = np.zeros_like(X)
    ny, nx = X.shape

    # Compute for each grid point
    for i in range(ny):
        for j in range(nx):
            G[i, j] = signed_distance_to_ellipse(
                X[i, j], Y[i, j], cx, cy, a, b
            )

    return G
```

---

## Computational Cost

**Per point:**
- Simple approximation: 1 evaluation (current)
- Newton method: ~5-10 iterations × (trig functions + arithmetic)
- **~50-100× slower for initialization**

**For 121×121 grid:**
- Current: < 0.001s
- True SDF: ~0.1-0.5s

This is **one-time cost** at initialization, totally acceptable!

---

## Expected Improvement

With true signed distance (perfect |∇G| = 1.0 initially):

**Best case scenario:**
- If initialization is the only problem → Could see 10-50× improvement
- Similar to what we see for circle

**Realistic scenario:**
- Initial condition helps but curvature variation remains
- Likely 5-15× improvement (better than current 1.4×, less than circle's 1632×)

**Why not as good as circle?**
Even with perfect |∇G| = 1.0 initially, the ellipse has:
1. **Non-uniform curvature** → different S_L·κ contribution
2. **Non-uniform propagation** → errors accumulate differently
3. **Geometric coupling** → x and y derivatives interact differently

The true signed distance gives us the **best possible starting point**, but doesn't eliminate the fundamental difficulty of maintaining |∇G| = 1.0 on an ellipse.

---

## Alternative: Use Existing Library

**Option 7b: Use scikit-image**

```python
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.segmentation import inverse_gaussian_gradient
from scipy.ndimage import distance_transform_edt

def ellipse_sdf_approximate(X, Y, cx, cy, a, b):
    # Create binary mask of ellipse
    mask = ((X-cx)/a)**2 + ((Y-cy)/b)**2 < 1.0

    # Compute distance transform
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)

    # Combine with proper sign
    sdf = np.where(mask, -dist_inside, dist_outside)

    return sdf
```

**Pros:**
- 5 lines of code!
- Uses fast Euclidean distance transform
- Already optimized

**Cons:**
- Approximate (uses pixel-level distance, not true geometric distance)
- Depends on grid resolution
- Not as accurate as true analytical solution

---

## Implementation Decision Tree

```
Do you need publication-quality results?
├─ YES → Implement full Newton method (3-4 hours)
│        Gets theoretically perfect |∇G| = 1.0
│
└─ NO → Use scipy distance transform (30 min)
         Gets |∇G| ≈ 0.95-0.99 (good enough)
```

**My recommendation:**
1. **Start with scipy approach** (30 min) - see if it helps
2. **If results promising** → implement full Newton (3 hrs) for final version
3. **If still only 2-3× improvement** → proves curvature is fundamental limit

---

## What This Will Prove

**Scenario A: Dramatic improvement (50-100×)**
→ Proves initialization was the bottleneck
→ Shows higher-order scheme truly superior for ellipses
→ Great demonstration result!

**Scenario B: Modest improvement (5-15×)**
→ Proves curvature variation is fundamental challenge
→ Still shows benefit of proper initialization
→ Honest scientific result

**Scenario C: Minimal improvement (1.5-3×)**
→ Proves ellipse is inherently harder than circle
→ Demonstrates limitations of any scheme
→ Important negative result for understanding

All three outcomes are **scientifically valuable**!

---

## Bottom Line

**True signed distance initialization:**
- Gives us the absolute best possible starting condition
- Requires Newton iteration (moderate complexity)
- Takes 3-4 hours to implement properly
- OR 30 minutes using scipy approximation
- Will definitively answer: "Is initialization or curvature the bottleneck?"

**Would you like me to:**
1. Implement the full Newton method (rigorous, 3-4 hrs)
2. Try scipy distance transform first (quick test, 30 min)
3. Do something else from the options list?
