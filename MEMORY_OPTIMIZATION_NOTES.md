# Memory Optimization for FTF Computation

## Current Approach (Implemented)

**Status**: Memory usage reduced from 14.5 GB → 0.44 GB (33× improvement)

**Method**: Adaptive `save_interval` to store ~1500 snapshots instead of all 48,334 timesteps

```python
target_snapshots = 1500
save_interval = max(1, int(np.ceil(n_steps / target_snapshots)))
# Result: ~0.44 GB for 201×201 grid
```

**Pros**:
- Simple implementation (no solver modification needed)
- Still provides good temporal resolution for FTF analysis
- Works with existing visualization code

**Cons**:
- Still stores full 2D fields (40,401 values) when only 1 scalar is needed per snapshot
- Memory scales with grid resolution: O(nx × ny × snapshots)

---

## Ultimate Solution: On-the-Fly Computation

**Potential memory reduction**: 0.44 GB → **0.01 MB** (40,000× improvement!)

### Key Insight

For FTF computation, we only need **flame lengths** (1 scalar per timestep), not full 2D G fields.

**Current waste**:
- Store: 1500 snapshots × 40,401 grid points = 60,600,000 floats (≈484 MB)
- Actually use: 1500 flame lengths = 1,500 floats (≈12 KB)
- **Waste factor**: 40,400×

### Implementation Strategy

Modify `GEquationSolver2D.solve()` to accept optional callback:

```python
def solve(self, G_initial, t_final, dt,
          callback=None,  # NEW: called every save_interval steps
          save_interval=None,
          ...):

    # Inside time loop:
    if step % save_interval == 0:
        if callback is not None:
            # User computes what they need (e.g., flame length)
            result = callback(self.G, t, step)
            # Callback stores result in external list

        # Only save to G_history if user needs full fields
        if return_full_fields:  # NEW: optional flag
            G_history.append(self.G.copy())
        t_history.append(t)
```

### Usage Example

```python
# Storage for just the scalars we need
flame_lengths = []
time_points = []

def compute_flame_length(G, t, step):
    """Callback executed at each save_interval"""
    length = compute_contour_length(G, solver.X, solver.Y, iso_value=0.0, N=5)
    flame_lengths.append(length)
    time_points.append(t)
    return length

# Solve WITHOUT storing full fields
solver.solve(
    G0, t_final, dt,
    save_interval=33,
    callback=compute_flame_length,
    return_full_fields=False,  # Don't store G_history!
    ...
)

# Now flame_lengths and time_points contain what we need for FTF
# Memory usage: ~12 KB instead of 484 MB!
```

### Benefits

1. **Massive memory reduction**: 40,000× less memory
2. **Enable longer simulations**: Can run 10× longer without memory issues
3. **Enable finer grids**: Can use 500×500 grids instead of 201×201
4. **Parallel parameter sweeps**: Run 40 cases simultaneously instead of 1

### Visualization Strategy

For contour plots (optional), use one of these approaches:

**Option 1**: Save only specific snapshots to disk
```python
if t in contour_times:
    np.save(f'snapshot_t{t:.3f}.npy', G)
```

**Option 2**: Checkpoint + restart
```python
# Save final state, restart for visualization if needed
save_checkpoint('final.npz', solver, G_final, t_final, meta)
```

**Option 3**: Dual-mode callback
```python
def callback(G, t, step):
    # Always compute flame length
    length = compute_contour_length(G, solver.X, solver.Y, iso_value=0.0, N=5)
    flame_lengths.append(length)

    # Optionally save snapshots for visualization
    if abs(t - contour_time) < 0.5 * dt:
        snapshots_for_viz.append((t, G.copy()))

    return length
```

---

## Memory Comparison Table

| Approach | Memory (201×201) | Memory (500×500) | Notes |
|----------|------------------|------------------|-------|
| All snapshots (48k) | 14.5 GB | 90.7 GB | Original (infeasible) |
| Adaptive interval (1.5k) | 0.44 GB | 2.8 GB | Current (good) |
| On-the-fly (1.5k scalars) | 12 KB | 12 KB | Ultimate (grid-independent!) |

---

## When Each Approach Makes Sense

### Current Approach (save_interval reduction)
**Use when:**
- Need contour plots at many time points
- Quick implementation needed
- Grid is modest (≤300×300)
- Single run (not parameter sweep)

### Ultimate Approach (callbacks)
**Use when:**
- Running parameter sweeps (many cases)
- Large grids (>300×300)
- Long simulations (>100 seconds)
- Only need scalar outputs (FTF, lengths, areas, etc.)
- Production code for repeated use

---

## Implementation Priority

**Phase 1** (DONE): Adaptive save_interval
- ✅ Reduces memory by 33×
- ✅ Zero solver modification
- ✅ Sufficient for current use case

**Phase 2** (Future): Callback API
- Add `callback` parameter to `solve()`
- Add `return_full_fields` flag
- Implement in `GEquationSolver2D.solve()`
- Update test scripts to use callbacks

**Phase 3** (Future): Smart visualization
- Implement selective snapshot saving
- Create restart-based visualization workflow
- Add memory profiling tools

---

## Conclusion

The current adaptive save_interval approach provides **excellent bang-for-buck**:
- 33× memory reduction
- 5 minutes implementation time
- No solver modification needed
- Maintains all existing functionality

The callback approach would provide an additional **1000× improvement** but requires:
- Solver API changes
- Test code refactoring
- More complex user code

**Recommendation**: Keep current approach unless hitting memory limits with large grids or parameter sweeps.
