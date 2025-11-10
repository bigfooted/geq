# FTF Test Files - Overview

This directory contains three versions of the oscillating flame FTF test:

## Files

### 1. `test_ftf_linear_flame_single.py` (Original)
The original version with basic settings:
- Time scheme: RK2 (2nd-order)
- Spatial scheme: WENO5 (5th-order)
- Gradient scheme: Godunov (1st-order, implicit)
- Reinitialization: Optional, disabled by default
- Initial condition: Linear profile (|∇G| varies from 1/y₀ to 1/(Ly-y₀))

**Performance:** Fast but |∇G| drifts significantly (→ 37 without reinit)

### 2. `test_ftf_linear_flame_single_improved.py` (Experimental)
Enhanced version with FMM initialization and gradient scheme comparison:
- Includes Fast Marching Method for proper SDF initialization
- Allows comparison between Godunov and WENO5 gradient schemes
- Comprehensive initial/final gradient quality reporting
- Comparison mode to test both schemes

**Key finding:** WENO5 gradient scheme **EXPLODES** for oscillating flames!

### 3. `test_ftf_linear_flame_single_updated.py` (⭐ RECOMMENDED)
Production-ready version with recommended settings based on comprehensive testing:

**Settings:**
- ✅ Time scheme: **RK3** (3rd-order accuracy)
- ✅ Spatial scheme: **Upwind2** (2nd-order, robust for cusps)
- ✅ Gradient scheme: **Godunov** (1st-order, stable for cusps)
- ✅ Reinitialization: **Enabled** every 10 steps with Fast Marching
- ✅ Initial condition: **Proper SDF** (|∇G| = 1.0 everywhere)

**Performance:** Stable, accurate, and physically correct

## Recommendation

**Use `test_ftf_linear_flame_single_updated.py` for production work!**

This version incorporates all lessons learned from comprehensive testing:
1. ✅ Proper SDF initialization (|∇G| = 1.0)
2. ✅ Godunov gradient (handles flame cusps)
3. ✅ Frequent reinitialization (cusps degrade |∇G|)
4. ✅ Higher-order time integration (RK3)
5. ✅ Robust spatial scheme (Upwind2)

## Why Not WENO5 Gradient?

From testing, WENO5 gradient scheme is **catastrophically unstable** for oscillating flames:

| Configuration | Initial |∇G| | Final |∇G| | Stability | Notes |
|---------------|------------|------------|-----------|-------|
| **Godunov** | 1.000 | 1.387 | ✅ Stable | Recommended |
| **WENO5** | 1.000 | 60,900 | ❌ **EXPLODES** | Never use for oscillating flames! |

**Reason:** Oscillating velocity creates sharp flame **cusps** with discontinuous derivatives. WENO5 uses centered differences which require C² smoothness, leading to exponential |∇G| growth via Gibbs phenomenon.

## Usage Examples

### Basic run (recommended settings):
```bash
python test_ftf_linear_flame_single_updated.py f=10.0
```

### With custom parameters:
```bash
python test_ftf_linear_flame_single_updated.py f=10.0 drop_cycles=10 measure_cycles=5
```

### Test different gradient scheme (not recommended):
```bash
python test_ftf_linear_flame_single_updated.py godunov f=10.0  # Stable
python test_ftf_linear_flame_single_updated.py weno5 f=10.0    # Will EXPLODE!
```

### Comparison mode (from improved version):
```bash
python test_ftf_linear_flame_single_improved.py compare f=10.0
```

## Supporting Files

### `fastmarch.py` (New module)
Standalone Fast Marching Method implementation for computing proper signed distance functions.

**Features:**
- Solves Eikonal equation |∇φ| = 1
- O(N log N) complexity
- Most accurate SDF initialization method
- Can be imported and used in other codes

**Usage:**
```python
from fastmarch import fast_marching_method

# Create mask (True = inside, False = outside)
mask = (X - 0.5)**2 + (Y - 0.5)**2 < 0.1**2

# Compute SDF
phi = fast_marching_method(mask, dx)
# Result: |∇φ| ≈ 1.0 everywhere (0.17% error)
```

**Test FMM quality:**
```bash
python fastmarch.py
# Creates visualization: fastmarch_test.png
```

## Performance Comparison

| Version | Time/Step | |∇G| Drift | Stability | FTF Quality |
|---------|-----------|-----------|-----------|-------------|
| Original | 4.2 ms | High (→37) | ✅ Stable | Good |
| Updated (Godunov) | 35.7 ms | Low (→1.4) | ✅ Stable | Excellent |
| Improved (WENO5) | 37.8 ms | Catastrophic (→60,900) | ❌ **EXPLODES** | Invalid |

**Note:** Updated version is 8× slower but maintains excellent |∇G| quality throughout simulation.

## Documentation

For complete analysis and recommendations, see:
- `OSCILLATING_FLAME_RECOMMENDATIONS.md` - Detailed analysis of why WENO5 fails
- `FMM_COMPARISON.md` - Fast Marching Method vs other initialization methods
- `COMPLETE_SUMMARY.md` - Overview of all testing and findings

## Key Lessons

1. **Higher-order ≠ Better:** WENO5 gradient gives 1632× improvement for smooth circles but explodes for oscillating flames. Match scheme to physics!

2. **Initialization matters:** Bad initialization (|∇G| = 0.785) wastes potential. FMM initialization (|∇G| = 1.002) realizes full benefit.

3. **Cusps break centered differences:** Oscillating flames create sharp cusps that violate smoothness assumptions of high-order centered schemes.

4. **Godunov is robust:** First-order upwind gradient scheme handles discontinuities naturally and is the only stable option for oscillating flames.

5. **Fast Marching Method is gold standard:** FMM gives best SDF initialization (0.17% error vs 17% for EDT, 2.3% for reinit PDE).

## Contact

For questions about these implementations, see the comprehensive analysis documents in the project root directory.
