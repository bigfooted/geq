# 2D G-Equation Solver for Laminar Premixed Flames - Project Summary

**Date:** 2025-11-01 12:31:53 UTC  
**User:** bigfootedexport

---

## Project Overview

This project implements a 2D G-equation solver for modeling laminar premixed flame surfaces using the level set approach. The solver includes multiple time discretization schemes, reinitialization methods, and comprehensive test cases with analytical validation.

---

## Complete File Structure

### Core Solver Files

1. **`g_equation_solver.py`** - Basic solver implementation
   - First-order Euler and second-order RK2 time schemes
   - First-order upwind spatial discretization
   - Vectorized gradient computation
   - Supports flow velocity field

2. **`g_equation_solver_improved.py`** - Enhanced solver with reinitialization
   - All features from basic solver
   - PDE-based reinitialization
   - Fast marching reinitialization
   - Local (narrow-band) and global reinitialization options
   - Initial condition smoothing for sharp discontinuities
   - Default: LOCAL reinitialization every 50 steps

3. **`plotting_utils.py`** - Visualization utilities
   - Contour plot comparisons with analytical solutions
   - Radius and center evolution plots
   - Trajectory plots
   - 3D surface plots
   - Error analysis plots

### Test Cases

4. **`test_expanding_circle.py`** - Stationary expanding circle
   - Zero velocity field (u = 0, v = 0)
   - Smooth signed distance initial condition
   - Local reinitialization enabled by default (every 50 steps)
   - Analytical validation: R(t) = R₀ + S_L·t
   - Flame surface area computation and validation
   - Usage: `python test_expanding_circle.py rk2 t=2.0`

5. **`test_expanding_moving_circle.py`** - Moving expanding circle
   - Non-zero velocity field (u = 0.1, v = 0.0)
   - Tracks flame propagation and convection
   - Local reinitialization enabled by default
   - Analytical validation for radius and center position
   - Flame surface area computation
   - Usage: `python test_expanding_moving_circle.py rk2 t=3.0`

6. **`test_expanding_circle_sharp.py`** - Sharp initial condition test
   - Discontinuous IC: G = -1 inside, G = +1 outside
   - Tests numerical robustness
   - No reinitialization
   - Usage: `python test_expanding_circle_sharp.py rk2`

7. **`test_expanding_circle_sharp_improved.py`** - Sharp IC with improvements
   - All reinitialization and smoothing options
   - Supports LOCAL/GLOBAL reinitialization
   - Command-line control of all parameters
   - Usage: `python test_expanding_circle_sharp_improved.py rk2 smooth reinit=50 local t=3.0`

### Comparison and Diagnostic Tools

8. **`compare_time_schemes.py`** - Compare Euler vs RK2
   - Runs both test cases with both schemes
   - Error reduction analysis
   - Computation time comparison

9. **`compare_initial_conditions.py`** - Compare smooth vs sharp IC
   - Shows importance of IC smoothing
   - Error evolution comparison
   - Cumulative error analysis

10. **`compare_improvements.py`** - Comprehensive method comparison
    - Tests 10 different configurations
    - Includes LOCAL vs GLOBAL reinitialization
    - Smooth IC combinations
    - Error reduction factors
    - Usage: `python compare_improvements.py t=2.0`

11. **`test_reinitialization.py`** - Reinitialization diagnostic
    - LOCAL vs GLOBAL comparison
    - Multiple reinitialization frequencies
    - PDE vs Fast Marching methods
    - Gradient quality assessment
    - Usage: `python test_reinitialization.py t=3.0`

12. **`diagnose_reinitialization.py`** - Fine mesh diagnostic
    - Tests multiple simulation times
    - Optimal frequency determination
    - Gradient magnitude tracking
    - Error accumulation analysis

13. **`diagnose_reinitialization_coarse.py`** - Coarse mesh diagnostic
    - Large domain (up to 5×5)
    - Coarse grids (31×31, 51×51, 71×71)
    - Larger time steps (dt = 0.005 to 0.01)
    - Non-zero flow velocity
    - Long simulation times (up to 10s)
    - Tests when reinitialization becomes beneficial

14. **`requirements.txt`** - Python dependencies
    ```
    numpy>=1.21.0
    matplotlib>=3.4.0
    ```

15. **`README.md`** - Project documentation

---

## Key Equations

### G-Equation