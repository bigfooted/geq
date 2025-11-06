# 2D G-Equation Solver for Laminar Premixed Flames

This repository contains a Python implementation of the 2D G-equation solver using the level set approach for modeling laminar premixed flame surfaces.

## Overview

The G-equation describes the motion of a flame surface in premixed combustion:
## High-order schemes (RK3 + WENO5)

The improved solver `g_equation_solver_improved.py` now supports:

- Time schemes: `euler`, `rk2`, `rk3` (SSP RK3)
- Spatial schemes: `upwind` (first-order), `weno5` (fifth-order WENO, conservative form)

Usage example (class API):

```python
from g_equation_solver_improved import GEquationSolver2D

solver = GEquationSolver2D(nx=101, ny=101, Lx=0.5, Ly=1.0, S_L=0.1)
G0 = ...  # initial level set
G_hist, t_hist = solver.solve(
	G_initial=G0,
	t_final=2.0,
	dt=0.001,
	time_scheme='rk3',
	spatial_scheme='weno5',
	reinit_interval=50,
	reinit_method='fast_marching',
	reinit_local=True,
	velocity_updater=lambda s, t: (s.u_x*0, s.u_y*0 + 0.2),
	smooth_ic=True,
)
```

See `tests/test_compare_schemes/test_compare_schemes.py` for a passive-advection comparison between baseline (RK2 + upwind) and high-order (RK3 + WENO5) schemes.

### High-order nonhomogeneous linear flame test

We provide a ready-to-run script variant configured for RK3 + WENO5:

- `tests/test_linear_flame_nonhom_highorder/test_linear_flame_nonhom_highorder.py`

Run examples:

```bash
python tests/test_linear_flame_nonhom_highorder/test_linear_flame_nonhom_highorder.py t=2.0 reinit
python tests/test_linear_flame_nonhom_highorder/test_linear_flame_nonhom_highorder.py t=2.0 no_reinit
```

### Convergence study (RK3 + WENO5)

The helper script `scripts/convergence_linear_flame_nonhom_rk3_weno5.py` sweeps grid resolutions and reports:

- Velocity error in the middle band vs. expected analytical value U_mid = u_y - S_L
- Mean |∇G| near the interface
- Runtime and step counts

Run:

```bash
python scripts/convergence_linear_flame_nonhom_rk3_weno5.py
```

Outputs are saved under `results/`:

- `convergence_rk3_weno5_nonhom.csv`
- `convergence_rk3_weno5_velocity_error.png`
- `convergence_rk3_weno5_grad_deviation.png`

Notes:
- WENO5 here is implemented in conservative form (∇·(G u)), which is robust and works well with variable velocities. For constant advection of smooth fields at very small dt, first-order upwind may occasionally show lower L2 error; higher-order advantages are clearer with coarser grids or larger timesteps.

