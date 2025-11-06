Linear Flame (Three Regions, corrected IC and diagnostics)

Purpose
- Horizontal flame with three x-regions of vertical flow: left=0, mid=0.2, right=0.
- Initial G varies linearly in y (+1 at inlet → 0 at y0 → -1 at top). Bottom band Dirichlet G=+1 enforced.
- Computes flame positions per region, flame length vs time, and 3D surface; designed to match expected U = u_y − S_L per region.

Run
- From repository root: python tests/test_linear_flame_nonhom_correct/test_linear_flame_nonhom_correct.py [rk2|euler] [t=10.0] [no_reinit]
