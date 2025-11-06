# Linear Flame Nonhomogeneous (Small)

Two-region vertical flow test for the 2D G-equation solver.

- Initial condition: horizontal interface at y=y0 separating unburnt (G>0, below) and burnt (G<0, above)
- Velocity field:
  - u_x = 0 everywhere
  - u_y = 0.0 for x < 0.1 (left region)
  - u_y = 0.2 for x ≥ 0.1 (right region)
- Laminar flame speed: S_L = 0.1

Expected net flame speeds: U = u_y - S_L
- Left:  -0.1 (recedes)
- Right: +0.1 (advances)

What it checks
- Flame position in left and right regions vs time
- 2D contours and 3D surface of G

Run
- Default: RK2, local reinit every 50 steps
- Example:
  - python test_linear_flame_nonhom_small.py
  - python test_linear_flame_nonhom_small.py rk2 t=2.0
  - python test_linear_flame_nonhom_small.py euler no_reinit
