# Linear Flame Nonhomogeneous (Three Regions)

Three-region vertical flow test with optional time-dependent middle-band velocity.

- Initial condition: linear-in-y G profile with G=+1 at y=0, G=0 at y=y0, G=-1 at y=Ly
- Velocity field:
  - u_x = 0 everywhere
  - u_y = 0.0 for x ≤ 0.1 (left)
  - u_y = 0.2 for 0.1 < x < 0.9 (middle)
  - u_y = 0.0 for x ≥ 0.9 (right)
  - Optional time dependence in the middle band: u_y(y,t) = U_Y(1 + A sin(St (t - K y)))
- Laminar flame speed: S_L = 0.1, with S_L masked to 0 near the side-bottom corners
- Inlet Dirichlet: G=+1 pinned on a thin bottom band across the entire inlet

Outputs
- Flame position per region vs time, total flame length vs time
- Velocity snapshots grid with G=0 overlay
- 3D surface of G
- Composite flame length + representing velocity on twin axes

Run examples
- python test_linear_flame_nonhom.py
- python test_linear_flame_nonhom.py rk2 t=10.0
- python test_linear_flame_nonhom.py euler no_reinit
