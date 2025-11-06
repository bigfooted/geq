# Linear Flame (Uniform Vertical Flow)

Baseline linear flame test with uniform vertical flow.

- Initial condition: sharp horizontal interface at y=y0 separating unburnt (G>0) below and burnt (G<0) above
- Velocity field: u = (0, 0.2)
- Laminar flame speed: S_L = 0.1
- Expected net flame speed: U = u_y - S_L = 0.1 (upward)

What it checks
- Numerical vs analytical flame position y(t) = y0 + U t
- 2D contours and a 3D surface of G

Run
- python test_linear_flame.py
- python test_linear_flame.py rk2 t=2.0
- python test_linear_flame.py euler no_reinit
