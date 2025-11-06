Expanding Moving Circle (u = (0.1, 0))

Purpose
- Expanding flame front in a uniform horizontal flow. Verifies radius growth R(t) = R0 + S_L t and advection of the center.
- Compares numerical radius, center, and G=0 flame length against analytical values.

Notes
- Uses local (narrow-band) reinitialization by default.
- Saves contour snapshots, radius/center plots, surface area comparisons, and 3D surface at final time.

Run
- From repository root: python tests/test_expanding_moving_circle/test_expanding_moving_circle.py [euler|rk2] [t=1.5] [no_reinit]
