# Expanding Moving Circle (u = (0, 0.1))

This test simulates an expanding circle advected upward by a uniform vertical flow.

- u = (0.0, 0.1)
- Radius grows by S_L; center advects with u
- Compares numerical vs analytical radius and center; includes surface-length diagnostics

Run:
- From repo root: python tests/test_expanding_moving_circle_u_v/test_expanding_moving_circle_u_v.py [euler|rk2] [t=...] [no_reinit]
