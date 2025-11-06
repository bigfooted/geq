# Expanding Circle (No Flow)

This test simulates an expanding circular flame without background flow using the improved solver.

- u = (0, 0)
- R(t) = R0 + S_L t; circumference via marching squares
- Optional local reinitialization with diagnostics

Run:
- From repo root: python tests/test_expanding_circle/test_expanding_circle.py [euler|rk2] [t=...] [no_reinit]
