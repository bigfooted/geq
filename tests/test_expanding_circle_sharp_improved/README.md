# Expanding Circle (Sharp IC) — Improved

This test simulates an expanding circular flame with a sharp (discontinuous) initial condition using the improved G-equation solver.

Highlights:
- No background flow (u = 0)
- Laminar burning at speed S_L grows the radius: R(t) = R0 + S_L t
- Optional reinitialization (local/global) and initial-condition smoothing
- Compares numerical radius vs analytical radius; saves contour and surface plots

Run:
- From repo root: python tests/test_expanding_circle_sharp_improved/test_expanding_circle_sharp_improved.py [euler|rk2] [t=...] [reinit=...] [method=fast_marching|pde] [local|global] [smooth]