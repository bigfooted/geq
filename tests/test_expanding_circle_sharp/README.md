# Expanding Circle (Sharp IC)

This test simulates an expanding circular flame starting from a sharp (discontinuous) initial level set and no background flow.

- u = (0, 0)
- Radius grows as R(t) = R0 + S_L t
- Compares numerical and analytical radii; produces contours, profiles, and surface plots

Run:
- From repo root: python tests/test_expanding_circle_sharp/test_expanding_circle_sharp.py [euler|rk2]