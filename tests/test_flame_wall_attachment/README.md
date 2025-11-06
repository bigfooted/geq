Flame Wall Attachment (45° anchored flame)

Purpose
- Laminar premixed flame attached to a wall at the origin, stabilized at 45° where S_L = u_y.
- Enforces anchoring at (0,0) via a small Dirichlet region and compares against the analytical 45° line.

Notes
- Local (narrow-band) reinitialization recommended; anchoring enforced after each step and after reinit.
- Produces evolution snapshots, time-series analysis, and final comparison.

Run
- From repository root: python tests/test_flame_wall_attachment/test_flame_wall_attachment.py [rk2|euler] [t=1.0] [no_reinit]
