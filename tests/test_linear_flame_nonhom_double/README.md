Linear Flame (Two regions: left=0, right=0.2)

Purpose
- Horizontal flame with a single vertical velocity jump at x=0.1: left u_y=0.0, right u_y=0.2.
- Confirms regional flame speeds U_left = -0.1 and U_right = +0.1 for S_L=0.1.

Run
- From repository root: python tests/test_linear_flame_nonhom_double/test_linear_flame_nonhom_double.py [rk2|euler] [t=2.0] [no_reinit]
