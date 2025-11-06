This test compares the baseline scheme (RK2 time, upwind space) with the high-order scheme (SSP RK3 time, WENO5 space) on a smooth passive advection of a signed-distance circle under a constant velocity.

Scenario
- Domain: [0, 1] x [0, 1]
- Grid: 201 x 201
- Velocity: u = (0.5, 0.3) constant
- Initial level set: signed distance to circle centered at (0.3, 0.3) with radius 0.15
- Final time: T = 0.2
- CFL target ~ 0.4 w.r.t convective speed

We compute the analytic solution by translating the initial circle by u*T and compare the numerical level set to the exact signed distance, measuring:
- L2 error over the grid
- Mean |∇G| at the interface (should be close to 1)
- Runtime

Expectation: RK3+WENO5 achieves lower L2 error at similar dt (and/or allows a larger stable dt).
