"""
Reporting utilities to consolidate repeated print statements across tests.

Each helper prints a focused section. Keep arguments simple and flexible.
"""

from typing import List, Dict, Optional
import numpy as np


def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def sub_banner(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def print_solver_overview(test_title: str,
                          nx: int, ny: int, Lx: float, Ly: float,
                          S_L: float,
                          flow_description: List[str],
                          time_scheme: str,
                          reinit_description: Optional[str] = None) -> None:
    banner(test_title)
    print(f"Time Scheme: {time_scheme.upper()}")
    if reinit_description:
        print(reinit_description)
    print("=" * 80)
    print(f"Grid: {nx} x {ny}")
    print(f"Domain: [0, {Lx}] x [0, {Ly}]")
    print(f"Laminar flame speed S_L = {S_L}")
    for line in flow_description:
        print(line)


def print_initial_flame_verification(y0: float,
                                     initial_positions,
                                     regions: Optional[List[str]] = None) -> None:
    sub_banner("Initial flame position verification")
    print(f"  Expected y0 = {y0:.6f}")
    if isinstance(initial_positions, dict):
        overall = initial_positions.get('overall', np.nan)
        print(f"  Computed y0 (overall) = {overall:.6f}")
        if regions:
            for r in regions:
                val = initial_positions.get(r, np.nan)
                print(f"  Computed y0 ({r} region) = {val:.6f}")
    else:
        print(f"  Computed y0 = {float(initial_positions):.6f}")


def print_G_field_stats(G_initial: np.ndarray,
                        bottom_val: float,
                        top_val: float,
                        bottom_label: str = 'y=0',
                        top_label: str = 'y=Ly') -> None:
    sub_banner("Initial G field statistics")
    print(f"  G_min = {G_initial.min():.6f}")
    print(f"  G_max = {G_initial.max():.6f}")
    print(f"  G at {bottom_label} (bottom): {bottom_val:.6f} (should be > 0, unburnt)")
    print(f"  G at {top_label} (top): {top_val:.6f} (should be < 0, burnt)")


def print_velocity_field_verification(samples: List[Dict[str, float]]) -> None:
    sub_banner("Velocity field verification")
    for s in samples:
        label = s.get('label', 'sample')
        value = s.get('value', float('nan'))
        expected = s.get('expected', float('nan'))
        print(f"  {label}: {value:.6f} (should be {expected})")


def print_solve_start() -> None:
    sub_banner("Solving G-equation...")


def print_performance(elapsed_time: float,
                      n_total_steps: int,
                      snapshots_saved: int,
                      extraction_time: Optional[float] = None) -> None:
    sub_banner("Computation Performance")
    print(f"  Total simulation time: {elapsed_time:.4f} seconds")
    print(f"  Time steps computed: {n_total_steps}")
    print(f"  Snapshots saved: {snapshots_saved}")
    if n_total_steps > 0:
        print(f"  Time per step: {elapsed_time/n_total_steps*1000:.4f} ms")
    if extraction_time is not None:
        print(f"  Extraction time: {extraction_time:.4f} seconds")
        print(f"  Total elapsed time: {elapsed_time + extraction_time:.4f} seconds")


def print_region_stats(regions: List[Dict]) -> None:
    for r in regions:
        name = r.get('name', 'region')
        u_y = r.get('u_y', None)
        Uexp = r.get('U_expected', None)
        y0 = r.get('y0', None)
        y_final = r.get('y_final', None)
        disp = r.get('displacement', None)
        vel = r.get('velocity', None)
        verr = r.get('velocity_error', None)
        header = r.get('header', name)
        print(f"\n{header}")
        if u_y is not None and Uexp is not None:
            print(f"  u_y = {u_y}, Expected U = {Uexp:+.3f}")
        if y0 is not None:
            print(f"  Initial position: {y0:.6f}")
        if y_final is not None:
            print(f"  Final position: {y_final:.6f}")
        if disp is not None:
            print(f"  Displacement: {disp:.6f}")
        if vel is not None:
            print(f"  Computed velocity: {vel:+.6f}")
        if verr is not None:
            print(f"  Velocity error: {verr:.6f}")


def print_domain_info(nx: int, ny: int, Lx: float, Ly: float,
                      dx: float, dy: float,
                      cfl_conv: Optional[float] = None,
                      cfl_prop: Optional[float] = None) -> None:
    sub_banner("Domain Information")
    print(f"  Grid resolution: {nx} × {ny} = {nx*ny} points")
    print(f"  Domain size: [0, {Lx}] × [0, {Ly}]")
    print(f"  Grid spacing: dx = {dx:.6f}, dy = {dy:.6f}")
    if cfl_conv is not None:
        print(f"  CFL number (convection): {cfl_conv:.4f}")
    if cfl_prop is not None:
        print(f"  CFL number (propagation): {cfl_prop:.4f}")


def print_flame_accuracy(position_error: np.ndarray, y0: float,
                         analytical_positions: np.ndarray, t_final: float,
                         U_expected: float, y_final_num: float) -> None:
    sub_banner("Flame Position Accuracy")
    print(f"  Maximum absolute error: {np.max(position_error):.8f}")
    print(f"  Mean absolute error: {np.mean(position_error):.8f}")
    print(f"  RMS error: {np.sqrt(np.mean(position_error**2)):.8f}")
    print(f"  Final position error: {position_error[-1]:.8f}")

    print(f"\nFinal State (t = {t_final}s):")
    print(f"  Analytical position: {analytical_positions[-1]:.8f}")
    print(f"  Numerical position: {y_final_num:.8f}")
    print(f"  Distance traveled: {analytical_positions[-1] - y0:.8f}")
    print(f"  Expected velocity U = {U_expected:.4f}")
    print(f"  Computed velocity = {(y_final_num - y0) / t_final:.4f}")


def print_completion(message: str = "Test completed successfully!") -> None:
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80 + "\n")
