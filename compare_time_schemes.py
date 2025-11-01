"""
Comparison script to evaluate Euler vs RK2 time discretization schemes.
"""

import numpy as np
import matplotlib.pyplot as plt
from test_expanding_circle import test_expanding_circle
from test_expanding_moving_circle import test_expanding_moving_circle

def compare_schemes():
    """
    Run both test cases with both time schemes and compare results.
    """
    print("\n" + "="*70)
    print("COMPARISON OF TIME DISCRETIZATION SCHEMES: EULER vs RK2")
    print("="*70 + "\n")
    
    # Test 1: Stationary expanding circle
    print("\n" + "+"*70)
    print("TEST 1: STATIONARY EXPANDING CIRCLE (No Flow)")
    print("+"*70 + "\n")
    
    print("\n--- Running with EULER scheme ---")
    error_euler_stat, time_euler_stat = test_expanding_circle(time_scheme='euler')
    plt.close('all')
    
    print("\n--- Running with RK2 scheme ---")
    error_rk2_stat, time_rk2_stat = test_expanding_circle(time_scheme='rk2')
    plt.close('all')
    
    # Test 2: Moving expanding circle
    print("\n" + "+"*70)
    print("TEST 2: MOVING EXPANDING CIRCLE (With Flow)")
    print("+"*70 + "\n")
    
    print("\n--- Running with EULER scheme ---")
    r_err_euler, x_err_euler, y_err_euler, time_euler_mov = test_expanding_moving_circle(time_scheme='euler')
    plt.close('all')
    
    print("\n--- Running with RK2 scheme ---")
    r_err_rk2, x_err_rk2, y_err_rk2, time_rk2_mov = test_expanding_moving_circle(time_scheme='rk2')
    plt.close('all')
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print("\nTest 1: Stationary Expanding Circle")
    print("-" * 70)
    print(f"{'Metric':<40} {'EULER':<15} {'RK2':<15}")
    print("-" * 70)
    print(f"{'Max radius error:':<40} {np.max(error_euler_stat):.6f}      {np.max(error_rk2_stat):.6f}")
    print(f"{'Mean radius error:':<40} {np.mean(error_euler_stat):.6f}      {np.mean(error_rk2_stat):.6f}")
    print(f"{'Computation time (s):':<40} {time_euler_stat:.3f}          {time_rk2_stat:.3f}")
    print(f"{'Error reduction factor (RK2 vs Euler):':<40} {np.max(error_euler_stat)/np.max(error_rk2_stat):.2f}x")
    
    print("\nTest 2: Moving Expanding Circle")
    print("-" * 70)
    print(f"{'Metric':<40} {'EULER':<15} {'RK2':<15}")
    print("-" * 70)
    print(f"{'Max radius error:':<40} {np.max(r_err_euler):.6f}      {np.max(r_err_rk2):.6f}")
    print(f"{'Mean radius error:':<40} {np.mean(r_err_euler):.6f}      {np.mean(r_err_rk2):.6f}")
    print(f"{'Max x-center error:':<40} {np.max(x_err_euler):.6f}      {np.max(x_err_rk2):.6f}")
    print(f"{'Max y-center error:':<40} {np.max(y_err_euler):.6f}      {np.max(y_err_rk2):.6f}")
    print(f"{'Computation time (s):':<40} {time_euler_mov:.3f}          {time_rk2_mov:.3f}")
    print(f"{'Error reduction factor (RK2 vs Euler):':<40} {np.max(r_err_euler)/np.max(r_err_rk2):.2f}x")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("RK2 typically provides:")
    print("  • Lower numerical errors (better accuracy)")
    print("  • ~2x longer computation time (2 RHS evaluations per step)")
    print("  • Better suited for larger time steps")
    print("="*70 + "\n")


if __name__ == "__main__":
    compare_schemes()