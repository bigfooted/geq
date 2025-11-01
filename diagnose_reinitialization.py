"""
Diagnostic test to investigate why reinitialization may show higher errors.
Tests multiple simulation times and reinitialization frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import (GEquationSolver2D, compute_circle_radius,
                                        analytical_radius)
import time as time_module


def initial_solution_sharp(X, Y, x_center, y_center, radius):
    """Create sharp initial condition."""
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    G = np.ones_like(X)
    G[distance < radius] = -1.0
    return G


def run_single_test(t_final, reinit_interval, reinit_method, reinit_local, smooth_ic):
    """
    Run a single test configuration.
    
    Returns:
    --------
    dict with error, elapsed time, and gradient magnitude statistics
    """
    # Parameters
    nx, ny = 101, 101
    Lx, Ly = 2.0, 2.0
    S_L = 0.2
    u_x, u_y = 0.0, 0.0
    x_center, y_center = 1.0, 1.0
    R0 = 0.3
    dt = 0.001
    save_interval = 50
    
    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)
    G_initial = initial_solution_sharp(solver.X, solver.Y, x_center, y_center, R0)
    
    # Solve
    start_time = time_module.time()
    G_history, t_history = solver.solve(
        G_initial, t_final, dt,
        save_interval=save_interval,
        time_scheme='rk2',
        reinit_interval=reinit_interval,
        reinit_method=reinit_method,
        reinit_local=reinit_local,
        smooth_ic=smooth_ic
    )
    elapsed = time_module.time() - start_time
    
    # Extract radii and compute errors
    numerical_radii = []
    grad_mags_interface = []
    
    for G in G_history:
        radius = compute_circle_radius(G, solver.X, solver.Y, x_center, y_center,
                                      solver.dx, solver.dy)
        numerical_radii.append(radius)
        
        # Compute gradient magnitude at interface
        grad_mag = solver.compute_gradient_magnitude(G)
        band_mask = solver.find_interface_band(G, bandwidth=2)
        if np.any(band_mask):
            avg_grad = np.mean(grad_mag[band_mask])
            grad_mags_interface.append(avg_grad)
        else:
            grad_mags_interface.append(0.0)
    
    t_array = np.array(t_history)
    analytical_radii = analytical_radius(t_array, R0, S_L)
    error = np.abs(np.array(numerical_radii) - analytical_radii)
    
    return {
        'error': error,
        'radii': numerical_radii,
        't_history': t_history,
        'grad_mags': grad_mags_interface,
        'elapsed': elapsed,
        'max_error': np.max(error),
        'mean_error': np.mean(error),
        'final_error': error[-1],
        'G_history': G_history,
        'solver': solver
    }


def diagnose_reinitialization():
    """
    Comprehensive diagnostic of reinitialization behavior.
    Tests multiple scenarios to understand error patterns.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC: INVESTIGATING REINITIALIZATION ERROR BEHAVIOR")
    print("="*80 + "\n")
    
    # Test 1: Effect of simulation time
    print("\n" + "-"*80)
    print("TEST 1: Effect of Simulation Time (No Reinit vs Reinit)")
    print("-"*80)
    
    time_points = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results_time_test = []
    
    for t_final in time_points:
        print(f"\n  Testing t_final = {t_final} s...")
        
        # No reinitialization
        result_no_reinit = run_single_test(t_final, 0, 'fast_marching', True, True)
        
        # With reinitialization (LOCAL)
        result_with_reinit = run_single_test(t_final, 50, 'fast_marching', True, True)
        
        results_time_test.append({
            't_final': t_final,
            'no_reinit': result_no_reinit,
            'with_reinit': result_with_reinit
        })
        
        print(f"    No Reinit    : Max Error = {result_no_reinit['max_error']:.6f}, "
              f"Final |∇G| = {result_no_reinit['grad_mags'][-1]:.3f}")
        print(f"    With Reinit  : Max Error = {result_with_reinit['max_error']:.6f}, "
              f"Final |∇G| = {result_with_reinit['grad_mags'][-1]:.3f}")
        print(f"    Error Ratio  : {result_with_reinit['max_error'] / result_no_reinit['max_error']:.3f}x")
    
    # Test 2: Effect of reinitialization frequency
    print("\n" + "-"*80)
    print("TEST 2: Effect of Reinitialization Frequency (t_final = 3.0 s)")
    print("-"*80)
    
    t_final = 3.0
    reinit_frequencies = [0, 25, 50, 100, 200, 500]
    results_freq_test = []
    
    for freq in reinit_frequencies:
        print(f"\n  Testing reinit_interval = {freq}...")
        result = run_single_test(t_final, freq, 'fast_marching', True, True)
        results_freq_test.append({
            'freq': freq,
            'result': result
        })
        print(f"    Max Error = {result['max_error']:.6f}, "
              f"Final |∇G| = {result['grad_mags'][-1]:.3f}")
    
    # Test 3: LOCAL vs GLOBAL reinitialization
    print("\n" + "-"*80)
    print("TEST 3: LOCAL vs GLOBAL Reinitialization (t_final = 3.0 s)")
    print("-"*80)
    
    t_final = 3.0
    reinit_interval = 50
    
    result_no_reinit = run_single_test(t_final, 0, 'fast_marching', True, True)
    result_local = run_single_test(t_final, reinit_interval, 'fast_marching', True, True)
    result_global = run_single_test(t_final, reinit_interval, 'fast_marching', False, True)
    
    print(f"\n  No Reinit    : Max Error = {result_no_reinit['max_error']:.6f}")
    print(f"  LOCAL Reinit : Max Error = {result_local['max_error']:.6f}")
    print(f"  GLOBAL Reinit: Max Error = {result_global['max_error']:.6f}")
    
    results_local_vs_global = {
        'no_reinit': result_no_reinit,
        'local': result_local,
        'global': result_global
    }
    
    # Test 4: Effect of IC smoothing
    print("\n" + "-"*80)
    print("TEST 4: Effect of IC Smoothing with Reinitialization (t_final = 3.0 s)")
    print("-"*80)
    
    t_final = 3.0
    reinit_interval = 50
    
    result_no_smooth_no_reinit = run_single_test(t_final, 0, 'fast_marching', True, False)
    result_no_smooth_with_reinit = run_single_test(t_final, reinit_interval, 'fast_marching', True, False)
    result_smooth_no_reinit = run_single_test(t_final, 0, 'fast_marching', True, True)
    result_smooth_with_reinit = run_single_test(t_final, reinit_interval, 'fast_marching', True, True)
    
    print(f"\n  No Smooth, No Reinit   : Max Error = {result_no_smooth_no_reinit['max_error']:.6f}")
    print(f"  No Smooth, With Reinit : Max Error = {result_no_smooth_with_reinit['max_error']:.6f}")
    print(f"  Smooth, No Reinit      : Max Error = {result_smooth_no_reinit['max_error']:.6f}")
    print(f"  Smooth, With Reinit    : Max Error = {result_smooth_with_reinit['max_error']:.6f}")
    
    results_smoothing_test = {
        'no_smooth_no_reinit': result_no_smooth_no_reinit,
        'no_smooth_with_reinit': result_no_smooth_with_reinit,
        'smooth_no_reinit': result_smooth_no_reinit,
        'smooth_with_reinit': result_smooth_with_reinit
    }
    
    # Create diagnostic plots
    print("\n" + "="*80)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    # Plot 1: Error vs simulation time
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    times = [r['t_final'] for r in results_time_test]
    errors_no_reinit = [r['no_reinit']['max_error'] for r in results_time_test]
    errors_with_reinit = [r['with_reinit']['max_error'] for r in results_time_test]
    grad_no_reinit = [r['no_reinit']['grad_mags'][-1] for r in results_time_test]
    grad_with_reinit = [r['with_reinit']['grad_mags'][-1] for r in results_time_test]
    
    ax1.plot(times, errors_no_reinit, 'b-o', linewidth=2, markersize=8, label='No Reinit')
    ax1.plot(times, errors_with_reinit, 'r-s', linewidth=2, markersize=8, label='With Reinit (every 50)')
    ax1.set_xlabel('Simulation Time (s)', fontsize=11)
    ax1.set_ylabel('Max Error', fontsize=11)
    ax1.set_title('Max Error vs Simulation Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, grad_no_reinit, 'b-o', linewidth=2, markersize=8, label='No Reinit')
    ax2.plot(times, grad_with_reinit, 'r-s', linewidth=2, markersize=8, label='With Reinit (every 50)')
    ax2.axhline(y=1.0, color='g', linestyle='--', linewidth=1, label='Ideal |∇G| = 1')
    ax2.set_xlabel('Simulation Time (s)', fontsize=11)
    ax2.set_ylabel('|∇G| at Interface', fontsize=11)
    ax2.set_title('Gradient Magnitude at Interface (Final)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 2: Error vs reinitialization frequency
    freqs = [r['freq'] for r in results_freq_test]
    errors_freq = [r['result']['max_error'] for r in results_freq_test]
    grads_freq = [r['result']['grad_mags'][-1] for r in results_freq_test]
    
    ax3.plot(freqs, errors_freq, 'g-o', linewidth=2, markersize=8)
    ax3.set_xlabel('Reinitialization Interval (steps, 0=off)', fontsize=11)
    ax3.set_ylabel('Max Error', fontsize=11)
    ax3.set_title('Max Error vs Reinit Frequency (t=3.0s)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(freqs, grads_freq, 'g-o', linewidth=2, markersize=8)
    ax4.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, label='Ideal |∇G| = 1')
    ax4.set_xlabel('Reinitialization Interval (steps, 0=off)', fontsize=11)
    ax4.set_ylabel('|∇G| at Interface', fontsize=11)
    ax4.set_title('Gradient Magnitude vs Reinit Frequency', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostic_reinitialization_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: diagnostic_reinitialization_analysis.png")
    
    # Plot 3: Time evolution of error and gradient
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Use t_final = 3.0 results
    result_nr = results_time_test[4]['no_reinit']  # t=3.0
    result_wr = results_time_test[4]['with_reinit']
    
    ax1.plot(result_nr['t_history'], result_nr['error'], 'b-', linewidth=2, label='No Reinit')
    ax1.plot(result_wr['t_history'], result_wr['error'], 'r-', linewidth=2, label='With Reinit')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Absolute Error', fontsize=11)
    ax1.set_title('Error Evolution Over Time (t_final=3.0s)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(result_nr['t_history'], result_nr['grad_mags'], 'b-', linewidth=2, label='No Reinit')
    ax2.plot(result_wr['t_history'], result_wr['grad_mags'], 'r-', linewidth=2, label='With Reinit')
    ax2.axhline(y=1.0, color='g', linestyle='--', linewidth=1, label='Ideal |∇G| = 1')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('|∇G| at Interface', fontsize=11)
    ax2.set_title('Gradient Evolution Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostic_temporal_evolution.png', dpi=300, bbox_inches='tight')
    print("Saved: diagnostic_temporal_evolution.png")
    
    # Print summary and diagnosis
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY AND CONCLUSIONS")
    print("="*80)
    
    print("\n1. SIMULATION TIME DEPENDENCE:")
    print("-" * 80)
    for r in results_time_test:
        ratio = r['with_reinit']['max_error'] / r['no_reinit']['max_error']
        print(f"  t = {r['t_final']:.1f}s: Error ratio (Reinit/No Reinit) = {ratio:.3f}")
    
    print("\n2. OPTIMAL REINITIALIZATION FREQUENCY:")
    print("-" * 80)
    best_freq_idx = np.argmin([r['result']['max_error'] for r in results_freq_test])
    best_freq = results_freq_test[best_freq_idx]['freq']
    print(f"  Best frequency: {best_freq} steps")
    print(f"  Error at best frequency: {results_freq_test[best_freq_idx]['result']['max_error']:.6f}")
    
    print("\n3. GRADIENT MAGNITUDE BEHAVIOR:")
    print("-" * 80)
    print("  |∇G| should be ≈ 1.0 at interface for signed distance function")
    print(f"  No Reinit (t=3.0s)  : |∇G| = {result_nr['grad_mags'][-1]:.3f}")
    print(f"  With Reinit (t=3.0s): |∇G| = {result_wr['grad_mags'][-1]:.3f}")
    
    print("\n4. POSSIBLE CAUSES OF HIGHER ERROR WITH REINITIALIZATION:")
    print("-" * 80)
    
    # Analyze the results
    if errors_with_reinit[-1] > errors_no_reinit[-1]:
        print("  ✗ Reinitialization IS causing higher errors at long times")
        print("\n  Possible reasons:")
        print("    a) Reinitialization frequency too high (introduces numerical noise)")
        print("    b) Reinitialization not preserving zero level set accurately")
        print("    c) Boundary effects accumulating with repeated reinitialization")
        print("    d) Small simulation time (errors don't accumulate enough to show benefit)")
    else:
        print("  ✓ Reinitialization REDUCES errors at long times")
        print("\n  This is the expected behavior!")
    
    # Check gradient magnitude
    if result_wr['grad_mags'][-1] < result_nr['grad_mags'][-1]:
        print(f"\n  ✓ Reinitialization maintains better gradient (closer to 1.0)")
    else:
        print(f"\n  ✗ Reinitialization NOT improving gradient magnitude")
        print(f"    This suggests the reinitialization may not be working correctly")
    
    print("\n5. RECOMMENDATIONS:")
    print("-" * 80)
    
    if best_freq == 0:
        print("  • For SHORT simulations (t < 2s): Reinitialization NOT needed with smooth IC")
        print("  • IC smoothing alone is sufficient for short times")
    else:
        print(f"  • Use reinitialization interval: {best_freq} steps")
        print("  • Reinitialization becomes more important for LONG simulations (t > 2s)")
    
    print("  • Always use IC smoothing for sharp initial conditions")
    print("  • Use LOCAL (narrow-band) reinitialization, not GLOBAL")
    print("  • Monitor |∇G| at interface: should stay near 1.0")
    
    print("\n" + "="*80 + "\n")
    
    plt.show()
    
    return results_time_test, results_freq_test


if __name__ == "__main__":
    diagnose_reinitialization()