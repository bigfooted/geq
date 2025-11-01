"""
Diagnostic test for reinitialization on coarse mesh with flow.
Tests larger domain, coarser grid, larger time step, and non-zero velocity.
This regime should show clearer benefits of reinitialization.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import (GEquationSolver2D, compute_circle_radius,
                                        compute_circle_center, analytical_radius,
                                        analytical_center)
import time as time_module


def initial_solution_sharp(X, Y, x_center, y_center, radius):
    """Create sharp initial condition."""
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    G = np.ones_like(X)
    G[distance < radius] = -1.0
    return G


def run_single_test_with_flow(nx, ny, Lx, Ly, dt, t_final, u_x, u_y, 
                               reinit_interval, reinit_method, reinit_local, 
                               smooth_ic, time_scheme='rk2'):
    """
    Run a single test configuration with flow.
    
    Returns:
    --------
    dict with error, elapsed time, and diagnostic information
    """
    S_L = 0.2
    x_center_0 = 0.3 * Lx  # Start away from edge to allow movement
    y_center_0 = 0.5 * Ly  # Centered in y
    R0 = 0.3
    save_interval = max(1, int(0.05 / dt))  # Save roughly every 0.05 time units
    
    print(f"\n  Grid: {nx}x{ny}, Domain: {Lx}x{Ly}, dt={dt}, u=({u_x},{u_y})")
    print(f"  dx={Lx/(nx-1):.4f}, dy={Ly/(ny-1):.4f}, save_interval={save_interval}")
    
    # Create solver
    solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)
    G_initial = initial_solution_sharp(solver.X, solver.Y, x_center_0, y_center_0, R0)
    
    # Solve
    start_time = time_module.time()
    G_history, t_history = solver.solve(
        G_initial, t_final, dt,
        save_interval=save_interval,
        time_scheme=time_scheme,
        reinit_interval=reinit_interval,
        reinit_method=reinit_method,
        reinit_local=reinit_local,
        smooth_ic=smooth_ic
    )
    elapsed = time_module.time() - start_time
    
    print(f"  Completed in {elapsed:.2f}s, {len(t_history)} snapshots")
    
    # Extract radii, centers, and gradients
    numerical_radii = []
    numerical_x_centers = []
    numerical_y_centers = []
    grad_mags_interface = []
    grad_mags_global = []
    
    for G in G_history:
        # Compute center
        x_c, y_c = compute_circle_center(G, solver.X, solver.Y, solver.dx, solver.dy)
        numerical_x_centers.append(x_c)
        numerical_y_centers.append(y_c)
        
        # Compute radius
        radius = compute_circle_radius(G, solver.X, solver.Y, x_c, y_c,
                                      solver.dx, solver.dy)
        numerical_radii.append(radius)
        
        # Compute gradient magnitude at interface
        grad_mag = solver.compute_gradient_magnitude(G)
        band_mask = solver.find_interface_band(G, bandwidth=2)
        if np.any(band_mask):
            avg_grad_interface = np.mean(grad_mag[band_mask])
        else:
            avg_grad_interface = 0.0
        grad_mags_interface.append(avg_grad_interface)
        
        # Global average gradient (for comparison)
        avg_grad_global = np.mean(grad_mag)
        grad_mags_global.append(avg_grad_global)
    
    # Compute analytical solutions
    t_array = np.array(t_history)
    analytical_radii = analytical_radius(t_array, R0, S_L)
    analytical_x_centers, analytical_y_centers = analytical_center(
        t_array, x_center_0, y_center_0, u_x, u_y
    )
    
    # Compute errors
    radius_error = np.abs(np.array(numerical_radii) - analytical_radii)
    x_center_error = np.abs(np.array(numerical_x_centers) - analytical_x_centers)
    y_center_error = np.abs(np.array(numerical_y_centers) - analytical_y_centers)
    
    print(f"  Max radius error: {np.max(radius_error):.6f}")
    print(f"  Max x-center error: {np.max(x_center_error):.6f}")
    print(f"  Final |∇G|_interface: {grad_mags_interface[-1]:.3f}")
    
    return {
        'radius_error': radius_error,
        'x_center_error': x_center_error,
        'y_center_error': y_center_error,
        'radii': numerical_radii,
        'x_centers': numerical_x_centers,
        'y_centers': numerical_y_centers,
        't_history': t_history,
        'grad_mags_interface': grad_mags_interface,
        'grad_mags_global': grad_mags_global,
        'elapsed': elapsed,
        'max_radius_error': np.max(radius_error),
        'max_x_error': np.max(x_center_error),
        'mean_radius_error': np.mean(radius_error),
        'final_radius_error': radius_error[-1],
        'G_history': G_history,
        'solver': solver,
        'analytical_radii': analytical_radii,
        'analytical_x_centers': analytical_x_centers,
        'analytical_y_centers': analytical_y_centers
    }


def diagnose_reinitialization_coarse():
    """
    Comprehensive diagnostic of reinitialization on coarse mesh with flow.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC: REINITIALIZATION ON COARSE MESH WITH FLOW")
    print("="*80 + "\n")
    
    # Test configurations: coarse mesh, larger domain, flow
    test_configs = [
        # Configuration 1: Coarse mesh, moderate flow
        {
            'name': 'Coarse Mesh (51x51), Moderate Flow',
            'nx': 51, 'ny': 51, 
            'Lx': 4.0, 'Ly': 4.0,
            'dt': 0.005,  # Larger time step
            'u_x': 0.1, 'u_y': 0.0,
            't_finals': [2.0, 5.0, 10.0]
        },
        # Configuration 2: Very coarse mesh, strong flow
        {
            'name': 'Very Coarse Mesh (31x31), Strong Flow',
            'nx': 31, 'ny': 31,
            'Lx': 5.0, 'Ly': 5.0,
            'dt': 0.01,  # Even larger time step
            'u_x': 0.2, 'u_y': 0.0,
            't_finals': [2.0, 5.0, 10.0]
        },
        # Configuration 3: Medium mesh for comparison
        {
            'name': 'Medium Mesh (71x71), Moderate Flow',
            'nx': 71, 'ny': 71,
            'Lx': 4.0, 'Ly': 4.0,
            'dt': 0.003,
            'u_x': 0.1, 'u_y': 0.0,
            't_finals': [2.0, 5.0, 10.0]
        }
    ]
    
    all_results = []
    
    for config_idx, config in enumerate(test_configs):
        print("\n" + "="*80)
        print(f"CONFIGURATION {config_idx + 1}: {config['name']}")
        print("="*80)
        
        config_results = {
            'config': config,
            'time_results': []
        }
        
        for t_final in config['t_finals']:
            print(f"\n{'='*80}")
            print(f"Simulation Time: t_final = {t_final} s")
            print(f"{'='*80}")
            
            # Test 1: No reinitialization
            print("\n  [1/4] No Reinitialization (with Smooth IC)...")
            result_no_reinit = run_single_test_with_flow(
                config['nx'], config['ny'], config['Lx'], config['Ly'],
                config['dt'], t_final, config['u_x'], config['u_y'],
                0, 'fast_marching', True, True
            )
            
            # Test 2: Reinitialization every 50 steps
            print("\n  [2/4] Reinitialization every 50 steps (LOCAL)...")
            result_reinit_50 = run_single_test_with_flow(
                config['nx'], config['ny'], config['Lx'], config['Ly'],
                config['dt'], t_final, config['u_x'], config['u_y'],
                50, 'fast_marching', True, True
            )
            
            # Test 3: Reinitialization every 100 steps
            print("\n  [3/4] Reinitialization every 100 steps (LOCAL)...")
            result_reinit_100 = run_single_test_with_flow(
                config['nx'], config['ny'], config['Lx'], config['Ly'],
                config['dt'], t_final, config['u_x'], config['u_y'],
                100, 'fast_marching', True, True
            )
            
            # Test 4: GLOBAL reinitialization for comparison
            print("\n  [4/4] Reinitialization every 50 steps (GLOBAL)...")
            result_reinit_global = run_single_test_with_flow(
                config['nx'], config['ny'], config['Lx'], config['Ly'],
                config['dt'], t_final, config['u_x'], config['u_y'],
                50, 'fast_marching', False, True
            )
            
            time_result = {
                't_final': t_final,
                'no_reinit': result_no_reinit,
                'reinit_50': result_reinit_50,
                'reinit_100': result_reinit_100,
                'reinit_global': result_reinit_global
            }
            
            config_results['time_results'].append(time_result)
            
            # Print comparison
            print(f"\n  COMPARISON at t_final = {t_final}s:")
            print(f"  {'Method':<30} {'Max R Err':<12} {'Max X Err':<12} {'Final |∇G|':<12}")
            print(f"  {'-'*66}")
            print(f"  {'No Reinit':<30} {result_no_reinit['max_radius_error']:<12.6f} "
                  f"{result_no_reinit['max_x_error']:<12.6f} "
                  f"{result_no_reinit['grad_mags_interface'][-1]:<12.3f}")
            print(f"  {'Reinit every 50 (LOCAL)':<30} {result_reinit_50['max_radius_error']:<12.6f} "
                  f"{result_reinit_50['max_x_error']:<12.6f} "
                  f"{result_reinit_50['grad_mags_interface'][-1]:<12.3f}")
            print(f"  {'Reinit every 100 (LOCAL)':<30} {result_reinit_100['max_radius_error']:<12.6f} "
                  f"{result_reinit_100['max_x_error']:<12.6f} "
                  f"{result_reinit_100['grad_mags_interface'][-1]:<12.3f}")
            print(f"  {'Reinit every 50 (GLOBAL)':<30} {result_reinit_global['max_radius_error']:<12.6f} "
                  f"{result_reinit_global['max_x_error']:<12.6f} "
                  f"{result_reinit_global['grad_mags_interface'][-1]:<12.3f}")
        
        all_results.append(config_results)
    
    # Create comprehensive comparison plots
    print("\n" + "="*80)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    # Plot 1: Error vs time for each configuration
    for config_idx, config_results in enumerate(all_results):
        config = config_results['config']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"{config['name']}\nRadius Error, Center Error, and Gradient Evolution", 
                    fontsize=14, fontweight='bold')
        
        for time_idx, time_result in enumerate(config_results['time_results']):
            if time_idx >= 3:
                break
                
            t_final = time_result['t_final']
            
            # Radius error evolution
            ax = axes[0, time_idx]
            ax.plot(time_result['no_reinit']['t_history'], 
                   time_result['no_reinit']['radius_error'],
                   'b-', linewidth=2, label='No Reinit')
            ax.plot(time_result['reinit_50']['t_history'],
                   time_result['reinit_50']['radius_error'],
                   'r-', linewidth=2, label='Reinit 50 (LOCAL)')
            ax.plot(time_result['reinit_100']['t_history'],
                   time_result['reinit_100']['radius_error'],
                   'g-', linewidth=2, label='Reinit 100 (LOCAL)')
            ax.plot(time_result['reinit_global']['t_history'],
                   time_result['reinit_global']['radius_error'],
                   'm--', linewidth=2, label='Reinit 50 (GLOBAL)')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Radius Error', fontsize=10)
            ax.set_title(f't_final = {t_final}s', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Gradient magnitude evolution
            ax = axes[1, time_idx]
            ax.plot(time_result['no_reinit']['t_history'],
                   time_result['no_reinit']['grad_mags_interface'],
                   'b-', linewidth=2, label='No Reinit')
            ax.plot(time_result['reinit_50']['t_history'],
                   time_result['reinit_50']['grad_mags_interface'],
                   'r-', linewidth=2, label='Reinit 50 (LOCAL)')
            ax.plot(time_result['reinit_100']['t_history'],
                   time_result['reinit_100']['grad_mags_interface'],
                   'g-', linewidth=2, label='Reinit 100 (LOCAL)')
            ax.plot(time_result['reinit_global']['t_history'],
                   time_result['reinit_global']['grad_mags_interface'],
                   'm--', linewidth=2, label='Reinit 50 (GLOBAL)')
            ax.axhline(y=1.0, color='orange', linestyle=':', linewidth=2, label='Ideal |∇G|=1')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('|∇G| at Interface', fontsize=10)
            ax.set_title(f'Gradient at t_final = {t_final}s', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'diagnostic_coarse_config{config_idx+1}_evolution.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Saved: diagnostic_coarse_config{config_idx+1}_evolution.png")
    
    # Plot 2: Summary comparison across all configurations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    for config_idx, config_results in enumerate(all_results):
        config_name = config_results['config']['name']
        t_finals = [tr['t_final'] for tr in config_results['time_results']]
        
        # Max radius errors
        no_reinit_errors = [tr['no_reinit']['max_radius_error'] 
                           for tr in config_results['time_results']]
        reinit_50_errors = [tr['reinit_50']['max_radius_error']
                           for tr in config_results['time_results']]
        reinit_100_errors = [tr['reinit_100']['max_radius_error']
                            for tr in config_results['time_results']]
        
        # Final gradients
        no_reinit_grads = [tr['no_reinit']['grad_mags_interface'][-1]
                          for tr in config_results['time_results']]
        reinit_50_grads = [tr['reinit_50']['grad_mags_interface'][-1]
                          for tr in config_results['time_results']]
        
        linestyle = ['-', '--', '-.'][config_idx]
        
        ax1.plot(t_finals, no_reinit_errors, 'b' + linestyle, linewidth=2, 
                marker='o', markersize=6, label=f'{config_name} - No Reinit')
        ax1.plot(t_finals, reinit_50_errors, 'r' + linestyle, linewidth=2,
                marker='s', markersize=6, label=f'{config_name} - Reinit 50')
        
        ax2.plot(t_finals, no_reinit_grads, 'b' + linestyle, linewidth=2,
                marker='o', markersize=6, label=f'{config_name} - No Reinit')
        ax2.plot(t_finals, reinit_50_grads, 'r' + linestyle, linewidth=2,
                marker='s', markersize=6, label=f'{config_name} - Reinit 50')
        
        # Error ratios
        error_ratios = [reinit_50_errors[i] / no_reinit_errors[i] 
                       for i in range(len(t_finals))]
        ax3.plot(t_finals, error_ratios, linestyle, linewidth=2, marker='o',
                markersize=6, label=config_name)
    
    ax1.set_xlabel('Simulation Time (s)', fontsize=11)
    ax1.set_ylabel('Max Radius Error', fontsize=11)
    ax1.set_title('Max Error vs Time (All Configurations)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.axhline(y=1.0, color='green', linestyle=':', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Simulation Time (s)', fontsize=11)
    ax2.set_ylabel('|∇G| at Interface', fontsize=11)
    ax2.set_title('Final Gradient vs Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal Error')
    ax3.set_xlabel('Simulation Time (s)', fontsize=11)
    ax3.set_ylabel('Error Ratio (Reinit/No Reinit)', fontsize=11)
    ax3.set_title('Error Ratio: Values < 1 mean Reinit is Better', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Bar chart: final comparison at longest time
    ax4.axis('off')
    summary_text = "SUMMARY at Longest Simulation Time:\n\n"
    for config_idx, config_results in enumerate(all_results):
        longest_result = config_results['time_results'][-1]
        t_final = longest_result['t_final']
        config_name = config_results['config']['name']
        
        err_no = longest_result['no_reinit']['max_radius_error']
        err_50 = longest_result['reinit_50']['max_radius_error']
        err_100 = longest_result['reinit_100']['max_radius_error']
        ratio = err_50 / err_no
        
        summary_text += f"{config_name} (t={t_final}s):\n"
        summary_text += f"  No Reinit: {err_no:.6f}\n"
        summary_text += f"  Reinit 50: {err_50:.6f} ({ratio:.3f}x)\n"
        summary_text += f"  Reinit 100: {err_100:.6f}\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('diagnostic_coarse_summary_all_configs.png', dpi=300, bbox_inches='tight')
    print("Saved: diagnostic_coarse_summary_all_configs.png")
    
    # Print final analysis
    print("\n" + "="*80)
    print("FINAL ANALYSIS AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. ERROR BEHAVIOR:")
    for config_idx, config_results in enumerate(all_results):
        config_name = config_results['config']['name']
        print(f"\n  {config_name}:")
        for tr in config_results['time_results']:
            t = tr['t_final']
            err_no = tr['no_reinit']['max_radius_error']
            err_50 = tr['reinit_50']['max_radius_error']
            ratio = err_50 / err_no
            
            if ratio < 1.0:
                improvement = "✓ BETTER"
            elif ratio > 1.1:
                improvement = "✗ WORSE"
            else:
                improvement = "≈ SIMILAR"
            
            print(f"    t={t:4.1f}s: Ratio={ratio:.3f} {improvement}")
    
    print("\n2. GRADIENT MAINTENANCE:")
    for config_idx, config_results in enumerate(all_results):
        config_name = config_results['config']['name']
        longest = config_results['time_results'][-1]
        grad_no = longest['no_reinit']['grad_mags_interface'][-1]
        grad_50 = longest['reinit_50']['grad_mags_interface'][-1]
        
        print(f"\n  {config_name} (t={longest['t_final']}s):")
        print(f"    No Reinit: |∇G| = {grad_no:.3f} (deviation: {abs(grad_no-1.0):.3f})")
        print(f"    Reinit 50: |∇G| = {grad_50:.3f} (deviation: {abs(grad_50-1.0):.3f})")
        
        if abs(grad_50 - 1.0) < abs(grad_no - 1.0):
            print(f"    ✓ Reinitialization IMPROVES gradient")
        else:
            print(f"    ✗ Reinitialization DEGRADES gradient")
    
    print("\n3. CONCLUSIONS:")
    print("-" * 80)
    print("  If reinitialization maintains better gradients BUT has higher errors:")
    print("    → The reinitialization itself may be introducing small position errors")
    print("    → This can happen if zero level set is not perfectly preserved")
    print("    → Trade-off: Better gradient properties vs. slight position drift")
    print("\n  Recommendations:")
    print("    • For COARSE meshes: Reinitialization may help at long times (t > 5s)")
    print("    • For FINE meshes with smooth IC: Skip reinitialization for t < 3s")
    print("    • If using reinit: LOCAL is always better than GLOBAL")
    print("    • Monitor BOTH error and gradient to make informed choice")
    print("    • Consider using reinit only when |∇G| deviates > 20% from 1.0")
    print("="*80 + "\n")
    
    plt.show()
    
    return all_results


if __name__ == "__main__":
    diagnose_reinitialization_coarse()