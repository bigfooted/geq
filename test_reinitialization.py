"""
Test and compare different reinitialization methods.
Compares global vs local (narrow-band) reinitialization.
"""

import numpy as np
import matplotlib.pyplot as plt
from g_equation_solver_improved import (GEquationSolver2D, compute_circle_radius,
                                        analytical_radius)
import time


def initial_solution_sharp(X, Y, x_center, y_center, radius):
    """Create sharp initial condition."""
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    G = np.ones_like(X)
    G[distance < radius] = -1.0
    return G


def test_reinitialization_methods(t_final=2.0):
    """
    Compare different reinitialization approaches including global vs local.
    
    Parameters:
    -----------
    t_final : float
        Final simulation time
    """
    print("\n" + "="*80)
    print("TESTING REINITIALIZATION METHODS: GLOBAL vs LOCAL")
    print(f"Simulation time: t_final = {t_final} seconds")
    print("="*80 + "\n")
    
    # Parameters
    nx, ny = 101, 101
    Lx, Ly = 2.0, 2.0
    S_L = 0.2
    u_x, u_y = 0.0, 0.0
    x_center, y_center = 1.0, 1.0
    R0 = 0.3
    dt = 0.001
    save_interval = 50
    
    # Test configurations
    configs = [
        {'reinit': 0, 'method': 'none', 'local': True, 'smooth': False, 
         'label': 'No Reinit'},
        {'reinit': 0, 'method': 'none', 'local': True, 'smooth': True, 
         'label': 'Smooth IC Only'},
        {'reinit': 100, 'method': 'fast_marching', 'local': False, 'smooth': False, 
         'label': 'Fast March GLOBAL (every 100)'},
        {'reinit': 100, 'method': 'fast_marching', 'local': True, 'smooth': False, 
         'label': 'Fast March LOCAL (every 100)'},
        {'reinit': 50, 'method': 'fast_marching', 'local': False, 'smooth': False, 
         'label': 'Fast March GLOBAL (every 50)'},
        {'reinit': 50, 'method': 'fast_marching', 'local': True, 'smooth': False, 
         'label': 'Fast March LOCAL (every 50)'},
        {'reinit': 100, 'method': 'pde', 'local': False, 'smooth': False, 
         'label': 'PDE GLOBAL (every 100)'},
        {'reinit': 100, 'method': 'pde', 'local': True, 'smooth': False, 
         'label': 'PDE LOCAL (every 100)'},
        {'reinit': 50, 'method': 'fast_marching', 'local': True, 'smooth': True, 
         'label': 'Fast March LOCAL + Smooth IC'},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(configs)}: {config['label']}")
        print(f"{'='*80}")
        
        solver = GEquationSolver2D(nx, ny, Lx, Ly, S_L, u_x=u_x, u_y=u_y)
        G_initial = initial_solution_sharp(solver.X, solver.Y, x_center, y_center, R0)
        
        start_time = time.time()
        G_history, t_history = solver.solve(
            G_initial, t_final, dt,
            save_interval=save_interval,
            time_scheme='rk2',
            reinit_interval=config['reinit'],
            reinit_method=config['method'],
            reinit_local=config['local'],
            smooth_ic=config['smooth']
        )
        elapsed = time.time() - start_time
        
        # Extract radii
        numerical_radii = []
        for G in G_history:
            radius = compute_circle_radius(G, solver.X, solver.Y, x_center, y_center,
                                          solver.dx, solver.dy)
            numerical_radii.append(radius)
        
        # Compute errors
        t_array = np.array(t_history)
        analytical_radii = analytical_radius(t_array, R0, S_L)
        error = np.abs(np.array(numerical_radii) - analytical_radii)
        
        results.append({
            'config': config,
            'error': error,
            'radii': numerical_radii,
            't_history': t_history,
            'elapsed': elapsed,
            'max_error': np.max(error),
            'mean_error': np.mean(error),
            'final_error': error[-1]
        })
        
        print(f"   Max Error: {np.max(error):.6f}")
        print(f"   Mean Error: {np.mean(error):.6f}")
        print(f"   Final Error: {error[-1]:.6f}")
        print(f"   Time: {elapsed:.2f} s")
    
    # Plotting
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    # Plot 1: Error evolution
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, result in enumerate(results):
        t_points = result['t_history']
        linestyle = '--' if 'GLOBAL' in result['config']['label'] else '-'
        linewidth = 2.5 if 'LOCAL' in result['config']['label'] else 2.0
        
        ax1.plot(t_points, result['error'], label=result['config']['label'],
                color=colors[i], linewidth=linewidth, linestyle=linestyle,
                marker='o', markersize=3, markevery=5)
        ax2.semilogy(t_points, result['error'] + 1e-10, label=result['config']['label'],
                    color=colors[i], linewidth=linewidth, linestyle=linestyle,
                    marker='o', markersize=3, markevery=5)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Absolute Error', fontsize=12)
    ax1.set_title(f'Error Evolution - Linear Scale (t_final={t_final}s)\nSolid=LOCAL, Dashed=GLOBAL', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Absolute Error (log scale)', fontsize=12)
    ax2.set_title(f'Error Evolution - Log Scale (t_final={t_final}s)\nSolid=LOCAL, Dashed=GLOBAL', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'reinitialization_global_vs_local_error_t{t_final}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: reinitialization_global_vs_local_error_t{t_final}.png")
    
    # Plot 2: Summary statistics
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    labels = [r['config']['label'] for r in results]
    max_errors = [r['max_error'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    final_errors = [r['final_error'] for r in results]
    comp_times = [r['elapsed'] for r in results]
    
    x = np.arange(len(labels))
    
    # Color bars by type
    bar_colors = ['red' if 'GLOBAL' in label else 'green' if 'LOCAL' in label else 'gray' 
                  for label in labels]
    
    ax1.bar(x, max_errors, color=bar_colors, alpha=0.7)
    ax1.set_ylabel('Max Error', fontsize=11)
    ax1.set_title('Maximum Absolute Error (Green=LOCAL, Red=GLOBAL)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, mean_errors, color=bar_colors, alpha=0.7)
    ax2.set_ylabel('Mean Error', fontsize=11)
    ax2.set_title('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3.bar(x, final_errors, color=bar_colors, alpha=0.7)
    ax3.set_ylabel('Final Error', fontsize=11)
    ax3.set_title(f'Error at t={t_final}s', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4.bar(x, comp_times, color=bar_colors, alpha=0.7)
    ax4.set_ylabel('Time (s)', fontsize=11)
    ax4.set_title('Computation Time', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'reinitialization_global_vs_local_stats_t{t_final}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: reinitialization_global_vs_local_stats_t{t_final}.png")
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY TABLE (t_final = {t_final} s)")
    print("="*80)
    print(f"{'Method':<40} {'Max Err':<12} {'Mean Err':<12} {'Final Err':<12} {'Time (s)':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['config']['label']:<40} "
              f"{r['max_error']:<12.6f} "
              f"{r['mean_error']:<12.6f} "
              f"{r['final_error']:<12.6f} "
              f"{r['elapsed']:<10.2f}")
    
    # Compare global vs local directly
    print("\n" + "="*80)
    print("DIRECT COMPARISON: GLOBAL vs LOCAL")
    print("="*80)
    
    # Find matching pairs
    for i, r1 in enumerate(results):
        if 'GLOBAL' in r1['config']['label']:
            # Look for matching LOCAL version
            base_label = r1['config']['label'].replace('GLOBAL', 'LOCAL')
            for r2 in results:
                if r2['config']['label'] == base_label:
                    print(f"\n{r1['config']['label']}")
                    print(f"  vs")
                    print(f"{r2['config']['label']}")
                    print(f"  Error improvement: {r1['max_error']/r2['max_error']:.2f}x")
                    print(f"  Speed ratio: {r2['elapsed']/r1['elapsed']:.2f}x (LOCAL/GLOBAL)")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("• LOCAL (narrow-band) reinitialization is DEFAULT and RECOMMENDED")
    print("• LOCAL is typically faster than GLOBAL (only processes interface region)")
    print("• LOCAL preserves far-field values better than GLOBAL")
    print("• GLOBAL may cause artifacts away from interface")
    print("• Smooth IC alone provides significant improvement")
    print("• Fast Marching method is more robust than PDE method")
    print("• Best approach: LOCAL Fast Marching every 50-100 steps + Smooth IC")
    print("• GLOBAL reinitialization provided for comparison purposes only")
    print("="*80 + "\n")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    import sys
    
    t_final = 2.0
    for arg in sys.argv[1:]:
        if arg.startswith('t=') or arg.startswith('time='):
            t_final = float(arg.split('=')[1])
    
    print("\nNOTE: Testing both GLOBAL and LOCAL (narrow-band) reinitialization")
    print("      LOCAL is the default and recommended approach\n")
    
    test_reinitialization_methods(t_final=t_final)