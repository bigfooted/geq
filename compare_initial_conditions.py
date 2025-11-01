"""
Comparison script to evaluate smooth vs sharp initial conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from test_expanding_circle import test_expanding_circle
from test_expanding_circle_sharp import test_expanding_circle_sharp


def compare_initial_conditions(time_scheme='euler'):
    """
    Run tests with both smooth and sharp initial conditions and compare results.
    
    Parameters:
    -----------
    time_scheme : str
        Time discretization scheme: 'euler' or 'rk2'
    """
    print("\n" + "="*70)
    print(f"COMPARISON OF INITIAL CONDITIONS (Time Scheme: {time_scheme.upper()})")
    print("="*70 + "\n")
    
    # Test 1: Smooth initial condition (signed distance)
    print("\n" + "+"*70)
    print("TEST 1: SMOOTH INITIAL CONDITION (Signed Distance)")
    print("+"*70 + "\n")
    
    error_smooth, time_smooth = test_expanding_circle(time_scheme=time_scheme)
    plt.close('all')
    
    # Test 2: Sharp initial condition (discontinuous)
    print("\n" + "+"*70)
    print("TEST 2: SHARP INITIAL CONDITION (G=-1 inside, G=+1 outside)")
    print("+"*70 + "\n")
    
    error_sharp, time_sharp = test_expanding_circle_sharp(time_scheme=time_scheme)
    plt.close('all')
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print(f"\nTime Scheme: {time_scheme.upper()}")
    print("-" * 70)
    print(f"{'Metric':<45} {'SMOOTH IC':<15} {'SHARP IC':<15}")
    print("-" * 70)
    print(f"{'Max radius error:':<45} {np.max(error_smooth):.6f}      {np.max(error_sharp):.6f}")
    print(f"{'Mean radius error:':<45} {np.mean(error_smooth):.6f}      {np.mean(error_sharp):.6f}")
    print(f"{'Final radius error:':<45} {error_smooth[-1]:.6f}      {error_sharp[-1]:.6f}")
    print(f"{'Computation time (s):':<45} {time_smooth:.3f}          {time_sharp:.3f}")
    
    if np.max(error_sharp) > 0:
        print(f"{'Error ratio (Sharp/Smooth):':<45} {np.max(error_sharp)/np.max(error_smooth):.2f}x")
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error comparison
    time_points = np.linspace(0, 1.5, len(error_smooth))
    ax1.semilogy(time_points, error_smooth, 'b-', linewidth=2, label='Smooth IC')
    ax1.semilogy(time_points, error_sharp, 'r--', linewidth=2, label='Sharp IC')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Absolute Error (log scale)', fontsize=11)
    ax1.set_title('Radius Error Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Cumulative error
    cumulative_smooth = np.cumsum(error_smooth)
    cumulative_sharp = np.cumsum(error_sharp)
    ax2.plot(time_points, cumulative_smooth, 'b-', linewidth=2, label='Smooth IC')
    ax2.plot(time_points, cumulative_sharp, 'r--', linewidth=2, label='Sharp IC')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Cumulative Error', fontsize=11)
    ax2.set_title('Cumulative Error Accumulation', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error statistics
    categories = ['Max Error', 'Mean Error', 'Final Error']
    smooth_stats = [np.max(error_smooth), np.mean(error_smooth), error_smooth[-1]]
    sharp_stats = [np.max(error_sharp), np.mean(error_sharp), error_sharp[-1]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax3.bar(x - width/2, smooth_stats, width, label='Smooth IC', color='blue', alpha=0.7)
    ax3.bar(x + width/2, sharp_stats, width, label='Sharp IC', color='red', alpha=0.7)
    ax3.set_ylabel('Error', fontsize=11)
    ax3.set_title('Error Statistics Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Computation time
    ax4.bar(['Smooth IC', 'Sharp IC'], [time_smooth, time_sharp], 
           color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Time (s)', fontsize=11)
    ax4.set_title('Computation Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'initial_condition_comparison_{time_scheme}.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: initial_condition_comparison_{time_scheme}.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print("Smooth Initial Condition (Signed Distance):")
    print("  • Better representation of continuous level set function")
    print("  • Lower numerical errors")
    print("  • Consistent with level set theory")
    print("\nSharp Initial Condition (Discontinuous):")
    print("  • Creates numerical challenges (discontinuity)")
    print("  • Naturally smooths out due to numerical diffusion")
    print("  • May have larger errors, especially initially")
    print("  • Useful for testing robustness of the scheme")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check if time scheme is provided as command line argument
    if len(sys.argv) > 1:
        scheme = sys.argv[1].lower()
        if scheme not in ['euler', 'rk2']:
            print("Usage: python compare_initial_conditions.py [euler|rk2]")
            print("Defaulting to 'euler'")
            scheme = 'euler'
    else:
        scheme = 'euler'
    
    compare_initial_conditions(time_scheme=scheme)