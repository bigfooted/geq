import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from FTF_exact import FTF_UCO, FTF_CCO

# simulated data
omega_sim=[1,2,3,4, 5, 10, 20, 50, 100]
gain_sim = [0.9416,0.851,0.7115, 0.5408,0.358, 0.186, 0.0662, 0.0252, 0.01223]
phase_sim=[0.5286,1.055, 1.5781, 2.093, 2.59, 2.153, -2.7666+2*np.pi, -1.0847+2*np.pi, -2.3596+4*np.pi]

gain_sim_weno5 = [0.9416, 0.851, 0.7115, 0.5536, 0.358, 0.186, 0.0662, 0.0252, 0.01223]
phase_sim_weno5= [0.5286, 1.055, 1.5781, 2.0914, 2.59, 2.153, -2.7666+2*np.pi, -1.0847+2*np.pi, -2.3596+4*np.pi]


def test_plot_ftf_uco_vs_cco():
    """
    Overlay FTF_UCO and FTF_CCO (α = 25°, 50°, 75°, 88°) on Bode-style plots
    for f = 0.1..100 Hz. Saves PNG and shows figure.
    """
    # Angular frequency axis ω in rad/s (0.1 .. 100)
    omega = np.logspace(np.log10(0.1), np.log10(100.0), 400)
    alpha_deg_list = [0.1, 25, 50, 75, 88, 89.9]

    phase_uco_rad, gain_uco = FTF_UCO(omega, phase_units='rad')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot UCO baseline
    ax1.semilogx(omega, gain_uco, 'k-', lw=2, label='|H| UCO')
    ax2.semilogx(omega, np.unwrap(phase_uco_rad), 'k-', lw=2, label='∠H UCO (rad, unwrapped)')
    # G-equation simulated data
    ax1.semilogx(omega_sim, gain_sim, 'o', label='|H| Simulated')
    ax2.semilogx(omega_sim, phase_sim, 'o', label='∠H| Simulated rad')
    ax1.semilogx(omega_sim, gain_sim_weno5, 'o', markerfacecolor='red', label='|H| Simulated WENO5')
    ax2.semilogx(omega_sim, phase_sim_weno5, 'o', markerfacecolor='red', label='∠H| Simulated WENO5 rad')

    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_deg_list)))
    for (a_deg, c) in zip(alpha_deg_list, colors):
        phase_rad, gain = FTF_CCO(omega, a_deg, phase_units='rad', radians=False)
        phase_rad = np.unwrap(phase_rad)
        ax1.semilogx(omega, gain, color=c, lw=1.8, label=f'|H| CCO α={a_deg}°')
        ax2.semilogx(omega, phase_rad, color=c, lw=1.8, linestyle='--', label=f'∠H CCO α={a_deg}°')


    ax1.set_ylabel('Gain |H|')
    ax1.grid(True, which='both', alpha=0.3)
    ax2.set_xlabel('ω (rad/s)')
    ax2.set_ylabel('Phase (rad, unwrapped)')
    ax2.set_ylim(0.0, 8.0)
    ax2.axhline(np.pi, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='π')
    ax2.axhline(2.0 * np.pi, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='2π')
    ax2.grid(True, which='both', alpha=0.3)

    ax1.legend(loc='best', ncols=2, fontsize=9)
    ax2.legend(loc='best', ncols=2, fontsize=9)

    fig.suptitle('Analytical FTF Comparison: UCO vs CCO (α=0.1°,25°,50°,75°,88°,89.9°)')
    fig.tight_layout()
    fig.savefig('ftf_uco_vs_cco_bode.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Combined Nyquist: UCO + selected CCO traces
    fig_nyq, axn = plt.subplots(1, 1, figsize=(6.5, 6.5))
    # UCO curve
    _, _, H_uco = FTF_UCO(omega, phase_units='rad', return_complex=True)
    axn.plot(H_uco.real, H_uco.imag, 'k-', lw=2.2, label='UCO')
    # CCO curves
    colors = plt.cm.plasma(np.linspace(0, 1, len(alpha_deg_list)))
    for a_deg, c in zip(alpha_deg_list, colors):
        _, _, H_cco = FTF_CCO(omega, a_deg, phase_units='rad', radians=False, return_complex=True)
        axn.plot(H_cco.real, H_cco.imag, color=c, lw=1.4, label=f'CCO α={a_deg}°')
    axn.axhline(0, color='k', linewidth=0.8)
    axn.axvline(0, color='k', linewidth=0.8)
    # Unit circle and critical point (-1,0)
    th = np.linspace(0, 2*np.pi, 512)
    axn.plot(np.cos(th), np.sin(th), linestyle=':', color='gray', linewidth=1.0, alpha=0.7, label='Unit circle')
    axn.plot(-1.0, 0.0, 'ro', markersize=6)
    axn.set_aspect('equal', adjustable='box')
    axn.set_xlabel('Re(H)')
    axn.set_ylabel('Im(H)')
    axn.grid(True, alpha=0.3)
    axn.legend(loc='best', ncols=2, fontsize=8)
    fig_nyq.suptitle('Nyquist: UCO vs CCO')
    fig_nyq.tight_layout()
    fig_nyq.savefig('ftf_uco_vs_cco_nyquist.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    test_plot_ftf_uco_vs_cco()
