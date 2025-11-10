import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from FTF_exact import FTF_CCO


def test_plot_ftf_cco():
    """
    Plot analytical FTF_CCO gain and phase over frequency f = 0.1..100 Hz (log x-axis)
    for alpha = 25, 50, 75, 88 degrees. Saves PNG and shows figure.
    """
    # Unified omega axis (rad/s) 0.1 .. 100 for all alpha values
    alpha_deg_list = [0.1, 25, 50, 75, 88, 89.9]
    omega = np.logspace(np.log10(0.1), np.log10(100.0), 400)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Gain subplot
    for a_deg in alpha_deg_list:
        phase_rad, gain = FTF_CCO(omega, a_deg, phase_units='rad', radians=False)
        phase_rad = np.unwrap(phase_rad)
        ax1.semilogx(omega, gain, lw=2, label=f'|H|, α={a_deg}°')
    ax1.set_ylabel('Gain |H|')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(loc='best', ncols=2)

    # Phase subplot
    for a_deg in alpha_deg_list:
        phase_rad, gain = FTF_CCO(omega, a_deg, phase_units='rad', radians=False)
        phase_rad = np.unwrap(phase_rad)
        ax2.semilogx(omega, phase_rad, lw=2, label=f'∠H, α={a_deg}°')
    ax2.set_xlabel('ω (rad/s)')
    ax2.set_ylabel('Phase (rad, unwrapped)')
    ax2.set_ylim(0.0, 8.0)
    ax2.axhline(np.pi, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='π')
    ax2.axhline(2.0 * np.pi, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='2π')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='best', ncols=2)

    fig.suptitle('Analytical FTF_CCO (α = 0.1°, 25°, 50°, 75°, 88°, 89.9°)')
    fig.tight_layout()
    fig.savefig('ftf_cco_bode.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Nyquist plot for CCO (multiple α)
    fig_nyq, axn = plt.subplots(1, 1, figsize=(6.5, 6.5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_deg_list)))
    for a_deg, c in zip(alpha_deg_list, colors):
        _, _, H = FTF_CCO(omega, a_deg, phase_units='rad', radians=False, return_complex=True)
        axn.plot(H.real, H.imag, color=c, lw=1.8, label=f'α={a_deg}°')
    axn.axhline(0, color='k', linewidth=0.8)
    axn.axvline(0, color='k', linewidth=0.8)
    # Unit circle and critical point (-1, 0)
    th = np.linspace(0, 2*np.pi, 512)
    axn.plot(np.cos(th), np.sin(th), linestyle=':', color='gray', linewidth=1.0, alpha=0.7, label='Unit circle')
    axn.plot(-1.0, 0.0, 'ro', markersize=6)
    axn.set_aspect('equal', adjustable='box')
    axn.set_xlabel('Re(H)')
    axn.set_ylabel('Im(H)')
    axn.grid(True, alpha=0.3)
    axn.legend(loc='best', ncols=2)
    fig_nyq.suptitle('Nyquist: FTF_CCO (multiple α)')
    fig_nyq.tight_layout()
    fig_nyq.savefig('ftf_cco_nyquist.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    test_plot_ftf_cco()
