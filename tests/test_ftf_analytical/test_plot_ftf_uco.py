import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from FTF_exact import FTF_UCO


def test_plot_ftf_uco():
    """
    Plot analytical FTF_UCO gain and phase over frequency f = 0..100 Hz with log x-axis.
    Saves figure to 'ftf_uco_bode.png' and shows it.
    """
    # Angular frequency ω in rad/s: 0.1 .. 100 on a log grid
    omega = np.logspace(np.log10(0.1), np.log10(100.0), 400)

    phase_rad, gain = FTF_UCO(omega, phase_units='rad')
    # Unwrap phase to avoid jumps at ±π
    phase_rad = np.unwrap(phase_rad)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Gain
    ax1.semilogx(omega, gain, 'b-', lw=2, label='|H(ω)|')
    ax1.set_ylabel('Gain |H|')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(loc='best')

    # Phase
    ax2.semilogx(omega, phase_rad, 'r--', lw=2, label='∠H (rad, unwrapped)')
    ax2.set_xlabel('ω (rad/s)')
    ax2.set_ylabel('Phase (rad, unwrapped)')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='best')

    fig.suptitle('Analytical FTF_UCO (uniformly perturbed conical flame)\nSchuller–Durox–Candel, Combust. Flame 134 (2003) 21–34')
    fig.tight_layout()
    fig.savefig('ftf_uco_bode.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Nyquist plot (Re(H) vs Im(H))
    phase_rad2, gain2, H = FTF_UCO(omega, phase_units='rad', return_complex=True)
    fig_nyq, axn = plt.subplots(1, 1, figsize=(6, 6))
    axn.plot(H.real, H.imag, 'b-', lw=2, label='UCO')
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
    axn.legend(loc='best')
    fig_nyq.suptitle('Nyquist: FTF_UCO')
    fig_nyq.tight_layout()
    fig_nyq.savefig('ftf_uco_nyquist.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    test_plot_ftf_uco()
