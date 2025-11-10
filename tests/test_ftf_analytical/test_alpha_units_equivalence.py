import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from FTF_exact import FTF_CCO


def test_alpha_units_equivalence():
    omega = np.array([0.0, 0.1, 1.0, 10.0]) * 2*np.pi  # rad/s
    alpha_deg_values = [5, 25, 50, 75]
    for a_deg in alpha_deg_values:
        phase_d_deg, gain_d = FTF_CCO(omega, a_deg, phase_units='deg', radians=False)
        phase_r_deg, gain_r = FTF_CCO(omega, np.deg2rad(a_deg), phase_units='deg', radians=True)
        assert np.allclose(gain_d, gain_r, rtol=1e-12, atol=1e-12)
        assert np.allclose(phase_d_deg, phase_r_deg, rtol=1e-12, atol=1e-10)

if __name__ == '__main__':
    test_alpha_units_equivalence()
    print('alpha units equivalence test: OK')
