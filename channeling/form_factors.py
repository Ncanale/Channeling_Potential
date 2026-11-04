"""Atomic form factors in reciprocal space, as potential-energy Fourier transforms.

Convention: U~_j(q) is the 3D Fourier transform of the potential energy [eV*A^3] of a
projectile of charge +1 in the field of one neutral atom, with q the momentum transfer
(q = 4 pi s, s = sin(theta)/lambda).

Peng/Doyle-Turner:   U~(q) = 2 pi e^2 a0 f_e(s),   f_e(s) = sum_i a_i exp(-b_i s^2)
   Mott-Bethe: f_e = (2/(a0 q^2)) (Z - f_x), which tends to the Coulomb limit
   U~ = 4 pi Z e^2/q^2. Coefficients: Peng, Ren, Dudarev, Whelan, Acta Cryst. A52,
   257 (1996), DOI 10.1107/S0108767395014371, fit range 0 < s < 6 1/A.
Moliere:             U~(q) = 4 pi Z e^2 sum_i alpha_i/(q^2 + kappa_i^2),  kappa_i = beta_i/a_TF
   alpha = (0.1, 0.55, 0.35), beta = (6.0, 1.2, 0.3), a_TF = 0.8853 a0 Z^(-1/3)
   (Moliere, Z. Naturforsch. A 2, 133 (1947), DOI 10.1515/zna-1947-0302).
"""
import json
import os
from functools import lru_cache

import numpy as np
from pymatgen.core import Element

E2 = 14.39964     # e^2 [eV*A] (Gaussian units)
A0 = 0.529177     # Bohr radius [A]; e^2*a0 = hbar^2/m_e
MOLIERE_ALPHA = np.array([0.1, 0.55, 0.35])
MOLIERE_BETA = np.array([6.0, 1.2, 0.3])
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@lru_cache(maxsize=None)
def peng_table():
    with open(os.path.join(DATA, "scattering_factors.json")) as f:
        return json.load(f)


class FormFactor:
    """Callable U~(q^2) [eV*A^3] for an element.

    `decay` is the Gaussian exponent c in exp(-c q^2) that bounds the tail of the form factor
    (0 for the power-law Moliere form factor).
    """

    def __init__(self, element, model="peng"):
        self.element, self.model = element, model
        self.Z = Element(element).Z
        if model == "peng":
            tab = peng_table()
            if element not in tab:
                raise SystemExit(f"No Peng 1996 scattering factor for {element}")
            self.a = np.array(tab[element]["a"])
            self.b = np.array(tab[element]["b"])
            self.decay = self.b.min() / (16 * np.pi ** 2)
        elif model == "moliere":
            self.kappa = MOLIERE_BETA / (0.8853 * A0 / self.Z ** (1 / 3))
            self.decay = 0.0
        else:
            raise ValueError(f"unknown model {model!r}")

    def __call__(self, q2):
        q2 = np.asarray(q2, float)
        flat = q2.ravel()[None]
        if self.model == "peng":
            v = 2 * np.pi * E2 * A0 * np.sum(self.a[:, None] * np.exp(-self.b[:, None] * flat / (16 * np.pi ** 2)), axis=0)
        else:
            v = 4 * np.pi * self.Z * E2 * np.sum(MOLIERE_ALPHA[:, None] / (flat + self.kappa[:, None] ** 2), axis=0)
        return v.reshape(q2.shape)
