"""Atomic form factors in reciprocal space, as potential-energy Fourier transforms.

Convention: U~_j(q) is the 3D Fourier transform of the potential energy [eV*A^3] of a
projectile of charge +1 in the field of one neutral atom, with q the momentum transfer
(q = 4 pi s, s = sin(theta)/lambda).

Peng/Doyle-Turner:   U~(q) = 2 pi e^2 a0 f_e(s),   f_e(s) = sum_i a_i exp(-b_i s^2)
   Mott-Bethe: f_e = (2/(a0 q^2)) (Z - f_x), which tends to the Coulomb limit
   U~ = 4 pi Z e^2/q^2. Coefficients: Peng, Ren, Dudarev, Whelan, Acta Cryst. A52,
   257 (1996), DOI 10.1107/S0108767395014371, fit range 0 < s < 6 1/A.
   model "doyle-turner": the same form with the 4 Gaussians of Doyle and Turner, Acta Cryst. A24, 390 (1968),
   DOI 10.1107/S0567739468000756, fit range 0 < s < 2 1/A (values as in De Graef, Introduction to Conventional
   TEM (2003), Table 12.1, and in pymatgen/diffsims).
   model "dpw": the same form with the 5 Gaussians of Dudarev, Peng and Whelan, Surf. Sci. 330, 86 (1995),
   DOI 10.1016/0039-6028(95)00464-5, Tables 1-2 (real part, relativistic Hartree-Fock, fit range 0 < s < 5 1/A,
   50 elements; data/dudarev_peng_whelan_1995.json). This is the atomic model of the Geant4 dataset
   G4CHANNELING 2.0 (Sytov's CRYSTAL code cites it, NIM B 355, 383 (2015)); the files of the dataset are reproduced
   with it and the Mott-Bethe density (output/g4_comparison).
Moliere:             U~(q) = 4 pi Z e^2 sum_i alpha_i/(q^2 + kappa_i^2),  kappa_i = beta_i/a_TF
   alpha = (0.1, 0.55, 0.35), beta = (6.0, 1.2, 0.3), a_TF = 0.8853 a0 Z^(-1/3)
   (Moliere, Z. Naturforsch. A 2, 133 (1947), DOI 10.1515/zna-1947-0302).

Electron density of the atom, n~(q) [electrons], two routes:
  "mott-bethe": n~ = Z - (a0 q^2 / 2) f_e(s) from the electron form factor above. Exact for an exact f_e, but the
      factor q^2 amplifies the fit error of f_e at large q: with small thermal amplitudes (diamond, u = 0.04 A) the
      density oscillates on a 0.1 A scale.
  "xray": n~ = f_x(s) = sum_i a_i exp(-b_i s^2) + c, X-ray form factor of Waasmaier and Kirfel, Acta Cryst. A51,
      416 (1995), DOI 10.1107/S0108767394013292 (values as in cctbx, eltbx/xray_scattering/wk1995.cpp).
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


# Doyle-Turner 1968: a_i, b_i [A^2] of f_e(s) = sum a_i exp(-b_i s^2)
DOYLE_TURNER = {
    "C": {"a": [0.731, 1.195, 0.456, 0.125], "b": [36.995, 11.297, 2.814, 0.346]},
    "Si": {"a": [2.129, 2.533, 0.835, 0.322], "b": [57.775, 16.476, 2.880, 0.386]},
    "Ge": {"a": [2.447, 2.702, 1.616, 0.601], "b": [55.893, 14.393, 2.446, 0.342]},
}
# Waasmaier-Kirfel 1995: a_i, b_i [A^2], c of f_x(s) = sum a_i exp(-b_i s^2) + c
WAASMAIER_KIRFEL = {
    "C": {"a": [2.657506, 1.078079, 1.490909, -4.241070, 0.713791],
          "b": [14.780758, 0.776775, 42.086842, -0.000294, 0.239535], "c": 4.297983},
    "Si": {"a": [5.275329, 3.191038, 1.511514, 1.356849, 2.519114],
           "b": [2.631338, 33.730728, 0.081119, 86.288643, 1.170087], "c": 0.145073},
    "Ge": {"a": [16.540613, 1.567900, 3.727829, 3.345098, 6.785079],
           "b": [2.866618, 0.012198, 13.432163, 58.866047, 0.210974], "c": 0.018726},
    "W": {"a": [31.507900, 15.682498, 37.960129, 4.885509, 16.792112],
          "b": [1.629485, 9.446448, 0.000898, 59.980675, 0.160798], "c": -32.864574},
}


GAUSSIAN_MODELS = ("peng", "doyle-turner", "dpw")


@lru_cache(maxsize=None)
def peng_table():
    with open(os.path.join(DATA, "scattering_factors.json")) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def dpw_table():
    """Dudarev, Peng, Whelan (1995), Tables 1-2: {element: {"a": [...], "b": [...]}} (a in A, b in A^2)."""
    with open(os.path.join(DATA, "dudarev_peng_whelan_1995.json")) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


class FormFactor:
    """Callable U~(q^2) [eV*A^3] for an element.

    `decay` is the Gaussian exponent c in exp(-c q^2) that bounds the tail of the form factor
    (0 for the power-law Moliere form factor).
    """

    def __init__(self, element, model="peng"):
        self.element, self.model = element, model
        self.Z = Element(element).Z
        if model in GAUSSIAN_MODELS:
            tab = {"peng": peng_table, "doyle-turner": lambda: DOYLE_TURNER, "dpw": dpw_table}[model]()
            if element not in tab:
                raise SystemExit(f"No {model} scattering factor for {element}")
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
        if self.model in GAUSSIAN_MODELS:
            v = 2 * np.pi * E2 * A0 * np.sum(self.a[:, None] * np.exp(-self.b[:, None] * flat / (16 * np.pi ** 2)), axis=0)
        else:
            v = 4 * np.pi * self.Z * E2 * np.sum(MOLIERE_ALPHA[:, None] / (flat + self.kappa[:, None] ** 2), axis=0)
        return v.reshape(q2.shape)


def atom_model(element):
    """(potential model, density route) of an element for the FastSim files: Dudarev-Peng-Whelan 1995 with the
    Mott-Bethe density, as in the Geant4 dataset G4CHANNELING 2.0, for the 50 elements of their table; Peng 1996 with
    the Mott-Bethe density for all others."""
    return ("dpw", "mott-bethe") if element in dpw_table() else ("peng", "mott-bethe")


def density_q(ff, q2, route="mott-bethe"):
    """Electron density of one atom in reciprocal space n~(q) [electrons], without Debye-Waller factor.
    ff: FormFactor of the potential (Gaussian models), used by the Mott-Bethe route."""
    q2 = np.asarray(q2, float)
    s2 = q2 / (16 * np.pi ** 2)
    if route == "xray":
        t = WAASMAIER_KIRFEL[ff.element]
        return np.sum(np.array(t["a"])[:, None] * np.exp(-np.array(t["b"])[:, None] * s2.ravel()[None]), axis=0
                      ).reshape(q2.shape) + t["c"]
    fe = np.sum(ff.a[:, None] * np.exp(-ff.b[:, None] * s2.ravel()[None]), axis=0).reshape(q2.shape)
    return ff.Z - 0.5 * A0 * q2 * fe


def density_r(ff, u, r, route="mott-bethe"):
    """Electron density n(r) [A^-3] of one atom smeared with the isotropic thermal amplitude u [A]; negative values
    of the fit set to 0. Mott-Bethe: n = n_nucleus + lap(V)/(4 pi e^2), V from the Gaussians of ff."""
    r = np.asarray(r, float)
    if route == "xray":
        t = WAASMAIER_KIRFEL[ff.element]
        terms = list(zip(t["a"], t["b"]))
        nuc = t["c"]
        n = np.zeros_like(r)
        for a, b in terms:
            al = b / (16 * np.pi ** 2) + u * u / 2
            n += a * (4 * np.pi * al) ** -1.5 * np.exp(-r * r / (4 * al))
    else:
        nuc = ff.Z
        n = np.zeros_like(r)
        for a, b in zip(ff.a, ff.b):
            al = b / (16 * np.pi ** 2) + u * u / 2
            g = 2 * np.pi * E2 * A0 * a * (4 * np.pi * al) ** -1.5 * np.exp(-r * r / (4 * al))
            n += g * (r * r / (4 * al * al) - 3 / (2 * al)) / (4 * np.pi * E2)
    n += nuc * (2 * np.pi * u * u) ** -1.5 * np.exp(-r * r / (2 * u * u))
    return np.clip(n, 0.0, None)
