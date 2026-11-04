"""Thermal displacement amplitudes u(T) of the atoms.

u is the 1D rms displacement [A]; B = 8 pi^2 <u^2> is the Debye-Waller factor.

Tier A  Peng, Ren, Dudarev, Whelan, Acta Cryst. A52, 456 (1996), DOI 10.1107/S010876739600089X,
        Table 1 (B(T) from measured phonon DOS). The printed table spans 110-280 K (the part used
        here). Inside that range B is interpolated linearly in T; outside it the Debye formula is
        used with the Debye-Waller temperature theta_M fitted to the nearest table row, so the
        result is continuous. Such values are flagged `extrapolated`.
Tier B  Debye model, Lonsdale form, for mass M = A amu and Debye-Waller temperature theta:
            <u^2> = 145.5262/(A theta) * [phi(x)/x + 1/4],  x = theta/T,
            phi(x) = (1/x) int_0^x t/(e^t - 1) dt,   145.5262 = 3 hbar^2/(amu k_B) [A^2 K]
        (Gemmell, Rev. Mod. Phys. 46, 129 (1974), DOI 10.1103/RevModPhys.46.129).
        Zero point <u^2>(0) = 145.5262/(4 A theta); high T <u^2> -> 145.5262 T/(A theta^2).
Tier C1 Trots, Senyshyn, Schwarz, J. Solid State Chem. 183, 1245 (2010), DOI 10.1016/j.jssc.2010.03.039:
        isotropic B_iso(T) per species of PbWO4 from neutron powder diffraction, fitted as
            B = 8 pi^2 * 145.5262/(A theta) * phi(x)/x + B_static.
        NOTE: Eq. (1) of the paper also carries a zero-point term. Reading Fig. 4 of the paper
        (B(298 K) about 0.55, 0.70, 0.73 A^2 above B_static for W, Pb, O) matches the formula only
        when the zero-point term is omitted (B_static absorbs it), so it is omitted here.
        The figure values were read by eye: treat the result as +-10 %.
        Also an Einstein model for light atoms: <u^2> = 48.509/(2 A theta_E) coth(theta_E/2T).
Tier C2 Anisotropic displacement tensor U_cart (A^2), e.g. from phonopy; see load_phonopy_tdispmat.
"""
import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pymatgen.core import Element
from scipy.constants import atomic_mass, hbar
from scipy.constants import k as KB
from scipy.integrate import quad
from scipy.optimize import brentq

C_DEBYE = 3 * hbar ** 2 / (atomic_mass * KB) * 1e20   # 145.5262 A^2 K
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class ThermalResult:
    u: float                     # isotropic 1D rms amplitude [A]
    tier: str
    source: str
    extrapolated: bool = False
    Ucart: np.ndarray = None     # optional 3x3 displacement tensor [A^2]; overrides u in the Fourier method

    def tensor(self):
        return self.Ucart if self.Ucart is not None else self.u ** 2 * np.eye(3)


def mass(el):
    return float(Element(el).atomic_mass)


def phi(x):
    """Debye function phi(x) = (1/x) int_0^x t/(e^t-1) dt."""
    if x < 1e-6:
        return 1 - x / 4
    return quad(lambda t: t / np.expm1(t), 0, x)[0] / x


def u2_debye(M, theta, T, zero_point=True):
    """Debye-model <u_x^2> [A^2] at temperature T [K] (T <= 0 gives the zero-point value)."""
    if T <= 0:
        return C_DEBYE / (4 * M * theta) if zero_point else 0.0
    x = theta / T
    return C_DEBYE / (M * theta) * (phi(x) / x + (0.25 if zero_point else 0.0))


def u2_einstein(M, theta_E, T):
    z = theta_E / (2 * max(T, 1e-9))
    return (C_DEBYE / 3) / (2 * M * theta_E) / np.tanh(z)


def theta_from_B(M, B, T):
    """Debye-Waller temperature theta_M that reproduces B = 8 pi^2 <u^2> at T."""
    target = B / (8 * np.pi ** 2)
    return brentq(lambda th: u2_debye(M, th, T) - target, 20.0, 30000.0)


@lru_cache(maxsize=None)
def _dw_table():
    with open(os.path.join(DATA, "debye_waller.json")) as f:
        return json.load(f)["elements"]


def peng_B(el, T):
    """Tier A B(T) [A^2] and an `extrapolated` flag (Debye outside the table rows)."""
    tab = _dw_table()[el]
    Ts, Bs = np.array(tab["T"], float), np.array(tab["B"], float)
    M = mass(el)
    if Ts[0] <= T <= Ts[-1]:
        return float(np.interp(T, Ts, Bs)), False
    edge = 0 if T < Ts[0] else -1
    theta = theta_from_B(M, Bs[edge], Ts[edge])
    return 8 * np.pi ** 2 * u2_debye(M, theta, T), True


def u_of_T(el, T, spec=None):
    """Resolve the amplitude of element `el` at temperature T [K].

    spec: dict from materials.yaml / CLI ({"model": peng|debye|trots|einstein|fixed|adp|missing, ...}).
    """
    spec = spec or {"model": "peng"}
    model = spec.get("model", "peng")
    M = mass(el)
    if model == "peng":
        tab = _dw_table().get(el)
        if tab is None:
            raise SystemExit(f"No Peng Table 1 data for {el}: add a thermal model in materials.yaml or pass --u {el}=<A>")
        if "T" not in tab:   # Debye-only entry (Ir)
            return ThermalResult(np.sqrt(u2_debye(M, tab["theta_M"], T)), "B", tab["source"], True)
        B, extra = peng_B(el, T)
        return ThermalResult(np.sqrt(B / (8 * np.pi ** 2)), "A", "Peng 1996 Table 1 (DOI 10.1107/S010876739600089X)", extra)
    if model == "debye":
        return ThermalResult(np.sqrt(u2_debye(M, spec["theta"], T)), "B", spec.get("source", "Debye model"), True)
    if model == "trots":
        u2 = u2_debye(M, spec["theta"], T, zero_point=False) + spec["B_static"] / (8 * np.pi ** 2)
        return ThermalResult(np.sqrt(u2), "C1", spec.get("source", "Trots 2010"), T > 298)
    if model == "einstein":
        u2 = u2_einstein(M, spec["theta_E"], T) + spec.get("B_static", 0.0) / (8 * np.pi ** 2)
        return ThermalResult(np.sqrt(u2), "C1", spec.get("source", "Einstein model"))
    if model == "fixed":
        return ThermalResult(float(spec["u"]), "fixed", spec.get("source", "user value"))
    if model == "adp":
        U = np.array(spec["Ucart"], float)
        return ThermalResult(float(np.sqrt(np.trace(U) / 3)), "C2", spec.get("source", "displacement tensor"), Ucart=U)
    if model == "missing":
        raise SystemExit(f"No thermal-amplitude source available for {el} in this material "
                         f"({spec.get('note', 'see materials.yaml')}). Pass --u {el}=<value in A>.")
    raise ValueError(f"unknown thermal model {model!r}")


def load_phonopy_tdispmat(path, T):
    """Hook for tier C2: read a phonopy `thermal_displacement_matrices.yaml` and return the mean
    Cartesian 3x3 tensor [A^2] (average over atoms) at the temperature closest to T.

    Generate the file with phonopy (no DFT is run here), e.g.
        phonopy-load --tdispmat --tmin 4 --tmax 400 --tstep 20
    Requires pyyaml (imported lazily). UNTESTED against real phonopy output: check the layout of
    your phonopy version (components xx yy zz yz xz xy are assumed).
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    row = min(data["thermal_displacement_matrices"], key=lambda r: abs(r["temperature"] - T))
    mats = []
    for m in row["displacement_matrices"]:
        xx, yy, zz, yz, xz, xy = m
        mats.append([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return np.mean(mats, axis=0)
