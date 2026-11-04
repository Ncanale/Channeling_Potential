"""Data files for the Geant4 channeling FastSim model (G4ChannelingFastSimModel).

FastSim (Sytov et al., J. Korean Phys. Soc. 83, 132 (2023), DOI 10.1007/s40042-023-00834-6;
Geant4 source/parameterisations/channeling) integrates the transverse motion of the projectile in
the continuum potential of a plane (1D model) or string (2D model) and needs, as functions of the
transverse position within one periodic cell:
  - the electric field Ex (and Ey for strings) = dU/dx, dU/dy of the potential energy of a unit
    positive charge [eV/cm] (the model multiplies by -z/pv),
  - the nuclear density of each element, normalised to its mean (incoherent scattering on nuclei),
  - the electron density [cm^-3] (ionisation, single scattering on electrons),
  - the minimal energy of ionisation [eV] (lower cut of the energy transfer to an electron),
and, per element, the atomic concentration and the thermal amplitude, plus the well height Vmax and
the absolute potential minimum Vmin. The file layout and the spline conventions are documented in
fastsim_format.py. This module computes those functions with the reciprocal-space machinery of
potential_fourier.py, so they follow the temperature through u_j(T), then writes the file.

Conventions (checked against the Geant4 dataset G4CHANNELING 2.0, see tests and FASTSIM.md):
  - x = 0 is the potential minimum (channel centre). Planes: the plane is at dx/2.
  - axes: the cell is the rectangle dx x dy spanned by lattice vectors of the potential along
    x (default: the first of [1 -1 0], [0 1 -1], [1 0 -1] perpendicular to the axis, i.e. a <110> direction, as in the
    shipped files; then [0 1 0], [0 0 1], [1 0 0]) and
    y = z x x (z along the axis); the origin of the cell is the potential minimum.
  - Planes: Vmin is the minimum of the absolute potential energy (mean inner potential included,
    zero in vacuum). Axes: the shipped files carry 0, and so do ours.

Physics:
  U~_j(q) = 2 pi e^2 a0 f_e,j(q/4pi)  (Peng 1996 scattering factors, form_factors.py)
  electron density: rho~_j(q) = Z_j - (a0 q^2 / 2) f_e,j  (Mott-Bethe; Poisson equation, with the
      same Debye-Waller factor as the potential)
  nuclear density: Debye-Waller-smeared atomic positions
Minimal ionisation energy (T_min of the single scattering on electrons; the rate goes as n_e/T_min):
'fermi' (default) = local Fermi energy in Sytov's definition (PRAB 22, 064601 (2019); PhD thesis
Ferrara 2018): exp(<n_e ln V>/<n_e>) along the string / over the plane, from the 3D crystal
(emin_fermi.py). It reproduces the shipped G4CHANNELING files at the atoms and makes the Geant4 rocking
curves agree with them (FASTSIM.md). Alternatives: 'amorphous' = constant I of the crystal (electron-
weighted geometric mean of the Geant4 NIST mean excitation energies; it gives 2-5 % more channeling near
the critical angle); 'plasma' = I sqrt(n_el(x)/mean n_el) (floor 0.5 eV); or a number [eV].
"""
import os
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Element
from scipy.fft import ifft2
from scipy.optimize import minimize, minimize_scalar

from . import fastsim_format as fmt
from .emin_fermi import CrystalField3D, emin_axial, emin_planar
from .form_factors import A0, FormFactor
from .potential_fourier import _qmax, axial_setup, effective_period_factor, planar_setup
from .structure import plane_frame

# mean excitation energies [eV] of the Geant4 NIST materials (G4NistMaterialBuilder, ICRU 37)
I_MEAN = {"C": 81.0, "O": 95.0, "Si": 173.0, "Ge": 350.0, "W": 727.0, "Ir": 757.0, "Pb": 823.0, "Bi": 823.0}
CM = 1e-8            # 1 A in cm
DEFAULT_X_DIRECTIONS = [(1, -1, 0), (0, 1, -1), (1, 0, -1), (0, 1, 0), (0, 0, 1), (1, 0, 0)]


@dataclass
class FastSimInfo:
    path: str
    model: int
    dx: float          # [A]
    dy: float
    npx: int
    npy: int
    vmax: float
    vmin: float
    emin: str
    max_rel_err_field: float


# ------------------------------------------------------------------ helpers
def _element_order(structure, order):
    present = []
    for s in structure:
        sym = s.specie.symbol
        if sym not in present:
            present.append(sym)
    if order:
        if set(order) != set(present):
            raise SystemExit(f"element order {order} does not match the crystal elements {present}")
        return list(order)
    return present


def emin_constant(structure):
    """Electron-weighted geometric mean of the element mean excitation energies [eV]."""
    num = den = 0.0
    for s in structure:
        el = s.specie.symbol
        if el not in I_MEAN:
            raise SystemExit(f"no mean excitation energy for {el}: pass a number with --emin")
        Z = Element(el).Z
        num += Z * np.log(I_MEAN[el])
        den += Z
    return float(np.exp(num / den))


def _density_weights(structure, order, th, dw_only=True):
    """Per-element weight functions of q^2 for the electron density (Mott-Bethe from the Peng f_e)."""
    ff = {el: FormFactor(el, "peng") for el in order}

    def rho_el(el, G2):
        f = ff[el]
        fe = np.sum(f.a[:, None] * np.exp(-f.b[:, None] * G2[None] / (16 * np.pi ** 2)), axis=0)
        return Element(el).Z - 0.5 * A0 * G2 * fe

    return ff, rho_el


def _quantity_coeffs(S, ff, rho_el, order, emin_mode):
    """Fourier coefficients (K,) of U [eV A^3/A^3], n_el [A^-3] and the nuclear densities."""
    cU = sum(ff[el](S.G2) * S.sfdw[el] for el in order)
    cEl = sum(rho_el(el, S.G2) * S.sfdw[el] for el in order)
    cNuc = [S.sfdw[el] / S.natoms[el] for el in order]
    return cU, cEl, cNuc


def _emin_values(emin, n_el, Ic):
    """Map of the minimal ionisation energy [eV] on the nodes."""
    if isinstance(emin, (int, float)):
        return np.full_like(n_el, float(emin))
    if emin == "amorphous":
        return np.full_like(n_el, Ic)
    if emin == "plasma":
        nbar = n_el.mean()
        return np.maximum(Ic * np.sqrt(np.maximum(n_el, 0.0) / nbar), 0.5)
    raise SystemExit(f"unknown --emin {emin!r}: use fermi, amorphous, plasma or a number [eV]")


def parse_emin(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _thermal_arrays(order, th):
    u = np.array([th[el].u for el in order])
    return u


# ------------------------------------------------------------------ planar
def planar_fastsim(structure, hkl, th, order=None, step=None, emin="fermi"):
    """FastSimData (model 1) for the plane family (hkl). th: {el: ThermalResult}."""
    order = _element_order(structure, order)
    Ucart = {el: t.tensor() for el, t in th.items()}
    ff, rho_el = _density_weights(structure, order, th)
    qmax = _qmax({el: 0.0 for el in order}, Ucart)
    S = planar_setup(structure, hkl, Ucart, qmax)
    cU, cEl, cNuc = _quantity_coeffs(S, ff, rho_el, order, emin)
    cs = sum(np.abs(S.sfdw[el]) for el in order)
    k = effective_period_factor(S.m, cs)
    d = S.d_cell / k
    gn, m, Om = S.gn, S.m, S.Omega

    def ev(c, x, deriv=0):
        x = np.atleast_1d(x)
        ph = np.exp(1j * np.outer(x, m * gn))
        w = (1j * m * gn) ** deriv
        return (ph @ (c * w)).real / Om

    xs = np.linspace(0, d, 4001)
    x0 = xs[np.argmin(ev(cU, xs))]
    x0 = minimize_scalar(lambda x: ev(cU, x)[0], bounds=(x0 - d / 400, x0 + d / 400), method="bounded",
                         options=dict(xatol=1e-10)).x
    xs2 = x0 + np.linspace(0, d, 4001)
    vals = ev(cU, xs2)
    imax = int(np.argmax(vals))
    xmax = minimize_scalar(lambda x: -ev(cU, x)[0], bounds=(xs2[max(imax - 1, 0)], xs2[min(imax + 1, 4000)]),
                           method="bounded", options=dict(xatol=1e-10)).x
    umin = float(ev(cU, x0)[0])
    umax = float(max(ev(cU, xmax)[0], vals.max()))
    vmax, vmin = umax - umin, umin

    u_min_h = min(t.u for t in th.values())
    h = step if step else min(0.002, u_min_h / 15)
    npx = int(np.ceil(d / h))
    h = d / npx
    x = x0 + np.arange(npx + 1) * h
    Ex = ev(cU, x, 1) / CM                       # dU/dx in eV/cm
    nel = ev(cEl, x) * 1e24
    nucs = [ev(c, x) * Om for c in cNuc]           # cNuc = sfdw/natoms has no 1/Omega: undo the one of ev()
    if emin == "fermi":
        emv = emin_planar(CrystalField3D(structure, th, order), S.nhat, x)
    else:
        emv = _emin_values(emin, nel, emin_constant(structure))
    # fields at the two ends of the period must match (periodic): enforce exact periodicity
    for arr in (Ex, nel, emv, *nucs):
        arr[-1] = arr[0]
    blocks = [fmt.planar_block(Ex, h * CM), fmt.planar_block(nucs[0], h * CM), fmt.planar_block(nel, h * CM),
              fmt.planar_block(emv, h * CM)] + [fmt.planar_block(a, h * CM) for a in nucs[1:]]
    coef = np.concatenate(blocks, axis=1)
    N0 = np.array([S.natoms[el] / Om * 1e24 for el in order])
    u = _thermal_arrays(order, th) * CM
    data = fmt.FastSimData(1, len(order), N0, u, d * CM, 0.0, npx, 0, vmax, vmin, coef)
    return data


# ------------------------------------------------------------------ axial
def _hnf_2d(vectors):
    """Basis (2x2 integer, rows) of the lattice generated by integer 2D vectors."""
    rows = [tuple(int(a) for a in v) for v in vectors if (v[0], v[1]) != (0, 0)]
    rows = list({r for r in rows})
    piv = None
    while True:
        nz = [r for r in rows if r[0] != 0]
        if not nz:
            break
        p = min(nz, key=lambda r: abs(r[0]))
        new = []
        for r in rows:
            if r == p:
                new.append(r)
                continue
            if r[0] != 0:
                q = r[0] // p[0]
                r = (r[0] - q * p[0], r[1] - q * p[1])
            if r != (0, 0):
                new.append(r)
        rows = list(set(new))
        piv = p
        if all(r[0] == 0 or r == p for r in rows):
            break
    if piv is None:
        piv = (0, 0)
    g = 0
    for r in rows:
        if r[0] == 0:
            g = np.gcd(g, abs(r[1]))
    if g == 0:
        raise SystemExit("cannot build the 2D period lattice")
    return np.array([[piv[0], piv[1] % g], [0, g]])


def _rectangular_cell(A2, ex, ey, M=12):
    """Shortest lattice vectors along ex and ey, or None. A2: rows = lattice basis (2D, e-frame)."""
    m = np.arange(-M, M + 1)
    m1, m2 = np.meshgrid(m, m, indexing="ij")
    V = m1.ravel()[:, None] * A2[0] + m2.ravel()[:, None] * A2[1]
    n = np.linalg.norm(V, axis=1)
    ok = n > 1e-9
    out = []
    for e, o in ((ex, ey), (ey, ex)):
        par = ok & (np.abs(V @ o) < 1e-6 * n) & (V @ e > 0)
        if not par.any():
            return None
        out.append(float(n[par].min()))
    return out[0], out[1]


def axial_fastsim(structure, uvw, th, order=None, step=None, emin="fermi", xdir=None):
    """FastSimData (model 2) for strings along [uvw]. xdir: crystal direction (h k l) of the cell x axis."""
    order = _element_order(structure, order)
    uvw = np.asarray(uvw, int)
    Ucart = {el: t.tensor() for el, t in th.items()}
    ff, rho_el = _density_weights(structure, order, th)
    qmax = _qmax({el: 0.0 for el in order}, Ucart)
    S = axial_setup(structure, uvw, Ucart, qmax)
    cU, cEl, cNuc = _quantity_coeffs(S, ff, rho_el, order, emin)
    Om, G, e1, e2 = S.Omega, S.G, S.e1, S.e2
    L = structure.lattice.matrix

    # significant harmonics (pure structure factor, before the Debye-Waller damping)
    sig = np.zeros(len(G), bool)
    for el in order:
        idx = [i for i, s in enumerate(structure) if s.specie.symbol == el]
        sf = np.exp(-1j * (G @ S.cart[idx].T)).sum(axis=1)
        sig |= np.abs(sf) > 1e-6 * len(idx)
    # 2D period lattice of the potential from the support of the structure factors
    g2d = np.array([[S.g1 @ e1, S.g1 @ e2], [S.g2 @ e1, S.g2 @ e2]])
    B = _hnf_2d(np.c_[S.n1[sig], S.n2[sig]])
    Gnew = B @ g2d
    A2 = 2 * np.pi * np.linalg.inv(Gnew).T

    # x direction and rectangular cell
    t_hat = S.t_hat
    cands = [tuple(xdir)] if xdir is not None else DEFAULT_X_DIRECTIONS
    cell = None
    for dirn in cands:
        vec = np.array(dirn, float) @ L
        if abs(vec @ t_hat) > 1e-6 * np.linalg.norm(vec):
            continue
        ex3 = vec / np.linalg.norm(vec)
        ey3 = np.cross(t_hat, ex3)
        ex2, ey2 = np.array([ex3 @ e1, ex3 @ e2]), np.array([ey3 @ e1, ey3 @ e2])
        rc = _rectangular_cell(A2, ex2, ey2)
        if rc is not None:
            cell = (ex3, ey3, rc[0], rc[1])
            break
    if cell is None:
        raise SystemExit("no rectangular periodic cell along the tried x directions: pass --fastsim-x 'h k l'")
    ex3, ey3, Dx, Dy = cell

    p = G @ ex3 * Dx / (2 * np.pi)
    q = G @ ey3 * Dy / (2 * np.pi)
    P, Q = np.rint(p).astype(int), np.rint(q).astype(int)
    if np.abs(p[sig] - P[sig]).max() > 1e-5 or np.abs(q[sig] - Q[sig]).max() > 1e-5:
        raise SystemExit("rectangular cell is not commensurate with the potential (internal error)")
    P, Q, Gs = P[sig], Q[sig], G[sig]

    u_min_h = min(t.u for t in th.values())
    h = step if step else float(np.clip(u_min_h / 2.2, 0.005, 0.04))
    npx = max(int(np.ceil(Dx / h)), 2 * int(np.abs(P).max()) + 1)
    npy = max(int(np.ceil(Dy / h)), 2 * int(np.abs(Q).max()) + 1)
    hx, hy = Dx / npx, Dy / npy

    def direct(c, pts2d):
        """Re sum c exp(i G.(x ex + y ey))/Om at (x, y) points (n, 2)."""
        r = pts2d[:, :1] * ex3 + pts2d[:, 1:] * ey3
        return (np.exp(1j * (Gs @ r.T)) * c[sig][:, None]).sum(axis=0).real / Om

    def grid(c, r0, deriv=None):
        """Values on the (npx, npy) node grid for coefficients c, origin r0 (x, y)."""
        r03 = r0[0] * ex3 + r0[1] * ey3
        cc = c[sig] * np.exp(1j * (Gs @ r03))
        if deriv == "x":
            cc = cc * 1j * (Gs @ ex3)
        elif deriv == "y":
            cc = cc * 1j * (Gs @ ey3)
        C = np.zeros((npx, npy), complex)
        np.add.at(C, (P % npx, Q % npy), cc)
        return (npx * npy / Om) * ifft2(C).real

    # origin = potential minimum
    U0 = grid(cU, (0.0, 0.0))
    i0, j0 = np.unravel_index(np.argmin(U0), U0.shape)
    res = minimize(lambda r: direct(cU, np.array([r]))[0], np.array([i0 * hx, j0 * hy]), method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-10, maxiter=400))
    r0 = tuple(res.x)
    umin = float(direct(cU, np.array([r0]))[0])
    proj = S.cart - np.outer(S.cart @ t_hat, t_hat)
    uq = np.unique(np.round(proj, 4), axis=0)
    ustr = (np.exp(1j * (Gs @ uq.T)) * cU[sig][:, None]).sum(axis=0).real / Om
    Ug = grid(cU, r0)
    umax = float(max(ustr.max(), Ug.max()))
    vmax = umax - umin

    wrap = lambda a: np.pad(a, ((0, 1), (0, 1)), mode="wrap")
    Ex = wrap(grid(cU, r0, "x")) / CM
    Ey = wrap(grid(cU, r0, "y")) / CM
    nel = wrap(grid(cEl, r0)) * 1e24
    # nuclear density: coefficients are sfdw/natoms (dimensionless, no 1/Om): undo the 1/Om of grid()
    nucs = [wrap(grid(c, r0)) * Om for c in cNuc]
    if emin == "fermi":
        origin = r0[0] * ex3 + r0[1] * ey3
        emv = wrap(emin_axial(CrystalField3D(structure, th, order), origin, ex3, ey3, t_hat,
                              np.arange(npx) * hx, np.arange(npy) * hy))
    else:
        emv = _emin_values(emin, nel, emin_constant(structure))
    # emin as a node function
    qs = [emv, Ex, Ey, nucs[0], nel] + nucs[1:]
    nq = len(qs)
    coef = np.zeros((npy, npx + 1, 2, 3 * nq))
    for iq, f in enumerate(qs):
        blk = fmt.axial_block(f, hx * CM, hy * CM)
        coef[:, :, :, 3 * iq:3 * iq + 3] = blk
    N0 = np.array([S.natoms[el] / Om * 1e24 for el in order])
    u = _thermal_arrays(order, th) * CM
    return fmt.FastSimData(2, len(order), N0, u, Dx * CM, Dy * CM, npx, npy, vmax, 0.0, coef), dict(
        r0=r0, ex=ex3, ey=ey3, vmin_abs=umin)


# ------------------------------------------------------------------ driver
def write_files(structure, uvw, hkl, th, outdir, g4name, order=None, step=None, emin="fermi", xdir=None,
                planar=True, axial=True, info_extra=""):
    """Write <g4name>_planes_<hkl>.dat and <g4name>_axes_<uvw>.dat in `outdir`; returns FastSimInfo list."""
    os.makedirs(outdir, exist_ok=True)
    out = []
    emin_mode = parse_emin(emin) if isinstance(emin, str) else emin
    lab = lambda v: "".join(str(int(a)) for a in v)
    if planar:
        d = planar_fastsim(structure, hkl, th, order, step, emin_mode)
        path = os.path.join(outdir, f"{g4name}_planes_{lab(hkl)}.dat")
        fmt.write(path, d)
        out.append(_info(path, d, emin_mode))
    if axial:
        d, _ = axial_fastsim(structure, uvw, th, order, step, emin_mode, xdir)
        path = os.path.join(outdir, f"{g4name}_axes_{lab(uvw)}.dat")
        fmt.write(path, d)
        out.append(_info(path, d, emin_mode))
    return out


def _info(path, d, emin):
    return FastSimInfo(path, d.model, d.dx / CM, d.dy / CM, d.npx, d.npy, d.vmax, d.vmin, str(emin), float("nan"))
