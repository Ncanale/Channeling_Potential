"""Reciprocal-space continuum potentials (default method).

For atoms j at Cartesian positions r_j in a cell of volume Omega, with thermal tensors U_j:

    U(r) = (z/Omega) sum_G S(G) exp(i G.r),
    S(G) = sum_j U~_j(|G|) exp(-1/2 G^T U_j G) exp(-i G.r_j)

U~_j is the form factor of form_factors.py. exp(-1/2 G^T U_j G) is the exact thermal average of
the potential of an atom vibrating as a Gaussian (the Debye-Waller factor); for isotropic motion
U_j = u_j^2 I. Averaging along an axis t keeps only the G with G.t = 0, i.e. Miller indices h with
h.[uvw] = 0 (2D reciprocal sublattice): the result is a periodic 2D map. Averaging over a plane
family (hkl) keeps G = m g_hkl: a periodic 1D profile. Because the sum is periodic and exact, it
handles mixed strings and planes, unequal plane spacings (diamond (111)), anisotropic displacement
tensors and partial structures without special cases. The axial map is evaluated on one cell by an
FFT in the fractional in-plane coordinates (oversampled by zero padding, so exact interpolation).

The result for charge z = +1 is returned; for z = -1 the potential changes sign (U -> -U), which
leaves depth, field and Lindhard angle unchanged. `axial_setup` / `planar_setup` expose the structure
factors (dw_j(G) * sum_{atoms of j} exp(-i G.r)) so that other periodic quantities, such as electron and
nuclear densities (see fastsim.py), can be built with the same machinery.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import ifft2, next_fast_len
from scipy.ndimage import map_coordinates

from .structure import plane_frame

LOG_TOL = np.log(1e7)   # drop Fourier components below 1e-7 of the form factor amplitude


def _qmax(ff, Ucart):
    """Cutoff |G| such that the Gaussian tail of every species is below 1e-7.

    ff: {el: FormFactor} (its `decay` adds to the thermal exponent) or {el: 0.0} for pure Debye-Waller
    decay (densities)."""
    q = 0.0
    for el, f in ff.items():
        lam = np.linalg.eigvalsh(Ucart[el]).min()
        c = (f.decay if hasattr(f, "decay") else float(f)) + 0.5 * lam
        q = max(q, np.sqrt(LOG_TOL / c))
    return q


def _shortest_sublattice_basis(uvw, recip):
    """Two integer vectors spanning {h in Z^3 : h.uvw = 0} (Gauss-reduced in |G|)."""
    M = 2 * int(np.abs(uvw).max()) + 2
    while True:
        rng = np.arange(-M, M + 1)
        h = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
        h = h[(h @ uvw == 0) & np.any(h != 0, axis=1)]
        G = h @ recip
        order = np.argsort(np.linalg.norm(G, axis=1))
        h = h[order]
        b1 = h[0]
        b2 = next((x for x in h if np.linalg.norm(np.cross(b1, x)) > 1e-9), None)
        if b2 is not None and abs(np.linalg.norm(np.cross(b1, b2)) - np.linalg.norm(uvw)) < 1e-9:
            return b1, b2
        M *= 2


@dataclass
class AxialSetup:
    uvw: np.ndarray
    Omega: float
    t_hat: np.ndarray
    period: float
    e1: np.ndarray
    e2: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    a1_3: np.ndarray            # in-plane direct vectors (3D), dual to g1, g2
    a2_3: np.ndarray
    n1: np.ndarray              # integer indices of the retained G = n1 g1 + n2 g2
    n2: np.ndarray
    G: np.ndarray               # (K, 3) Cartesian
    G2: np.ndarray
    sfdw: dict                  # el -> exp(-1/2 G^T U G) * sum_atoms exp(-i G.r)   (K,)
    natoms: dict                # el -> atoms of the element in the cell
    cart: np.ndarray
    elements: list
    qmax: float


def axial_setup(structure, uvw, Ucart, qmax):
    uvw = np.asarray(uvw, int)
    L = structure.lattice.matrix
    Omega = structure.volume
    recip = 2 * np.pi * np.linalg.inv(L).T            # rows: reciprocal basis vectors
    t = uvw @ L
    period = np.linalg.norm(t)
    t_hat = t / period
    e1, e2 = plane_frame(t_hat)
    b1, b2 = _shortest_sublattice_basis(uvw, recip)
    g1, g2 = b1 @ recip, b2 @ recip
    gm = np.array([g1, g2])
    a1_3, a2_3 = 2 * np.pi * np.linalg.solve(gm @ gm.T, gm)   # a_i . g_j = 2 pi delta_ij
    n1m = int(np.ceil(qmax * np.linalg.norm(a1_3) / (2 * np.pi)))
    n2m = int(np.ceil(qmax * np.linalg.norm(a2_3) / (2 * np.pi)))
    n1, n2 = np.meshgrid(np.arange(-n1m, n1m + 1), np.arange(-n2m, n2m + 1), indexing="ij")
    n1, n2 = n1.ravel(), n2.ravel()
    G = n1[:, None] * g1 + n2[:, None] * g2
    G2 = np.einsum("ij,ij->i", G, G)
    keep = G2 <= qmax ** 2
    n1, n2, G, G2 = n1[keep], n2[keep], G[keep], G2[keep]
    cart = structure.cart_coords
    elements = [s.specie.symbol for s in structure]
    sfdw, natoms = {}, {}
    for el in sorted(set(elements)):
        idx = [i for i, e in enumerate(elements) if e == el]
        dw = np.exp(-0.5 * np.einsum("ki,ij,kj->k", G, Ucart[el], G))
        sfdw[el] = dw * np.exp(-1j * (G @ cart[idx].T)).sum(axis=1)
        natoms[el] = len(idx)
    return AxialSetup(uvw, Omega, t_hat, period, e1, e2, g1, g2, a1_3, a2_3, n1, n2, G, G2, sfdw, natoms,
                      cart, elements, qmax)


@dataclass
class AxialResult:
    U: np.ndarray                # (M1, M2) potential [eV] on the fractional grid of one cell, z = +1
    a1: np.ndarray               # in-plane period vectors [A] in the (e1, e2) frame (2D)
    a2: np.ndarray
    emax: float                  # maximum transverse field |grad U| [eV/A]
    u_max: float                 # max of U, evaluated exactly at atomic strings
    u_min: float
    r_max: np.ndarray            # 3D Cartesian positions of the maximum and of the minimum
    r_min: np.ndarray
    period: float
    shares: dict = field(default_factory=dict)   # species share of (U_max - U_min)
    frame: tuple = None
    n_g: int = 0

    @property
    def depth(self):
        return self.u_max - self.u_min

    def sample(self, X, Y):
        """U on arbitrary Cartesian points (X, Y) of the (e1, e2) frame, periodic, cubic interpolation."""
        A = np.array([[self.a1[0], self.a2[0]], [self.a1[1], self.a2[1]]])
        s = np.linalg.solve(A, np.array([X.ravel(), Y.ravel()]))
        m1, m2 = self.U.shape
        out = map_coordinates(self.U, [s[0] * m1, s[1] * m2], order=3, mode="grid-wrap")
        return out.reshape(X.shape)


def axial_fourier(structure, uvw, ff, Ucart, oversample=2):
    """Axial (2D) potential of strings along [uvw]. ff: {el: FormFactor}, Ucart: {el: 3x3 tensor [A^2]}."""
    S = axial_setup(structure, uvw, Ucart, _qmax(ff, Ucart))
    e1, e2, G, G2, n1, n2, Omega = S.e1, S.e2, S.G, S.G2, S.n1, S.n2, S.Omega
    c_el = {el: ff[el](G2) * S.sfdw[el] for el in S.sfdw}
    c = sum(c_el.values())
    a1 = np.array([S.a1_3 @ e1, S.a1_3 @ e2])
    a2 = np.array([S.a2_3 @ e1, S.a2_3 @ e2])
    n1m, n2m = int(np.abs(n1).max()), int(np.abs(n2).max())

    # grid on one cell via FFT, oversampled (zero padding)
    m1 = next_fast_len(max(oversample * (2 * n1m + 1), 32))
    m2 = next_fast_len(max(oversample * (2 * n2m + 1), 32))
    C = np.zeros((m1, m2), complex)
    C[n1 % m1, n2 % m2] = c
    U = (m1 * m2 / Omega) * ifft2(C).real
    # exact transverse gradient (spectral derivative)
    Cx = np.zeros((m1, m2), complex)
    Cy = np.zeros((m1, m2), complex)
    Cx[n1 % m1, n2 % m2] = 1j * (G @ e1) * c
    Cy[n1 % m1, n2 % m2] = 1j * (G @ e2) * c
    Ux = (m1 * m2 / Omega) * ifft2(Cx).real
    Uy = (m1 * m2 / Omega) * ifft2(Cy).real
    emax = float(np.hypot(Ux, Uy).max())

    def direct(coeffs, r):
        """U at 3D Cartesian points r (P, 3) from coefficient vector `coeffs`."""
        r = np.atleast_2d(r)
        return (np.exp(1j * (G @ r.T)) * coeffs[:, None]).sum(axis=0).real / Omega

    # maximum: exact evaluation at the projected atomic strings; minimum: grid
    cart = S.cart
    proj = cart - np.outer(cart @ S.t_hat, S.t_hat)
    uniq = np.unique(np.round(proj, 4), axis=0)
    u_str = direct(c, uniq)
    k = int(np.argmax(u_str))
    u_max, r_max = float(u_str[k]), uniq[k]
    i1, i2 = np.unravel_index(np.argmin(U), U.shape)
    r_min = (i1 / m1) * S.a1_3 + (i2 / m2) * S.a2_3
    u_min = float(U.min())
    if u_str.max() < U.max():     # grid maximum off the strings (should not happen): keep the larger
        u_max = float(U.max())
    shares = {}
    tot = u_max - float(direct(c, r_min)[0])
    for el, ce in c_el.items():
        shares[el] = float((direct(ce, r_max)[0] - direct(ce, r_min)[0]) / tot)
    return AxialResult(U, a1, a2, emax, u_max, u_min, r_max, r_min, S.period, shares, (e1, e2), len(c))


@dataclass
class PlanarSetup:
    hkl: np.ndarray
    Omega: float
    nhat: np.ndarray
    gn: float                   # |g_hkl|
    d_cell: float               # 2 pi / |g_hkl|
    m: np.ndarray               # harmonic indices
    G: np.ndarray
    G2: np.ndarray
    sfdw: dict
    natoms: dict
    qmax: float


def planar_setup(structure, hkl, Ucart, qmax):
    hkl = np.asarray(hkl, int)
    L = structure.lattice.matrix
    recip = 2 * np.pi * np.linalg.inv(L).T
    g = hkl @ recip
    gn = np.linalg.norm(g)
    mm = int(np.ceil(qmax / gn))
    m = np.arange(-mm, mm + 1)
    G = m[:, None] * g
    G2 = (m * gn) ** 2
    cart = structure.cart_coords
    elements = [s.specie.symbol for s in structure]
    sfdw, natoms = {}, {}
    for el in sorted(set(elements)):
        idx = [i for i, e in enumerate(elements) if e == el]
        dw = np.exp(-0.5 * np.einsum("ki,ij,kj->k", G, Ucart[el], G))
        sfdw[el] = dw * np.exp(-1j * (G @ cart[idx].T)).sum(axis=1)
        natoms[el] = len(idx)
    return PlanarSetup(hkl, structure.volume, g / gn, gn, 2 * np.pi / gn, m, G, G2, sfdw, natoms, qmax)


def effective_period_factor(m, c, kmax=12):
    """Largest k <= kmax such that only harmonics m multiple of k are non-zero: the physical period is d/k
    (structure-factor extinctions, e.g. (110) of a diamond lattice)."""
    cm = np.abs(c)
    return max([1] + [k for k in range(2, kmax + 1) if np.all(cm[(m % k) != 0] < 1e-9 * cm.max())])


@dataclass
class PlanarResult:
    x: np.ndarray                # [A], two periods centred on the origin
    U: np.ndarray                # [eV], z = +1, minimum shifted to zero
    d: float                     # effective period of the plane family [A] (d_cell / k, see planar_fourier)
    emax: float                  # maximum field |dU/dx| [eV/A]
    shares: dict = field(default_factory=dict)
    n_g: int = 0

    @property
    def depth(self):
        return float(self.U.max() - self.U.min())


def planar_fourier(structure, hkl, ff, Ucart, n_x=2001):
    """Planar (1D) potential of the plane family (hkl) (coprime integer Miller indices)."""
    S = planar_setup(structure, hkl, Ucart, _qmax(ff, Ucart))
    m, gn, G2, Omega = S.m, S.gn, S.G2, S.Omega
    c_el = {el: ff[el](G2) * S.sfdw[el] for el in S.sfdw}
    c = sum(c_el.values())
    k_eff = effective_period_factor(m, c)
    d = S.d_cell / k_eff
    x = np.linspace(-d, d, n_x)
    ph = np.exp(1j * np.outer(m * gn, x))
    U = (c[:, None] * ph).sum(axis=0).real / Omega
    dU = ((1j * m * gn * c)[:, None] * ph).sum(axis=0).real / Omega
    one = (x >= 0) & (x < d)     # one period for shares and depth bookkeeping
    imax = np.where(one)[0][np.argmax(U[one])]
    imin = np.where(one)[0][np.argmin(U[one])]
    shares = {}
    tot = U[imax] - U[imin]
    for el, ce in c_el.items():
        Ue = (ce[:, None] * ph[:, [imax, imin]]).sum(axis=0).real / Omega
        shares[el] = float((Ue[0] - Ue[1]) / tot)
    return PlanarResult(x, U - U[one].min(), d, float(np.abs(dU).max()), shares, len(m))
