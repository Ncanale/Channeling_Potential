"""Position-dependent minimal ionisation energy T_min for the Geant4 FastSim data files.

In the FastSim model (G4VChannelingFastSimCrystalData::CoulombElectronScattering) T_min is the lower cut of
the energy transferred to an electron in single scattering: the rate of such events is proportional to
n_e(x) / T_min(x), while the transfers below the cut are in the 'amorphous' half of the Bethe formula
(IonizationLosses). Sytov, Tikhomirov, Bandiera, Phys. Rev. Accel. Beams 22, 064601 (2019),
DOI 10.1103/PhysRevAccelBeams.22.064601, and A. Sytov, PhD thesis (Ferrara, 2018, sect. 2.2), take it
equal to the Fermi energy at the point of scattering, estimated as "the average logarithm of the
interplanar or interaxial potential, averaged over the electron density", with the Thomas-Fermi relation
they cite (Landau-Lifshitz, QM sect. 70): the electrons of a neutral atom at distance r from its nucleus
have Fermi energy e phi(r), phi being the potential of that atom (zero at infinity).

'fermi' (AtomicFermiField, default): every electron gets the Fermi energy of its own atom,

    ln T_min(x) = < sum_j n_j(r) ln V_j(r) > / < sum_j n_j(r) >,

with n_j and V_j the electron density and the potential energy (unit positive charge) of atom j alone,
thermally smeared (Peng 1996 scattering factors, isotropic amplitude u_j; negative densities of the fit
set to zero), and < > the average along the string (axes) or over the plane (planes) at the transverse
position x. It reproduces the shipped G4CHANNELING 2.0 maps point by point (Si: mean of ln(ours/shipped)
within 0.012, rms 0.04-0.09, from 0.28 eV in the <110> channel to 770 eV on the strings; FASTSIM.md).
Signature of this definition, also present in the shipped files: the electron-weighted geometric mean of
T_min over the cell does not depend on the orientation and equals the atomic value
exp(int n ln V d3r / Z) (Si 81.9 eV, Ge 208 eV, W 724 eV at the shipped thermal amplitudes).

'fermi-crystal' (CrystalField3D): the same average with the total crystal potential V(r) (neighbours and
mean inner potential included) and the crystal electron density, ln T_min = <n_e ln V>/<n_e>. It agrees at
the atoms but is 2-7 times higher than the shipped maps in the channels, where the potential of the
neighbours dominates.

Both are periodic over the structure (CIF) cell: one 3D FFT, then periodic linear interpolation on a grid
of spacing ~ sigma_min / 2.5 (sigma_min: width of the sharpest thermally averaged Gaussian).
"""
import numpy as np
from pymatgen.core import Element
from scipy.ndimage import map_coordinates

from .form_factors import A0, E2, FormFactor

TINY = 1e-12


def _grid(structure, th, order, spacing, max_points):
    L = np.array(structure.lattice.matrix, float)                    # rows: lattice vectors [A]
    ff = {el: FormFactor(el, "peng") for el in order}
    U = {el: th[el].tensor() for el in order}
    if spacing is None:
        bmin = min(ff[el].b.min() / (4 * np.pi ** 2) + 2 * np.linalg.eigvalsh(U[el]).min() for el in order)
        spacing = max(np.sqrt(bmin / 2) / 2.5, 0.015)
    lens = np.linalg.norm(L, axis=1)
    N = np.maximum(16, np.ceil(lens / spacing)).astype(int)
    while N.prod() > max_points:                                     # memory guard (large cells)
        N = np.maximum(16, (N * 0.9).astype(int))
    N += N % 2
    return L, ff, U, N, float(spacing)


class _Periodic:
    """Periodic linear interpolation of two real-space arrays A, W on the structure-cell grid."""

    def sample(self, pts):
        shp = pts.shape[:-1]
        c = ((pts.reshape(-1, 3) @ self.Linv) * self.N).T
        a = map_coordinates(self.A, c, order=1, mode="grid-wrap").reshape(shp)
        w = map_coordinates(self.W, c, order=1, mode="grid-wrap").reshape(shp)
        return a, w


class CrystalField3D(_Periodic):
    """'fermi-crystal': V(r) [eV] and n_e(r) [A^-3] of the whole crystal; T = exp(<n_e ln V>/<n_e>)."""

    def __init__(self, structure, th, order, spacing=None, max_points=2.5e7):
        self.L, ff, U, N, self.spacing = _grid(structure, th, order, spacing, max_points)
        self.Linv, self.N = np.linalg.inv(self.L), N
        h, k, l = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in N), indexing="ij")
        B = 2 * np.pi * self.Linv.T                                  # reciprocal vectors (rows)
        Gx = h * B[0, 0] + k * B[1, 0] + l * B[2, 0]
        Gy = h * B[0, 1] + k * B[1, 1] + l * B[2, 1]
        Gz = h * B[0, 2] + k * B[1, 2] + l * B[2, 2]
        G2 = Gx ** 2 + Gy ** 2 + Gz ** 2
        Om = abs(np.linalg.det(self.L))
        VG = np.zeros(G2.shape, complex)
        nG = np.zeros(G2.shape, complex)
        for el in order:
            S = np.zeros(G2.shape, complex)
            for s in structure:
                if s.specie.symbol != el:
                    continue
                f = s.frac_coords
                S += np.exp(-2j * np.pi * (h * f[0] + k * f[1] + l * f[2]))
            Ue = U[el]
            DW = np.exp(-0.5 * (Ue[0, 0] * Gx * Gx + Ue[1, 1] * Gy * Gy + Ue[2, 2] * Gz * Gz
                                + 2 * (Ue[0, 1] * Gx * Gy + Ue[0, 2] * Gx * Gz + Ue[1, 2] * Gy * Gz)))
            Ut = ff[el](G2)                                          # eV A^3
            VG += Ut * DW * S
            nG += (Element(el).Z - G2 * Ut / (4 * np.pi * E2)) * DW * S
            del S, DW, Ut
        ntot = N.prod()
        self.A = self.V = np.real(np.fft.ifftn(VG)) * ntot / Om
        self.W = self.ne = np.real(np.fft.ifftn(nG)) * ntot / Om

    def tmin(self, pts):
        """T_min [eV] of the lines (or planes) of points pts (n, m, 3), averaged over the m points."""
        V, ne = self.sample(pts)
        w = np.clip(ne, TINY, None)
        return np.exp((w * np.log(np.clip(V, TINY, None))).sum(1) / w.sum(1))


def atom_tf(ff, u, r):
    """Electron density n(r) [A^-3] (negative values of the fit set to 0) and potential energy V(r) [eV]
    of a unit positive charge, for one atom thermally smeared with the isotropic amplitude u [A].
    Peng Gaussians: U~(q) = 2 pi e^2 a0 sum a_i exp(-b_i q^2/16pi^2); n from Poisson,
    n = n_nucleus + lap(V) / (4 pi e^2)."""
    V = np.zeros_like(r)
    lap = np.zeros_like(r)
    for a, b in zip(ff.a, ff.b):
        al = b / (16 * np.pi ** 2) + u * u / 2                        # U~ ~ exp(-al q^2)
        g = 2 * np.pi * E2 * A0 * a * (4 * np.pi * al) ** -1.5 * np.exp(-r * r / (4 * al))
        V += g
        lap += g * (r * r / (4 * al * al) - 3 / (2 * al))
    n = ff.Z * (2 * np.pi * u * u) ** -1.5 * np.exp(-r * r / (2 * u * u)) + lap / (4 * np.pi * E2)
    return np.clip(n, 0.0, None), V


def _hankel(f, r, q, block=256):
    """3D Fourier transform of a radial function on a uniform grid r: 4 pi int f r^2 j0(q r) dr."""
    dr = r[1] - r[0]
    out = np.empty(len(q))
    fr2 = f * r * r
    for i in range(0, len(q), block):
        out[i:i + block] = 4 * np.pi * dr * (np.sinc(np.outer(q[i:i + block], r) / np.pi) * fr2).sum(1)
    return out


def atom_radial_grid(ff, u):
    al_max = ff.b.max() / (16 * np.pi ** 2) + u * u / 2
    al_min = ff.b.min() / (16 * np.pi ** 2) + u * u / 2
    dr = min(u, np.sqrt(2 * al_min)) / 40
    R = np.sqrt(160 * al_max)                                        # Gaussian tails below e^-40
    return np.arange(0.0, R + dr, dr)


def atomic_fermi_mean(element, u):
    """exp( int n ln V d3r / int n d3r ) [eV] of one atom (thermal amplitude u [A]): the electron-weighted
    geometric mean of T_min over any cell of a crystal of this element ('fermi' definition)."""
    ff = FormFactor(element, "peng")
    r = atom_radial_grid(ff, u)
    n, V = atom_tf(ff, u, r)
    w = r * r * n
    return float(np.exp((w * np.log(np.clip(V, 1e-300, None))).sum() / w.sum()))


class AtomicFermiField(_Periodic):
    """'fermi': Q(r) = sum_j n_j ln V_j [eV-log A^-3] and W(r) = sum_j n_j [A^-3]; T = exp(<Q>/<W>)."""

    def __init__(self, structure, th, order, spacing=None, max_points=2.5e7):
        self.L, ff, U, N, self.spacing = _grid(structure, th, order, spacing, max_points)
        self.Linv, self.N = np.linalg.inv(self.L), N
        f0, f1, f2 = (np.fft.fftfreq(N[0], 1.0 / N[0]), np.fft.fftfreq(N[1], 1.0 / N[1]),
                      np.fft.rfftfreq(N[2], 1.0 / N[2]))
        h, k, l = f0[:, None, None], f1[None, :, None], f2[None, None, :]
        B = 2 * np.pi * self.Linv.T
        Gm = np.sqrt(sum((h * B[0, c] + k * B[1, c] + l * B[2, c]) ** 2 for c in range(3)))
        qt = np.linspace(0.0, Gm.max() * 1.0001, 8192)
        QG = np.zeros(Gm.shape, complex)
        WG = np.zeros(Gm.shape, complex)
        for el in order:
            u = float(np.sqrt(np.trace(U[el]) / 3))                    # isotropic equivalent amplitude
            r = atom_radial_grid(ff[el], u)
            n, V = atom_tf(ff[el], u, r)
            q = n * np.log(np.clip(V, 1e-300, None))
            S = 0
            for s in structure:
                if s.specie.symbol == el:
                    fc = s.frac_coords
                    S = S + (np.exp(-2j * np.pi * f0 * fc[0])[:, None, None] * np.exp(-2j * np.pi * f1 * fc[1])[None, :, None]
                             * np.exp(-2j * np.pi * f2 * fc[2])[None, None, :])
            QG += np.interp(Gm, qt, _hankel(q, r, qt)) * S
            WG += np.interp(Gm, qt, _hankel(n, r, qt)) * S
            del S
        scale = N.prod() / abs(np.linalg.det(self.L))
        self.A = self.Q = np.fft.irfftn(QG, s=tuple(N), axes=(0, 1, 2)) * scale
        self.W = np.fft.irfftn(WG, s=tuple(N), axes=(0, 1, 2)) * scale

    def tmin(self, pts):
        """T_min [eV] of the lines (or planes) of points pts (n, m, 3), averaged over the m points."""
        Q, W = self.sample(pts)
        return np.exp(Q.sum(1) / np.clip(W.sum(1), TINY, None))


def _lattice_vectors(L, M=4):
    m = np.arange(-M, M + 1)
    i, j, k = np.meshgrid(m, m, m, indexing="ij")
    v = np.c_[i.ravel(), j.ravel(), k.ravel()] @ L
    n = np.linalg.norm(v, axis=1)
    return v[n > 1e-9], n[n > 1e-9]


def period_along(L, t_hat):
    """Shortest lattice vector length along t_hat [A]."""
    v, n = _lattice_vectors(L)
    par = np.abs(np.abs(v @ t_hat) - n) < 1e-6 * n
    return float(n[par].min())


def inplane_cell(L, n_hat):
    """Two lattice vectors spanning the in-plane lattice perpendicular to n_hat (minimal area)."""
    v, n = _lattice_vectors(L)
    inp = np.abs(v @ n_hat) < 1e-6 * n
    v, n = v[inp], n[inp]
    order = np.argsort(n)
    v1 = v[order[0]]
    best = None
    for idx in order[1:]:
        a = np.linalg.norm(np.cross(v1, v[idx]))
        if a > 1e-6 and (best is None or a < best[0] - 1e-9):
            best = (a, v[idx])
    return v1, best[1]


def emin_axial(F, origin, ex3, ey3, t_hat, xs, ys, chunk=2.0e6):
    """T_min [eV] on nodes (len(xs), len(ys)); node r = origin + x ex3 + y ey3, averaged along t_hat."""
    Lz = period_along(F.L, t_hat)
    nz = max(64, int(np.ceil(Lz / F.spacing)))
    z = (np.arange(nz) + 0.5) / nz * Lz
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    base = origin + X.ravel()[:, None] * ex3 + Y.ravel()[:, None] * ey3
    out = np.empty(len(base))
    step = max(1, int(chunk // nz))
    for s in range(0, len(base), step):
        out[s:s + step] = F.tmin(base[s:s + step, None, :] + z[None, :, None] * t_hat)
    return out.reshape(X.shape)


def emin_planar(F, n_hat, xs, chunk=2.0e6):
    """T_min [eV] at positions x along n_hat (node r = x n_hat), averaged over the plane."""
    v1, v2 = inplane_cell(F.L, n_hat)
    n1 = max(32, int(np.ceil(np.linalg.norm(v1) / F.spacing)))
    n2 = max(32, int(np.ceil(np.linalg.norm(v2) / F.spacing)))
    s1, s2 = np.meshgrid((np.arange(n1) + 0.5) / n1, (np.arange(n2) + 0.5) / n2, indexing="ij")
    plane = s1.ravel()[:, None] * v1 + s2.ravel()[:, None] * v2
    out = np.empty(len(xs))
    step = max(1, int(chunk // len(plane)))
    for s in range(0, len(xs), step):
        out[s:s + step] = F.tmin(np.asarray(xs[s:s + step])[:, None, None] * n_hat + plane[None])
    return out


FIELDS = {"fermi": AtomicFermiField, "fermi-crystal": CrystalField3D}
