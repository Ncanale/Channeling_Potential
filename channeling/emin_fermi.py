"""Position-dependent minimal ionisation energy for the Geant4 FastSim data files ('fermi' mode).

In the FastSim model (G4VChannelingFastSimCrystalData::CoulombElectronScattering) this function is
T_min, the lower cut of the energy transferred to an electron in single scattering: the rate of such
events is proportional to n_e(x) / T_min(x). Sytov, Tikhomirov, Bandiera, Phys. Rev. Accel. Beams 22,
064601 (2019), DOI 10.1103/PhysRevAccelBeams.22.064601, and A. Sytov, PhD thesis (Ferrara, 2018,
sect. 2.2), define it as the Fermi energy at the point of scattering, estimated (Thomas-Fermi,
Landau-Lifshitz QM sect. 70) as "the average logarithm of the potential, averaged over the electron
density". The formula itself is not published; we implement the literal reading

    T_min(x_perp) = exp( < n_e(r) ln V(r) > / < n_e(r) > ),

where V(r) is the full 3D crystal potential energy of a unit positive charge (thermally averaged atoms,
Peng 1996 scattering factors, mean inner potential included, so V > 0), n_e(r) the 3D electron density
(Poisson), and < > the average over the coordinates along the string (axes) or over the plane (planes)
at the transverse position x_perp. Against the shipped G4CHANNELING 2.0 Si files this reproduces the
values at the atoms (300-800 eV) within about 10 %, and the channel values within a factor of about 2
(the shipped ones are lower); see FASTSIM.md. With it the Geant4 rocking curves agree with those of
the shipped files, whereas the constant (amorphous) cut gives 2-5 % more channeling at the shoulders.

The 3D fields come from one FFT on the structure (CIF) cell and are sampled by periodic linear
interpolation on a grid of spacing ~ sigma_min / 2.5 (sigma_min: width of the sharpest thermally
averaged Gaussian of the atomic potential).
"""
import numpy as np
from pymatgen.core import Element
from scipy.ndimage import map_coordinates

from .form_factors import A0, E2, FormFactor

TINY = 1e-12


class CrystalField3D:
    """V(r) [eV] and n_e(r) [A^-3] of the whole crystal, periodic over the structure cell."""

    def __init__(self, structure, th, order, spacing=None, max_points=2.5e7):
        self.L = np.array(structure.lattice.matrix, float)          # rows: lattice vectors [A]
        self.Linv = np.linalg.inv(self.L)
        ff = {el: FormFactor(el, "peng") for el in order}
        U = {el: th[el].tensor() for el in order}
        if spacing is None:
            bmin = min(ff[el].b.min() / (4 * np.pi ** 2) + 2 * np.linalg.eigvalsh(U[el]).min() for el in order)
            spacing = max(np.sqrt(bmin / 2) / 2.5, 0.015)
        lens = np.linalg.norm(self.L, axis=1)
        N = np.maximum(16, np.ceil(lens / spacing)).astype(int)
        while N.prod() > max_points:                                 # memory guard (large cells)
            N = np.maximum(16, (N * 0.9).astype(int))
        N += N % 2
        self.N, self.spacing = N, float(spacing)
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
        self.V = np.real(np.fft.ifftn(VG)) * ntot / Om
        self.ne = np.real(np.fft.ifftn(nG)) * ntot / Om

    def sample(self, pts):
        """V and n_e at Cartesian points (..., 3)."""
        shp = pts.shape[:-1]
        fr = pts.reshape(-1, 3) @ self.Linv
        c = (fr * self.N).T
        V = map_coordinates(self.V, c, order=1, mode="grid-wrap").reshape(shp)
        ne = map_coordinates(self.ne, c, order=1, mode="grid-wrap").reshape(shp)
        return V, ne


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


def _log_average(V, ne, axis):
    w = np.clip(ne, TINY, None)
    return np.exp((w * np.log(np.clip(V, TINY, None))).sum(axis) / w.sum(axis))


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
        pts = base[s:s + step, None, :] + z[None, :, None] * t_hat
        V, ne = F.sample(pts)
        out[s:s + step] = _log_average(V, ne, axis=1)
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
        pts = np.asarray(xs[s:s + step])[:, None, None] * n_hat + plane[None]
        V, ne = F.sample(pts)
        out[s:s + step] = _log_average(V, ne, axis=1)
    return out
