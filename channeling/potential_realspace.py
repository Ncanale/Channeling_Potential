"""Real-space Doyle-Turner/Peng strings and planes (reference method).

Each atom of species j is a Gaussian sum thermally smeared with the isotropic 1D amplitude u_j:

  axial  U(rho) = sum_strings lam * 2 e^2 a0 sum_i a_i/s_i exp(-|rho - rho_s|^2/s_i)
  planar U(x)   = sum_planes sigma * 2 sqrt(pi) e^2 a0 sum_i a_i/sqrt(s_i) exp(-(x-x_p)^2/s_i)

with s_i = B_i + 2 u_j^2, B_i = b_i/(4 pi^2) (Peng coefficients), lam = atoms per unit length of the
string and sigma = atoms per unit area of the plane, both computed exactly from the cell.
The derivation: the 3D potential U(r) = 2 pi e^2 a0 sum_i a_i (pi s_i)^(-3/2) exp(-r^2/s_i)
integrated along the string (axial) or over the plane (planar).
Atoms are summed in a cylinder (axial) or over plane images (planar) larger than the plotted window
by TAIL, so there are no edge artefacts inside the window. Isotropic u only: use the Fourier method
for anisotropic displacement tensors and for the Moliere model.
"""
import numpy as np

from .form_factors import A0, E2, peng_table
from .structure import plane_frame

TAIL = 6.0   # [A] distance beyond which an atom's contribution is negligible (Gaussians, b <= 140)


def _coeff(el, u):
    tab = peng_table()[el]
    a = np.array(tab["a"])
    s2 = np.array(tab["b"]) / (4 * np.pi ** 2) + 2 * u ** 2
    return a, s2


def _U_string(r2, a, s2, lam):
    return 2 * E2 * A0 * lam * sum(a[i] / s2[i] * np.exp(-r2 / s2[i]) for i in range(len(a)))


def _U_plane(x, a, s2, sigma):
    return 2 * np.sqrt(np.pi) * E2 * A0 * sigma * sum(a[i] / np.sqrt(s2[i]) * np.exp(-x ** 2 / s2[i])
                                                       for i in range(len(a)))


def axial_realspace(structure, uvw, u, window=None, n=401):
    """Return dict(xx, yy, U, depth, period). u: {element: isotropic u [A]}."""
    uvw = np.asarray(uvw, int)
    L = structure.lattice.matrix
    frac = structure.frac_coords % 1.0
    elements = [s.specie.symbol for s in structure]
    coeff = {el: _coeff(el, u[el]) for el in set(elements)}
    t_vec = uvw @ L
    T = np.linalg.norm(t_vec)
    t_hat = t_vec / T
    e1, e2 = plane_frame(t_hat)
    W = window or 0.75 * max(structure.lattice.abc)
    R = np.sqrt(2) * W + TAIL
    n_max = int(np.ceil((R + T) / min(np.linalg.norm(L, axis=1)))) + 1
    rng = np.arange(-n_max, n_max + 1)
    shifts = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    strings = {}                                  # (element, x, y) -> atoms in one period T
    for f, el in zip(frac, elements):
        r = (f + shifts) @ L
        z = r @ t_hat
        keep = (z >= -1e-6) & (z < T - 1e-6)
        rx, ry = r[keep] @ e1, r[keep] @ e2
        near = rx ** 2 + ry ** 2 <= R ** 2
        for x, y in zip(rx[near], ry[near]):
            key = (el, round(x, 3) + 0.0, round(y, 3) + 0.0)
            strings[key] = strings.get(key, 0) + 1
    grid = np.linspace(-W, W, n)
    xx, yy = np.meshgrid(grid, grid)
    U = np.zeros_like(xx)
    for (el, x0, y0), cnt in strings.items():
        U += _U_string((xx - x0) ** 2 + (yy - y0) ** 2, *coeff[el], cnt / T)
    # the grid may miss the exact string positions: evaluate the maximum there
    on_window = [(x0, y0) for (_, x0, y0) in strings if abs(x0) <= W and abs(y0) <= W]
    at_str = [sum(_U_string((x0 - x1) ** 2 + (y0 - y1) ** 2, *coeff[el], c / T) for (el, x1, y1), c in strings.items())
              for x0, y0 in on_window]
    u_max = max([U.max()] + at_str)
    return dict(xx=xx, yy=yy, U=U - U.min(), depth=float(u_max - U.min()), period=T)


def planar_realspace(structure, hkl, u, n=2001):
    """Return dict(x, U, d, depth). Plane family (hkl); u: {element: isotropic u [A]}."""
    hkl = np.asarray(hkl, int)
    L = structure.lattice.matrix
    frac = structure.frac_coords % 1.0
    elements = [s.specie.symbol for s in structure]
    coeff = {el: _coeff(el, u[el]) for el in set(elements)}
    G = hkl @ np.linalg.inv(L).T
    d = 1 / np.linalg.norm(G)
    planes = {}
    for f, el in zip(frac, elements):
        xpos = round(((f @ hkl) % 1.0) * d, 4) % round(d, 4)
        planes[(el, xpos)] = planes.get((el, xpos), 0) + 1
    x = np.linspace(-d, d, n)
    m_max = int(np.ceil(TAIL / d)) + 2
    U = np.zeros_like(x)
    for (el, x0), cnt in planes.items():
        sigma = cnt / structure.volume * d
        for m in range(-m_max, m_max + 1):
            U += _U_plane(x - x0 - m * d, *coeff[el], sigma)
    return dict(x=x, U=U - U.min(), d=d, depth=float(U.max() - U.min()))
