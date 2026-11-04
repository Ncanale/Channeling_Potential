"""Reader, writer and spline coefficients of the Geant4 channeling FastSim data files.

The files are read by G4ChannelingFastSimCrystalData::SetMaterialProperties (Geant4,
source/parameterisations/channeling; dataset G4CHANNELING 2.0, README of the dataset). File name:
    <material>_planes_<hkl>.dat   or   <material>_axes_<uvw>.dat        e.g. Si_planes_110.dat, W_axes_111.dat
where <material> is the Geant4 material name without its first three characters ("G4_Si" -> "Si").

Layout (whitespace separated numbers, lengths in cm, energies in eV, densities in cm^-3)
    N0[1..nel]   atomic concentration of each element
    u[1..nel]    1D rms thermal amplitude of each element
    planar: dx  npx            axial: dx dy  npx npy        (period and number of intervals)
    Vmax         height of the potential well (0 at the minimum)
    Vmin         minimal potential inside the crystal (0 = vacuum), written for planes, 0 for axes
    then cubic-spline coefficients of
      planar, for every interval i = 0..npx-1:  4 numbers (a, b, c, d) for each of
          electric field Ex [eV/cm], nuclear density of element 0 (normalised to its mean),
          electron density [cm^-3], minimal ionisation energy [eV], nuclear density of elements 1..
          p(s) = a + b s + c s^2 + d s^3 with s = x - x_{i+1} <= 0 [cm]  (a = value at the right node)
      axial, for j = 0..npy-1, i = 0..npx, k = 0, 1: 3 numbers (A, B, C) for each of
          minimal ionisation energy, Ex, Ey, nuclear density of element 0, electron density,
          nuclear density of elements 1..
          (tensor-product cubic spline; k = 0: spline in y of f(x_i, y), k = 1: of d^2f/dx^2 at x_i)
In both cases the coordinates run over one periodic cell, x in [0, dx) (and y in [0, dy)).

Coefficient definitions of the axial case, derived from G4ChannelingFastSimInterpolation::Spline2D /
SetCoefficients2D with hx = dx/npx, hy = dy/npy, for a function f sampled on the nodes (i, j):
    k = 0:  A = f_ij - hy^2/6 f_yy,ij      B = f_i,j+1 - hy^2/6 f_yy,i,j+1      C = f_yy,ij / 6
    k = 1:  A = f_xx,ij - hy^2/6 f_xxyy,ij  B = f_xx,i,j+1 - hy^2/6 f_xxyy,i,j+1  C = f_xxyy,ij / 6
where f_xx are the second derivatives of the x cubic spline (periodic) through f at fixed y_j, f_yy those
of the y spline (natural: the Geant4 constructor sets the coefficient C beyond the last node to 0, and
the shipped files have C = 0 at the first node) and f_xxyy the y-spline second derivatives of f_xx.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

# quantity indices
PLANAR = {"Ex": 0, "nuc0": 1, "el": 2, "emin": 3}
AXIAL = {"emin": 0, "Ex": 1, "Ey": 2, "nuc0": 3, "el": 4}


@dataclass
class FastSimData:
    model: int                 # 1 planes, 2 axes
    nel: int
    N0: np.ndarray             # [cm^-3]
    u: np.ndarray              # [cm]
    dx: float                  # [cm]
    dy: float                  # [cm] (0 for planes)
    npx: int
    npy: int
    vmax: float                # [eV]
    vmin: float                # [eV]
    coef: np.ndarray           # planar: (npx, 4*(3+nel)); axial: (npy, npx+1, 2, 3*(4+nel))

    def index(self, name):
        table = PLANAR if self.model == 1 else AXIAL
        if name in table:
            return table[name]
        if name.startswith("nuc"):
            i = int(name[3:])
            return (3 if self.model == 1 else 4) + i if i > 0 else table["nuc0"]
        raise KeyError(name)


def infer_nel(n_tokens, model):
    for nel in range(1, 12):
        if model == 1:
            head = 2 * nel + 4
            per = 4 * (3 + nel)
            if (n_tokens - head) > 0 and (n_tokens - head) % per == 0:
                return nel
        else:
            head = 2 * nel + 6
            per = 6 * (4 + nel)
            if (n_tokens - head) > 0 and (n_tokens - head) % per == 0:
                return nel
    raise ValueError("cannot infer the number of elements: file does not match the FastSim layout")


def read(path, model=None, nel=None):
    """Read a FastSim data file. model: 1 planes / 2 axes (default from the file name)."""
    t = np.array(open(path).read().split(), float)
    if model is None:
        model = 1 if "_planes_" in str(path) else 2
    nel = nel or infer_nel(len(t), model)
    N0, u = t[:nel], t[nel:2 * nel]
    p = 2 * nel
    if model == 1:
        dx, npx = t[p], int(t[p + 1])
        dy, npy = 0.0, 0
        vmax, vmin = t[p + 2], t[p + 3]
        coef = t[p + 4:].reshape(npx, 4 * (3 + nel))
    else:
        dx, dy = t[p], t[p + 1]
        npx, npy = int(t[p + 2]), int(t[p + 3])
        vmax, vmin = t[p + 4], t[p + 5]
        coef = t[p + 6:].reshape(npy, npx + 1, 2, 3 * (4 + nel))
    return FastSimData(model, nel, N0, u, dx, dy, npx, npy, vmax, vmin, coef)


def evaluate(d, name, x, y=None):
    """Evaluate a stored function (Python port of G4ChannelingFastSimInterpolation::GetIF), cm units."""
    q = d.index(name)
    x = np.mod(np.asarray(x, float), d.dx)
    hx = d.dx / d.npx
    if d.model == 1:
        i = np.minimum((x // hx).astype(int), d.npx - 1)
        s = x - (i + 1) * hx
        a, b, c, dd = (d.coef[i, 4 * q + m] for m in range(4))
        return a + s * (b + s * (c + dd * s))
    y = np.mod(np.asarray(y, float), d.dy)
    hy = d.dy / d.npy
    mx = np.minimum((x // hx).astype(int), d.npx - 1)
    my = np.minimum((y // hy).astype(int), d.npy - 1)
    t1, t2 = x - mx * hx, (mx + 1) * hx - x
    u1, u2 = y - my * hy, (my + 1) * hy - y
    k = 3 * q
    f = d.coef

    def ysp(kk, ix):
        """y-spline of the k-th group at x node ix (C at node npy is 0: natural boundary)."""
        A, B, C = (f[my, ix, kk, k + m] for m in range(3))
        Cn = np.where(my + 1 < d.npy, f[np.minimum(my + 1, d.npy - 1), ix, kk, k + 2], 0.0)
        s = 6 * hx * hy
        return (C / s) * u2 ** 3 + (Cn / s) * u1 ** 3 + (A / s) * u2 + (B / s) * u1

    g1, g2 = ysp(0, mx), ysp(0, mx + 1)
    h1, h2 = ysp(1, mx), ysp(1, mx + 1)
    return (h1 * t2 ** 3 + h2 * t1 ** 3 + (6 * g1 - h1 * hx ** 2) * t2 + (6 * g2 - h2 * hx ** 2) * t1)


# ---------------------------------------------------------------- coefficient generation
def spline_planar(values):
    """Cubic coefficients (a, b, c, d) per interval about the right node, from node values f_0..f_npx
    (f_npx = f_0, periodic). Returns array (npx, 4) with s = x - x_{i+1} in units of the step h=1."""
    n = len(values) - 1
    cs = CubicSpline(np.arange(n + 1), values, bc_type="periodic")
    c3, c2, c1, c0 = cs.c                                   # p(t) = c3 t^3 + c2 t^2 + c1 t + c0, t = x - x_i
    # shift to the right node: t = s + 1
    a = c3 + c2 + c1 + c0
    b = 3 * c3 + 2 * c2 + c1
    c = 3 * c3 + c2
    return np.stack([a, b, c, c3], axis=1)


def planar_block(values, h):
    """Coefficients (npx, 4) in physical units: s in cm, f in the unit of `values`; h = step [cm]."""
    co = spline_planar(values)
    scale = np.array([1.0, 1.0 / h, 1.0 / h ** 2, 1.0 / h ** 3])
    return co * scale


def axial_block(f, hx, hy):
    """Coefficients for a function sampled on nodes f[i, j], i = 0..npx (f[npx] = f[0]), j = 0..npy.
    Returns array (npy, npx+1, 2, 3) of (A, B, C) for k = 0, 1 (units: f in file units, lengths in cm)."""
    npx, npy = f.shape[0] - 1, f.shape[1] - 1
    x = np.arange(npx + 1) * hx
    y = np.arange(npy + 1) * hy
    fxx = CubicSpline(x, f, axis=0, bc_type="periodic")(x, 2)
    fyy = CubicSpline(y, f, axis=1, bc_type="natural")(y, 2)
    fxxyy = CubicSpline(y, fxx, axis=1, bc_type="natural")(y, 2)
    out = np.zeros((npy, npx + 1, 2, 3))
    for k, (g, gyy) in enumerate(((f, fyy), (fxx, fxxyy))):
        A = g - hy ** 2 / 6 * gyy
        out[:, :, k, 0] = A[:, :npy].T
        out[:, :, k, 1] = A[:, 1:npy + 1].T
        out[:, :, k, 2] = (gyy / 6)[:, :npy].T
    return out


def fmt(v):
    return f"{v:.9E}"


def write(path, d_planar_or_axial):
    """Write a FastSimData object in the Geant4 layout."""
    d = d_planar_or_axial
    lines = [fmt(v) for v in d.N0] + [fmt(v) for v in d.u]
    if d.model == 1:
        lines += [fmt(d.dx), str(d.npx), fmt(d.vmax), fmt(d.vmin)]
        body = [" ".join(fmt(v) for v in row) for row in d.coef]
    else:
        lines += [fmt(d.dx), fmt(d.dy), str(d.npx), str(d.npy), fmt(d.vmax), fmt(d.vmin)]
        body = []
        for j in range(d.npy):
            for i in range(d.npx + 1):
                for k in range(2):
                    body.append(" ".join(fmt(v) for v in d.coef[j, i, k]))
    with open(path, "w") as f:
        f.write("\n".join(lines + body) + "\n")
