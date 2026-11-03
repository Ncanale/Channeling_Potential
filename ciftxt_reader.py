"""Axial (2D) and planar (1D) continuum channeling potentials from a CIF file.

Doyle-Turner-type model: each atom is described by the five-Gaussian electron
scattering factor of Peng et al. (1996) (peng1996.json), thermally smeared with
the 1D rms vibration amplitude u, and averaged along atomic strings / planes:

  axial  U(rho) = sum_j  lambda_j * 2 e^2 a0 sum_i a_i/(B_i+2u^2) exp(-|rho-rho_j|^2/(B_i+2u^2))
  planar U(x)   = sum_j sigma_j * 2 sqrt(pi) e^2 a0 sum_i a_i/sqrt(B_i+2u^2) exp(-(x-x_j)^2/(B_i+2u^2))

with B_i = b_i/(4 pi^2), lambda_j = atoms per unit length of string j and
sigma_j = atoms per unit area of plane j. Works for any lattice and for
polyatomic crystals; string/plane densities are computed exactly from the cell.

Usage:
    python ciftxt_reader.py [material] [axis] [--plane "h k l"] [--outdir DIR] [--u El=val ...] [--show]
    python ciftxt_reader.py Si "1 1 0"
    python ciftxt_reader.py PbWO4 "0 0 1" --outdir output/PbWO4/001
The CIF is read from <material>.cif. For cubic crystals the plane defaults to
the same indices as the axis; give --plane otherwise.
"""
import argparse
import json
import os
from math import gcd

import numpy as np
import matplotlib

E2 = 14.39964   # e^2 [eV*A] (Gaussian units)
A0 = 0.529177   # Bohr radius [A]; e^2*a0 = hbar^2/m_e
TAIL = 6.0      # [A] distance beyond which an atom's contribution is negligible

# 1D rms thermal vibration amplitudes at room temperature [A]
U_THERMAL = {
    "Si": 0.075,  # Sytov thesis / Biryukov
    "Ge": 0.085,  # Baier-Katkov
    "C": 0.040,   # diamond, Biryukov
    "W": 0.050,   # Biryukov
    # PbWO4: approximate, from room-temperature isotropic displacement parameters
    "Pb": 0.094,
    "O": 0.110,
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("material", nargs="?", default="Si", help="material name; reads <material>.cif")
parser.add_argument("axis", nargs="?", default="1 0 0", help='crystal axis [u v w] as "u v w"')
parser.add_argument("--plane", default=None, help='plane (h k l); default: same indices as the axis')
parser.add_argument("--outdir", default=".", help="where to save the PDF plots")
parser.add_argument("--u", nargs="*", default=[], metavar="El=val", help="override thermal amplitudes, e.g. --u W=0.05")
parser.add_argument("--show", action="store_true", help="also open interactive plot windows")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "peng1996.json")) as f:
    PENG = json.load(f)
for item in args.u:
    el, val = item.split("=")
    U_THERMAL[el] = float(val)


def parse_indices(text):
    v = np.array([int(t) for t in text.split()])
    if v.shape != (3,) or not v.any():
        raise SystemExit(f'Invalid indices "{text}": expected three integers, not all zero')
    return v // gcd(gcd(*map(abs, v[:2])), abs(v[2]))


uvw = parse_indices(args.axis)
hkl = parse_indices(args.plane) if args.plane else uvw
label_ax = "".join(map(str, uvw))
label_pl = "".join(map(str, hkl))
os.makedirs(args.outdir, exist_ok=True)

# ----------------------------------------------------------------- structure
structure = Structure.from_file(os.path.join(here, args.material + ".cif"))
L = structure.lattice.matrix             # rows: a, b, c [A]
frac = structure.frac_coords % 1.0
elements = [site.specie.symbol for site in structure]
V_cell = structure.volume
print(f"{args.material}: {structure.composition.reduced_formula}, {len(structure)} atoms per cell, "
      f"V = {V_cell:.3f} A^3, N = {len(structure) / V_cell:.4e} A^-3")

species = sorted(set(elements))
coeff = {}
for el in species:
    if el not in PENG:
        raise SystemExit(f"No scattering factor parameters for {el}")
    if el not in U_THERMAL:
        raise SystemExit(f"No thermal amplitude for {el}: pass --u {el}=<value in A>")
    a = np.array(PENG[el]["a"])
    s2 = np.array(PENG[el]["b"]) / (4 * np.pi ** 2) + 2 * U_THERMAL[el] ** 2   # B_i + 2u^2
    coeff[el] = (a, s2)
    print(f"  {el}: u = {U_THERMAL[el]} A, f_e(0) = {a.sum():.3f} A")


def U_string(r2, el, lam):
    """Axial potential [eV] of a string of element el with lam atoms/A, at squared distance r2."""
    a, s2 = coeff[el]
    return 2 * E2 * A0 * lam * sum(a[i] / s2[i] * np.exp(-r2 / s2[i]) for i in range(len(a)))


def U_plane(x, el, sigma):
    """Planar potential [eV] of a plane of element el with sigma atoms/A^2, at distance x."""
    a, s2 = coeff[el]
    return 2 * np.sqrt(np.pi) * E2 * A0 * sigma * sum(a[i] / np.sqrt(s2[i]) * np.exp(-x ** 2 / s2[i])
                                                       for i in range(len(a)))


# ================================================================= AXIAL
t_vec = uvw @ L                          # lattice translation along the axis
T = np.linalg.norm(t_vec)                # string period [A]
t_hat = t_vec / T
e1 = np.cross(t_hat, [0, 0, 1] if abs(t_hat[2]) < 0.9 else [1, 0, 0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(t_hat, e1)

W = 0.75 * max(structure.lattice.abc)    # half-width of the plotted window
R = np.sqrt(2) * W + TAIL                # radius of the cylinder of atoms used
n_max = int(np.ceil((R + T) / min(np.linalg.norm(L, axis=1)))) + 1
rng = np.arange(-n_max, n_max + 1)
shifts = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T

strings = {}                             # (element, x, y) -> atoms in one period T
for f, el in zip(frac, elements):
    r = (f + shifts) @ L
    z = r @ t_hat
    keep = (z >= -1e-6) & (z < T - 1e-6)
    rx, ry = r[keep] @ e1, r[keep] @ e2
    near = rx ** 2 + ry ** 2 <= R ** 2
    for x, y in zip(rx[near], ry[near]):
        key = (el, round(x, 3) + 0.0, round(y, 3) + 0.0)
        strings[key] = strings.get(key, 0) + 1

grid = np.linspace(-W, W, 401)
xx, yy = np.meshgrid(grid, grid)
U_2D = np.zeros_like(xx)
for (el, x0, y0), count in strings.items():
    U_2D += U_string((xx - x0) ** 2 + (yy - y0) ** 2, el, count / T)

# the grid may miss the exact string positions: evaluate the maximum there
on_window = [(x0, y0) for (_, x0, y0) in strings if abs(x0) <= W and abs(y0) <= W]
U_at_strings = [sum(U_string((x0 - x1) ** 2 + (y0 - y1) ** 2, el, c / T) for (el, x1, y1), c in strings.items())
                for x0, y0 in on_window]
U_min = U_2D.min()
U_max = max([U_2D.max()] + U_at_strings)
axial_depth = U_max - U_min
n_str = len({(x, y) for (_, x, y) in strings})
print(f"axial <{args.axis}>: period T = {T:.3f} A, depth U_max - U_min = {axial_depth:.2f} eV")

fig, ax = plt.subplots()
cs = ax.contourf(xx, yy, U_2D - U_min, np.linspace(0, U_2D.max() - U_min, 150), cmap="jet")
fig.colorbar(cs, label="U, eV")
ax.set_title(f"{args.material} <{label_ax}>   depth {axial_depth:.1f} eV")
ax.set_xlabel(r"x, $\AA$")
ax.set_ylabel(r"y, $\AA$")
ax.set_aspect("equal")
fig.savefig(os.path.join(args.outdir, f"{args.material}{label_ax}_axial.pdf"))

# ================================================================= PLANAR
G = hkl @ np.linalg.inv(L).T             # reciprocal lattice vector (no 2 pi)
d = 1 / np.linalg.norm(G)                # interplanar period [A]
planes = {}                              # (element, position in [0, d)) -> atoms per cell
for f, el in zip(frac, elements):
    xpos = round(((f @ hkl) % 1.0) * d, 4) % round(d, 4)
    planes[(el, xpos)] = planes.get((el, xpos), 0) + 1

x_pl = np.linspace(-d, d, 2001)          # two periods, centred on a plane
m_max = int(np.ceil(TAIL / d)) + 2
U_1D = np.zeros_like(x_pl)
for (el, x0), count in planes.items():
    sigma = count / V_cell * d           # atoms per A^2 in each plane of this family
    for m in range(-m_max, m_max + 1):
        U_1D += U_plane(x_pl - x0 - m * d, el, sigma)
U_1D -= U_1D.min()
planar_depth = U_1D.max()
positions = sorted({x for (_, x) in planes})
print(f"planar ({args.plane or args.axis}): period d = {d:.4f} A, planes per period at {positions}, "
      f"depth = {planar_depth:.2f} eV")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_pl, U_1D, label="DT (Peng 1996)")
ax.set_title(f"{args.material} ({label_pl})   depth {planar_depth:.2f} eV")
ax.set_ylabel("U, eV")
ax.set_xlabel(r"x, $\AA$")
ax.legend()
ax.grid()
fig.savefig(os.path.join(args.outdir, f"{args.material}{label_pl}_planar.pdf"))

print(f"RESULT {args.material} <{label_ax}> {axial_depth:.2f} eV  ({label_pl}) {planar_depth:.2f} eV")
if args.show:
    plt.show()
