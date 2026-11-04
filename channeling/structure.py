"""Crystal structure, materials table and axis frames."""
import os
from math import gcd

import numpy as np
import yaml
from pymatgen.core import Lattice, Structure

from .thermal import u_of_T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def materials():
    with open(os.path.join(DATA, "materials.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def material_info(name):
    """Entry of materials.yaml, or a default one if only <name>.cif exists."""
    info = dict(materials().get(name, {}))
    info.setdefault("cif", name + ".cif")
    info.setdefault("thermal", {})
    return info


def load_structure(name, T=None, alpha=None, T_ref=293.0):
    """Load <name>.cif from the repository root.

    alpha: optional linear thermal expansion coefficient(s) [1/K], one value or three (a, b, c);
    the cell edges are scaled by 1 + alpha (T - T_ref). Off by default (no expansion).
    """
    path = os.path.join(ROOT, material_info(name)["cif"])
    if not os.path.exists(path):
        raise SystemExit(f"CIF not found: {path}")
    s = Structure.from_file(path)
    if alpha is not None and T is not None:
        al = np.broadcast_to(np.array(alpha, float), (3,))
        a, b, c = s.lattice.abc
        f = 1 + al * (T - T_ref)
        s = Structure(Lattice.from_parameters(a * f[0], b * f[1], c * f[2], *s.lattice.angles),
                      s.species, s.frac_coords)
    return s


def resolve_thermal(name, elements, T, overrides=None):
    """{element: ThermalResult} for the species of a material at temperature T [K]."""
    specs = material_info(name)["thermal"]
    out = {}
    for el in sorted(set(elements)):
        spec = {"model": "fixed", "u": overrides[el], "source": "--u"} if overrides and el in overrides else specs.get(el)
        out[el] = u_of_T(el, T, spec)
    return out


def parse_indices(text):
    v = np.array([int(t) for t in str(text).split()])
    if v.shape != (3,) or not v.any():
        raise SystemExit(f'Invalid indices "{text}": expected three integers, not all zero')
    return v // gcd(gcd(int(abs(v[0])), int(abs(v[1]))), int(abs(v[2])))


def plane_frame(t_hat):
    """Orthonormal vectors (e1, e2) spanning the plane perpendicular to the unit axis t_hat."""
    e1 = np.cross(t_hat, [0, 0, 1] if abs(t_hat[2]) < 0.9 else [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(t_hat, e1)
