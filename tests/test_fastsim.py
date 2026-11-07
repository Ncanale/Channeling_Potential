"""Tests of the Geant4 FastSim data-file generator. Run from the repository root:  python -m pytest -q

The agreement with the official Geant4 dataset (G4CHANNELING 2.0) is checked only when the environment
variable G4CHANNELINGDATA points to the unpacked dataset directory (otherwise these tests are skipped).
"""
import os

import numpy as np
import pytest

from channeling import emin_fermi, fastsim, fastsim_format as F
from channeling import potential_fourier as pf
from channeling.form_factors import FormFactor, atom_model
from channeling.structure import load_structure, parse_indices, resolve_thermal
from channeling.thermal import ThermalResult

G4DATA = os.environ.get("G4CHANNELINGDATA", "")


def thermal(material, T=293.0):
    s = load_structure(material)
    return s, resolve_thermal(material, [x.specie.symbol for x in s], T)


# ------------------------------------------------------------------ format and splines
def test_planar_spline_roundtrip_synthetic(tmp_path):
    n, d = 400, 2.0e-8
    h = d / n
    x = np.arange(n + 1) * h
    f = np.cos(2 * np.pi * x / d) + 0.3 * np.sin(4 * np.pi * x / d)
    blk = F.planar_block(f, h)
    coef = np.concatenate([blk] * 4, axis=1)            # Ex, nuc0, el, emin all the same function
    data = F.FastSimData(1, 1, np.array([5e22]), np.array([7.5e-10]), d, 0.0, n, 0, 21.0, 7.0, coef)
    path = tmp_path / "X_planes_110.dat"
    F.write(path, data)
    back = F.read(path)
    xs = np.random.default_rng(0).uniform(0, d, 500)
    ref = np.cos(2 * np.pi * xs / d) + 0.3 * np.sin(4 * np.pi * xs / d)
    assert back.npx == n and back.nel == 1
    assert np.abs(F.evaluate(back, "Ex", xs) - ref).max() < 2e-6


def test_axial_spline_roundtrip_synthetic(tmp_path):
    nx, ny, dx, dy = 80, 100, 3.0e-8, 4.0e-8
    hx, hy = dx / nx, dy / ny
    X, Y = np.meshgrid(np.arange(nx + 1) * hx, np.arange(ny + 1) * hy, indexing="ij")
    func = lambda x, y: np.cos(2 * np.pi * x / dx) * np.cos(2 * np.pi * y / dy) + 0.2 * np.sin(2 * np.pi * x / dx)
    f = func(X, Y)
    blk = F.axial_block(f, hx, hy)
    coef = np.zeros((ny, nx + 1, 2, 15))
    for q in range(5):
        coef[:, :, :, 3 * q:3 * q + 3] = blk
    data = F.FastSimData(2, 1, np.array([5e22]), np.array([7.5e-10]), dx, dy, nx, ny, 100.0, 0.0, coef)
    path = tmp_path / "X_axes_110.dat"
    F.write(path, data)
    back = F.read(path)
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(0, dx, 600), rng.uniform(0, dy, 600)
    err = np.abs(F.evaluate(back, "Ex", xs, ys) - func(xs, ys))
    assert err.max() < 5e-3            # natural boundary condition in y limits the accuracy at the edges
    assert np.median(err) < 5e-4


def test_inferred_number_of_elements():
    assert F.infer_nel(2 * 3 + 4 + 100 * 4 * (3 + 3), 1) == 3
    assert F.infer_nel(2 * 1 + 6 + 10 * 11 * 6 * (4 + 1), 2) == 1


# ------------------------------------------------------------------ generated files
def test_planar_file_consistent_with_fourier_potential(tmp_path):
    s, th = thermal("Si")
    hkl = (1, 1, 0)
    d = fastsim.planar_fastsim(s, hkl, th)
    P = pf.planar_fourier(s, hkl, {"Si": FormFactor("Si", atom_model("Si")[0])}, {"Si": th["Si"].tensor()})
    assert d.vmax == pytest.approx(P.depth, rel=1e-4)
    assert d.N0[0] == pytest.approx(len(s) / s.volume * 1e24, rel=1e-9)
    assert d.u[0] == pytest.approx(th["Si"].u * 1e-8)
    x = np.linspace(0, d.dx, 2001)
    # field equals dU/dx of the independent planar profile (x = 0 is the minimum of U)
    xp = P.x
    dU = np.gradient(P.U, xp) * 1e8
    xmin = xp[(xp >= 0) & (xp < P.d)][np.argmin(P.U[(xp >= 0) & (xp < P.d)])]
    ex = F.evaluate(d, "Ex", x[:-1])
    pos = ((xmin + x[:-1] * 1e8 + P.d) % (2 * P.d)) - P.d          # wrap into the tabulated range
    ref = np.interp(pos, xp, dU)
    # rms criterion: the numerical gradient of the reference is unreliable on the sharp peak at the plane
    assert np.sqrt(np.mean((ex - ref) ** 2)) < 2e-3 * np.abs(ref).max()
    # normalisations: mean nuclear density 1, mean electron density Z * N0
    assert F.evaluate(d, "nuc0", x[:-1]).mean() == pytest.approx(1.0, abs=2e-3)
    assert F.evaluate(d, "el", x[:-1]).mean() == pytest.approx(14 * d.N0[0], rel=2e-3)
    # the field integrates to the well height: potential energy difference between centre and plane
    cum = np.cumsum(F.evaluate(d, "Ex", x[:-1])) * (x[1] - x[0])
    assert cum.max() - cum.min() == pytest.approx(d.vmax, rel=2e-3)


def test_axial_file_consistent_with_fourier_potential(tmp_path):
    s, th = thermal("Si")
    uvw = (1, 1, 0)
    d, info = fastsim.axial_fastsim(s, uvw, th)
    A = pf.axial_fourier(s, parse_indices("1 1 0"), {"Si": FormFactor("Si")}, {"Si": th["Si"].tensor()})
    assert d.vmax == pytest.approx(A.depth, rel=2e-3)
    assert d.dx * 1e8 == pytest.approx(s.lattice.a / np.sqrt(2), rel=1e-3) and d.dy * 1e8 == pytest.approx(s.lattice.a, rel=1e-3)
    # compare the file field with finite differences of the independent Fourier grid
    e1, e2 = A.frame
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, d.dx * 1e8, 200), rng.uniform(0.05, d.dy * 1e8 - 0.05, 200)
    r = np.array(info["r0"])[0] * info["ex"] + np.array(info["r0"])[1] * info["ey"]
    pts = r[None] + x[:, None] * info["ex"] + y[:, None] * info["ey"]
    X, Y = pts @ e1, pts @ e2
    h = 1e-3
    gx = (A.sample(X + h * (info["ex"] @ e1), Y + h * (info["ex"] @ e2)) - A.sample(X - h * (info["ex"] @ e1), Y - h * (info["ex"] @ e2))) / (2 * h)
    ex = F.evaluate(d, "Ex", x * 1e-8, y * 1e-8) * 1e-8
    assert np.sqrt(np.mean((ex - gx) ** 2)) < 5e-3 * np.abs(gx).max()
    assert F.evaluate(d, "nuc0", *np.meshgrid(np.linspace(0, d.dx, 120, endpoint=False), np.linspace(0, d.dy, 200, endpoint=False))).mean() == pytest.approx(1.0, abs=5e-3)


def test_compound_files_keep_element_order_and_ratios(tmp_path):
    s, th = thermal("PbWO4")
    d = fastsim.planar_fastsim(s, (0, 0, 1), th, order=["O", "Pb", "W"])
    assert d.nel == 3
    assert d.N0[0] / d.N0[1] == pytest.approx(4.0) and d.N0[1] == pytest.approx(d.N0[2])
    x = np.linspace(0, d.dx, 3000, endpoint=False)
    for name in ("nuc0", "nuc1", "nuc2"):
        assert F.evaluate(d, name, x).mean() == pytest.approx(1.0, abs=5e-3)
    zsum = 8 * d.N0[0] + 82 * d.N0[1] + 74 * d.N0[2]
    assert F.evaluate(d, "el", x).mean() == pytest.approx(zsum, rel=3e-3)
    path = tmp_path / "PbWO4_planes_001.dat"
    F.write(path, d)
    assert F.read(path).nel == 3
    with pytest.raises(SystemExit):
        fastsim.planar_fastsim(s, (0, 0, 1), th, order=["O", "Pb"])


def test_compound_axial_cell_is_rectangular_and_periodic(tmp_path):
    s, th = thermal("PbWO4")
    d, _ = fastsim.axial_fastsim(s, (0, 0, 1), th, order=["O", "Pb", "W"])
    assert d.nel == 3 and d.dx > 0 and d.dy > 0
    path = tmp_path / "PbWO4_axes_001.dat"
    F.write(path, d)
    back = F.read(path)
    xs = np.linspace(0, back.dx, 90, endpoint=False)
    ys = np.linspace(0, back.dy, 90, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    assert F.evaluate(back, "nuc0", X, Y).mean() == pytest.approx(1.0, abs=1e-2)
    # periodicity: shifting by one cell changes nothing
    assert np.allclose(F.evaluate(back, "el", X, Y), F.evaluate(back, "el", X + back.dx, Y + back.dy))


def test_emin_models():
    s = load_structure("Si")
    assert fastsim.emin_constant(s) == pytest.approx(173.0)
    th = {"Si": ThermalResult(0.075, "fixed", "t")}
    c = fastsim.planar_fastsim(s, (1, 1, 0), th, emin="amorphous")
    assert np.allclose(F.evaluate(c, "emin", np.linspace(0, c.dx, 50)), 173.0, atol=1e-6)
    v = fastsim.planar_fastsim(s, (1, 1, 0), th, emin=50.0)
    assert np.allclose(F.evaluate(v, "emin", np.linspace(0, v.dx, 50)), 50.0, atol=1e-6)
    p = fastsim.planar_fastsim(s, (1, 1, 0), th, emin="plasma")
    e = F.evaluate(p, "emin", np.linspace(0, p.dx, 200))
    assert e.min() >= 0.5 and e.max() > 3 * e.min()      # ~ sqrt of the density contrast


def test_emin_fermi_planar():
    """Sytov's local Fermi energy (atomic Thomas-Fermi): a few eV in the channel, ~330 eV on the plane, periodic,
    and its electron-weighted geometric mean over the cell is the atomic value (orientation independent)."""
    s = load_structure("Si")
    th = {"Si": ThermalResult(0.075, "fixed", "t")}
    d = fastsim.planar_fastsim(s, (1, 1, 0), th, emin="fermi")
    x = np.linspace(0, d.dx, 2000, endpoint=False)
    e = F.evaluate(d, "emin", x)
    assert 2 < e.min() < 5 and 250 < e.max() < 450
    assert abs(x[np.argmax(e)] - d.dx / 2) < 0.05 * d.dx      # plane at dx/2, channel centre at 0
    assert np.allclose(F.evaluate(d, "emin", x + d.dx), e)
    el = F.evaluate(d, "el", x)
    i_cell = np.exp((el * np.log(e)).sum() / el.sum())
    assert i_cell == pytest.approx(emin_fermi.atomic_fermi_mean("Si", 0.075), rel=0.01)
    assert emin_fermi.atomic_fermi_mean("Si", 0.075) == pytest.approx(81.9, rel=0.005)   # shipped Si maps: 81.8-82.0 eV


def test_emin_fermi_crystal_vs_atomic():
    """At the channel centre the crystal potential (neighbours, mean inner potential) gives a higher T_min
    than the potential of the own atom; on the plane the two coincide within ~10 %."""
    s = load_structure("Si")
    th = {"Si": ThermalResult(0.075, "fixed", "t")}
    n_hat = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    d = s.lattice.a / (2 * np.sqrt(2))                       # (110) spacing; an atomic plane through the origin
    xs = np.array([0.0, d / 2])
    tc = emin_fermi.emin_planar(emin_fermi.CrystalField3D(s, th, ["Si"]), n_hat, xs)
    ta = emin_fermi.emin_planar(emin_fermi.AtomicFermiField(s, th, ["Si"]), n_hat, xs)
    assert 250 < ta[0] < 450 and ta[0] == pytest.approx(tc[0], rel=0.15)
    assert 2 < ta[1] < 5 and 5 < tc[1] < 15

# ------------------------------------------------------------------ agreement with the Geant4 dataset
needs_g4 = pytest.mark.skipif(not os.path.isdir(G4DATA), reason="set G4CHANNELINGDATA to the unpacked G4CHANNELING dataset")


@needs_g4
@pytest.mark.parametrize("material,name,hkl,u", [("Si", "Si", (1, 1, 0), 0.075), ("Ge", "Ge", (1, 1, 0), 0.085),
                                                 ("Si", "Si", (1, 1, 1), 0.075)])
def test_planar_agrees_with_official_files(material, name, hkl, u):
    s = load_structure(material)
    th = {material: ThermalResult(u, "fixed", "test")}
    mine = fastsim.planar_fastsim(s, hkl, th)
    ref = F.read(os.path.join(G4DATA, f"{name}_planes_{''.join(map(str, hkl))}.dat"))
    assert mine.dx == pytest.approx(ref.dx, rel=1e-3)
    assert mine.vmax == pytest.approx(ref.vmax, rel=5e-3)
    assert mine.vmin == pytest.approx(ref.vmin, rel=2e-2)
    x = np.linspace(0, ref.dx, 600, endpoint=False)
    for q, tol in (("Ex", 0.02), ("nuc0", 0.01), ("el", 0.03)):
        a, b = F.evaluate(mine, q, x), F.evaluate(ref, q, x)
        assert np.sqrt(np.mean((a - b) ** 2)) < tol * np.abs(b).max()


@needs_g4
@pytest.mark.parametrize("hkl", [(1, 1, 0), (1, 1, 1), (1, 0, 0)])
def test_emin_fermi_matches_official_planes(hkl):
    """'fermi' reproduces the shipped Si map point by point, channels (1.5-5 eV) and planes (~300 eV)."""
    s = load_structure("Si")
    th = {"Si": ThermalResult(0.075, "fixed", "test")}
    mine = fastsim.planar_fastsim(s, hkl, th, emin="fermi")
    ref = F.read(os.path.join(G4DATA, "Si_planes_%d%d%d.dat" % hkl))
    x = (np.arange(1000) + 0.5) / 1000 * ref.dx
    a, b = F.evaluate(mine, "emin", x), F.evaluate(ref, "emin", x)
    r = np.log(a / b)
    assert abs(r.mean()) < 0.02 and np.sqrt(np.mean(r ** 2)) < 0.07
    assert a.min() == pytest.approx(b.min(), rel=0.15) and a.max() == pytest.approx(b.max(), rel=0.05)


def _si_axis_110():
    if not hasattr(_si_axis_110, "cache"):
        s = load_structure("Si")
        th = {"Si": ThermalResult(0.075, "fixed", "test")}
        mine, _ = fastsim.axial_fastsim(s, (1, 1, 0), th)
        _si_axis_110.cache = mine, F.read(os.path.join(G4DATA, "Si_axes_110.dat"))
    return _si_axis_110.cache


def _grid(ref, n=120):
    return np.meshgrid((np.arange(n) + 0.5) / n * ref.dx, (np.arange(n) + 0.5) / n * ref.dy, indexing="ij")


@needs_g4
def test_axial_agrees_with_official_file_point_by_point():
    """Si <110>: same cell, origin and orientation as the official file: field and densities coincide directly."""
    mine, ref = _si_axis_110()
    assert mine.dx == pytest.approx(ref.dx, rel=1e-3) and mine.dy == pytest.approx(ref.dy, rel=1e-3)
    X, Y = _grid(ref)
    a = F.evaluate(ref, "el", X, Y)
    assert np.sqrt(np.mean((F.evaluate(mine, "el", X, Y) - a) ** 2)) / a.max() < 0.005
    E = np.hypot(F.evaluate(ref, "Ex", X, Y), F.evaluate(ref, "Ey", X, Y)).max()
    for q in ("Ex", "Ey"):
        assert np.sqrt(np.mean((F.evaluate(mine, q, X, Y) - F.evaluate(ref, q, X, Y)) ** 2)) / E < 0.005


@pytest.mark.parametrize("name, uvw, el", [("Si_axes_100", (1, 0, 0), "Si"), ("Si_axes_111", (1, 1, 1), "Si"),
                                           ("W_axes_111", (1, 1, 1), "W")])
@needs_g4
def test_axial_origin_convention(name, uvw, el):
    """Strings on the main diagonal with their mean at the cell centre, as in the official files (also bcc <111>,
    where the rectangular cell holds two strings but the cubic cell projects onto one)."""
    ref = F.read(os.path.join(G4DATA, name + ".dat"))
    u = ref.u[0] / 1e-8
    mine, _ = fastsim.axial_fastsim(load_structure(el), uvw, {el: ThermalResult(u, "fixed", "test")}, emin=0.0)
    X, Y = _grid(ref)
    a = F.evaluate(ref, "nuc0", X, Y)
    assert np.sqrt(np.mean((F.evaluate(mine, "nuc0", X, Y) - a) ** 2)) / a.max() < 0.01


@needs_g4
def test_emin_fermi_matches_official_axis():
    """Si <110>: 'fermi' reproduces the shipped map from 0.28 eV (channel) to ~750 eV (strings)."""
    mine, ref = _si_axis_110()
    X, Y = _grid(ref)
    a = F.evaluate(mine, "emin", X, Y)
    b = F.evaluate(ref, "emin", X, Y)
    r = np.log(a / b)
    assert abs(r.mean()) < 0.03 and np.sqrt(np.mean(r ** 2)) < 0.12
    assert a.min() == pytest.approx(b.min(), rel=0.2) and a.max() == pytest.approx(b.max(), rel=0.05)


def test_atom_model_is_dudarev_peng_whelan():
    """Dudarev-Peng-Whelan 1995 (Tables 1-2) + Mott-Bethe for the elements of their table, Peng otherwise."""
    from channeling.form_factors import dpw_table
    t = dpw_table()
    assert len(t) == 50 and t["C"]["a"][0] == pytest.approx(0.6617) and t["W"]["b"][4] == pytest.approx(0.07291)
    assert atom_model("Si") == ("dpw", "mott-bethe") and atom_model("Pb") == ("dpw", "mott-bethe")
    assert atom_model("Bi") == ("peng", "mott-bethe")


@pytest.mark.parametrize("name, el, u, tmax, nmax", [("Si_planes_110", "Si", 0.075, 0.015, 0.002),
                                                     ("Ge_planes_110", "Ge", 0.085, 0.015, 0.002),
                                                     ("Diamond_planes_110", "C", 0.040, 0.015, 0.002)])
@needs_g4
def test_atomic_model_per_element_matches_official_planes(name, el, u, tmax, nmax):
    """With Dudarev-Peng-Whelan 1995 and the Mott-Bethe density the shipped T_min and n_e are reproduced
    (rms of (ours - official)/official below tmax; n_e rms below nmax of its maximum)."""
    ref = F.read(os.path.join(G4DATA, name + ".dat"))
    d = fastsim.planar_fastsim(load_structure(el), (1, 1, 0), {el: ThermalResult(u, "fixed", "test")}, emin="fermi")
    x = (np.arange(1000) + 0.5) / 1000 * ref.dx
    t = F.evaluate(d, "emin", x) / F.evaluate(ref, "emin", x) - 1
    a = F.evaluate(ref, "el", x)
    assert np.sqrt(np.mean(t ** 2)) < tmax
    assert np.sqrt(np.mean((F.evaluate(d, "el", x) - a) ** 2)) / a.max() < nmax
