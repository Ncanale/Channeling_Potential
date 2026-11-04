"""Regression tests. Run from the repository root:  python -m pytest -q"""
import numpy as np
import pytest
from scipy.special import erfc, erfcx

from channeling import observables as obs
from channeling import potential_fourier as pf
from channeling import potential_realspace as pr
from channeling import thermal as th
from channeling.form_factors import A0, E2, FormFactor, MOLIERE_ALPHA, MOLIERE_BETA
from channeling.structure import load_structure, parse_indices, resolve_thermal


def setup(material, T=293.0, u=None, model="peng"):
    s = load_structure(material)
    els = [x.specie.symbol for x in s]
    if u is None:
        res = resolve_thermal(material, els, T)
        u = {e: r.u for e, r in res.items()}
    ff = {e: FormFactor(e, model) for e in u}
    Uc = {e: v ** 2 * np.eye(3) for e, v in u.items()}
    return s, u, ff, Uc


# ------------------------------------------------------------------ thermal
@pytest.mark.parametrize("el,lo,hi", [("Ge", 0.085, 0.089), ("W", 0.043, 0.047), ("Si", 0.075, 0.081)])
def test_u_293K_matches_peng_table_extrapolation(el, lo, hi):
    assert lo < th.u_of_T(el, 293.0).u < hi


def test_peng_table_rows_reproduced_exactly():
    assert th.u_of_T("Ge", 210.0).u == pytest.approx(np.sqrt(0.4302 / (8 * np.pi ** 2)))
    assert not th.u_of_T("Ge", 210.0).extrapolated
    assert th.u_of_T("Ge", 77.0).extrapolated


def test_debye_zero_point_and_classical_limits():
    M, theta = 72.63, 282.0
    assert th.u2_debye(M, theta, 1e-3) == pytest.approx(th.C_DEBYE / (4 * M * theta), rel=1e-6)
    assert th.u2_debye(M, theta, 1e5) == pytest.approx(th.C_DEBYE * 1e5 / (M * theta ** 2), rel=1e-3)
    assert th.C_DEBYE == pytest.approx(145.5262, rel=1e-5)


@pytest.mark.parametrize("el", ["Ge", "W", "Si", "C", "Pb", "Ir"])
def test_u_monotonic_in_T_with_finite_zero_point(el):
    T = np.linspace(1, 600, 120)
    u = np.array([th.u_of_T(el, t).u for t in T])
    assert u[0] > 0
    assert np.all(np.diff(u) > -1e-9)


# ------------------------------------------------------------------ Fourier vs real space
@pytest.mark.parametrize("material", ["Si", "Ge"])
@pytest.mark.parametrize("axis,plane", [("1 0 0", "1 1 0"), ("1 1 0", "1 1 1"), ("1 1 1", "1 0 0")])
def test_fourier_matches_realspace(material, axis, plane):
    s, u, ff, Uc = setup(material)
    A = pf.axial_fourier(s, parse_indices(axis), ff, Uc)
    P = pf.planar_fourier(s, parse_indices(plane), ff, Uc)
    Ar = pr.axial_realspace(s, parse_indices(axis), u)
    Pr = pr.planar_realspace(s, parse_indices(plane), u)
    assert A.depth == pytest.approx(Ar["depth"], rel=5e-3)
    assert P.depth == pytest.approx(Pr["depth"], rel=5e-3)


def moliere_plane_closed_form(x, u, Z, sigma):
    """Thermally averaged Moliere potential of one plane (closed form, erfcx for stability)."""
    x = np.abs(np.asarray(x, float))
    a = 0.8853 * A0 / Z ** (1 / 3)
    t = 0
    for al, be in zip(MOLIERE_ALPHA, MOLIERE_BETA):
        k = be / a
        z1, z2 = (u * k - x / u) / np.sqrt(2), (u * k + x / u) / np.sqrt(2)
        A = np.where(z1 >= 0, np.exp(-x ** 2 / (2 * u * u)) * erfcx(np.maximum(z1, 0)),
                     np.exp(np.minimum(k * k * u * u / 2 - k * x, 50)) * erfc(np.minimum(z1, 0)))
        t += al / (2 * be) * (A + np.exp(-x ** 2 / (2 * u * u)) * erfcx(z2))
    return 2 * np.pi * sigma * Z * E2 * a * t


def test_moliere_fourier_matches_closed_form_planar():
    u = {"Ge": 0.085}
    s, _, ff, Uc = setup("Ge", u=u, model="moliere")
    P = pf.planar_fourier(s, parse_indices("1 1 0"), ff, Uc)
    N = len(s) / s.volume
    d = P.d
    ref = sum(moliere_plane_closed_form(P.x - m * d, 0.085, 32, N * d) for m in range(-30, 31))
    ref -= ref.min()
    assert np.abs(P.U - ref).max() < 5e-3 * ref.max()


# ------------------------------------------------------------------ reference values (Peng, DT)
def planar_xc(material, plane, u, which="Uxc"):
    s, uu, ff, Uc = setup(material, u={material: u})
    P = pf.planar_fourier(s, parse_indices(plane), ff, Uc)
    return obs.planar_U_at_xc(P.x, P.U, P.d, u)


@pytest.mark.parametrize("material,u,ref_planar,ref_axial", [
    ("Ge", 0.0869, 20.6, (145, 211, 167)),
    ("W", 0.045, 88.7, (820, 578, 947)),
])
def test_reference_values_from_literature_pass(material, u, ref_planar, ref_axial):
    assert planar_xc(material, "1 1 0", u)[0] == pytest.approx(ref_planar, rel=0.02)
    s, uu, ff, Uc = setup(material, u={material: u})
    for axis, ref in zip(("1 0 0", "1 1 0", "1 1 1"), ref_axial):
        A = pf.axial_fourier(s, parse_indices(axis), ff, Uc)
        assert A.depth == pytest.approx(ref, rel=0.02)


def test_depth_decreases_with_temperature():
    s = load_structure("Ge")
    depths = []
    for T in (77, 150, 293, 400):
        u = {"Ge": th.u_of_T("Ge", T).u}
        ff = {"Ge": FormFactor("Ge")}
        Uc = {"Ge": u["Ge"] ** 2 * np.eye(3)}
        depths.append(pf.axial_fourier(s, parse_indices("1 1 0"), ff, Uc).depth)
    assert np.all(np.diff(depths) < 0)


# ------------------------------------------------------------------ polyatomic, anisotropy, hooks
def test_pbwo4_is_scheelite_and_pb_w_dominate_axial_001():
    s, u, ff, Uc = setup("PbWO4")
    assert len(s) == 24
    a, b, c = s.lattice.abc
    assert a == pytest.approx(b) and a == pytest.approx(5.47, abs=0.02) and c == pytest.approx(12.06, abs=0.03)
    A = pf.axial_fourier(s, parse_indices("0 0 1"), ff, Uc)
    assert sum(A.shares.values()) == pytest.approx(1.0)
    assert abs(A.shares["O"]) < 0.10
    assert A.shares["Pb"] + A.shares["W"] > 0.9


def test_vibration_along_the_axis_does_not_change_axial_potential():
    """Only G with G.t = 0 enter the axial average, so U_zz must be irrelevant for the axis z = [001]."""
    s, u, ff, Uc = setup("Si")
    ref = pf.axial_fourier(s, parse_indices("0 0 1"), ff, Uc)
    Ua = {"Si": np.diag([u["Si"] ** 2, u["Si"] ** 2, 0.5])}
    aniso = pf.axial_fourier(s, parse_indices("0 0 1"), ff, Ua)
    assert aniso.depth == pytest.approx(ref.depth, rel=1e-6)


def test_missing_thermal_source_is_reported_not_invented():
    with pytest.raises(SystemExit):
        resolve_thermal("SiC", ["Si", "C"], 293.0)


def test_thermal_expansion_is_off_by_default_and_scales_when_requested():
    a0 = load_structure("Si").lattice.a
    assert load_structure("Si", T=393.0).lattice.a == pytest.approx(a0)
    assert load_structure("Si", T=393.0, alpha=2.6e-6).lattice.a == pytest.approx(a0 * (1 + 2.6e-6 * 100))
