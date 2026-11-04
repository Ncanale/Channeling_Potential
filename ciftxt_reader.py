"""Axial (2D) and planar (1D) continuum channeling potentials from a CIF file, versus temperature.

Thin command-line wrapper around the `channeling` package (see channeling/__init__.py).
Each atom is described by the Peng et al. (1996) five-Gaussian electron scattering factor (or the
Moliere form factor), thermally smeared by u_j(T) (Debye-Waller factor, exact for Gaussians) and
averaged along atomic strings / planes. Works for any lattice and polyatomic crystals.

Usage:
    python ciftxt_reader.py Si "1 1 0"                         # 293 K, axis <110>, plane (110)
    python ciftxt_reader.py PbWO4 "0 0 1" --plane "0 0 1" --T-scan 77,150,293
    python ciftxt_reader.py Ge --all-axes --T-scan 77,293 --summary output/summary.csv
    python ciftxt_reader.py W "1 1 1" --model moliere --u W=0.05
    python ciftxt_reader.py --list-axes Si                      # used by scanner.sh

The CIF is read from <material>.cif in the repository root. Output (per temperature, in
<outdir>/T<T>K/): PDF plots, an .npz with the maps and a log; one row per (material, axis, T) is
appended to summary.csv. Depth, maximum field and Lindhard angle do not depend on the projectile
charge sign (U -> -U); --charge only flips the plotted potential.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

from channeling import fastsim
from channeling import observables as obs
from channeling.form_factors import FormFactor
from channeling.potential_fourier import axial_fourier, planar_fourier
from channeling.potential_realspace import axial_realspace, planar_realspace
from channeling.structure import load_structure, material_info, parse_indices, resolve_thermal

CSV_FIELDS = ["material", "axis", "plane", "T_K", "method", "model", "energy_GeV",
              "axial_depth_eV", "axial_Emax_GeV_cm", "axial_theta_L_urad", "planar_depth_eV", "planar_d_A",
              "planar_Emax_GeV_cm", "planar_theta_L_urad", "axial_shares", "planar_shares",
              "u_A", "tier", "extrapolated"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("material", nargs="?", default="Si", help="material name; reads <material>.cif")
    p.add_argument("axis", nargs="?", default=None, help='crystal axis [u v w] as "u v w" (default: first default axis)')
    p.add_argument("--plane", default=None, help="plane (h k l); default: the default plane paired with the axis, "
                                                 "or the same indices as the axis")
    p.add_argument("--all-axes", action="store_true", help="run all default axes/planes of the material")
    p.add_argument("--list-axes", action="store_true", help="print 'axis|plane|dirname' lines and exit")
    p.add_argument("--T", type=float, default=None, help="temperature [K] (default 293)")
    p.add_argument("--T-scan", default=None, help="comma separated temperatures [K], e.g. 77,150,293")
    p.add_argument("--method", choices=["fourier", "realspace"], default="fourier")
    p.add_argument("--model", choices=["peng", "moliere"], default="peng", help="form factor model")
    p.add_argument("--charge", type=int, choices=[1, -1], default=1, help="projectile charge sign (plot only)")
    p.add_argument("--energy", type=float, default=120.0, help="projectile energy [GeV] for the Lindhard angle")
    p.add_argument("--outdir", default=".", help="where to save plots/npz/log")
    p.add_argument("--summary", default=None, help="summary csv (default: <outdir>/summary.csv)")
    p.add_argument("--u", nargs="*", default=[], metavar="El=val", help="fixed amplitudes [A], e.g. --u W=0.05")
    p.add_argument("--theta", nargs="*", default=[], metavar="El=val", help="Debye-Waller temperature [K], e.g. --theta W=347")
    p.add_argument("--alpha", type=float, nargs="+", default=None, help="linear thermal expansion [1/K], 1 or 3 values (off by default)")
    p.add_argument("--fastsim", action="store_true", help="also write the Geant4 channeling FastSim data files (see FASTSIM.md)")
    p.add_argument("--fastsim-dir", default=None, help="FastSim output root (default <outdir>/fastsim); files go to <dir>/T<T>K/")
    p.add_argument("--g4-name", default=None, help="Geant4 material name without 'G4_' for the file names (default from materials.yaml)")
    p.add_argument("--fastsim-step", type=float, default=None, help="interpolation step [A] (default from the thermal amplitude)")
    p.add_argument("--fastsim-x", default=None, help='crystal direction "h k l" of the cell x axis for axes (default <110>-type)')
    p.add_argument("--emin", default="amorphous", help="minimal ionisation energy: amorphous | plasma | number [eV]")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--show", action="store_true", help="also open interactive plot windows")
    return p


def axes_for(material, args):
    info = material_info(material)
    if args.axis is not None:
        ax = parse_indices(args.axis)
        pl = parse_indices(args.plane) if args.plane else ax
        return [(ax, pl)]
    axes, planes = info.get("axes"), info.get("planes")
    if not axes:
        raise SystemExit(f"{material}: give an axis, e.g. python ciftxt_reader.py {material} \"1 0 0\"")
    pairs = [(parse_indices(a), parse_indices(p)) for a, p in zip(axes, planes)]
    if args.all_axes:
        return pairs
    return pairs[:1]


def temperatures(args):
    if args.T_scan:
        return [float(t) for t in args.T_scan.split(",")]
    return [args.T if args.T is not None else 293.0]


def label(v):
    return "".join(map(str, v))


def overrides(args):
    out = {}
    for item in args.u:
        el, val = item.split("=")
        out[el] = {"model": "fixed", "u": float(val), "source": "--u"}
    for item in args.theta:
        el, val = item.split("=")
        out[el] = {"model": "debye", "theta": float(val), "source": "--theta"}
    return out


def thermal_for(material, elements, T, ov):
    """Species -> ThermalResult at T, honouring the command-line overrides."""
    from channeling.thermal import u_of_T
    res = resolve_thermal(material, [e for e in elements if e not in ov], T)
    for el, spec in ov.items():
        if el in elements:
            res[el] = u_of_T(el, T, spec)
    return res


def run_one(material, uvw, hkl, T, args, outdir, ov, fastsim_root=None):
    structure = load_structure(material, T=T, alpha=args.alpha)
    elements = [s.specie.symbol for s in structure]
    th = thermal_for(material, elements, T, ov)
    ff = {el: FormFactor(el, args.model) for el in th}
    Ucart = {el: t.tensor() for el, t in th.items()}
    aniso = any(t.Ucart is not None for t in th.values())
    if args.method == "realspace" and (args.model != "peng" or aniso):
        raise SystemExit("--method realspace supports only the Peng model with isotropic u; use --method fourier")
    print(f"{material} T = {T:g} K  axis <{label(uvw)}>  plane ({label(hkl)})  method={args.method} model={args.model}")
    for el, t in th.items():
        print(f"  {el}: u = {t.u:.4f} A  tier {t.tier}{' (extrapolated)' if t.extrapolated else ''}  [{t.source}]")

    if args.method == "fourier":
        A = axial_fourier(structure, uvw, ff, Ucart)
        P = planar_fourier(structure, hkl, ff, Ucart)
        W = 0.75 * max(structure.lattice.abc)
        grid = np.linspace(-W, W, 401)
        xx, yy = np.meshgrid(grid, grid)
        Uwin = A.sample(xx, yy) - A.u_min
        a_depth, a_emax, a_shares = A.depth, A.emax, A.shares
        p_x, p_U, p_d, p_depth, p_emax, p_shares = P.x, P.U, P.d, P.depth, P.emax, P.shares
    else:
        u = {el: t.u for el, t in th.items()}
        A = axial_realspace(structure, uvw, u)
        P = planar_realspace(structure, hkl, u)
        xx, yy, Uwin = A["xx"], A["yy"], A["U"]
        gy, gx = np.gradient(Uwin, yy[:, 0], xx[0])
        a_depth, a_emax, a_shares = A["depth"], float(np.hypot(gx, gy).max()), {}
        p_x, p_U, p_d, p_depth = P["x"], P["U"], P["d"], P["depth"]
        p_emax, p_shares = float(np.abs(np.gradient(p_U, p_x)).max()), {}

    row = dict(material=material, axis=label(uvw), plane=label(hkl), T_K=T, method=args.method, model=args.model,
               energy_GeV=args.energy, axial_depth_eV=round(a_depth, 3),
               axial_Emax_GeV_cm=round(obs.field_GeV_per_cm(a_emax), 2),
               axial_theta_L_urad=round(obs.lindhard_angle_urad(a_depth, args.energy), 2),
               planar_depth_eV=round(p_depth, 3), planar_d_A=round(p_d, 4),
               planar_Emax_GeV_cm=round(obs.field_GeV_per_cm(p_emax), 2),
               planar_theta_L_urad=round(obs.lindhard_angle_urad(p_depth, args.energy), 2),
               axial_shares=";".join(f"{k}:{v:.3f}" for k, v in a_shares.items()),
               planar_shares=";".join(f"{k}:{v:.3f}" for k, v in p_shares.items()),
               u_A=";".join(f"{k}:{t.u:.4f}" for k, t in th.items()),
               tier=";".join(f"{k}:{t.tier}" for k, t in th.items()),
               extrapolated=";".join(f"{k}:{int(t.extrapolated)}" for k, t in th.items()))
    print(f"  axial <{label(uvw)}>: depth {a_depth:.2f} eV, E_max {row['axial_Emax_GeV_cm']} GeV/cm, "
          f"theta_L({args.energy:g} GeV) {row['axial_theta_L_urad']} urad")
    print(f"  planar ({label(hkl)}): d = {p_d:.4f} A, depth {p_depth:.2f} eV, E_max {row['planar_Emax_GeV_cm']} GeV/cm")
    if len(th) > 1 and a_shares:
        print("  species share of the axial depth: " + ", ".join(f"{k} {v:.1%}" for k, v in a_shares.items()))
    print(f"RESULT {material} <{label(uvw)}> {a_depth:.2f} eV  ({label(hkl)}) {p_depth:.2f} eV  T={T:g}K")

    if args.fastsim:
        info = material_info(material)
        g4name = args.g4_name or info.get("g4_name", material)
        xdir = parse_indices(args.fastsim_x) if args.fastsim_x else None
        for r in fastsim.write_files(structure, uvw, hkl, th, os.path.join(fastsim_root, f"T{T:g}K"), g4name,
                                     order=info.get("g4_elements"), step=args.fastsim_step, emin=args.emin,
                                     xdir=None if xdir is None else tuple(int(a) for a in xdir)):
            kind = "planes" if r.model == 1 else "axes"
            size = f"{r.dx:.4f} A, {r.npx} intervals" if r.model == 1 else                 f"{r.dx:.4f} x {r.dy:.4f} A, {r.npx} x {r.npy} intervals"
            print(f"  FastSim {kind}: {r.path}  [{size}, Vmax {r.vmax:.2f} eV, Vmin {r.vmin:.2f} eV, emin {r.emin}]")

    tdir = os.path.join(outdir, f"T{T:g}K")
    os.makedirs(tdir, exist_ok=True)
    np.savez(os.path.join(tdir, f"{material}{label(uvw)}_T{T:g}K.npz"), axial_x=xx, axial_y=yy, axial_U=Uwin,
             planar_x=p_x, planar_U=p_U, T=T)
    if not args.no_plots:
        import matplotlib.pyplot as plt
        sign = 1 if args.charge > 0 else -1
        Z = Uwin if sign > 0 else Uwin.max() - Uwin
        fig, ax = plt.subplots()
        cs = ax.contourf(xx, yy, Z, np.linspace(0, Z.max(), 150), cmap="jet")
        fig.colorbar(cs, label="U, eV")
        ax.set_title(f"{material} <{label(uvw)}>  {T:g} K  depth {a_depth:.1f} eV")
        ax.set_xlabel(r"x, $\AA$")
        ax.set_ylabel(r"y, $\AA$")
        ax.set_aspect("equal")
        fig.savefig(os.path.join(tdir, f"{material}{label(uvw)}_axial.pdf"))
        pU = p_U if sign > 0 else p_U.max() - p_U
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(p_x, pU, label=f"{args.model} / {args.method}")
        ax.set_title(f"{material} ({label(hkl)})  {T:g} K  depth {p_depth:.2f} eV")
        ax.set_ylabel("U, eV")
        ax.set_xlabel(r"x, $\AA$")
        ax.legend()
        ax.grid()
        fig.savefig(os.path.join(tdir, f"{material}{label(hkl)}_planar.pdf"))
        if not args.show:
            plt.close("all")
    return row, (p_x, p_U)


def scan_plots(material, uvw, hkl, rows, profiles, outdir, args):
    import matplotlib.pyplot as plt
    Ts = [r["T_K"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    for T, (x, U) in zip(Ts, profiles):
        ax.plot(x, U, label=f"{T:g} K")
    ax.set_title(f"{material} ({label(hkl)}) planar potential vs T")
    ax.set_xlabel(r"x, $\AA$")
    ax.set_ylabel("U, eV")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(outdir, f"{material}{label(hkl)}_planar_Tscan.pdf"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(Ts, [r["axial_depth_eV"] for r in rows], "o-", label=f"axial <{label(uvw)}>")
    axs[0].plot(Ts, [r["planar_depth_eV"] for r in rows], "s-", label=f"planar ({label(hkl)})")
    axs[0].set_xlabel("T, K")
    axs[0].set_ylabel("depth, eV")
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(Ts, [r["axial_Emax_GeV_cm"] for r in rows], "o-", label=f"axial <{label(uvw)}>")
    axs[1].plot(Ts, [r["planar_Emax_GeV_cm"] for r in rows], "s-", label=f"planar ({label(hkl)})")
    axs[1].set_xlabel("T, K")
    axs[1].set_ylabel(r"$E_{max}$, GeV/cm")
    axs[1].legend()
    axs[1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{material}{label(uvw)}_depth_field_vs_T.pdf"))
    plt.close("all")


def write_summary(path, new_rows):
    """Append rows to summary.csv, replacing earlier rows with the same (material, axis, plane, T, method, model)."""
    key = lambda r: (str(r["material"]), str(r["axis"]), str(r["plane"]), f'{float(r["T_K"]):g}', r["method"], r["model"])
    old = []
    if os.path.exists(path):
        with open(path, newline="") as fh:
            old = list(csv.DictReader(fh))
    drop = {key(r) for r in new_rows}
    rows = [r for r in old if key(r) not in drop] + new_rows
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    args = build_parser().parse_args()
    if args.list_axes:
        info = material_info(args.material)
        if not info.get("axes"):
            raise SystemExit(f"{args.material}: no default axes in materials.yaml")
        for a, p in zip(info["axes"], info["planes"]):
            print(f"{a}|{p}|{label(parse_indices(a))}")
        return
    if not args.show:
        matplotlib.use("Agg")
    ov = overrides(args)
    base_out = args.outdir
    pairs = axes_for(args.material, args)
    fastsim_root = args.fastsim_dir or os.path.join(base_out, "fastsim")
    Ts = temperatures(args)
    summary_path = args.summary or os.path.join(base_out, "summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    new_rows = []
    for uvw, hkl in pairs:
        outdir = os.path.join(base_out, label(uvw)) if len(pairs) > 1 else base_out
        rows, profiles = [], []
        for T in Ts:
            row, prof = run_one(args.material, uvw, hkl, T, args, outdir, ov, fastsim_root)
            rows.append(row)
            profiles.append(prof)
        new_rows += rows
        if len(Ts) > 1 and not args.no_plots:
            scan_plots(args.material, uvw, hkl, rows, profiles, outdir, args)
    write_summary(summary_path, new_rows)
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
