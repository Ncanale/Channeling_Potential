"""Radiation in W <111> 100 um aligned, e- 855 MeV, Geant4 11.4.3 FastSim + Baier-Katkov: Nostri vs Geant4 (ufficiali).
usage: radiation_comparison.py [seeds, default 1,...,20] [label format, default rW_s{s}_{d}_W111] [electrons per seed]
Observables (per incident electron): photon spectrum E dN/dE, radiated energy, energy of the exiting electron.
Residuals: (Nostri - Geant4) / Geant4, from the summed spectra.
Errors: seed-to-seed std / sqrt(n_seeds), paired for the differences (same seeds for both sets).
Output: output/g4_comparison/06_geant4_radiation_W111.png, radiation_W111.csv"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
BUILD = r'\\wsl.localhost\Ubuntu-24.04\home\tacab\chtest\ch2\build'
SEEDS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(range(1, 21))
FMT = sys.argv[2] if len(sys.argv) > 2 else 'rW_s{s}_{d}_W111'
N_INC = int(sys.argv[3]) if len(sys.argv) > 3 else 2000                # incident electrons per seed
SURF, INK, INK2, BASE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#c3c2b7', '#e1e0d9'
NOSTRI, GEANT = '#ff7f0e', INK2
REL = '(Nostri − Geant4) / Geant4'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 9.5,
    'axes.edgecolor': BASE, 'axes.linewidth': 0.8, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'], 'font.size': 9,
    'legend.frameon': False, 'legend.fontsize': 8.5,
})


def load(path):
    """Bin centres, counts and sum of x*w per bin (underflow/overflow dropped)."""
    lo = hi = None
    rows = []
    for line in open(path):
        if line.startswith('#axis fixed'):
            _, _, _, lo, hi = line.split()
            lo, hi = float(lo), float(hi)
        elif line[0] != '#' and not line.startswith('entries'):
            rows.append([float(v) for v in line.split(',')])
    r = np.array(rows)
    nb = len(r) - 2
    c = (np.arange(nb) + 0.5) * (hi - lo) / nb + lo
    return c, r[1:-1, 1], r[1:-1, 3], r[0, 1] + r[-1, 1]


def f(s, d, h):
    return os.path.join(BUILD, FMT.format(s=s, d=d) + f'_h1_{h}.csv')


EDGES = np.unique(np.r_[0, np.round(np.geomspace(1, 860, 31))])     # MeV, 1 MeV histogram bins merged
data = {d: [] for d in ('off', 'now')}
for s in SEEDS:
    for d in data:
        cp, wp, xp, _ = load(f(s, d, 'Eprim'))
        Ne = wp.sum()                                               # primaries reaching the detector
        cg, wg, xg, ovf = load(f(s, d, 'Spectrum'))
        idx = np.digitize(cg, EDGES) - 1
        IE = np.array([xg[idx == k].sum() for k in range(len(EDGES) - 1)]) / N_INC / np.diff(EDGES)   # E dN/dE
        NG = np.array([wg[idx == k].sum() for k in range(len(EDGES) - 1)]) / N_INC
        band = (cg >= 178) & (cg < 438)
        data[d].append(dict(Eband=xg[band].sum() / N_INC, Ne=Ne, Erad=xg.sum() / N_INC, Ngam=wg.sum() / N_INC, IE=IE, NG=NG, ovf=ovf,
                            Eout=xp.sum() / Ne, Eprim=wp / N_INC, cp=cp, lost=1 - Ne / N_INC))


def ms(v):
    v = np.asarray(v)
    return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))


rows = []
print(f'{len(SEEDS)} seeds, electrons per seed (off/now): '
      f'{[int(x["Ne"]) for x in data["off"]]} / {[int(x["Ne"]) for x in data["now"]]}')
for key, lab, unit in [('Erad', 'radiated energy per e-', 'MeV'), ('Eband', 'radiated energy 178-438 MeV', 'MeV'),
                       ('Ngam', 'photons per e-', ''),
                       ('Eout', 'mean energy of the exiting e-', 'MeV'),
                       ('lost', 'primaries not reaching the detector', '')]:
    o, oe = ms([x[key] for x in data['off']])
    n, ne = ms([x[key] for x in data['now']])
    dd, de = ms([a[key] - b[key] for a, b in zip(data['now'], data['off'])])
    print(f'{lab:32s} Geant4 {o:8.3f} +- {oe:.3f}   Nostri {n:8.3f} +- {ne:.3f}   '
          f'diff {dd:+.3f} +- {de:.3f} {unit} ({100 * dd / o:+.2f}% +- {100 * de / o:.2f}%, {dd / de:+.1f} sigma)')
    rows.append([lab, unit, o, oe, n, ne, dd, de])
print('overflow photons (> 860 MeV):', [x['ovf'] for x in data['off']], [x['ovf'] for x in data['now']])

IEo, IEoe = ms([x['IE'] for x in data['off']])
IEn, IEne = ms([x['IE'] for x in data['now']])
# ratio of the summed spectra; error from the seed scatter of the paired residual now_i - R off_i (delta method)
An = np.array([x['IE'] for x in data['now']])
Ao = np.array([x['IE'] for x in data['off']])
with np.errstate(invalid='ignore', divide='ignore'):
    Rm = An.mean(0) / Ao.mean(0)
    Re = (An - Rm * Ao).std(0, ddof=1) / np.sqrt(len(SEEDS)) / Ao.mean(0)
Ec = np.sqrt(np.maximum(EDGES[:-1], 0.5) * EDGES[1:])
ok = np.isfinite(Re) & (Re > 0)
chi2 = np.sum(((Rm[ok] - 1) / Re[ok]) ** 2)
from scipy import stats  # noqa: E402
print(f'spectrum ratio Nostri/Geant4, chi2 vs 1: {chi2:.1f} / {ok.sum()} bins, p {stats.chi2.sf(chi2, ok.sum()):.3f}; '
      f'bins above 1: {(Rm[ok] > 1).sum()}')
for k in range(len(Ec)):
    print(f'   {EDGES[k]:6.0f}-{EDGES[k + 1]:4.0f} MeV  E dN/dE Geant4 {IEo[k]:.4f}  ratio {Rm[k]:.3f} +- {Re[k]:.3f}')

with open(os.path.join(OUT, 'radiation_W111.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['quantity', 'unit', 'geant4', 'geant4_err', 'nostri', 'nostri_err', 'difference', 'difference_err'])
    for r in rows:
        w.writerow([r[0], r[1]] + [f'{v:.5g}' for v in r[2:]])
    w.writerow([])
    w.writerow(['E_low_MeV', 'E_high_MeV', 'geant4_EdNdE', 'geant4_err', 'nostri_EdNdE', 'nostri_err',
                'relative_difference_percent', 'relative_difference_err_percent'])
    for k in range(len(Ec)):
        w.writerow([EDGES[k], EDGES[k + 1]] + [f'{v:.5g}' for v in (IEo[k], IEoe[k], IEn[k], IEne[k],
                                                                    100 * (Rm[k] - 1), 100 * Re[k])])

fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex='col', gridspec_kw={'height_ratios': [2.2, 1]})
ax, axr = axs[0, 0], axs[1, 0]
for IE, IEe, col, ls, lab in [(IEo, IEoe, GEANT, '--', 'Geant4 (ufficiali)'), (IEn, IEne, NOSTRI, '-', 'Nostri (Thomas-Fermi per atomo)')]:
    ax.stairs(IE, EDGES, color=col, ls=ls, lw=1.6, label=lab)
    ax.errorbar(Ec, IE, yerr=IEe, color=col, ls='none', elinewidth=1)
ax.set_xscale('symlog', linthresh=1, linscale=0.4)
ax.set_xlim(0, 860)
ax.set_ylabel('E dN/dE per electron')
ax.legend(loc='lower left')
axr.axhline(0, color=INK2, lw=0.8)
Ec[0] = 0.5 * (EDGES[0] + EDGES[1])
axr.errorbar(Ec[ok], 100 * (Rm[ok] - 1), yerr=100 * Re[ok], color=NOSTRI, marker='o', ms=4, lw=1.2, elinewidth=1)
axr.set_ylabel(f'{REL}  [%]', fontsize=8)
axr.set_xlabel('photon energy [MeV]  (linear below 1 MeV, logarithmic above)')
ax.set_title(f'Photon spectrum      {REL}: χ² = {chi2:.1f} / {ok.sum()} bins', loc='left')

ax, axr = axs[0, 1], axs[1, 1]
cp = data['off'][0]['cp']                                     # 2 MeV histogram bins
lo_edges = cp - 1
# variable bins from the top down: at least NMIN Geant4 electrons (all seeds) and at least 10 MeV wide,
# so that every bin has a residual with an error of ~5 % or less
NMIN = 400
tot = np.sum([x['Eprim'] for x in data['off']], axis=0) * N_INC
edges = [860.0]
acc, top = 0.0, 860.0
for i in np.where(lo_edges < 860)[0][::-1]:
    acc += tot[i]
    if acc >= NMIN and top - lo_edges[i] >= 10:
        edges.append(lo_edges[i])
        acc, top = 0.0, lo_edges[i]
if edges[-1] > 0:
    edges[-1] = 0.0                                           # merge the remainder into the last bin
eb = np.array(edges[::-1])
idx = np.searchsorted(lo_edges, eb[:-1])
def reb(v):
    return np.add.reduceat(v, idx)
Bo = np.array([reb(x['Eprim']) for x in data['off']]) / np.diff(eb)    # fraction per MeV
Bn = np.array([reb(x['Eprim']) for x in data['now']]) / np.diff(eb)
Po, Poe = ms(Bo)
Pn, Pne = ms(Bn)
Pr = Bn.mean(0) / Bo.mean(0)
Pre = (Bn - Pr * Bo).std(0, ddof=1) / np.sqrt(len(SEEDS)) / Bo.mean(0)
chi2p = np.sum(((Pr - 1) / Pre) ** 2)
print(f'exiting electron energy, (N-G)/G: chi2 vs 0 {chi2p:.1f} / {len(Pr)} bins, p {stats.chi2.sf(chi2p, len(Pr)):.3f}')
for P, Pe, col, ls in [(Po, Poe, GEANT, '--'), (Pn, Pne, NOSTRI, '-')]:
    ax.stairs(P, eb, color=col, ls=ls, lw=1.6)
    ax.errorbar(0.5 * (eb[1:] + eb[:-1]), P, yerr=Pe, color=col, ls='none', elinewidth=1)
ax.set_yscale('log')
ax.set_ylabel('fraction of electrons per MeV')
ax.set_title(f'Energy of the exiting electron      {REL}: χ² = {chi2p:.1f} / {len(Pr)} bins', loc='left')
axr.axhline(0, color=INK2, lw=0.8)
ec = 0.5 * (eb[1:] + eb[:-1])
axr.errorbar(ec, 100 * (Pr - 1), yerr=100 * Pre, xerr=0.5 * np.diff(eb), color=NOSTRI, marker='o', ms=4, lw=0,
             elinewidth=1)
axr.set_ylabel(f'{REL}  [%]', fontsize=8)
axr.set_xlabel(f'electron energy [MeV]  (variable bins, ≥ {NMIN} Geant4 electrons each)')
for a in axs.ravel():
    a.tick_params(length=0, labelsize=8, colors=INK2)
er = rows[0]
fig.suptitle('Geant4 11.4.3 radiation: e− 855 MeV, W <111> 100 µm aligned, u = 0.050 Å: Nostri vs Geant4 (ufficiali)',
             x=0.07, ha='left', fontsize=11.5)
fig.text(0.07, 0.01, f'{len(SEEDS)} seeds × {N_INC} electrons per set (normalised per incident electron), same seeds for both sets; '
         f'errors: seed-to-seed std / √{len(SEEDS)} (paired for differences and ratios).\n'
         f'Radiated energy per electron: Geant4 {er[2]:.2f} MeV, Nostri {er[4]:.2f} MeV, difference {100 * er[6] / er[2]:+.2f} % ± {100 * er[7] / er[2]:.2f} %.',
         fontsize=8, color=INK2)
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.13, wspace=0.22, hspace=0.1)
fig.savefig(os.path.join(OUT, '06_geant4_radiation_W111.png'), dpi=150)
print('figure written')
