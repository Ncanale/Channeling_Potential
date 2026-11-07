"""Radiation tests of the Geant4 channeling examples with planar files: Nostri (modello atomico g4) vs Geant4 (ufficiali).

  mami: examples ch1 / ch2 run.mac - e- 855 MeV, divergence 30 x 70 µrad, Si (111) bent 0.905 mrad, 30.5 µm
        (MAMI: Mazzolari 2014, Bandiera 2015)
  dia:  example ch2 run_Radiation.mac - e- 500 MeV, divergence 50 x 50 µrad, diamond (110) straight, 50 µm

usage: radiation_planar.py <mami|dia> [electrons per seed]
Labels rad_{case}{seed}_{off|now}; uses every seed complete for both sets.
Observables per incident electron: photon spectrum E dN/dE, photons, radiated energy; exit-angle distribution in x
(mami: channeled = deflected fraction 700-1200 µrad, volume reflection < -300 µrad).
Residuals (Nostri - Geant4) / Geant4 from the summed histograms; errors from the seed-to-seed scatter of the
paired residual (delta method), as for figure 06.
Output: 11_geant4_radiation_Si111_bent.png / 12_geant4_radiation_diamond110.png, radiation_<case>.csv
"""
import csv
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rocking_comparison as rc  # noqa: E402


def load(path):
    """Bin centres, counts and sum of x*w per bin, overflow counts (as in radiation_comparison.py)."""
    lo = hi = None
    r = []
    for line in open(path):
        if line.startswith('#axis fixed'):
            _, _, _, lo, hi = line.split()
            lo, hi = float(lo), float(hi)
        elif line[0] != '#' and not line.startswith('entries'):
            r.append([float(v) for v in line.split(',')])
    r = np.array(r)
    nb = len(r) - 2
    c = (np.arange(nb) + 0.5) * (hi - lo) / nb + lo
    return c, r[1:-1, 1], r[1:-1, 3], r[0, 1] + r[-1, 1]



CASE = sys.argv[1]
N_INC = int(sys.argv[2]) if len(sys.argv) > 2 else (50000 if CASE == 'mami' else 10000)
INFO = {'mami': ('e− 855 MeV, Si (111) bent 0.905 mrad, 30.5 µm, divergence 30 × 70 µrad (examples ch1/ch2, MAMI)',
                 '11_geant4_radiation_Si111_bent.png'),
        'dia': ('e− 500 MeV, diamond (110) straight, 50 µm, divergence 50 × 50 µrad (example ch2 run_Radiation)',
                '12_geant4_radiation_diamond110.png')}[CASE]
names = set(os.listdir(rc.BUILD))
fn = lambda s, d, h: f'rad_{CASE}{s}_{d}_h1_{h}.csv'
seeds = sorted({int(m.group(1)) for n in names if (m := re.match(rf'rad_{CASE}(\d+)_off_h1_Spectrum\.csv$', n))})
seeds = [s for s in seeds if s < 900 and all(fn(s, d, h) in names for d in ('off', 'now') for h in ('Spectrum', 'thx'))]
ns = len(seeds)
if ns < 3:
    sys.exit(f'{CASE}: not enough complete seeds ({ns})')

EDGES = np.unique(np.r_[1, np.round(np.geomspace(1, 860, 26))])
WIN = {'mami': [('channeled, 700 to 1200 µrad', 700, 1200), ('volume reflection, -2000 to -300 µrad', -2000, -300),
                ('undeflected, -300 to 300 µrad', -300, 300), ('dechanneled, 300 to 700 µrad', 300, 700)],
       'dia': [('|θx| < 100 µrad', -100, 100), ('|θx| < 300 µrad', -300, 300)]}[CASE]
D = {d: [] for d in ('off', 'now')}
for s in seeds:
    for d in D:
        cg, wg, xg, ovf = load(os.path.join(rc.BUILD, fn(s, d, 'Spectrum')))
        idx = np.digitize(cg, EDGES) - 1
        ok = (idx >= 0) & (idx < len(EDGES) - 1)
        IE = np.array([xg[ok & (idx == k)].sum() for k in range(len(EDGES) - 1)]) / N_INC / np.diff(EDGES)
        c, w = rc.load_hist(os.path.join(rc.BUILD, fn(s, d, 'thx')))
        c, w = np.asarray(c), np.asarray(w, float)
        D[d].append(dict(IE=IE, Erad=xg.sum() / N_INC, Ngam=wg.sum() / N_INC, thx=w / N_INC,
                         **{lab: w[(c >= lo) & (c < hi)].sum() / N_INC for lab, lo, hi in WIN}))
cth = c


def ms(v):
    v = np.asarray(v)
    return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))


def ratio(key):
    A = np.array([x[key] for x in D['now']])
    B = np.array([x[key] for x in D['off']])
    with np.errstate(invalid='ignore', divide='ignore'):
        R = A.mean(0) / B.mean(0)
        E = (A - R * B).std(0, ddof=1) / np.sqrt(ns) / B.mean(0)
    return R, E, B.mean(0), A.mean(0)


rows = []
print(f'{CASE}: {ns} seeds x {N_INC} electrons per set')
for key in ['Erad', 'Ngam'] + [w[0] for w in WIN]:
    R, E, g, n = ratio(key)
    print(f'   {key:40s} Geant4 {g:.5g}  Nostri {n:.5g}  (N-G)/G = {100 * (R - 1):+.2f} % +- {100 * E:.2f} %  ({(R - 1) / E:+.1f} sigma)')
    rows.append([key, g, n, 100 * (R - 1), 100 * E])

Rs, Es, Gs, Ns = ratio('IE')
ok = np.isfinite(Es) & (Es > 0)
chi2s = np.sum(((Rs[ok] - 1) / Es[ok]) ** 2)
print(f'   photon spectrum: chi2 vs 1 = {chi2s:.1f} / {ok.sum()} bins, p = {stats.chi2.sf(chi2s, ok.sum()):.2f}')

# exit angle: merge 25 µrad bins into variable bins with >= 2000 Geant4 electrons in total
tot = np.sum([x['thx'] for x in D['off']], axis=0) * N_INC
edges_i, acc = [0], 0.0
for i in range(len(tot)):
    acc += tot[i]
    if acc >= 2000:
        edges_i.append(i + 1)
        acc = 0.0
edges_i[-1] = len(tot)
idx = np.array(edges_i[:-1])
for d in D:
    for x in D[d]:
        x['thb'] = np.add.reduceat(x['thx'], idx)
Rt, Et, Gt, Nt = ratio('thb')
okt = np.isfinite(Et) & (Et > 0)
chi2t = np.sum(((Rt[okt] - 1) / Et[okt]) ** 2)
print(f'   exit angle x: chi2 vs 1 = {chi2t:.1f} / {okt.sum()} bins, p = {stats.chi2.sf(chi2t, okt.sum()):.2f}')
be = np.r_[cth[idx] - 12.5, cth[-1] + 12.5]

with open(os.path.join(HERE, f'radiation_{CASE}.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['quantity', 'geant4', 'nostri', 'relative_difference_percent', 'relative_difference_err_percent'])
    for r in rows:
        w.writerow([r[0]] + [f'{v:.5g}' for v in r[1:]])
    w.writerow([])
    w.writerow(['E_low_MeV', 'E_high_MeV', 'geant4_EdNdE', 'nostri_EdNdE', 'relative_difference_percent', 'err_percent'])
    for k in range(len(Rs)):
        w.writerow([EDGES[k], EDGES[k + 1]] + [f'{v:.5g}' for v in (Gs[k], Ns[k], 100 * (Rs[k] - 1), 100 * Es[k])])

fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex='col', gridspec_kw={'height_ratios': [2.2, 1]})
ax, axr = axs[0, 0], axs[1, 0]
for v, col, ls, lab in [(Gs, rc.GEANT, '--', 'Geant4 (ufficiali)'), (Ns, rc.NOSTRI, '-', 'Nostri (modello atomico g4)')]:
    ax.stairs(v, EDGES, color=col, ls=ls, lw=1.6, label=lab)
ax.set_xscale('log')
ax.set_ylabel('E dN/dE per electron')
ax.legend(loc='upper right')
Ec = np.sqrt(EDGES[:-1] * EDGES[1:])
axr.axhline(0, color=rc.INK2, lw=0.8)
axr.errorbar(Ec[ok], 100 * (Rs[ok] - 1), yerr=100 * Es[ok], color=rc.NOSTRI, marker='o', ms=4, lw=1.2, elinewidth=1)
axr.set_ylabel(f'{rc.REL}  [%]', fontsize=8)
axr.set_xlabel('photon energy [MeV]')
ax.set_title(f'Photon spectrum      χ² = {chi2s:.1f} / {ok.sum()} bins', loc='left')
ax, axr = axs[0, 1], axs[1, 1]
for v, col, ls in [(Gt, rc.GEANT, '--'), (Nt, rc.NOSTRI, '-')]:
    ax.stairs(v / np.diff(be), be, color=col, ls=ls, lw=1.6)
lim = (-1500, 2000) if CASE == 'mami' else (-1500, 1500)
ax.set_xlim(*lim)
ax.set_ylabel('fraction of electrons per µrad')
ax.set_title(f'Exit angle θx      χ² = {chi2t:.1f} / {okt.sum()} bins', loc='left')
cb = 0.5 * (be[1:] + be[:-1])
axr.axhline(0, color=rc.INK2, lw=0.8)
axr.errorbar(cb[okt], 100 * (Rt[okt] - 1), yerr=100 * Et[okt], color=rc.NOSTRI, marker='o', ms=3, lw=0, elinewidth=1)
axr.set_ylabel(f'{rc.REL}  [%]', fontsize=8)
axr.set_xlabel('θx [µrad]  (variable bins, ≥ 2000 Geant4 electrons each)')
for a in axs.ravel():
    a.tick_params(length=0, labelsize=8, colors=rc.INK2)
fig.suptitle(f'Geant4 11.4.3 radiation: {INFO[0]}: Nostri vs Geant4 (ufficiali)', x=0.07, ha='left', fontsize=11)
txt = '; '.join(f'{r[0]} {r[3]:+.2f} % ± {r[4]:.2f} %' for r in rows)
fig.text(0.07, 0.01, f'{ns} seeds × {N_INC} electrons per set, same seeds; errors from the seed-to-seed scatter (paired). '
         f'(Nostri − Geant4)/Geant4:\n{txt}'.replace('Erad', 'radiated energy').replace('Ngam', 'photons'),
         fontsize=7.5, color=rc.INK2, wrap=True)
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15, wspace=0.22, hspace=0.1)
fig.savefig(os.path.join(HERE, INFO[1]), dpi=150)
print('figure written')
