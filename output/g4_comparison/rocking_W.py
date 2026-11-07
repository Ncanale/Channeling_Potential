"""Geant4 axial rocking curves in W <111>: Nostri vs Geant4 (ufficiali).

e- 855 MeV, 10 µm, no radiation, 13 tilts (0, ±500 ... ±3000 µrad), same seeds for both sets; labels wr{seed}_{off|now}_{tilt}.
In W the ±100 µrad fraction used for Si, Ge and diamond hardly changes with the tilt (multiple scattering of 1.2 mrad,
Lindhard angle 1.44 mrad), so three observables are compared:
  - fraction of the exit-angle histogram in the best ±500 µrad window;
  - mean exit angle (the potential pulls the beam towards the axis);
  - rms width of the exit-angle distribution.
Errors: seed-to-seed scatter / √n, the variance pooled over the 13 tilts (relative for the fraction and the width,
absolute for the mean angle), paired for the difference.
Output: 10_geant4_rocking_W111.png, rocking_W111.csv
"""
import csv
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rocking_comparison as rc  # noqa: E402


def fname(s, tag, t):
    return f'wr{s}_{tag}_{t}_h1_thx.csv'


def observables(path):
    c, w = rc.load_hist(path)
    w = np.asarray(w, float)
    n = w.sum()
    m = (c * w).sum() / n
    return np.array([rc.window_fraction(c, w, 500.0), m, np.sqrt((w * (c - m) ** 2).sum() / n)])


names = set(os.listdir(rc.BUILD))
pat = re.compile(r'wr(\d+)_off_(-?\d+)_h1_thx\.csv$')
seeds = sorted({int(m.group(1)) for n in names if (m := pat.match(n))})
tilts = sorted({int(m.group(2)) for n in names if (m := pat.match(n))})
seeds = [s for s in seeds if all(fname(s, tag, t) in names for tag in ('off', 'now') for t in tilts)]
ns = len(seeds)
if ns < 2:
    sys.exit(f'not enough complete seeds yet ({ns})')

# O[(tag, t)] = array (seed, observable)
O = {(tag, t): np.array([observables(os.path.join(rc.BUILD, fname(s, tag, t))) for s in seeds])
     for tag in ('off', 'now') for t in tilts}
D = {t: O[('now', t)] - O[('off', t)] for t in tilts}
G = {t: O[('off', t)].mean(0) for t in tilts}
REL = [True, False, True]                     # observable compared in relative terms?
NAMES = ['fraction within ±500 µrad', 'mean exit angle [µrad]', 'rms exit angle [µrad]']


def pooled(arrs, k, rel):
    v = [a[:, k].var(ddof=1) / (G[t][k] ** 2 if rel else 1) for t, a in arrs.items()]
    return np.mean(v)


rows = []
summary = []
for k in range(3):
    vd = pooled(D, k, REL[k])
    vg = pooled({t: O[('off', t)] for t in tilts}, k, REL[k])
    vn = pooled({t: O[('now', t)] for t in tilts}, k, REL[k])
    sc = lambda t: G[t][k] if REL[k] else 1.0
    d = np.array([D[t][:, k].mean() for t in tilts])
    e = np.array([np.sqrt(vd / ns) * sc(t) for t in tilts])
    r = np.array([100 * D[t][:, k].mean() / G[t][k] if REL[k] else D[t][:, k].mean() for t in tilts])
    re_ = np.array([100 * np.sqrt(vd / ns) if REL[k] else np.sqrt(vd / ns) for t in tilts])
    chi2 = float(np.sum((d / e) ** 2))
    mean_r, err_r = r.mean(), np.sqrt(np.sum(re_ ** 2)) / len(r)
    per_seed = np.mean([D[t][:, k] / (G[t][k] if REL[k] else 1) for t in tilts], axis=0) * (100 if REL[k] else 1)
    unit = '%' if REL[k] else 'µrad'
    print(f'{NAMES[k]}: {ns} seeds; single-run scatter G {np.sqrt(vg) * (100 if REL[k] else 1):.2f}, '
          f'N {np.sqrt(vn) * (100 if REL[k] else 1):.2f}, difference {np.sqrt(vd) * (100 if REL[k] else 1):.2f} {unit}')
    print(f'   mean difference over {len(tilts)} tilts {mean_r:+.3f} +- {err_r:.3f} {unit} '
          f'(from per-seed means +- {per_seed.std(ddof=1) / np.sqrt(ns):.3f}); chi2 vs 0 = {chi2:.1f} / {len(tilts)}, '
          f'beyond 2 sigma: {int(np.sum(np.abs(d / e) > 2))}')
    summary.append((k, r, re_, mean_r, err_r, chi2, unit))
    for i, t in enumerate(tilts):
        rows.append([NAMES[k], t, f'{G[t][k]:.6g}', f'{O[("now", t)][:, k].mean():.6g}', f'{d[i]:+.4g}', f'{e[i]:.3g}',
                     f'{r[i]:+.4g}', f'{re_[i]:.3g}', unit, ns, 20000])

with open(os.path.join(HERE, 'rocking_W111.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['observable', 'tilt_urad', 'geant4', 'nostri', 'difference', 'difference_err', 'residual', 'residual_err',
                'residual_unit', 'n_seeds', 'events_per_seed'])
    w.writerows(rows)

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
RES_LABEL = [f'{rc.REL}  [%]', 'Nostri − Geant4  [µrad]', f'{rc.REL}  [%]']
for k in range(3):
    ax, axd = axs[0, k], axs[1, k]
    for ax_ in (ax, axd):
        ax_.tick_params(length=0, labelsize=8, colors=rc.INK2)
    for tag, lab, col, ls in rc.SETS:
        y = [O[(tag, t)][:, k].mean() for t in tilts]
        ax.plot(tilts, y, color=col, ls=ls, marker='o', ms=4.5, lw=1.5)
    if k == 1:
        ax.plot(tilts, tilts, color=rc.BASE, lw=1, zorder=0)
        ax.text(tilts[-1], tilts[-1], ' = tilt', color=rc.INK2, fontsize=8, va='center')
    ax.set_title(NAMES[k], loc='left', fontsize=10)
    _, r, re_, mean_r, err_r, chi2, unit = summary[k]
    axd.axhline(0, color=rc.INK2, lw=0.8)
    axd.errorbar(tilts, r, yerr=re_, color=rc.NOSTRI, marker='o', ms=4.5, lw=1.5, elinewidth=1, capsize=0)
    axd.set_ylabel(RES_LABEL[k], fontsize=8)
    axd.set_xlabel('tilt [µrad]')
    axd.text(0.02, 0.04, f'mean {mean_r:+.2f} ± {err_r:.2f} {unit}, χ² = {chi2:.1f} / {len(tilts)}',
             transform=axd.transAxes, fontsize=8, color=rc.INK2)
lim = max(3 * np.max(summary[0][2]), 1)
axs[1, 0].set_ylim(-lim, lim)
axs[1, 2].set_ylim(-max(3 * np.max(summary[2][2]), 1), max(3 * np.max(summary[2][2]), 1))
handles = [Line2D([0], [0], color=col, ls=ls, marker='o', ms=5, lw=1.5, label=lab) for tag, lab, col, ls in rc.SETS]
fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.955))
fig.suptitle('Geant4 11.4.3 axial rocking curves: e− 855 MeV, W <111> 10 µm, u = 0.050 Å: Nostri vs Geant4 (ufficiali)',
             x=0.07, ha='left', fontsize=11.5)
fig.text(0.07, 0.01, f'Exit-angle histogram in x; mean of {ns} seeds × 20 000 events per point, same seeds for both sets. '
         f'Errors: seed-to-seed scatter / √{ns},\nvariance pooled over the {len(tilts)} tilts (relative for fraction and '
         'width, absolute for the mean angle), paired for the difference. Error bars on the curves are smaller than the markers.',
         fontsize=8, color=rc.INK2)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.13, wspace=0.28, hspace=0.1)
fig.savefig(os.path.join(HERE, '10_geant4_rocking_W111.png'), dpi=150)
print('figure written')
