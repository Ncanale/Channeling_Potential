"""Geant4 axial rocking curves in Ge and diamond: Nostri vs Geant4 (ufficiali), same settings as Si (figure 05).

Labels gc{seed}_{off|now}_{Ge|Diamond}_{axis}_{tilt}; uses every seed and tilt complete for both sets and all axes.
Output: 08_geant4_rocking_Ge.png, 09_geant4_rocking_Diamond.png, rocking_fraction_GeC.csv
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

LATS = rc.LATS
U = {'Ge': 0.085, 'Diamond': 0.040}
FIG = {'Ge': '08_geant4_rocking_Ge.png', 'Diamond': '09_geant4_rocking_Diamond.png'}


def fname(s, tag, mat, lat, t):
    return f'gc{s}_{tag}_{mat}_{lat}_{t}_h1_thx.csv'


def available(mat):
    names = set(os.listdir(rc.BUILD))
    pat = re.compile(rf'gc(\d+)_off_{mat}_100_(-?\d+)_h1_thx\.csv$')
    seeds = sorted({int(m.group(1)) for n in names if (m := pat.match(n))})
    tilts = sorted({int(m.group(2)) for n in names if (m := pat.match(n))})
    ok = lambda s, t: all(fname(s, tag, mat, lat, t) in names for tag in ('off', 'now') for lat in LATS)
    return [s for s in seeds if all(ok(s, t) for t in tilts)], tilts      # only seeds complete for every tilt


rows = []
for mat in ['Ge', 'Diamond']:
    seeds, tilts = available(mat)
    if len(seeds) < 2 or not tilts:
        print(f'{mat}: not enough runs yet ({len(seeds)} seeds)')
        continue
    ns = len(seeds)
    F = {(lat, tag, t): np.array([rc.window_fraction(*rc.load_hist(os.path.join(rc.BUILD, fname(s, tag, mat, lat, t))))
                                  for s in seeds]) for lat in LATS for tag in ('off', 'now') for t in tilts}
    # With few seeds the scatter of a single point is poorly known (ns - 1 degrees of freedom), so the variance is
    # pooled over the 39 points in relative terms (the relative scatter of a ±100 µrad fraction from 20 000 events is
    # nearly the same everywhere): 39 (ns - 1) degrees of freedom. The binomial expectation is printed as a check.
    keys = [(lat, t) for lat in LATS for t in tilts]
    mean_off = {k: F[(k[0], 'off', k[1])].mean() for k in keys}
    rv = lambda tag: np.mean([F[(k[0], tag, k[1])].var(ddof=1) / mean_off[k] ** 2 for k in keys])
    rvd = np.mean([(F[(k[0], 'now', k[1])] - F[(k[0], 'off', k[1])]).var(ddof=1) / mean_off[k] ** 2 for k in keys])
    rock = {(lat, tag, t): (F[(lat, tag, t)].mean(), np.sqrt(rv(tag) / ns) * mean_off[(lat, t)])
            for lat in LATS for tag in ('off', 'now') for t in tilts}
    diff = {k: ((F[(k[0], 'now', k[1])] - F[(k[0], 'off', k[1])]).mean(), np.sqrt(rvd / ns) * mean_off[k]) for k in keys}
    dm = np.array([diff[k][0] for k in keys])
    de = np.array([diff[k][1] for k in keys])
    rel = 100 * dm / np.array([mean_off[k] for k in keys])
    rel_m, rel_e = rel.mean(), 100 * np.sqrt(rvd / ns / len(keys))         # points independent
    per_seed = np.mean([(F[(k[0], 'now', k[1])] - F[(k[0], 'off', k[1])]) / mean_off[k] for k in keys], axis=0)
    rel_e_seed = 100 * per_seed.std(ddof=1) / np.sqrt(ns)                  # check: scatter of the per-seed means
    chi2 = np.sum((dm / de) ** 2)
    binom = 100 * np.sqrt(np.mean([(1 - mean_off[k]) / (mean_off[k] * 20000) for k in keys]))
    print(f'{mat}: {len(tilts)} tilts x 3 axes x {ns} seeds; single run relative scatter {100 * np.sqrt(rv("off")):.2f} % '
          f'(Geant4), {100 * np.sqrt(rv("now")):.2f} % (Nostri), binomial {binom:.2f} %; difference {100 * np.sqrt(rvd):.2f} %')
    print(f'   (Nostri - Geant4)/Geant4 = {rel_m:+.3f} % +- {rel_e:.3f} % (from per-seed means: +- {rel_e_seed:.3f} %), '
          f'chi2 vs 0 = {chi2:.1f} / {len(dm)}, points beyond 2 sigma: {int(np.sum(np.abs(dm / de) > 2))}')
    for lat in LATS:
        for t in tilts:
            go, ge = rock[(lat, 'off', t)]
            no, ne = rock[(lat, 'now', t)]
            d_, e_ = diff[(lat, t)]
            rows.append([mat, f'<{lat}>', t, f'{go:.5f}', f'{ge:.5f}', f'{no:.5f}', f'{ne:.5f}', f'{d_:+.5f}', f'{e_:.5f}',
                         f'{100 * d_ / go:+.3f}', f'{100 * e_ / go:.3f}', ns, 20000])

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for j, lat in enumerate(LATS):
        ax, axd = axs[0, j], axs[1, j]
        for ax_ in (ax, axd):
            ax_.tick_params(length=0, labelsize=8, colors=rc.INK2)
        for tag, lab, col, ls in rc.SETS:
            ax.errorbar(tilts, [rock[(lat, tag, t)][0] for t in tilts], yerr=[rock[(lat, tag, t)][1] for t in tilts],
                        color=col, ls=ls, marker='o', ms=4.5, lw=1.5, elinewidth=1, capsize=0)
        ax.set_title(f'<{lat}>', loc='left')
        if j == 0:
            ax.set_ylabel('fraction within ±100 µrad')
        axd.axhline(0, color=rc.INK2, lw=0.8)
        axd.errorbar(tilts, [100 * diff[(lat, t)][0] / rock[(lat, 'off', t)][0] for t in tilts],
                     yerr=[100 * diff[(lat, t)][1] / rock[(lat, 'off', t)][0] for t in tilts],
                     color=rc.NOSTRI, marker='o', ms=4.5, lw=1.5, elinewidth=1, capsize=0)
        axd.set_ylim(-3, 3)
        axd.set_xlabel('tilt [µrad]')
        if j == 0:
            axd.set_ylabel(f'{rc.REL}  [%]', fontsize=8)
    handles = [Line2D([0], [0], color=col, ls=ls, marker='o', ms=5, lw=1.5, label=lab) for tag, lab, col, ls in rc.SETS]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f'Geant4 11.4.3 axial rocking curves: e− 855 MeV, {mat} 30 µm, u = {U[mat]:.3f} Å: Nostri vs Geant4 (ufficiali)',
                 x=0.07, ha='left', fontsize=11.5)
    fig.text(0.07, 0.01, f'Fraction of the exit-angle histogram in the best ±100 µrad window; mean of {ns} seeds × 20 000 events '
             f'per point, same seeds for both sets.\nErrors: seed-to-seed scatter / √{ns}, relative variance pooled over the {len(dm)} points (paired for the difference). '
             f'Mean {rc.REL} over the {len(dm)} points: {rel_m:+.2f} % ± {rel_e:.2f} %, χ² = {chi2:.1f} / {len(dm)}.',
             fontsize=8, color=rc.INK2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.13, wspace=0.22, hspace=0.1)
    fig.savefig(os.path.join(HERE, FIG[mat]), dpi=150)
    plt.close(fig)

if rows:
    with open(os.path.join(HERE, 'rocking_fraction_GeC.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['material', 'axis', 'tilt_urad', 'geant4_fraction', 'geant4_err', 'nostri_fraction', 'nostri_err',
                    'difference', 'difference_err', 'relative_difference_percent', 'relative_difference_err_percent',
                    'n_seeds', 'events_per_seed'])
        w.writerows(rows)
    print('figures written')
