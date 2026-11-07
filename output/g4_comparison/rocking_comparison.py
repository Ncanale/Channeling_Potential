"""Geant4 axial rocking curves in Si: Nostri vs Geant4 (ufficiali), same build, same seeds for both sets.

Metric: fraction of the exit-angle histogram (x) inside the best +-100 urad window, 20000 events per point and seed.
Uses every tilt for which all seeds, axes and both sets exist (labels nw_* for seed 1, sd{seed}_* for seeds 2-14).
Errors: seed-to-seed std / sqrt(n_seeds), paired for the difference.
Output: 05_geant4_rocking_axes.png, rocking_fraction.csv
"""
import csv
import os
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))
BUILD = r'\\wsl.localhost\Ubuntu-24.04\home\tacab\chtest\ch2\build'
SEEDS = list(range(1, 15))
LATS = [100, 110, 111]
SURF, INK, INK2, BASE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#c3c2b7', '#e1e0d9'
NOSTRI, GEANT = '#ff7f0e', INK2
SETS = [('off', 'Geant4 (ufficiali)', GEANT, '--'), ('now', 'Nostri (Thomas-Fermi per atomo)', NOSTRI, '-')]
REL = '(Nostri − Geant4) / Geant4'

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 9.5,
    'axes.edgecolor': BASE, 'axes.linewidth': 0.8, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'], 'font.size': 9,
    'legend.frameon': False, 'legend.fontsize': 8.5,
})


def load_hist(path):
    lo = hi = None
    rows = []
    for line in open(path):
        if line.startswith('#axis fixed'):
            _, _, _, lo, hi = line.split()
            lo, hi = float(lo), float(hi)
        elif line[0] != '#' and not line.startswith('entries'):
            rows.append([float(v) for v in line.split(',')])
    r = np.array(rows)[1:-1]
    c = (np.arange(len(r)) + 0.5) * (hi - lo) / len(r) + lo
    return c, r[:, 1]


def window_fraction(c, w, half=100.0):
    k = int(round(2 * half / (c[1] - c[0])))
    return np.convolve(w, np.ones(k), 'same').max() / w.sum()


def fname(s, tag, lat, t):
    return f"{'nw' if s == 1 else f'sd{s}'}_{tag}_{lat}_{t}_h1_thx.csv"


def available_tilts():
    names = set(os.listdir(BUILD))
    cand = {int(m.group(1)) for n in names if (m := re.match(r'nw_off_100_(-?\d+)_h1_thx\.csv$', n))}
    return sorted(t for t in cand if all(fname(s, tag, lat, t) in names
                                         for s in SEEDS for tag in ('off', 'now') for lat in LATS))


def load_all():
    """F[(lat, tag, tilt)] = array of the fraction per seed."""
    tilts = available_tilts()
    F = {(lat, tag, t): np.array([window_fraction(*load_hist(os.path.join(BUILD, fname(s, tag, lat, t)))) for s in SEEDS])
         for lat in LATS for tag in ('off', 'now') for t in tilts}
    return tilts, F


if __name__ == '__main__':
    TILTS, F = load_all()
    ns = len(SEEDS)
    rock = {k: (v.mean(), v.std(ddof=1) / np.sqrt(ns)) for k, v in F.items()}
    diff = {}
    for lat in LATS:
        for t in TILTS:
            dd = F[(lat, 'now', t)] - F[(lat, 'off', t)]
            diff[(lat, t)] = (dd.mean(), dd.std(ddof=1) / np.sqrt(ns))
    dm = np.array([v[0] for v in diff.values()])
    de = np.array([v[1] for v in diff.values()])
    npts = len(dm)
    per_seed = np.mean([F[(l, 'now', t)] - F[(l, 'off', t)] for l in LATS for t in TILTS], axis=0)
    g_mean = np.mean([rock[k][0] for k in rock if k[1] == 'off'])
    rel_m, rel_e = 100 * per_seed.mean() / g_mean, 100 * per_seed.std(ddof=1) / np.sqrt(ns) / g_mean
    chi2 = np.sum((dm / de) ** 2)
    print(f'tilts {TILTS}: {npts} points x {ns} seeds')
    print(f'mean over points, (Nostri - Geant4) / Geant4: {rel_m:+.3f} % +- {rel_e:.3f} %   chi2 vs 0: {chi2:.1f} / {npts}')

    with open(os.path.join(OUT, 'rocking_fraction.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['axis', 'tilt_urad', 'geant4_fraction', 'geant4_err', 'nostri_fraction', 'nostri_err',
                    'difference', 'difference_err', 'relative_difference_percent', 'relative_difference_err_percent',
                    'n_seeds', 'events_per_seed'])
        for lat in LATS:
            for t in TILTS:
                go, ge = rock[(lat, 'off', t)]
                no, ne = rock[(lat, 'now', t)]
                d_, e_ = diff[(lat, t)]
                w.writerow([f'<{lat}>', t, f'{go:.5f}', f'{ge:.5f}', f'{no:.5f}', f'{ne:.5f}',
                            f'{d_:+.5f}', f'{e_:.5f}', f'{100 * d_ / go:+.3f}', f'{100 * e_ / go:.3f}', ns, 20000])

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for j, lat in enumerate(LATS):
        ax, axd = axs[0, j], axs[1, j]
        for ax_ in (ax, axd):
            ax_.tick_params(length=0, labelsize=8, colors=INK2)
        for tag, lab, col, ls in SETS:
            ax.errorbar(TILTS, [rock[(lat, tag, t)][0] for t in TILTS], yerr=[rock[(lat, tag, t)][1] for t in TILTS],
                        color=col, ls=ls, marker='o', ms=4.5, lw=1.5, elinewidth=1, capsize=0)
        ax.set_title(f'<{lat}>', loc='left')
        if j == 0:
            ax.set_ylabel('fraction within ±100 µrad')
        axd.axhline(0, color=INK2, lw=0.8)
        axd.errorbar(TILTS, [100 * diff[(lat, t)][0] / rock[(lat, 'off', t)][0] for t in TILTS],
                     yerr=[100 * diff[(lat, t)][1] / rock[(lat, 'off', t)][0] for t in TILTS],
                     color=NOSTRI, marker='o', ms=4.5, lw=1.5, elinewidth=1, capsize=0)
        axd.set_ylim(-2.5, 2.5)
        axd.set_xlabel('tilt [µrad]')
        if j == 0:
            axd.set_ylabel(f'{REL}  [%]', fontsize=8)
    handles = [Line2D([0], [0], color=col, ls=ls, marker='o', ms=5, lw=1.5, label=lab) for tag, lab, col, ls in SETS]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle('Geant4 11.4.3 axial rocking curves: e− 855 MeV, Si 30 µm, u = 0.075 Å: Nostri vs Geant4 (ufficiali)',
                 x=0.07, ha='left', fontsize=11.5)
    fig.text(0.07, 0.01, f'Fraction of the exit-angle histogram in the best ±100 µrad window; mean of {ns} seeds × 20 000 events '
             f'per point, same seeds for both sets.\nErrors: seed-to-seed std / √{ns} (paired for the difference). '
             f'Mean {REL} over the {npts} points: {rel_m:+.2f} % ± {rel_e:.2f} %, χ² = {chi2:.1f} / {npts}.',
             fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.13, wspace=0.22, hspace=0.1)
    fig.savefig(os.path.join(OUT, '05_geant4_rocking_axes.png'), dpi=150)
    print('figure written')
