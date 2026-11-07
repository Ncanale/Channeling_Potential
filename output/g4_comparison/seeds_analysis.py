"""Seed-to-seed scatter of the rocking-curve comparison: Nostri vs Geant4 (ufficiali), seeds 1-4."""
import os

import numpy as np

BUILD = r'\\wsl.localhost\Ubuntu-24.04\home\tacab\chtest\ch2\build'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rocking_comparison import available_tilts  # noqa: E402
TILTS = available_tilts()   # every tilt run for all 14 seeds
LATS = [100, 110, 111]
SEEDS = list(range(1, 15))


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
    n = w.sum()
    k = int(round(2 * half / (c[1] - c[0])))
    cs = np.convolve(w, np.ones(k), 'same')
    return cs.max() / n


def label(seed, d, lat, t):
    prefix = 'nw' if seed == 1 else f'sd{seed}'
    return f'{prefix}_{d}_{lat}_{t}_h1_thx.csv'


# frac[seed][set][(lat, tilt)]
frac = {s: {d: {} for d in ('off', 'now')} for s in SEEDS}
for s in SEEDS:
    for d in ('off', 'now'):
        for lat in LATS:
            for t in TILTS:
                c, w = load_hist(os.path.join(BUILD, label(s, d, lat, t)))
                frac[s][d][(lat, t)] = window_fraction(c, w)

points = [(lat, t) for lat in LATS for t in TILTS]
mean_off = np.array([np.mean([frac[s]['off'][p] for p in points]) for s in SEEDS])
mean_now = np.array([np.mean([frac[s]['now'][p] for p in points]) for s in SEEDS])
diff_mean = mean_now - mean_off
print(f"per seed: mean fraction ({len(TILTS) * len(LATS)} points)")
for i, s in enumerate(SEEDS):
    print(f'  seed {s}: Geant4 {mean_off[i]:.4f}   Nostri {mean_now[i]:.4f}   Nostri-Geant4 {diff_mean[i]:+.4f}')
print(f'seed-to-seed std: Geant4 {mean_off.std(ddof=1):.4f}, Nostri {mean_now.std(ddof=1):.4f}, difference {diff_mean.std(ddof=1):.4f}')
rel = 100 * diff_mean / mean_off
print(f'difference in % of the Geant4 value, per seed: ' + ' '.join(f'{x:+.2f}%' for x in rel))
print(f'mean difference {diff_mean.mean():+.4f} +- {diff_mean.std(ddof=1) / np.sqrt(len(SEEDS)):.4f} (sem over seeds)')
print(f'5 % of the Geant4 value would be {0.05 * mean_off.mean():.4f} in fraction')

# per point: difference per seed, scatter across seeds
pd = np.array([[frac[s]['now'][p] - frac[s]['off'][p] for p in points] for s in SEEDS])
print(f'per point ({len(points)} points x {len(SEEDS)} seeds): rms of the difference {np.sqrt((pd ** 2).mean()):.4f}')
print(f'per point, seed-to-seed std of the difference, median over points: {np.median(pd.std(axis=0, ddof=1)):.4f}')
pm = pd.mean(axis=0)
ps = pd.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
chi2 = np.sum((pm / ps) ** 2)
print(f'chi2 of per-point mean difference vs 0: {chi2:.1f} / {len(points)} dof')
for i, p in enumerate(points):
    off = np.mean([frac[s]['off'][p] for s in SEEDS])
    print(f'  <{p[0]}> {p[1]:+5d}: Geant4 {off:.4f}  diff {pm[i]:+.4f} +- {ps[i]:.4f}  ({100 * pm[i] / off:+.2f}%, {pm[i] / ps[i]:+.1f} sigma)')
for lat in LATS:
    idx = [i for i, p in enumerate(points) if p[0] == lat]
    per_axis = pd[:, idx].mean(axis=1)
    print(f'  <{lat}>: difference per seed {per_axis.round(4)}  mean {per_axis.mean():+.4f}  std {per_axis.std(ddof=1):.4f}')
