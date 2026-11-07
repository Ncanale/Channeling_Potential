"""Full exit-angle distribution, Nostri vs Geant4 (ufficiali), 14 seeds summed per point (280k events each).
Two-sample chi2 on the histogram (bins merged to >= 20 counts), and window fractions at several half-widths
with their seed-to-seed error."""
import os

import numpy as np
from scipy import stats

BUILD = r'\\wsl.localhost\Ubuntu-24.04\home\tacab\chtest\ch2\build'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rocking_comparison import available_tilts  # noqa: E402
TILTS = available_tilts()   # every tilt run for all 14 seeds
LATS = [100, 110, 111]
SEEDS = list(range(1, 15))
HALF = [25, 50, 100, 200, 400, 1000]


def load(path):
    lo = hi = None
    rows = []
    for line in open(path):
        if line.startswith('#axis fixed'):
            _, _, _, lo, hi = line.split()
            lo, hi = float(lo), float(hi)
        elif line[0] != '#' and not line.startswith('entries'):
            rows.append([float(v) for v in line.split(',')])
    r = np.array(rows)[1:-1]
    return (np.arange(len(r)) + 0.5) * (hi - lo) / len(r) + lo, r[:, 1]


def label(s, d, lat, t):
    return f"{'nw' if s == 1 else f'sd{s}'}_{d}_{lat}_{t}_h1_thx.csv"


def frac_centered(c, w, half, centre):
    return w[np.abs(c - centre) <= half].sum() / w.sum()


def merge(a, b, nmin=20):
    oa, ob, ca, cb = [], [], 0.0, 0.0
    for x, y in zip(a, b):
        ca += x
        cb += y
        if ca + cb >= nmin:
            oa.append(ca)
            ob.append(cb)
            ca = cb = 0.0
    if ca + cb > 0 and oa:
        oa[-1] += ca
        ob[-1] += cb
    return np.array(oa), np.array(ob)


chi_tot, dof_tot = 0.0, 0
rows = []
for lat in LATS:
    for t in TILTS:
        H = {d: [] for d in ('off', 'now')}
        for s in SEEDS:
            for d in H:
                c, w = load(os.path.join(BUILD, label(s, d, lat, t)))
                H[d].append(w)
        A, B = np.sum(H['off'], 0), np.sum(H['now'], 0)
        centre = c[np.argmax(np.convolve(A + B, np.ones(8), 'same'))]
        a, b = merge(A, B)
        Na, Nb = a.sum(), b.sum()
        chi = np.sum((np.sqrt(Nb / Na) * a - np.sqrt(Na / Nb) * b) ** 2 / (a + b))
        dof = len(a) - 1
        chi_tot += chi
        dof_tot += dof
        mean_off = np.sum(A * c) / A.sum()
        rms_off = np.sqrt(np.sum(A * (c - mean_off) ** 2) / A.sum())
        rms_now = np.sqrt(np.sum(B * (c - np.sum(B * c) / B.sum()) ** 2) / B.sum())
        fr = []
        for h in HALF:
            dd = np.array([frac_centered(c, H['now'][i], h, centre) - frac_centered(c, H['off'][i], h, centre)
                           for i in range(len(SEEDS))])
            fo = frac_centered(c, A, h, centre)
            fr.append((dd.mean() / fo, dd.std(ddof=1) / np.sqrt(len(SEEDS)) / fo))
        rows.append((lat, t, chi, dof, stats.chi2.sf(chi, dof), rms_off, rms_now, fr))
        print(f'<{lat}> {t:+5d}: chi2 {chi:7.1f}/{dof:4d}  p {stats.chi2.sf(chi, dof):.3f}  '
              f'rms off {rms_off:7.1f} now {rms_now:7.1f} urad  ({100 * (rms_now / rms_off - 1):+.2f}%)')
print(f'TOTAL chi2 {chi_tot:.1f} / {dof_tot}  p {stats.chi2.sf(chi_tot, dof_tot):.3f}')
print('\nwindow fraction (centred on the peak), relative difference Nostri/Geant4 - 1, mean over all points:')
for k, h in enumerate(HALF):
    r = np.array([x[7][k][0] for x in rows])
    e = np.array([x[7][k][1] for x in rows])
    wm = np.sum(r / e ** 2) / np.sum(1 / e ** 2)
    we = 1 / np.sqrt(np.sum(1 / e ** 2))
    print(f'  +-{h:5d} urad: weighted mean {100 * wm:+.3f}% +- {100 * we:.3f}%   chi2 vs 0 {np.sum((r / e) ** 2):.1f}/{len(rows)}   '
          f'max |pt| {100 * np.abs(r).max():.2f}%')
