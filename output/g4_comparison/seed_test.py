"""Seed test: is Nostri distinguishable from Geant4 (ufficiali) run with a different seed?

The reference is Geant4 against itself: two runs with the official files and different seeds differ only by
statistics. If our files give the same physics, Nostri vs Geant4 (different seeds) must look like that null test.

Rocking (Si, all tilts available for 14 seeds):
  (a) per point, the 14 per-seed values of Geant4 and Nostri, as (X_s - <G>) / <G>, <G> = mean of Geant4 over seeds;
  (b) distribution over points of (G_s - G_s+1) / <G>  (null test)  and  (N_s - G_s+1) / <G>  (different seeds).
Radiation (W <111> 100 um, 20 seeds x 2000 e-):
  (c) per-seed radiated energy per incident electron, Geant4 and Nostri;  (d) the same for 178-438 MeV.
Two-sample tests: Kolmogorov-Smirnov and Welch t on the per-seed values / on the two difference distributions.
Output: 07_seed_test.png, seed_test.csv
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rocking_comparison as rc  # noqa: E402

BUILD = rc.BUILD
SURF, INK, INK2, GEANT, NOSTRI = rc.SURF, rc.INK, rc.INK2, rc.GEANT, rc.NOSTRI
W_SEEDS = list(range(1, 21))
N_INC = 2000

# ---------------------------------------------------------------- rocking
TILTS, F = rc.load_all()
pts = [(lat, t) for lat in rc.LATS for t in TILTS]
S = len(rc.SEEDS)
G = np.array([F[(lat, 'off', t)] for lat, t in pts])          # (points, seeds)
N = np.array([F[(lat, 'now', t)] for lat, t in pts])
Gm = G.mean(1, keepdims=True)
null = 100 * (G - np.roll(G, -1, axis=1)) / Gm               # G_s - G_s+1 (cyclic)
cross = 100 * (N - np.roll(G, -1, axis=1)) / Gm              # N_s - G_s+1
same = 100 * (N - G) / Gm                                    # N_s - G_s (same seed)
ks_r = stats.ks_2samp(null.ravel(), cross.ravel())
sd_g = 100 * (G / Gm - 1).std(1, ddof=1)                      # seed scatter per point
sd_n = 100 * (N / Gm - 1).std(1, ddof=1)
t_pts = np.array([stats.ttest_ind(N[i], G[i], equal_var=False).pvalue for i in range(len(pts))])
print(f'rocking: {len(pts)} points x {S} seeds')
print(f'  null (G_s - G_s+1)/<G>:  mean {null.mean():+.3f} %  std {null.std(ddof=1):.3f} %')
print(f'  cross (N_s - G_s+1)/<G>: mean {cross.mean():+.3f} %  std {cross.std(ddof=1):.3f} %   KS p = {ks_r.pvalue:.3f}')
print(f'  same seed (N_s - G_s)/<G>: mean {same.mean():+.3f} %  std {same.std(ddof=1):.3f} %')
print(f'  seed scatter per point, median: Geant4 {np.median(sd_g):.2f} %, Nostri {np.median(sd_n):.2f} %')
print(f'  per-point Welch t-test N vs G: p < 0.05 in {np.sum(t_pts < 0.05)} of {len(pts)} points '
      f'(expected by chance {0.05 * len(pts):.1f}); KS of p-values vs uniform p = {stats.kstest(t_pts, "uniform").pvalue:.3f}')


# ---------------------------------------------------------------- radiation
def rad(s, d):
    path = os.path.join(BUILD, f'rW_s{s}_{d}_W111_h1_Spectrum.csv')
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
    band = (c >= 178) & (c < 438)
    return r[:, 3].sum() / N_INC, r[band, 3].sum() / N_INC


R = {d: np.array([rad(s, d) for s in W_SEEDS]) for d in ('off', 'now')}
rad_rows = []
for k, name in enumerate(['radiated energy per e-', 'radiated energy 178-438 MeV per e-']):
    g, n = R['off'][:, k], R['now'][:, k]
    ks, tt = stats.ks_2samp(g, n), stats.ttest_ind(n, g, equal_var=False)
    ftest = stats.levene(g, n)
    print(f'radiation, {name}: Geant4 {g.mean():.3f} (seed std {g.std(ddof=1):.3f})  Nostri {n.mean():.3f} '
          f'(seed std {n.std(ddof=1):.3f}) MeV;  KS p {ks.pvalue:.3f}  t p {tt.pvalue:.3f}  Levene p {ftest.pvalue:.3f}')
    rad_rows.append((name, g, n, ks.pvalue, tt.pvalue, ftest.pvalue))

with open(os.path.join(HERE, 'seed_test.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['test', 'geant4_mean', 'geant4_seed_std', 'nostri_mean', 'nostri_seed_std', 'p_KS', 'p_t', 'note'])
    w.writerow(['rocking: null (G_s - G_s+1)/<G> vs cross (N_s - G_s+1)/<G> [%]', f'{null.mean():.4f}',
                f'{null.std(ddof=1):.4f}', f'{cross.mean():.4f}', f'{cross.std(ddof=1):.4f}', f'{ks_r.pvalue:.4f}', '',
                f'{len(pts)} points x {S} seeds'])
    w.writerow(['rocking: per-point seed scatter, median [%]', '', f'{np.median(sd_g):.4f}', '', f'{np.median(sd_n):.4f}',
                '', '', ''])
    for name, g, n, pk, pt, pl in rad_rows:
        w.writerow([f'radiation W<111>: {name} [MeV]', f'{g.mean():.4f}', f'{g.std(ddof=1):.4f}', f'{n.mean():.4f}',
                    f'{n.std(ddof=1):.4f}', f'{pk:.4f}', f'{pt:.4f}', f'{len(W_SEEDS)} seeds x {N_INC} e-; Levene p {pl:.3f}'])

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.5, 9.5))
gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.38,
                      left=0.07, right=0.98, top=0.88, bottom=0.12)
ax = fig.add_subplot(gs[0, :2])
x = np.arange(len(pts))
jit = np.linspace(-0.12, 0.12, S)
ax.scatter((x[:, None] - 0.2 + jit).ravel(), (100 * (G / Gm - 1)).ravel(), s=7, color=GEANT, alpha=0.7, lw=0,
           label='Geant4 (ufficiali), one dot per seed')
ax.scatter((x[:, None] + 0.2 + jit).ravel(), (100 * (N / Gm - 1)).ravel(), s=7, color=NOSTRI, alpha=0.8, lw=0,
           label='Nostri, one dot per seed')
ax.axhline(0, color=INK2, lw=0.8)
for b in range(1, len(rc.LATS)):
    ax.axvline(b * len(TILTS) - 0.5, color=INK2, lw=0.8, ls=':')
ax.set_xticks(x, [str(t) for lat, t in pts], rotation=90, fontsize=7)
for b, lat in enumerate(rc.LATS):
    ax.text(b * len(TILTS) + (len(TILTS) - 1) / 2, 1.02, f'<{lat}>', transform=ax.get_xaxis_transform(),
            ha='center', fontsize=9)
ax.set_xlabel('tilt [µrad]')
ax.set_ylabel('(X_seed − mean Geant4) / mean Geant4  [%]')
ax.set_title('(a) Rocking, fraction within ±100 µrad: every seed of both sets around the Geant4 mean', loc='left', pad=16)
ax.legend(loc='lower left', fontsize=8, markerscale=2)
ax.grid(axis='x', visible=False)

ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(-8, 8, 41)
ax.hist(null.ravel(), bins=bins, histtype='step', color=GEANT, ls='--', lw=1.6, density=True,
        label=f'Geant4 seed s − Geant4 seed s+1\nmean {null.mean():+.2f} %, std {null.std(ddof=1):.2f} %')
ax.hist(cross.ravel(), bins=bins, histtype='step', color=NOSTRI, lw=1.8, density=True,
        label=f'Nostri seed s − Geant4 seed s+1\nmean {cross.mean():+.2f} %, std {cross.std(ddof=1):.2f} %')
ax.set_xlabel('difference / mean Geant4  [%]')
ax.set_ylabel('density')
ax.set_title(f'(b) Rocking: difference between two runs\nKS p = {ks_r.pvalue:.2f}', loc='left')
ax.legend(loc='upper left', fontsize=7.2)
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)

for k, (name, g, n, pk, pt, pl) in enumerate(rad_rows):
    ax = fig.add_subplot(gs[1, :1] if k == 0 else gs[1, 1:])
    for i, (v, col, lab) in enumerate([(g, GEANT, 'Geant4 (ufficiali)'), (n, NOSTRI, 'Nostri')]):
        xs = i + np.linspace(-0.18, 0.18, len(v))
        ax.scatter(xs, v, s=16, color=col, lw=0, alpha=0.85)
        sem = v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(i + 0.32, v.mean(), yerr=v.std(ddof=1), color=col, marker='_', ms=14, lw=1, capsize=0)
        ax.errorbar(i + 0.32, v.mean(), yerr=sem, color=col, lw=3.5, capsize=0)
    ax.set_xticks([0, 1], ['Geant4\n(ufficiali)', 'Nostri'])
    ax.set_xlim(-0.5, 1.6)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('MeV per incident electron')
    title = 'radiated energy' if k == 0 else 'radiated energy, photons 178–438 MeV'
    ax.set_title(f'({"cd"[k]}) Radiation W <111>: {title}, one dot per seed\n'
                 f'difference {100 * (n.mean() / g.mean() - 1):+.1f} %;  KS p = {pk:.2f},  t-test p = {pt:.2f}',
                 loc='left', fontsize=9)
for a in fig.axes:
    a.tick_params(length=0, labelsize=8, colors=INK2)
fig.suptitle('Seed test: Nostri against Geant4 (ufficiali) compared with Geant4 against itself', x=0.07, ha='left',
             fontsize=12)
fig.text(0.07, 0.015,
         f'(a, b) Si 30 µm, e− 855 MeV, {len(pts)} points × {S} seeds × 20 000 events. Geant4 with a different seed is the '
         f'reference: two Geant4 runs differ only by statistics.\n(c, d) W 100 µm, e− 855 MeV, {len(W_SEEDS)} seeds × '
         f'{N_INC} electrons per set. Thin bar: seed-to-seed std; thick bar: error of the mean (std / √{len(W_SEEDS)}). '
         f'p-values: two-sample Kolmogorov-Smirnov and Welch t-test.', fontsize=8, color=INK2)
fig.savefig(os.path.join(HERE, '07_seed_test.png'), dpi=150)
print('figure written')
