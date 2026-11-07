"""Summary figure: how well Nostri agree with Geant4 (ufficiali), with the seed-to-seed spread as reference.

Left: data (agreement_table.csv), rms of (Nostri - Geant4)/Geant4 for T_min and of the field and densities
      in % of the maximum, per material.
Right: physics results of the Geant4 simulations, (Nostri - Geant4)/Geant4 with its error (values from the
      README tables, Parts 2-6), against the spread of a single run between seeds (grey band): a difference
      smaller than that band is not distinguishable from the seed-to-seed scatter.
Output: 11_summary_agreement.png
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e1e0d9'
NOSTRI, GEANT = '#ff7f0e', '#52514e'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': '#c3c2b7', 'axes.grid': True,
                     'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
                     'font.size': 9})

# ---- left: data, from agreement_table.csv (fractions -> %)
rows = list(csv.DictReader(open(os.path.join(HERE, 'agreement_table.csv'), encoding='utf-8')))
groups = {'Si': [], 'Ge': [], 'diamond': [], 'W': []}
for r in rows:
    key = {'Si': 'Si', 'Ge': 'Ge', 'Diamond': 'diamond', 'W': 'W'}[r['file'].split('_')[0]]
    groups[key].append(r)
labels, tmin, field, dens = [], [], [], []
for k, v in groups.items():
    labels.append(k)
    tmin.append(100 * np.max([float(x['T_rms']) for x in v]))
    field.append(100 * np.max([float(x['E_rel_max']) for x in v]))
    dens.append(100 * np.max([float(x['ne_rel_max']) for x in v]))

# ---- right: physics results (README), residual %, error %, single-run scatter %
PHYS = [  # label, residual, error, single-run scatter (seed-to-seed)
    ('Si rocking, 39 points, 14 seeds', 0.08, 0.09, 1.7),
    ('Ge rocking, 39 points, 4 seeds', 0.06, 0.19, 1.5),
    ('Diamond rocking, 39 points, 12 seeds', 0.22, 0.09, 1.3),
    ('W rocking, fraction ±500 µrad, 12 seeds', -0.02, 0.11, 0.9),
    ('W rocking, mean exit angle, 12 seeds', -0.17, 0.18, 0.0),
    ('W rocking, rms width, 12 seeds', 0.13, 0.07, 0.6),
    ('W radiation, radiated energy, 20 seeds', 1.4, 1.1, 4.7),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={'width_ratios': [1, 1.25]})
y = np.arange(len(labels))
w = 0.26
ax1.barh(y - w, tmin, height=w, color=NOSTRI, label='T_min')
ax1.barh(y, dens, height=w, color='#fdbf6f', label='electron density (of max)')
ax1.barh(y + w, field, height=w, color='#8c6d31', label='field (of max)')
for yi, a, b, c in zip(y, tmin, dens, field):
    for off, v in [(-w, a), (0, b), (w, c)]:
        ax1.text(v + 0.02, yi + off, f'{v:.2f} %', va='center', fontsize=7.5, color=INK2)
ax1.set_yticks(y, labels)
ax1.invert_yaxis()
ax1.set_xlim(0, max(tmin) * 1.35)
ax1.set_xlabel('rms of (Nostri − Geant4) / Geant4 (T_min), or / max Geant4 [%]', labelpad=8)
ax1.set_title('Data, point by point (19 files, the worst file of each material)', loc='left', fontsize=9.5)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
ax1.tick_params(length=0)

names = [p[0] for p in PHYS]
yy = np.arange(len(PHYS))
res = np.array([p[1] for p in PHYS])
err = np.array([p[2] for p in PHYS])
scat = np.array([p[3] for p in PHYS])
for i, s in enumerate(scat):
    if s > 0:
        ax2.fill_betweenx([i - 0.38, i + 0.38], -s, s, color='#c3c2b7', alpha=0.45, lw=0)
ax2.axvline(0, color=INK2, lw=0.8)
ax2.errorbar(res, yy, xerr=err, fmt='o', color=NOSTRI, ms=6, elinewidth=1.4, capsize=3)
ax2.set_yticks(yy, names)
ax2.invert_yaxis()
ax2.set_xlabel('(Nostri − Geant4) / Geant4 [%]   (bar: 1σ; grey band: single-run seed spread)')
ax2.set_title('Physics: Geant4 runs with our files vs the official ones', loc='left', fontsize=9.5)
ax2.set_xlim(-4, 6)
ax2.tick_params(length=0)
for i, p in enumerate(PHYS):
    ax2.text(5.85, i, f'{p[1]:+.2f} ± {p[2]:.2f} %', va='center', ha='right', fontsize=7.5, color=INK2)

fig.suptitle('Nostri vs Geant4 (ufficiali): agreement summary', x=0.02, ha='left', fontsize=12)
fig.text(0.02, 0.01, 'Data: rms of point-by-point residuals of the 19 shipped files. Physics: summed residuals over the '
         'scan points, same seeds for both sets. Single-run scatter is the seed-to-seed spread of one simulation '
         '(README); a residual inside the grey band is indistinguishable from it.', fontsize=7.5, color=INK2, wrap=True)
fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.2, wspace=0.55)
fig.savefig(os.path.join(HERE, '11_summary_agreement.png'), dpi=150)
print('figure written')
