"""Nostri (current model: per-atom Thomas-Fermi T_min) against the Geant4 official files (G4CHANNELING 2.0).

Only the current model and the official files are compared: no earlier definitions, no earlier runs.
Thermal amplitudes as in the official files. Residuals: (Nostri - Geant4) / Geant4 for T_min and the electron
density; for the field and the nuclear density, which vanish inside the cell, (Nostri - Geant4) / max(Geant4).
Output: output/g4_comparison/01..04 PNG and agreement_table.csv
"""
import csv
import os
import sys
import time

PROJ = r'C:/Users/tacab/OneDrive - Istituto Nazionale di Fisica Nucleare/Cristalli/Calcolo potenziali/Channeling_Potential'
sys.path.insert(0, PROJ)

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D

from channeling import fastsim, fastsim_format as F
from channeling.structure import load_structure
from channeling.thermal import ThermalResult

OFF = r'\\wsl.localhost\Ubuntu-24.04\home\tacab\g4\install\share\Geant4\data\G4CHANNELING2.0' + '\\'
OUT = os.path.join(PROJ, 'output', 'g4_comparison')
os.makedirs(OUT, exist_ok=True)
CM = 1e-8  # 1 Å in cm

# colours: reference palette of the dataviz skill (palette.md, light mode)
SURF, INK, INK2, BASE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#c3c2b7', '#e1e0d9'
NOSTRI = '#ff7f0e'      # tab:orange
GEANT = INK2            # Geant4 official files: dashed, secondary ink
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
REL_T = '(Nostri − Geant4) / Geant4'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 9.5,
    'axes.edgecolor': BASE, 'axes.linewidth': 0.8, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.5, 'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'], 'font.size': 9,
    'legend.frameon': False, 'legend.fontsize': 8.5,
})

U = {'Si': 0.075, 'Ge': 0.085, 'C': 0.04, 'W': 0.05}     # thermal amplitudes [A], as in the official files
NAME = {'Si': 'Si', 'Ge': 'Ge', 'C': 'Diamond', 'W': 'W'}
ORDER = ['Si', 'Ge', 'C', 'W']
TH = {el: {el: ThermalResult(u, 'fixed', 'test')} for el, u in U.items()}
STRUCT = {}


def structure(el):
    if el not in STRUCT:
        STRUCT[el] = load_structure(el)
    return STRUCT[el]


def nostri(el, kind, hkl):
    """Our current model: per-atom Thomas-Fermi minimal ionisation energy (emin='fermi')."""
    if kind == 'planes':
        return fastsim.planar_fastsim(structure(el), hkl, TH[el], emin='fermi')
    d, _ = fastsim.axial_fastsim(structure(el), hkl, TH[el], emin='fermi')
    return d


def label(el, kind, hkl):
    h = ''.join(str(i) for i in hkl)
    return f'{NAME[el]} ({h})' if kind == 'planes' else f'{NAME[el]} <{h}>'


def style(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(length=0, labelsize=8, colors=INK2)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASE)


def points(ref, kind, n):
    if kind == 'planes':
        return (np.arange(n) + .5) / n * ref.dx, None
    return np.meshgrid((np.arange(n) + .5) / n * ref.dx, (np.arange(n) + .5) / n * ref.dy, indexing='ij')


def get(d, kind, name, X, Y):
    """Same cell, origin and orientation in both files (channeling.fastsim follows the official convention)."""
    if kind == 'planes':
        return F.evaluate(d, name, X)
    return F.evaluate(d, name, X, Y)


def field(d, kind, X, Y):
    if kind == 'planes':
        return get(d, kind, 'Ex', X, Y)
    return np.hypot(get(d, kind, 'Ex', X, Y), get(d, kind, 'Ey', X, Y))


def analyse(el, kind, hkl, fn, n):
    ref = F.read(OFF + fn)
    new = nostri(el, kind, hkl)
    X, Y = points(ref, kind, n)
    out = dict(el=el, kind=kind, hkl=hkl, ref=ref, X=X, Y=Y)
    out['T_ref'] = get(ref, kind, 'emin', X, Y)
    out['T_new'] = get(new, kind, 'emin', X, Y)
    out['ne_ref'] = get(ref, kind, 'el', X, Y)
    out['ne_new'] = get(new, kind, 'el', X, Y)
    out['nuc_ref'] = get(ref, kind, 'nuc0', X, Y)
    out['nuc_new'] = get(new, kind, 'nuc0', X, Y)
    out['E_ref'] = field(ref, kind, X, Y)
    out['E_new'] = field(new, kind, X, Y)
    rt = out['T_new'] / out['T_ref'] - 1
    rn = out['ne_new'] / out['ne_ref'] - 1
    ch, at = out['T_ref'] < 20, out['T_ref'] > 100
    out['summary'] = dict(
        file=fn, kind=kind, hkl=''.join(map(str, hkl)),
        T_mean=float(rt.mean()), T_rms=float(np.sqrt((rt ** 2).mean())),
        T_channel_median=float(np.median(rt[ch])), T_atom_median=float(np.median(rt[at])),
        ne_rel_rms=float(np.sqrt((rn ** 2).mean())),
        ne_rel_max=float(np.sqrt(((out['ne_new'] - out['ne_ref']) ** 2).mean()) / np.abs(out['ne_ref']).max()),
        nuc_rel_max=float(np.sqrt(((out['nuc_new'] - out['nuc_ref']) ** 2).mean()) / np.abs(out['nuc_ref']).max()),
        E_rel_max=float(np.sqrt(((out['E_new'] - out['E_ref']) ** 2).mean()) / np.abs(out['E_ref']).max()))
    return out


# ------------------------------------------------------------------ run on all official files
files = []
for fn in os.listdir(OFF):
    if fn.endswith('.dat'):
        name, kind, h = fn[:-4].split('_')
        el = next(k for k, v in NAME.items() if v == name)
        files.append((el, kind, tuple(int(c) for c in h), fn))
files.sort(key=lambda t: (ORDER.index(t[0]), t[1] == 'axes', t[2]))

results = {}
for el, kind, hkl, fn in files:
    t0 = time.time()
    results[fn] = analyse(el, kind, hkl, fn, 1000 if kind == 'planes' else 160)
    s = results[fn]['summary']
    print(f'{fn:24s} {time.time() - t0:5.1f} s   T_min (N-G)/G: rms {s["T_rms"]:.3f}   mean {s["T_mean"]:+.3f}', flush=True)

keys = ['file', 'kind', 'hkl', 'T_mean', 'T_rms', 'T_channel_median', 'T_atom_median',
        'ne_rel_rms', 'ne_rel_max', 'nuc_rel_max', 'E_rel_max']
with open(os.path.join(OUT, 'agreement_table.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for el, kind, hkl, fn in files:
        s = results[fn]['summary']
        w.writerow({k: (f'{v:.5f}' if isinstance(v, float) else v) for k, v in s.items()})


def legend_handles():
    return [Line2D([0], [0], color=GEANT, ls='--', lw=1.5, label='Geant4 (ufficiali)'),
            Line2D([0], [0], color=NOSTRI, lw=1.7, label='Nostri (Thomas-Fermi per atomo)')]


def fig_planes_si():
    fns = ['Si_planes_100.dat', 'Si_planes_110.dat', 'Si_planes_111.dat']
    fig = plt.figure(figsize=(12, 15.5))
    gs = fig.add_gridspec(8, 3, height_ratios=[3, 1.1] * 4, hspace=0.14, wspace=0.22,
                          left=0.08, right=0.98, top=0.90, bottom=0.04)
    main_lab = ['T_min [eV]', 'electron density [e/Å³]', 'field E_x [eV/Å]', 'nuclear density (mean = 1)']
    res_lab = ['(N − G) / G  [%]', '(N − G) / G  [%]', '(N − G) / max G  [%]', '(N − G) / max G  [%]']
    for j, fn in enumerate(fns):
        r = results[fn]
        s = r['summary']
        x = r['X'] / CM
        nuc_r = r['nuc_ref'] / r['nuc_ref'].mean()
        nuc_n = r['nuc_new'] / r['nuc_new'].mean()
        for i in range(4):
            ax = fig.add_subplot(gs[2 * i, j])
            axr = fig.add_subplot(gs[2 * i + 1, j], sharex=ax)
            style(ax)
            style(axr)
            if i == 0:
                ax.semilogy(x, r['T_ref'], color=GEANT, ls='--', lw=1.5)
                ax.semilogy(x, r['T_new'], color=NOSTRI, lw=1.7)
                axr.plot(x, 100 * (r['T_new'] / r['T_ref'] - 1), color=NOSTRI, lw=1.7)
                ax.set_title(f"Si ({''.join(map(str, r['hkl']))})    T_min: rms (N − G) / G = "
                             f"{100 * s['T_rms']:.1f} %", loc='left', fontsize=8.5)
            elif i == 1:
                ax.plot(x, r['ne_ref'] * 1e-24, color=GEANT, ls='--', lw=1.5)
                ax.plot(x, r['ne_new'] * 1e-24, color=NOSTRI, lw=1.7)
                axr.plot(x, 100 * (r['ne_new'] / r['ne_ref'] - 1), color=NOSTRI, lw=1.7)
            elif i == 2:
                ax.plot(x, r['E_ref'] * CM, color=GEANT, ls='--', lw=1.5)
                ax.plot(x, r['E_new'] * CM, color=NOSTRI, lw=1.7)
                axr.plot(x, 100 * (r['E_new'] - r['E_ref']) / np.abs(r['E_ref']).max(), color=NOSTRI, lw=1.7)
            else:
                ax.plot(x, nuc_r, color=GEANT, ls='--', lw=1.5)
                ax.plot(x, nuc_n, color=NOSTRI, lw=1.7)
                axr.plot(x, 100 * (nuc_n - nuc_r) / nuc_r.max(), color=NOSTRI, lw=1.7)
            axr.axhline(0, color=INK2, lw=0.8)
            if j == 0:
                ax.set_ylabel(main_lab[i])
                axr.set_ylabel(res_lab[i])
            if i < 3:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                axr.set_xlabel('position in the period [Å]')
    fig.legend(handles=legend_handles(), loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle('Si planes, u = 0.075 Å: Nostri vs Geant4 (ufficiali)', x=0.08, ha='left', fontsize=12)
    fig.savefig(os.path.join(OUT, '01_planes_Si.png'), dpi=150)
    plt.close(fig)


def fig_axes_maps():
    fns = ['Si_axes_100.dat', 'Si_axes_110.dat', 'Si_axes_111.dat']
    fig, axs = plt.subplots(3, 3, figsize=(12, 12.8), constrained_layout=True)
    nT = LogNorm(vmin=0.2, vmax=800)
    nR = TwoSlopeNorm(vmin=-5, vcenter=0.0, vmax=5)
    im_T = im_R = None
    for j, fn in enumerate(fns):
        r = results[fn]
        ext = [0, r['ref'].dx / CM, 0, r['ref'].dy / CM]
        rel = 100 * (r['T_new'] / r['T_ref'] - 1)
        nm = label(r['el'], 'axes', r['hkl'])
        rows = [(r['T_ref'], nT, SEQ, 'Geant4 (ufficiali)'), (r['T_new'], nT, SEQ, 'Nostri'),
                (rel, nR, DIV, f'{REL_T} [%]')]
        for i, (arr, norm, cmap, tag) in enumerate(rows):
            ax = axs[i, j]
            ax.grid(False)
            im = ax.imshow(arr.T, origin='lower', extent=ext, cmap=cmap, norm=norm, interpolation='bilinear')
            ax.set_title(f'{tag}: {nm}', loc='left', fontsize=8.5)
            ax.tick_params(length=0, labelsize=8, colors=INK2)
            if i == 2:
                ax.set_xlabel('x [Å]')
            if j == 0:
                ax.set_ylabel('y [Å]')
            if i == 1:
                im_T = im
            if i == 2:
                im_R = im
    fig.colorbar(im_T, ax=axs[:2, :], label='T_min [eV]', shrink=0.85)
    fig.colorbar(im_R, ax=axs[2, :], label=f'{REL_T} [%]  (saturated at ±5 %)', shrink=0.85, extend='both')
    fig.suptitle('Si axes: T_min over one cell, u = 0.075 Å', x=0.02, ha='left', fontsize=12)
    fig.savefig(os.path.join(OUT, '02_axes_Si_Tmin_maps.png'), dpi=150)
    plt.close(fig)


def fig_tmin_planes():
    rows = [('Si', 'Si'), ('Ge', 'Ge'), ('C', 'Diamond')]
    fig, axs = plt.subplots(3, 3, figsize=(12, 10), sharey='row')
    for i, (el, name) in enumerate(rows):
        for j, h in enumerate(['100', '110', '111']):
            r = results[f'{name}_planes_{h}.dat']
            x = r['X'] / CM
            ax = axs[i, j]
            ax.semilogy(x, r['T_ref'], color=GEANT, ls='--', lw=1.5)
            ax.semilogy(x, r['T_new'], color=NOSTRI, lw=1.7)
            ax.set_title(f"{name} ({h})      rms (N − G) / G: {100 * r['summary']['T_rms']:.1f} %", loc='left', fontsize=9)
            if i == 2:
                ax.set_xlabel('position in the period [Å]')
            if j == 0:
                ax.set_ylabel('T_min [eV]')
    fig.legend(handles=legend_handles(), loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle('T_min on the planes: Nostri vs Geant4 (ufficiali)', x=0.06, ha='left', fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.07, wspace=0.18, hspace=0.25)
    fig.savefig(os.path.join(OUT, '03_Tmin_planes_all.png'), dpi=150)
    plt.close(fig)


def fig_summary():
    names = [label(el, kind, hkl) for el, kind, hkl, fn in files]
    fns = [fn for el, kind, hkl, fn in files]
    y = np.arange(len(fns))
    fig, ax = plt.subplots(figsize=(8, 9))
    style(ax)
    v = [100 * results[f]['summary']['T_rms'] for f in fns]
    ax.barh(y, v, height=0.6, color=NOSTRI, edgecolor=SURF, linewidth=1.2)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.01 * max(v), yi, f'{vi:.1f} %', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.yaxis.grid(False)
    ax.set_xlabel(f'rms over the cell of {REL_T} for T_min  [%]')
    ax.set_title('T_min: Nostri vs Geant4 (ufficiali), all 19 files', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '04_Tmin_agreement_all_files.png'), dpi=150)
    plt.close(fig)


fig_planes_si()
fig_axes_maps()
fig_tmin_planes()
fig_summary()

print('\nNostri vs Geant4 (ufficiali): (N - G) / G of T_min, channels = Geant4 < 20 eV, atoms = Geant4 > 100 eV')
print('%-14s %8s %8s %9s %9s %9s %9s %9s' % ('file', 'mean', 'rms', 'channels', 'atoms', 'n_e rel', 'nuc rel', 'E rel'))
for el, kind, hkl, fn in files:
    s = results[fn]['summary']
    print('%-14s %+8.3f %8.3f %+9.3f %+9.3f %9.4f %9.4f %9.4f' % (
        label(el, kind, hkl), s['T_mean'], s['T_rms'], s['T_channel_median'], s['T_atom_median'],
        s['ne_rel_max'], s['nuc_rel_max'], s['E_rel_max']))
print('\nfigures in', OUT)
