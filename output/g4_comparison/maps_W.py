"""W <111>: Nostri vs Geant4 (ufficiali), point by point (same cell, origin and orientation in both files).
Our file is generated here with channeling.fastsim at the official thermal amplitude u = 0.050 A.
Field and densities: rms of (N - G) / max(G); electron density and T_min also as (N - G) / G."""
import os
import sys

import numpy as np

PROJ = r'C:/Users/tacab/OneDrive - Istituto Nazionale di Fisica Nucleare/Cristalli/Calcolo potenziali/Channeling_Potential'
sys.path.insert(0, PROJ)
from channeling import fastsim, fastsim_format as F
from channeling.structure import load_structure
from channeling.thermal import ThermalResult

OFF = r'\\wsl.localhost\Ubuntu-24.04\home\tacab\g4\install\share\Geant4\data\G4CHANNELING2.0\W_axes_111.dat'

ref = F.read(OFF)
mine, _ = fastsim.axial_fastsim(load_structure('W'), (1, 1, 1), {'W': ThermalResult(0.050, 'fixed', 'test')}, emin='fermi')
n = 240
X, Y = np.meshgrid((np.arange(n) + .5) / n * ref.dx, (np.arange(n) + .5) / n * ref.dy, indexing='ij')
print(f'cell official {ref.dx / 1e-8:.4f} x {ref.dy / 1e-8:.4f} A, ours {mine.dx / 1e-8:.4f} x {mine.dy / 1e-8:.4f} A')
print(f"{'quantity':10s} {'max G':>10s} {'max N/G-1':>10s} {'rms (N-G)/maxG':>15s} {'max |N-G|/maxG':>15s}")
Emax = np.hypot(F.evaluate(ref, 'Ex', X, Y), F.evaluate(ref, 'Ey', X, Y)).max()
for q in ['Ex', 'Ey', 'el', 'nuc0']:
    r_, m_ = F.evaluate(ref, q, X, Y), F.evaluate(mine, q, X, Y)
    norm = Emax if q in ('Ex', 'Ey') else r_.max()
    d = (m_ - r_) / norm
    print(f'{q:10s} {norm:10.4g} {100 * (m_.max() / r_.max() - 1):+9.2f}% {100 * np.sqrt(np.mean(d ** 2)):14.2f}% '
          f'{100 * np.abs(d).max():14.2f}%')
a = F.evaluate(ref, 'el', X, Y)
rn = F.evaluate(mine, 'el', X, Y) / a - 1
print(f'n_e   (N-G)/G: mean {100 * rn.mean():+.2f}%  rms {100 * np.sqrt(np.mean(rn ** 2)):.2f}%')
Tr, Tm = F.evaluate(ref, 'emin', X, Y), F.evaluate(mine, 'emin', X, Y)
rt = Tm / Tr - 1
print(f'T_min (N-G)/G: mean {100 * rt.mean():+.2f}%  rms {100 * np.sqrt(np.mean(rt ** 2)):.2f}%  '
      f'range official {Tr.min():.3g}-{Tr.max():.3g} eV, ours {Tm.min():.3g}-{Tm.max():.3g} eV')
print(f'Vmax official {ref.vmax:.2f} eV, ours {mine.vmax:.2f} eV ({100 * (mine.vmax / ref.vmax - 1):+.2f}%)')
