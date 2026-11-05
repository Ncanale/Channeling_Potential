# Data files for the Geant4 channeling FastSim model

`ciftxt_reader.py --fastsim` writes the text files read by `G4ChannelingFastSimModel`
(Geant4 `source/parameterisations/channeling`, class `G4ChannelingFastSimCrystalData`). They
contain the potential, the electron density and the other lattice functions for any crystal in a
CIF file and for any temperature (through the thermal amplitudes u(T) of `channeling/thermal.py`).

```
python ciftxt_reader.py Ge --all-axes --T-scan 77,293 --fastsim --outdir output/Ge
FASTSIM=1 ./scanner.sh -T 77,293 Si Ge C W Ir PbWO4        # files in output/fastsim/<material>/T<T>K/
python ciftxt_reader.py PbWO4 "0 0 1" --plane "0 0 1" --fastsim --emin 100 --fastsim-step 0.02
```

## What FastSim needs

The model integrates the transverse motion in the continuum potential of a plane (1D) or string (2D),
`x'' = -z U'(x)/(pv)`, and adds incoherent processes. From the data files it uses, as functions of the
position inside one periodic cell:

| quantity | used for | stored as |
|---|---|---|
| electric field `Ex` (`Ey` for strings) = dU/dx of a unit positive charge [eV/cm] | trajectory (the model multiplies by -z/pv) | spline |
| **electron density** [cm^-3] | single scattering on electrons, ionisation losses | spline |
| minimal ionisation energy T_min [eV] | lower cut of the energy transfer in single scattering on electrons (rate ∝ n_e/T_min) | spline |
| nuclear density of each element, normalised to its mean | incoherent Coulomb scattering on nuclei | spline |
| atomic concentration `N0` and thermal amplitude `u` of each element [cm^-3, cm] | scattering suppression by the Debye-Waller factor, screening | numbers |
| `Vmax` = height of the potential well [eV] | Lindhard angle, step length | number |
| `Vmin` = absolute potential minimum [eV] (planes; 0 for axes) | refraction at the lateral surface | number |

The potential itself is not stored: only its gradient and `Vmax`.

## File format (from the Geant4 reader and the dataset README, verified against `G4CHANNELING 2.0`)

Name: `<material>_planes_<hkl>.dat` or `<material>_axes_<uvw>.dat`, with `<material>` the Geant4 material name
without its first three characters (`G4_Si` becomes `Si`; `--g4-name` or `g4_name` in `materials.yaml` sets it).
FastSim looks for the files in `$G4CHANNELINGDATA` or in the directory given by
`/crystal/setChannelingDataPath <dir>/` (**the path must end with a slash**: Geant4 concatenates it with the
file name). Use one directory per temperature, e.g. `output/fastsim/Ge/T77K/`.

Numbers separated by white space, lengths in cm, energies in eV:

```
N0 (one per element)          u (one per element)
dx  npx                        (planes)        dx dy npx npy   (axes)       period(s) and number of intervals
Vmax   Vmin
planes: for each of the npx intervals, 4 coefficients (a b c d) of the cubic spline of Ex, nuc(elem 0),
        n_el, emin, nuc(elem 1), ...                p(s) = a + b s + c s^2 + d s^3,  s = x - x_right_node [cm]
axes:   for j in 0..npy-1, i in 0..npx, k in 0,1: 3 coefficients (A B C) of emin, Ex, Ey, nuc(elem 0), n_el, nuc(elem 1), ...
        (tensor-product spline, definitions in channeling/fastsim_format.py)
```

The element order must be the one of the Geant4 material (for `G4_PbWO4`: O, Pb, W; set `g4_elements` in
`materials.yaml`). The conventions of the cell (origin, orientation, rectangular cell) are described in
`channeling/fastsim.py`.

## How the functions are computed

All of them come from the same reciprocal-space sum as the potential (`channeling/potential_fourier.py`), with
the Peng et al. (1996) scattering factors and the Debye-Waller factor of each element at the chosen T:

* potential: `U~(q) = 2 pi e^2 a0 f_e(q)`; field: exact spectral derivative;
* electron density: Mott-Bethe / Poisson, `rho~(q) = Z - (a0 q^2/2) f_e(q)`, with the same Debye-Waller factor;
* nuclear density: atomic positions smeared by the Debye-Waller factor;
* minimal ionisation energy (`--emin fermi`, default; `channeling/emin_fermi.py`): the local Fermi energy as
  defined by Sytov (PRAB 22, 064601 (2019), DOI 10.1103/PhysRevAccelBeams.22.064601; PhD thesis, Ferrara 2018,
  sect. 2.2: "the average logarithm of the interplanar or interaxial potential, averaged over the electron
  density", with the Thomas-Fermi relation of Landau-Lifshitz QM sect. 70, E_F(r) = e phi(r) for the electrons
  of a neutral atom). Every electron gets the potential of its own atom:
  `ln T_min(x) = < sum_j n_j(r) ln V_j(r) > / < sum_j n_j(r) >`, with n_j, V_j the electron density and potential
  energy of atom j alone (thermally smeared, Peng 1996, isotropic u; negative densities of the fit set to 0),
  averaged along the string (axes) or over the plane (planes) at the transverse position x. The sum over atoms is
  done in reciprocal space (radial Fourier transform of n_j ln V_j and n_j, one 3D FFT of the structure cell).
  Alternatives: `--emin fermi-crystal` (same average with the total crystal potential, neighbours and mean inner
  potential included: 2-7x higher in the channels), `--emin amorphous` (constant I of the crystal),
  `--emin plasma`, `--emin <eV>`.

## Validation against the official Geant4 dataset (Si, Ge, diamond, W; G4CHANNELING 2.0)

With the same u as the shipped files (Si 0.075, Ge 0.085, C 0.04, W 0.05 A):

| | planes (110), (111) | axes <100>, <110>, <111> (after aligning the origin) |
|---|---|---|
| field Ex, Ey | 0.2-1.3 % rms-of-max | 0.07-0.4 % |
| nuclear density | 0.01-0.2 % | 0.0-0.1 % |
| electron density | 0.6-1.5 % (diamond 11 %) | 0.16-0.27 % (diamond 3.4 %) |
| Vmax | 0.03-0.3 % | 0.04-1 %; Si/Ge/C <110> are 2-6 % higher than the shipped value |
| Vmin (planes) | 0.1-1.7 % | |

The comparison (fields, densities and Si T_min) is repeated by `tests/test_fastsim.py` when `G4CHANNELINGDATA`
points to the unpacked dataset (`export G4CHANNELINGDATA=.../G4CHANNELING2.0`); without it those tests are skipped.

### Minimal ionisation energy

How the definition was identified: the electron-weighted geometric mean of the shipped T_min over the cell,
`exp(sum n_e ln T_min / sum n_e)`, is the same for every orientation of a material (Si 81.8-82.0 eV for the six
files, Ge 207.4-207.7, diamond 27.6-27.7, W 722.6). This holds when T_min is the line (plane) average of one
3D function weighted with n_e, and then the cell mean is the 3D mean of that function. With the crystal
potential the 3D mean is 106 eV for Si; with the potential of the own atom it is the atomic value
`exp(int n ln V d3r / Z)`: Si 81.9, Ge 208.2, C 27.5, W 724.0 eV at the shipped u, as in the files. A single
constant (offset or factor) cannot turn the crystal-potential map into the shipped one: the offset needed goes
from 1.8 to 5.9 eV depending on the channel, and the shipped value in the Si <110> channel (0.28 eV) is below the
minimum of the crystal potential (1.36 eV).

`fermi` against the shipped maps, ln(ours/shipped) on a 1000-point (planes) or 120x120 (axes, after aligning
the origin) grid:

| file | shipped range [eV] | ours [eV] | mean | rms | channels (shipped < 20 eV) | atoms (> 100 eV) |
|---|---|---|---|---|---|---|
| Si (100) | 5.10-294 | 4.59-299 | -0.002 | 0.049 | +0.005 | +0.009 |
| Si (110) | 3.08-335 | 3.24-335 | +0.005 | 0.049 | +0.030 | -0.001 |
| Si (111) | 1.51-291 | 1.46-291 | -0.001 | 0.042 | +0.004 | +0.001 |
| Si <100> | 2.03-766 | 2.01-773 | -0.004 | 0.091 | +0.022 | +0.006 |
| Si <110> | 0.28-745 | 0.28-750 | -0.012 | 0.077 | -0.019 | +0.008 |
| Si <111> | 2.44-763 | 2.42-768 | -0.001 | 0.091 | +0.033 | +0.001 |
| Ge (100), (110), (111) | 1.3-648 | 1.4-646 | 0.000 to +0.036 | 0.09-0.10 | +0.004 to +0.124 | -0.004 to +0.010 |
| Ge <100>, <110>, <111> | 0.25-1361 | 0.24-1356 | -0.020 to +0.026 | 0.16-0.17 | -0.106 to +0.044 | +0.017 to +0.025 |
| W <111> | 1.95-5835 | 2.16-5889 | -0.023 | 0.114 | -0.046 | -0.036 |
| diamond planes | 2.1-196 | 2.1-181 | +0.025 to +0.031 | 0.19-0.23 | -0.08 to +0.06 | -0.09 to -0.03 |
| diamond axes | 0.59-762 | 0.62-767 | -0.001 to +0.016 | 0.33-0.43 | 0.00 to +0.12 | +0.08 to +0.10 |

The rms is dominated by the steep flanks near the atoms (small differences of position and of the form factors).
For diamond the Peng density of carbon at u = 0.04 A is negative near the nucleus (Mott-Bethe from the fit at
large q; the same limitation as the 11 % on the planar electron density above), which spoils the intermediate
region; the Doyle-Turner (1968) coefficients are worse for C and Si.

## Validation inside Geant4 (11.4.3, example ch1 modified, Si, 855 MeV, 30 µm, u = 0.075 Å)

Exit-angle distribution of 855 MeV positrons (beam divergence 1 µrad), fraction within ±100 µrad of the peak,
own files against the shipped ones (20000 events per point):

| scan | constant T_min (`amorphous`) | `fermi-crystal` |
|---|---|---|
| axial rocking curves <100>, <110>, <111>, tilts ±200, ±500, ±700 µrad and 0 | +0.018 ± 0.002 (3.8 σ) | +0.003 ± 0.0015 (0.7 σ) |
| planar rocking curves (110), (111), ±400 µrad | +0.003 (0.6 σ mean) | |
| bent planes (110), (111), 0-25 mrad | +0.004 (0.8 σ mean) | |

These runs were made before the present `fermi` definition was identified; its maps coincide with the shipped
ones within the accuracy shown above, so the Geant4 results are expected to coincide too (not yet run).

Sensitivity of the axial curve near the critical angle (own files, one block scaled at a time): T_min x0.3 / x3
moves the fraction by -0.01 / +0.01; nuclear density ±10 % by ∓0.005; electron density ±10 % and field ±1 % by
∓0.003. The official files with only their T_min replaced by the constant reproduce the own-file results within
0.001: the earlier difference came entirely from T_min. Electrons (e-) give no channeling peak at this energy and
thickness (aligned crystals broaden the distribution); their position and angle distributions agree within the
statistics (two-seed test: seed-to-seed scatter 0.7 σ, own-vs-official 0.9 σ).

## Open points

* **Minimal ionisation energy.** The shipped maps are reproduced (see above); the remaining differences come
  from the form factors (the shipped files use the Doyle-Turner representation with the Dudarev-Peng-Whelan 1995
  coefficients, Surf. Sci. 330, 86 (1995), which we do not have; we use Peng 1996). For anisotropic thermal
  motion the atomic functions use the isotropic equivalent u = sqrt(tr U / 3). The Geant4 runs with `fermi` maps
  are still to be done.
* `Vmax` of the <110> axes of Si, Ge and diamond is higher than the shipped value (130.4 vs 125.9 eV for Si)
  although the fields agree to 0.1 %: the shipped value probably comes from a different evaluation of the
  string core; `Vmax` only enters the Lindhard angle and the step length.
* For axes `Vmin` is written as 0, as in the shipped files.
* The cell for axes must be a rectangle spanned by lattice vectors of the potential along x and y. Crystals
  without such a cell (oblique projected lattices) stop with a message; give another x direction with
  `--fastsim-x "h k l"`.
* Orientation: x is along a <110>-type direction perpendicular to the axis (as in the shipped files) and the
  maps of Si, Ge, diamond and W are mirror symmetric, so the sign convention is immaterial there. For
  lower-symmetry crystals (PbWO4) check that the sense of x and y matches your Geant4 crystal rotation.
* Files of the interpolation step: `--fastsim-step` (default: thermal amplitude / 2.2 for axes, min(0.002 A,
  u/15) for planes); at 77 K the W axes files are several tens of MB.
* Inside Geant4 only Si was tested (see above); Ge, W, diamond, PbWO4 files are validated only against the shipped
  data (Python port of `G4ChannelingFastSimInterpolation`) or not at all (PbWO4: no shipped file).
* `fermi` needs a 3D FFT of the structure cell: 2-17 s per file for Si, Ge, diamond, W and PbWO4; large cells use
  more memory (grid capped at 2.5e7 points). `fermi-crystal` is slower (about 1 min per axis for Si).
