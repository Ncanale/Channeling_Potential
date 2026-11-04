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
| minimal ionisation energy [eV] | lower cut of the energy transfer to an electron | spline |
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
* minimal ionisation energy: see below.

## Validation against the official Geant4 dataset (Si, Ge, diamond, W; G4CHANNELING 2.0)

With the same u as the shipped files (Si 0.075, Ge 0.085, C 0.04, W 0.05 A):

| | planes (110), (111) | axes <100>, <110>, <111> (after aligning the origin) |
|---|---|---|
| field Ex, Ey | 0.2-1.3 % rms-of-max | 0.07-0.4 % |
| nuclear density | 0.01-0.2 % | 0.0-0.1 % |
| electron density | 0.6-1.5 % (diamond 11 %) | 0.16-0.27 % (diamond 3.4 %) |
| Vmax | 0.03-0.3 % | 0.04-1 %; Si/Ge/C <110> are 2-6 % higher than the shipped value |
| Vmin (planes) | 0.1-1.7 % | |

The comparison is repeated by `tests/test_fastsim.py` when `G4CHANNELINGDATA` points to the unpacked dataset
(`export G4CHANNELINGDATA=.../G4CHANNELING2.0`); without it those four tests are skipped.

## Open points

* **Minimal ionisation energy.** The shipped files contain a position-dependent map (3 eV in the channel,
  330-650 eV at the planes for Si/Ge) whose definition could not be found and which is not a simple function
  of the local electron density (a fit over the nine planar files has 40 % scatter). `--emin amorphous`
  (default) writes the constant `I` of the crystal (electron-weighted geometric mean of the Geant4 NIST mean
  excitation energies), the cut of the amorphous material (also used by MBN Explorer, arXiv:2405.07633).
  `--emin plasma` writes `I sqrt(n_el(x)/mean n_el)` and `--emin 50` a number. This block changes the
  ionisation energy loss and the electron scattering of channeled particles: it should be settled with the
  authors of the model (A. Sytov) before quantitative use.
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
* Not tested inside Geant4 itself (no Geant4 installation here): the files were checked by reading them back
  with a Python port of `G4ChannelingFastSimInterpolation` and by comparison with the official files.
