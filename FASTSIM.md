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
`materials.yaml`). The cell follows the convention of the shipped files, so our maps coincide with them point by
point without translation or mirror: planes, x = 0 at the potential minimum and the atomic plane at dx/2; axes,
x along a <110>-type direction, y = z x x, and the atomic strings on the main diagonal of the cell with their mean
position at the centre (one string at (dx/2, dy/2); two at (dx/4, dy/4) and (3dx/4, 3dy/4); the <110> dumbbells of
the diamond lattice centred there). Details in `channeling/fastsim.py` (`_geant4_origin`).

## How the functions are computed

All of them come from the same reciprocal-space sum as the potential (`channeling/potential_fourier.py`), with
Gaussian electron scattering factors and the Debye-Waller factor of each element at the chosen T:

* potential: `U~(q) = 2 pi e^2 a0 f_e(q)`; field: exact spectral derivative;
* electron density: Mott-Bethe / Poisson, `rho~(q) = Z - (a0 q^2/2) f_e(q)`, with the same Debye-Waller factor
  (the X-ray form factor of Waasmaier-Kirfel, `rho~(q) = f_x(q)`, is available as an alternative route);
* atomic model (`form_factors.atom_model`): the Doyle-Turner representation of Dudarev, Peng and Whelan,
  Surf. Sci. 330, 86 (1995), DOI 10.1016/0039-6028(95)00464-5, Tables 1-2 (real part, 5 Gaussians fitted to
  relativistic Hartree-Fock values for 0 < s < 5 1/A, 50 elements, `channeling/data/dudarev_peng_whelan_1995.json`),
  with the Mott-Bethe density; Peng et al. 1996 for elements not in that table. This is the model of the shipped
  dataset: Sytov's CRYSTAL code, from which the dataset comes, cites it (Sytov and Tikhomirov, NIM B 355, 383
  (2015), DOI 10.1016/j.nimb.2015.02.042), and the f_e read off the shipped files (below) coincides with it.
* nuclear density: atomic positions smeared by the Debye-Waller factor;
* minimal ionisation energy (`--emin fermi`, default; `channeling/emin_fermi.py`): the local Fermi energy as
  defined by Sytov (PRAB 22, 064601 (2019), DOI 10.1103/PhysRevAccelBeams.22.064601; PhD thesis, Ferrara 2018,
  sect. 2.2: "the average logarithm of the interplanar or interaxial potential, averaged over the electron
  density", with the Thomas-Fermi relation of Landau-Lifshitz QM sect. 70, E_F(r) = e phi(r) for the electrons
  of a neutral atom). Every electron gets the potential of its own atom:
  `ln T_min(x) = < sum_j n_j(r) ln V_j(r) > / < sum_j n_j(r) >`, with n_j, V_j the electron density and potential
  energy of atom j alone (thermally smeared with the isotropic u, atomic model of the element as above;
  negative densities of the fit set to 0),
  averaged along the string (axes) or over the plane (planes) at the transverse position x. The sum over atoms is
  done in reciprocal space (radial Fourier transform of n_j ln V_j and n_j, one 3D FFT of the structure cell).
  Alternatives: `--emin fermi-crystal` (same average with the total crystal potential, neighbours and mean inner
  potential included: 2-7x higher in the channels), `--emin amorphous` (constant I of the crystal),
  `--emin plasma`, `--emin <eV>`.

## Validation against the official Geant4 dataset (Si, Ge, diamond, W; G4CHANNELING 2.0)

With the same u as the shipped files (Si 0.075, Ge 0.085, C 0.04, W 0.05 A):

Point by point, same grid for both files (no alignment), rms of (ours - shipped) in % of the shipped maximum:

| | Si | Ge | diamond | W <111> |
|---|---|---|---|---|
| field, planes / axes | 0.01-0.03 / 0.01 | 0.14-0.26 / 0.06-0.13 | 0.02-0.04 / 0.01-0.02 | 0.11 |
| nuclear density, planes / axes | 0.01-0.02 / 0.00-0.01 | 0.08-0.18 / 0.04-0.09 | 0.02-0.04 / 0.01-0.02 | 0.08 |
| electron density, planes / axes | 0.01 / 0.00-0.01 | 0.07-0.13 / 0.03-0.08 | 0.01-0.02 / 0.01-0.02 | 0.08 |

For Ge and W the residual follows the nuclear density, i.e. it is geometric: the lattice parameters differ by up to
0.05 % from the shipped ones. Vmax agrees within 0.06 % for the (100), (110) planes and the <100> axes; it is
0.2-0.5 % higher for the (111) planes and W <111>, 0.6-1.3 % for the <111> axes and 2-6 % for the <110> axes of Si,
Ge and diamond (see Open points). Vmin of the planes agrees within 0.2 %.

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

`fermi` against the shipped maps, (ours - shipped)/shipped in %, on a 1000-point (planes) or 160x160 (axes) grid
(`output/g4_comparison/agreement_table.csv`, per file):

| files | mean | rms | channels (shipped < 20 eV), median | atoms (shipped > 100 eV), median |
|---|---|---|---|---|
| Si planes (100), (110), (111) | +0.4 | 0.6-0.7 | +0.2 to +0.3 | -0.0 to +0.3 |
| Si axes <100>, <110>, <111> | +0.3 | 0.5 | +0.1 | +0.7 to +0.9 |
| Ge planes | +0.5 to +0.6 | 0.8-0.9 | +0.2 to +0.7 | +0.4 to +0.7 |
| Ge axes | +0.5 | 0.8 | +0.2 to +0.3 | +1.0 |
| diamond planes | +0.3 to +0.4 | 0.8-1.0 | +0.1 to +0.3 | -0.7 to -0.2 |
| diamond axes | +0.1 to +0.5 | 1.3-1.6 | -0.3 to +0.1 | +0.4 to +0.7 |
| W <111> | +0.3 | 0.7 | +0.3 | +0.5 |

With the form factors used before (Peng 1996; Doyle-Turner 1968 for Ge; X-ray density for diamond) the rms was 4-9 %
(Si), 3-7 % (Ge), 9-27 % (diamond) and 10.6 % (W): almost all of the difference came from the atomic form factor,
not from the definition of T_min.

How the atomic model of the shipped files was identified: dividing the Fourier harmonics of the shipped fields by
those of a file computed with a known f_e and the same u gives f_e(s) of the shipped files at the s values of the
reciprocal lattice (31-36 values up to 3.4-6 1/A from the 9 planar files, 99 from W_axes_111). The shipped nuclear
densities have exactly the declared u (ratio 1.0000); the shipped electron density is the Mott-Bethe density of the
same f_e (within 0.3 %); f_e is within 1-2 % of Doyle-Turner 1968 up to s ~ 1.9 1/A and has a tail above it, unlike
Peng 1996, Kirkland 2010 or Lobato 2014. A 5-Gaussian fit to these values gives the coefficients of Dudarev, Peng
and Whelan (1995) to a few per mille, which are then used as published (no number is taken from the shipped files).

## Validation inside Geant4 (11.4.3, example ch1 modified)

The physical results with our files and with the shipped ones, same build and same random seeds; the errors come
from the scatter between seeds (details, figures and scripts in `output/g4_comparison/README.md`):

| test | result |
|---|---|
| Si axial rocking curves <100>, <110>, <111>, e- 855 MeV, 30 µm, fraction within ±100 µrad, 14 seeds x 20000 events | (ours - shipped)/shipped = +0.08 % ± 0.09 % over 39 points (13 tilts), chi2 = 36.6 / 39 |
| Ge axial rocking curves, same settings, 4 seeds | +0.06 % ± 0.19 % over 39 points, chi2 = 36.8 / 39 |
| diamond axial rocking curves, same settings, 12 seeds | +0.22 % ± 0.09 % over 39 points, chi2 = 54.5 / 39; off axis (\|tilt\| > 200 µrad) +0.37 % ± 0.10 %, mostly <100> |
| W <111> rocking, e- 855 MeV, 10 µm, 13 tilts 0 to ±3000 µrad, 12 seeds x 20000 events | fraction within ±500 µrad -0.02 % ± 0.11 %; mean exit angle -0.17 % ± 0.18 %; rms width +0.13 % ± 0.07 % |
| W <111> 100 µm aligned, e- 855 MeV, radiation (Baier-Katkov), 20 seeds x 2000 events | radiated energy +1.4 % ± 1.1 %; photon spectrum chi2 = 35.3 / 28 bins |
| seed test: ours vs shipped compared with shipped vs shipped (different seeds) | indistinguishable (Kolmogorov-Smirnov p = 0.90 rocking, 0.34 radiation) |

All these runs used the form factors of the previous version (Peng 1996; Doyle-Turner 1968 for Ge; Peng with
X-ray density for diamond), with which the maps were 0.2-1.7 % (fields, densities) and 3-27 % (T_min) off the
shipped ones. Even so Si, Ge and W were indistinguishable from the shipped files within the statistics; diamond
showed a small but real difference (stable at +0.22 % with 4, 9 and 12 seeds while the error decreases). With
Dudarev-Peng-Whelan the maps agree to 0.04 % (diamond) and 0.26 % (Ge, W, geometric). The Si runs and the W radiation runs used the files of the previous version (cell
origin at the potential minimum), which differ from the present ones only by a translation of the cell, irrelevant
inside Geant4; Ge, diamond and the W rocking curves used the present version. Diamond needs a material whose name without `G4_` is
`Diamond` (density 3.515 g/cm3): `G4_C` is graphite and would look for a file `C`.

## Open points

* **Atomic model.** Dudarev-Peng-Whelan 1995 covers 50 elements; the others use Peng 1996. For anisotropic thermal
  motion the atomic functions of T_min use the isotropic equivalent u = sqrt(tr U / 3).
* `Vmax` of the <110> axes of Si, Ge and diamond is 2-6 % higher than the shipped value (130.6 vs 125.9 eV for Si),
  that of the <111> axes 0.6-1.3 % and of the (111) planes 0.3-0.5 %, although the fields agree to 0.03 %: the shipped value probably comes from a different evaluation of the
  string core; `Vmax` only enters the Lindhard angle and the step length.
* For axes `Vmin` is written as 0, as in the shipped files.
* The cell for axes must be a rectangle spanned by lattice vectors of the potential along x and y. Crystals
  without such a cell (oblique projected lattices) stop with a message; give another x direction with
  `--fastsim-x "h k l"`.
* Orientation: x is along a <110>-type direction perpendicular to the axis (as in the shipped files) and the
  maps of Si, Ge, diamond and W are mirror symmetric. For lower-symmetry crystals (PbWO4) check that the sense of
  x and y matches your Geant4 crystal rotation; the origin rule assumes that the strings can be placed on the main
  diagonal (true for the cubic crystals above).
* Files of the interpolation step: `--fastsim-step` (default: thermal amplitude / 2.2 for axes, min(0.002 A,
  u/15) for planes); at 77 K the W axes files are several tens of MB.
* Inside Geant4 Si, Ge, diamond (rocking) and W (rocking and radiation) were tested; diamond differs by +0.2-0.4 % in the
  rocking curves (see above). PbWO4 has no shipped file to compare with.
* `fermi` needs a 3D FFT of the structure cell: 2-17 s per file for Si, Ge, diamond, W and PbWO4; large cells use
  more memory (grid capped at 2.5e7 points). `fermi-crystal` is slower (about 1 min per axis for Si).
