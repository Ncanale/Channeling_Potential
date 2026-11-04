# Upgrade prompt: temperature-dependent channeling potentials for multi-element crystals

> Paste everything below the line into a coding agent opened in this repository.

---

## 0. Your task

Upgrade this repository so that it computes **axial (2D) and planar (1D) channeling potentials as a function of crystal temperature T**. Thermal vibrations must enter through physically sourced, temperature-dependent mean-square displacements. The code must handle **elemental crystals (Si, Ge, C-diamond, W, Ir)** and **multi-element crystals (PbWO4, SiC, BGO, CeF3, ...)**.

Use analytic expressions wherever they are exact, and say in the docstrings where an approximation is made. Every physical constant or tabulated coefficient must carry a source with its DOI in a comment or docstring. Do not invent coefficients: if you cannot find a value in a cited table, stop and report what is missing.

## 1. What the repository does today (read these files first)

- `ciftxt_reader.py` is the main script; `ciftxt_reader.ipynb` is the older notebook version.
  - It loads `<Material>.cif` with pymatgen, builds a fixed 2×2×2 supercell, projects atoms onto the plane perpendicular to the axis `"h k l"`, and rotates that plane to xy.
  - It derives `d_pl` (plane spacings) and `d_at` (atom spacing along one string) as the **mean of the distinct gaps**.
  - It evaluates the Doyle–Turner-type Gaussian-sum potential:
    - string: `U(ρ) = (2 e² a0 / d_at) Σ_i a_i/(B_i+2u²) · exp(-ρ²/(B_i+2u²))`
    - plane: `U(x) = 2√π e² a0 N d_pl Σ_i a_i/√(B_i+2u²) · exp(-x²/(B_i+2u²))`
    - with `B_i = b_i/4π²`, `e² = 14.40 eV·Å`, `a0 = 0.529 Å`.
  - These prefactors are correct: they follow from `Ũ(q) = 2π e² a0 f_e(s)` with `s = q/4π`. Keep them.
  - Coefficients are hard-coded for Si and Ge only, behind `if Material == "Si"`. The thermal amplitude `u` is a constant (Si 0.075 Å "from Sytov thesis", Ge 0.085 Å "from Baier–Katkov").
  - Outputs are PDFs in `output/<Mat>/<axis>/`.
- `scanner.sh` / `scanner_mac.sh` loop over the axes `100 110 111` for one material.
- `potential_plot.ipynb` is a standalone Si study. It contains a **thermally averaged Molière planar potential** (the erfc form in `U_M`) and a harmonic approximation. Keep the Molière model as an optional alternative.
- CIFs:
  - `Si.cif` and `Ge.cif`. `Ge.txt` and `Si.txt` are byte-identical copies used only by notebook cell 0.
  - `AMS_DATA (1).cif` is Si at **1573 K**.
  - `AMS_DATA (2).cif` is **zircon ZrSiO4**.
  - `PbWO4_mp-504591_primitive.cif` is described in §2.

## 2. Defects to fix along the way

1. **Wrong PbWO4 polymorph.** `PbWO4_mp-504591_primitive.cif` is monoclinic (a=5.687, b=5.108, c=13.953 Å, β=109.4°, P1 setting). That is the **raspite** structure. The scintillator PWO used in calorimeters (CMS, OREO) is tetragonal **scheelite/stolzite, I4₁/a (No. 88), a≈5.47 Å, c≈12.06 Å** (single-crystal refinement at 300 K, Chipaux et al. 2001: a=5.4698, c=12.0634). Replace it with a stolzite CIF (AMCSD/COD/ICSD) and keep the source in the CIF header.
2. **Single-species assumption.** One coefficient set, one `u`, one `d_at` and one `N` cannot describe PbWO4, SiC or BGO. Strings and planes of a compound have mixed compositions with different linear and areal densities.
3. **`d_at` and `d_pl` as the mean of distinct gaps.** This is correct only by coincidence (diamond <111>). Use **atoms per period / period length** for each string, and **atoms per unit area** for each plane, per species.
4. **Finite 2×2×2 cluster.** The axial map is not periodic, so edge strings feel fewer neighbours, and `U -= U.min()` picks up these boundary artefacts. Use periodic summation (see §4.3).
5. Material selection by string comparison, and hard-coded coefficients. Replace both with a data table keyed by element symbol.
6. The notebook multiplies the Doyle–Turner sum by `Z`. This is wrong because `Σa_i = f_e(0)` already contains the atom; the `.py` version correctly dropped it. Make sure it does not come back.
7. Find out which table the current Si/Ge coefficients come from. They look like the 5-Gaussian electron-scattering fits of Peng et al. (1996), DOI 10.1107/S0108767395014371, but this needs checking. Then switch every element to **one consistent table**: the Peng et al. 1996 neutral-atom fit valid for **0 < s < 6 Å⁻¹**. The large-s range matters for the near-nucleus potential.

## 3. Physics: how temperature enters

### 3.1 Exact analytic thermal averaging (keep it analytic)

For a Gaussian-sum form factor `f_j(s) = Σ_i a_ij exp(-b_ij s²)`, the Debye–Waller factor `exp(-B_j s²)` with `B_j = 8π²⟨u_j²⟩` is also Gaussian. The **thermally averaged potential is therefore exact and closed-form**: replace `b_ij → b_ij + B_j(T)`, which is `B_ij → B_ij + 2u_j²(T)`. This is what the current code does, but with constant `u`. Generalise it as follows:

- **Per species j:** each atom carries its own `u_j(T)`.
- **Anisotropic sites (PbWO4, CeF3, 4H-SiC, BGO):** use the displacement tensor `U_j` (Å²).
  - Planar potential with unit normal `n`: use `u_n² = nᵀ U_j n`.
  - Axial potential: the projected 2×2 covariance `Σ_j = P U_j Pᵀ`, where P projects on the plane ⟂ axis. Each Gaussian term becomes a 2D anisotropic Gaussian with covariance `(B_ij/2)·I + Σ_j`, normalised. Write this formula out explicitly in the docstring.
- The **Fourier method** in §4.3 does all of this automatically through `exp(-½ Gᵀ U_j G)`.

### 3.2 Where `u_j(T)` comes from (implement all three, in this priority order)

Report which tier was used for every species in the output.

**Tier A: tabulated or parametrised Debye–Waller factors (preferred for elements and zincblende)**
- Peng, Ren, Dudarev, Whelan (1996): B(T) for elemental crystals, 1–1000 K, from measured phonon DOS. This covers Si, Ge, C, W, Ir. DOI **10.1107/S010876739600089X**
- Gao & Peng (1999): polynomial parametrisation of B(T) for 68 elemental crystals and 17 zincblende compounds. Implement this directly as an analytic function of T. DOI **10.1107/S0108767399005176**
- Schowalter et al. (2009): ab-initio B(T) parametrisation for group-IV, III-V and II-VI semiconductors. Use it for **3C-SiC** (separate B for Si and C). DOI **10.1107/S0108767308031437**
- Sears & Shelley (1991): compilation for 46 elemental crystals. Use it as a cross-check. DOI **10.1107/S0108767391002970**
- Flensburg & Stewart (1999): Si at 293 K from lattice dynamics. Use it as a validation point. DOI **10.1103/PhysRevB.60.284**
- Conversion: `⟨u²⟩ = B/(8π²)` (1D mean-square). As a sanity check, Si at 293 K should give u ≈ 0.075–0.081 Å. The hard-coded 0.075 is the classic value, while recent channeling-radiation measurements give 0.078–0.081 Å. Report the difference rather than forcing agreement.

**Tier B: analytic Debye model (fallback; also used to inter/extrapolate)**

The 1D mean-square displacement for an atom of mass `M = A·amu` and Debye–Waller temperature `Θ` is:

```
⟨u_x²⟩(T) = (3ħ² / (M k_B Θ)) · [ Φ(x)/x + 1/4 ],   x = Θ/T,
Φ(x)      = (1/x) ∫₀ˣ t/(eᵗ−1) dt
3ħ²/(amu·k_B) = 145.52 Å²·K   →   ⟨u_x²⟩ [Å²] = 145.52/(A·Θ) · [Φ(x)/x + 1/4]
```

- Limits: zero-point `⟨u²⟩(0) = 145.52/(4AΘ)`; high-T `≈ 145.52·T/(AΘ²)`.
- Evaluate Φ with `scipy.integrate.quad`, or with the series for small x.
- **Use the Debye–Waller Θ (Θ_M) fitted to B(T), not the calorimetric Θ_D.** They differ by 10–20 %.
- Fit Θ_M to the Tier-A value at 293 K and report it.
- References: Gemmell (1974), DOI **10.1103/RevModPhys.46.129** (thermal averaging and Debye model in channeling); Appleton, Erginsoy, Gibson (1967), DOI **10.1103/PhysRev.161.330** (thermally averaged string potential).

**Tier C: compounds without tabulated B(T) (PbWO4, BGO, CeF3, 4H-SiC, LYSO)**
- **C1, preferred:** take experimental anisotropic displacement parameters U_ij per site at one or more temperatures from a diffraction refinement. Fit them with a per-site Debye model (heavy cations) or an Einstein model (light O/F, `⟨u²⟩ = ħ²/(2Mk_BΘ_E)·coth(Θ_E/2T)`), and keep a static (disorder) offset if the data require it.
  - For PbWO4, start from Trots et al. (2010), *Low temperature structural variations and molar heat capacity of stolzite, PbWO4*, J. Solid State Chem. 183, DOI **10.1016/j.jssc.2010.03.039** (2–290 K data).
  - Trots 2010 (checked in the PDF) reports only **isotropic** B_iso(T) per species from neutron powder diffraction, 3.5–298 K, fitted with the Lonsdale–Debye formula `B_iso = 8π²[145.55·T/(M·ν_D²)·φ + 36.39/(M·ν_D) + B_static]`. Fitted values: Θ_D(W)=172(5) K, B_static(W)=0.19 Å²; Θ_D(Pb)=141(5) K, B_static(Pb)=0.08 Å²; Θ_D(O)=438(5) K, B_static(O)=0.32 Å². Cell at ambient: a=5.46320(2), c=12.04821(8) Å. Use these as isotropic per-species inputs (so only the Gaussian width, no anisotropy). Note the large static terms, and that the authors flag 19 % scatter in Θ_D from powder ADPs. For anisotropic U_ij fall back to C2.
- **C2:** ab-initio thermal displacement matrices from phonopy (`TDISPMAT`), from Togo (2023), DOI **10.7566/JPSJ.92.012001**. Write the input-generation and parsing code, but do not run DFT. Leave a clear hook for the user to supply `thermal_displacement_matrices.yaml`.
- **C3, last resort:** isotropic Debye model per species, with Θ estimated from the compound's Debye temperature and mass scaling. Flag the result as approximate.

### 3.3 Thermal expansion (optional, small)

Allow `a(T)` via a linear expansion coefficient or a tabulated `a(T)`. This is a <0.1 % effect on spacings between 77 and 300 K for Si and Ge, but it is larger and anisotropic for PbWO4 (Trots 2010). Keep it switchable and off by default.

## 4. Implementation plan

### 4.1 Package layout

```
channeling/
  data/scattering_factors.json   # Peng 1996 (0<s<6 Å⁻¹) a_i,b_i per element, with DOI
  data/debye_waller.json         # Tier-A params (Gao–Peng / Schowalter) + Θ_M per element
  data/materials.yaml            # per material: CIF path, default axes/planes, per-site thermal model & source
  structure.py                   # CIF → conventional cell, sites, species, occupancies, ADPs
  thermal.py                     # u_j(T) / U_j(T): tiers A, B, C (+ provenance string)
  potential_realspace.py         # Gaussian-sum strings/planes, periodic images, per-species
  potential_fourier.py           # reciprocal-space method (§4.3)
  observables.py                 # well depth U0, Lindhard angle, max field, ...
ciftxt_reader.py                 # thin CLI wrapper, backward compatible
scanner.sh                       # loops over material × axis × T
```

### 4.2 Real-space method (generalise the current one)

- Find the **distinct strings** per axis using the lattice periodicity, not a finite cluster. For each string, store its composition and period. Then each string contributes `(2e²a0/period) Σ_atoms-in-period Σ_i …` with that atom's species coefficients and `u_j(T)`.
- Sum periodic images over the 2D cell until the result converges (default tolerance 1e-4 eV).
- Planes are handled the same way, with an areal density per species.

### 4.3 Reciprocal-space method (recommended default for compounds)

```
U(r) = (2π e² a0 z / Ω) Σ_{G≠0} Σ_j occ_j f_j(|G|/4π) exp(-½ Gᵀ U_j(T) G) exp(iG·(r − r_j))
```

- `G` includes the 2π; Ω is the cell volume; z is the projectile charge (sign!).
- **Axial:** keep only G with `G·n = 0`. This gives the potential averaged along the axis, as a 2D map.
- **Planar:** keep only `G = m·g_hkl`. This gives a 1D profile over one period.
- Truncate at |G|_max; check convergence.
- This method handles mixed strings and planes, unequal spacings (diamond (111)), partial occupancy, anisotropic ADPs and periodicity with no special cases.
- **Cross-check it against the real-space method for Si and Ge. They must agree to <0.5 %.**

### 4.4 CLI and outputs

- Command line: `python ciftxt_reader.py PbWO4 "0 0 1" --T 293`, `--T-scan 4,77,100,150,200,293,400,600`, `--method fourier|realspace`, `--model peng|moliere`, `--projectile-charge ±1`.
- Output files:
  - PDFs, as today.
  - `.npz` / `.csv` with the grid and U.
  - A `summary.csv` with one row per (material, axis/plane, T), containing:
    - `U0` (well depth) and `U_max − U_min`;
    - maximum transverse field `E_max` (eV/Å). This matters for strong-field radiation and pair production;
    - Lindhard angle `θ_L = √(2U0/pv)` at 120 GeV;
    - `u_j(T)` per species and the provenance tier.
- Overlay plots: U(x) for several T on the same axes; U0(T) and E_max(T) curves.
- `scanner.sh`: loop over materials × axes × temperatures, keeping `set -euo pipefail` and the `output/<Mat>/<axis>/T<T>K/` structure.

## 5. Materials, structures and default orientations

| Material | Structure (space group, a [Å]) | Default axes | Default planes | Thermal source |
|---|---|---|---|---|
| Si | diamond Fd-3m, 5.431 | <100>,<110>,<111> | (110),(111),(100) | Tier A (Peng 1996 / Gao–Peng 1999) |
| Ge | diamond Fd-3m, 5.658 | <100>,<110>,<111> | (110),(111) | Tier A |
| C (diamond) | Fd-3m, 3.567 | <100>,<110>,<111> | (110),(111) | Tier A |
| W | bcc Im-3m, 3.165 | <111>,<100>,<110> | (110),(100) | Tier A |
| Ir | fcc Fm-3m, 3.839 | <110>,<100>,<111> | (111),(100) | Tier A |
| PbWO4 (PWO) | scheelite I4₁/a, a 5.47, c 12.06 | [001],[100],[111] | (001),(100),(101) | Tier C1 (Trots 2010) → C2 |
| SiC | 3C: F-43m, 4.36; 4H: P6₃mc | 3C: <110>,<111>; 4H: [0001] | 3C: (111) | 3C: Tier A (Schowalter 2009); 4H: C2/C3 |
| Bi4Ge3O12 (BGO) | I-43d, 10.52 | <100>,<111> | (110) | Tier C |
| CeF3 | P-3c1 | [0001] | – | Tier C |

- Put lattice parameters in the CIFs, not in the code. The numbers above are only for checking.
- Download every CIF from AMCSD/COD and record the database ID in `materials.yaml`.
- Reference for PbWO4 [001] being the relevant axis: Bandiera et al., PRL 121, 021603 (2018), DOI **10.1103/PhysRevLett.121.021603**.
- Reference for W and Ir at 100–300 K: Kirsebom et al., NIM B 135, 143 (1998), DOI **10.1016/S0168-583X(97)00589-2**.

## 6. Validation (add these as `pytest` tests)

1. Si 293 K: `u` from Tier A ≈ 0.075–0.081 Å. Si (110) planar well depth ≈ 21–23 eV (21.1 eV Doyle-Turner, 22.6 eV Molière in the MAMI analysis, arXiv:2004.04539; compare with Biryukov–Chesnokov–Kotov, DOI 10.1007/978-3-662-03407-1). The axial and planar results must match the current `output/Si/*` within a few % when the same `u` is used.
2. Fourier vs real-space: <0.5 % for Si and Ge at all three axes.
3. The T-dependence must behave physically:
   - `u(T)` is monotonic, with a finite zero-point value at T→0;
   - the potential maximum at the string or plane and `E_max` **decrease** with T;
   - the well depth changes only weakly;
   - the 77 K vs 293 K difference is reported for Ge, W and Ir.
4. A compound sanity check: PbWO4 [001] strings must come out as Pb/W-dominated, with O contributing <~10 % of the depth. Print the per-species breakdown.
5. The zero-point limit must match the Debye formula `145.52/(4AΘ)`.

## 7. Constraints

- Python ≥3.10, numpy, scipy, pymatgen, matplotlib, pyyaml. No new heavy dependencies; phonopy is optional and imported lazily.
- Keep `python ciftxt_reader.py Si "1 1 0" --outdir ...` working with the current behaviour when `--T` is not given (default 293 K via Tier A).
- Vectorise; no Python loops over grid points.
- Do not delete user files. Move the unused/odd CIFs (`AMS_DATA (1).cif` = Si at 1573 K, `AMS_DATA (2).cif` = zircon, the raspite PbWO4) into `cif/unused/` with a README line, rather than removing them.
- Finish with a short report:
  - what was implemented;
  - which `u_j(T)` source was used for each material and site;
  - any coefficients you could not find;
  - a table of U0 and E_max at 77 K and 293 K for every material/axis.

## 8. References (all DOIs verified on Crossref)

| Topic | Reference | DOI |
|---|---|---|
| Atomic scattering factors (Gaussian fits) | Doyle & Turner, Acta Cryst. A24, 390 (1968) | 10.1107/S0567739468000756 |
| Scattering factors, all elements, 0<s<6 | Peng, Ren, Dudarev, Whelan, Acta Cryst. A52, 257 (1996) | 10.1107/S0108767395014371 |
| Debye–Waller B(T), elemental crystals | Peng, Ren, Dudarev, Whelan, Acta Cryst. A52, 456 (1996) | 10.1107/S010876739600089X |
| B(T) parametrisation (68 elements + zincblende) | Gao & Peng, Acta Cryst. A55, 926 (1999) | 10.1107/S0108767399005176 |
| B(T) group IV / III-V / II-VI semiconductors (SiC) | Schowalter et al., Acta Cryst. A65, 5 (2009) | 10.1107/S0108767308031437 |
| DW factor compilation | Sears & Shelley, Acta Cryst. A47, 441 (1991) | 10.1107/S0108767391002970 |
| Si DW from lattice dynamics | Flensburg & Stewart, PRB 60, 284 (1999) | 10.1103/PhysRevB.60.284 |
| Channeling review, thermal averaging | Gemmell, Rev. Mod. Phys. 46, 129 (1974) | 10.1103/RevModPhys.46.129 |
| Thermally averaged string potential | Appleton, Erginsoy, Gibson, Phys. Rev. 161, 330 (1967) | 10.1103/PhysRev.161.330 |
| Molière potential | Molière, Z. Naturforsch. A 2, 133 (1947) | 10.1515/zna-1947-0302 |
| Channeling at accelerators (book) | Biryukov, Chesnokov, Kotov, Springer (1997) | 10.1007/978-3-662-03407-1 |
| Strong-field effects in crystals (book) | Baier, Katkov, Strakhovenko, World Scientific (1998) | 10.1142/2216 |
| CRYSTALRAD (same potential model, used by the group) | Sytov, Tikhomirov, Bandiera, PRAB 22, 064601 (2019) | 10.1103/PhysRevAccelBeams.22.064601 |
| PbWO4 oriented, [001] | Bandiera et al., PRL 121, 021603 (2018) | 10.1103/PhysRevLett.121.021603 |
| W, Ir, Ge at 100–300 K | Kirsebom et al., NIM B 135, 143 (1998) | 10.1016/S0168-583X(97)00589-2 |
| PbWO4 low-T structure / expansion | Trots et al., J. Solid State Chem. 183 (2010) | 10.1016/j.jssc.2010.03.039 |
| Phonopy thermal displacements | Togo, J. Phys. Soc. Jpn. 92, 012001 (2023) | 10.7566/JPSJ.92.012001 |

## 9. Verified input parameters for Ge and W (added after a second literature pass)

All numbers below were checked against the cited primary source or reproduced by calculation. Status is stated per item. Do not replace them without a better source.

### 9.1 Electron scattering factors: Peng et al. 1996, DOI 10.1107/S0108767395014371

`f_e(s) = Σ a_i exp(-b_i s²)`, `s = sin θ/λ` in Å⁻¹, `a_i` in Å, `b_i` in Å². Values come from the machine-readable tables in the `dynasor` package, which say they are Peng 1996 Table 1 (`electron-parameters-kmax6/kmax2-peng-1996.json`). The PDF of the paper itself was not machine-readable, so the numbers were not compared against the printed table.

| El. | range | a1..a5 | b1..b5 |
|---|---|---|---|
| Ge | s ≤ 6 | 0.2135, 0.9761, 1.6555, 2.8938, 1.6356 | 0.0989, 0.9845, 4.5527, 21.5563, 70.3903 |
| Ge | s ≤ 2 | 0.4655, 1.3014, 1.6088, 2.6998, 1.3003 | 0.2647, 1.7926, 7.6071, 26.5541, 77.5238 |
| W | s ≤ 6 | 0.3661, 1.6191, 3.2455, 4.0856, 3.2064 | 0.0761, 0.7543, 4.0952, 18.2886, 68.0967 |
| W | s ≤ 2 | 0.9853, 2.1167, 3.3570, 3.7981, 2.2798 | 0.2569, 1.6745, 7.0098, 26.9234, 81.3910 |
| Ir | s ≤ 6 | 0.3510, 1.5620, 3.2946, 4.0615, 2.4382 | 0.0706, 0.6904, 3.8266, 16.0812, 58.7638 |

- The Ge and Si coefficients already in `ciftxt_reader.py` reproduce these `f_e(s)` to <1 % for s ≤ 2 Å⁻¹ (they differ at s = 3, 0.082 vs 0.088). They are therefore equivalent to Peng for channeling purposes; the Ge discrepancy with Biryukov is not caused by them. Still, replace them by the table above for traceability.
- Use the s ≤ 6 set by default. The s ≤ 2 set changes the planar W result by <3 %.
- Number density `N = n_atoms/a³`: Ge `8/5.658³ = 0.04416 Å⁻³`; W `2/3.1652³ = 0.06306 Å⁻³` (W a = 3.1652 Å at room temperature: standard value, take it from the CIF you download and check it).

### 9.2 Thermal amplitude u (1D rms) from Peng et al. 1996 Table 1 (DOI 10.1107/S010876739600089X)

Table 1 of that paper (B in Å², `B = 8π²⟨u²⟩`) was read from the page image. It covers only 85–280 K in the printed version, with rows at 100, 150, 200 and 250 K omitted. The first two rows (85 and 90 K) are inconsistent for Ge, so use rows ≥ 110 K. An extended 1–1000 K table is deposited with the IUCr (reference ZH0008 in the paper); try to obtain it, and until then use the Debye extrapolation below.

| | B(280 K) | slope near 280 K | B(293 K) extrap. | u(293 K) | Θ_M (Debye fit, 240–280 K rows) |
|---|---|---|---|---|---|
| Ge | 0.5734 | 0.00205 Å²/K | 0.600 | **0.087 Å** (0.0869 from the Debye fit) | 282 K |
| W | 0.1526 | 0.000545 Å²/K | 0.160 | **0.045 Å** (0.0447 from the Debye fit) | 347 K |
| Si (check) | 0.4634 | 0.00165 Å²/K | 0.485 | 0.078 Å | 523 K |

Θ_M implied by the table is not constant (Ge 304 K at 110 K, 282 K at 280 K; W 387 K at 110 K, 346 K at 280 K), so the Debye formula with a single Θ_M is an extrapolation of ±5–10 % in u at low T. Keep this as an explicit systematic uncertainty and expose Θ_M as a parameter.

| T | Ge u(Θ=282) | Ge u(Θ=304) | W u(Θ=310) | W u(Θ=347) | W u(Θ=377) |
|---|---|---|---|---|---|
| 77 K | 0.0508 | 0.0480 | 0.0297 | 0.0274 | 0.0258 |
| 150 K | 0.0643 | 0.0601 | 0.0371 | 0.0335 | 0.0312 |
| 293 K | 0.0870 | 0.0809 | 0.0499 | 0.0447 | 0.0413 |

- The values used in Biryukov's Tables 1.3/1.4 (Ge 0.085, W 0.050 Å) correspond to Θ_M ≈ 290 K (Ge) and ≈ 310 K (W). The W Peng value (0.045 Å) is ~10 % smaller, and that matters: W axial U0 changes by ~10 % and the (110) planar U(x_c) by ~8 % over u = 0.041–0.050 Å (see 9.4).
- An older X-ray measurement gives Θ_M(W) = 377 ± 11 K (https://www.sciencedirect.com/science/article/abs/pii/0025540869900282, DOI not checked).
- For Ge, channeling-radiation measurements (Buschhorn et al., PRB 55, 6196 (1997), DOI 10.1103/PhysRevB.55.6196) give Θ rising from 233 ± 13 K at 12 K to 292 ± 16 K at 223 K. This is the **opposite trend** to the phonon-DOS derived Θ_M(T) in Peng's table. At 77 K it means u between 0.048 and ~0.059 Å. Report that range in the outputs for Ge at low T.
- Ir (not in Peng Table 1): the Debye temperature Θ = 420 K is taken from Peng's Table 2, which cites Kittel (1971). It gives u(293 K) = 0.036 Å. It is only a Debye estimate: flag it as approximate.

### 9.3 Validation benchmark: thermally averaged Molière (reproduces the Biryukov tables)

Implement this model as `--model moliere` and use it as the regression test. Constants: `α = (0.1, 0.55, 0.35)`, `β = (6.0, 1.2, 0.3)`, `a_TF = 0.8853·a0·Z^(-1/3)` (gives 0.194, 0.148, 0.112 Å for Si, Ge, W, as in the book).
- Planar, one plane of areal density σ = N·d_p, Gaussian thermal width u (1D):
  `U(x) = 2π σ Z e² a_TF Σ_i α_i/(2β_i) · exp(κ_i²u²/2) · [exp(-κ_i x) erfc((uκ_i − x/u)/√2) + exp(κ_i x) erfc((uκ_i + x/u)/√2)]`, `κ_i = β_i/a_TF`.
  Evaluate it with `scipy.special.erfcx` and `|x|`, otherwise it overflows for small u (this broke the first W run).
- Axial, string of linear density ν: `U(r) = 2 Z e² ν Σ_i α_i ∫₀^∞ q J0(q r) exp(-q²u²/2)/(q²+κ_i²) dq` (2D Hankel transform of K0 times the thermal Gaussian).
- Linear density per string: count atoms of each projected string in a **half-open** slab of length equal to a multiple of the lattice period, `[-L/2+δ, L/2+δ)`. A closed or symmetric slab drops the end atom and underestimates ν by 25 %. This was a real bug in the first check.

Results (u as in the book; U in eV; "book" = Biryukov Tab. 1.3 and 1.4 as pasted by the user):

| | book | Molière (this model) | DT with Peng, same u |
|---|---|---|---|
| Ge (110) U(x_c) | 27 | 26.6 | 21.0 |
| Ge (111)L U(x_c) | 30 | 28.2 | 21.1 |
| Ge (111)S U(x_c) | 7.2 | 6.9 | 6.9 |
| W (100) U(x_c) | 63 | 61.5 | 53.0 |
| W (110) U(x_c) | 105 | 101.6 | 84.9 |
| Ge axial ⟨100⟩ / ⟨110⟩ / ⟨111⟩ | 157 / 203 / 185 | 161 / 238 / 186 | 148 / 214 / 170 |
| W axial ⟨100⟩ / ⟨110⟩ / ⟨111⟩ | 842 / 576 / 979 | 813 / 572 / 938 | 765 / 539 / 883 |
| Si axial ⟨100⟩ / ⟨110⟩ / ⟨111⟩ | 89 / 114 / 105 | 93 / 139 / 107 | 88 / 130 / 101 |

- Within ~5 % everywhere except the **diamond ⟨110⟩ axis** (Si +22 %, Ge +17 % over the book). Cause not found: either a different definition of U0 for that axis in the book, or a book value issue. Do not tune anything to hide it. Check the book's definition before drawing conclusions about ⟨110⟩.
- DT with Peng factors is 15–25 % lower than Molière for the planar potential at the channel edge (Ge (110), (111)L; W) and 5–8 % lower for axial U0. Coefficients are not the cause (see 9.1). The model is: Molière is Thomas–Fermi, Peng is a fit to relativistic Hartree–Fock scattering factors. Expose both, label which was used, and do not mix them in one comparison.

### 9.4 Reference outputs to reproduce (DT with Peng s ≤ 6 coefficients)

| | u [Å] | planar (110) U(x_c) [eV] | axial U0 ⟨100⟩ / ⟨110⟩ / ⟨111⟩ [eV] |
|---|---|---|---|
| Ge, 77 K | 0.0508 | 27.8 | 212 / 306 / 245 |
| Ge, 293 K | 0.0869 | 20.6 | 145 / 211 / 167 |
| W, 293 K | 0.041 | 92.1 | 870 / 613 / 1004 |
| W, 293 K | 0.045 | 88.7 | 820 / 578 / 947 |
| W, 293 K | 0.050 | 84.9 | 765 / 539 / 883 |

Here U0 = U(string core) − U(channel minimum). The core value is finite only because of the thermal smearing, which is why U0 depends so strongly on u. For the planar column U(x_c) is taken at `x_c = d_p/2 − 2u` from the channel centre, so x_c itself changes with u.
These come from a throwaway script, not from the new code. The new implementation must reproduce them to within 1 %, otherwise explain the difference.
