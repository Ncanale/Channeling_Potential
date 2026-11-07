# Confronto con Geant4: i nostri file FastSim contro quelli ufficiali

Questa cartella risponde a una domanda: i file di dati per il modello di channeling FastSim di Geant4 prodotti da
questo progetto danno gli stessi risultati di quelli ufficiali distribuiti con Geant4?

Il confronto è sempre e solo fra due insiemi di file:

- **Geant4 (ufficiali)**: il dataset G4CHANNELING 2.0 distribuito con Geant4 11.4.3 (curato da A. Sytov).
  I file sono quelli dell'installazione in WSL, `~/g4/install/share/Geant4/data/G4CHANNELING2.0/`.
- **Nostri**: i file scritti da `channeling.fastsim` con la definizione attuale di T_min (`emin='fermi'`,
  Thomas-Fermi per atomo, vedi `FASTSIM.md`) e il fattore di forma atomico di Dudarev, Peng e Whelan (1995) con la
  densità elettronica di Mott-Bethe. Ogni file è calcolato con la stessa ampiezza termica del corrispondente file
  ufficiale: Si 0,075 Å, Ge 0,085 Å, diamante 0,040 Å, W 0,050 Å. Nessun numero è preso dai file ufficiali.

Non ci sono confronti con versioni precedenti del nostro modello né fra versioni diverse dei file ufficiali.

Il confronto è su tre livelli:

1. **I dati** (figure 01–04): campo elettrico, densità elettronica, densità nucleare e T_min, punto per punto.
2. **I risultati fisici della simulazione** (figure 05, 06, 08, 09, 10): Geant4 viene eseguito con i file ufficiali e con i nostri,
   e si confrontano le grandezze misurabili. È questo il test che conta: i dati possono differire in punti a cui la
   simulazione è poco sensibile.
3. **Il test fra seed** (figura 07): la differenza fra noi e Geant4 si confronta con quella fra due esecuzioni di Geant4
   con seed diversi. Se non si distinguono, le nostre simulazioni sono equivalenti a quelle ufficiali entro la statistica.

### Convenzioni delle figure

- **Colori**: Geant4 (ufficiali) in grigio tratteggiato, Nostri in arancione (tab:orange).
- **Residuo**: sempre **(Nostri − Geant4) / Geant4**, in %, scritto per esteso sugli assi. N e G nelle etichette
  brevi stanno per Nostri e Geant4.
- **Eccezione**: per il campo elettrico e la densità nucleare, che si annullano all'interno della cella, il rapporto
  esploderebbe. Si usa **(Nostri − Geant4) / max(Geant4)**, indicato come "/ max G".

## In breve

| test | risultato | conclusione |
|---|---|---|
| campo, densità elettronica e nucleare (19 file) | scarto rms ≤ 0,04 % del massimo per Si e diamante, ≤ 0,26 % per Ge e W (residuo geometrico) | coincidono |
| T_min (19 file) | rms di (N − G)/G: Si 0,5–0,7 %, Ge 0,8–0,9 %, diamante 0,8–1,6 %, W 0,7 % | coincidono |
| rocking assiale, Si 30 µm, 39 punti × 14 seed | differenza media +0,08 % ± 0,09 %, χ² = 36,6 / 39 | nessuna differenza misurabile |
| rocking assiale, Ge 30 µm, 39 punti × 4 seed | differenza media +0,06 % ± 0,19 %, χ² = 36,8 / 39 | nessuna differenza misurabile |
| rocking assiale, diamante 30 µm, 39 punti × 12 seed | differenza media +0,22 % ± 0,09 %, χ² = 54,5 / 39 (p = 0,05); fuori asse (\|θ\| > 200 µrad) +0,37 % ± 0,10 % | differenza piccola ma misurabile |
| rocking assiale, W ⟨111⟩ 10 µm, 13 inclinazioni × 12 seed | frazione entro ±500 µrad −0,02 % ± 0,11 %; angolo medio di uscita −0,17 % ± 0,18 %; larghezza rms +0,13 % ± 0,07 % | nessuna differenza misurabile |
| radiazione, W ⟨111⟩ 100 µm, 40 000 e⁻ per lato | energia irraggiata +1,4 % ± 1,1 %, spettro χ² = 35,3 / 28 (p = 0,16) | nessuna differenza misurabile |
| test fra seed (rocking e radiazione) | noi contro Geant4 come Geant4 contro Geant4: KS p = 0,90 (rocking), 0,34 (radiazione) | indistinguibili |

I dati coincidono punto per punto con quelli ufficiali, T_min compreso: i file ufficiali sono costruiti con il
fattore di forma di Dudarev, Peng e Whelan (1995) e la densità di Mott-Bethe, e così ora i nostri (Parte 1, "Il
modello atomico").

Le simulazioni Geant4 delle Parti 2–6 sono state fatte prima, con i fattori di forma usati allora (Peng 1996; per Ge
Doyle-Turner 1968; per il diamante Peng con densità da raggi X), con cui i dati differivano dello 0,2–1,7 % (campo,
densità) e del 3–27 % (T_min). Già con quei file Si, Ge e W non si distinguevano dagli ufficiali entro la statistica
raccolta; il diamante, il più lontano nei dati, mostrava una differenza dello 0,2–0,4 % nelle rocking curve,
statisticamente reale ma molto sotto la precisione tipica delle misure di channeling (qualche %).

---

## Parte 1: i dati (figure 01–04)

Script: `make_comparison_plots.py`. Produce le figure 01–04 e `agreement_table.csv`.

### Che cosa contiene un file FastSim

Per ogni piano (hkl) o asse ⟨uvw⟩, su una cella periodica, il file contiene i coefficienti di spline di:

- **campo elettrico** E_x (piani) oppure E_x, E_y (assi), in eV/cm: guida la traiettoria della particella;
- **densità nucleare**, normalizzata alla media: pesa lo scattering coerente sui nuclei;
- **densità elettronica**, in cm⁻³: pesa lo scattering sugli elettroni;
- **T_min**, in eV: l'energia di ionizzazione minima locale. Sotto questa soglia il trasferimento di energia agli
  elettroni è trattato in modo medio (formula di Bethe), sopra come urti singoli. Il tasso di urti singoli è
  proporzionale a n_e / T_min.

### Stessa cella, stessa origine: confronto punto per punto

I nostri file usano la stessa cella, la stessa origine e lo stesso orientamento di quelli ufficiali, quindi le mappe
si confrontano direttamente, senza traslazioni né specchiature.

- **Piani**: x = 0 è il minimo del potenziale, al centro del canale; il piano atomico sta a dx/2. Era già così.
- **Assi**: la convenzione dei file ufficiali, ricavata da tutti i 10 file assiali (Si, Ge, diamante, W), è che le
  stringhe di atomi stiano sulla diagonale principale della cella, con la posizione media al centro:
  - una sola stringa per cella (⟨100⟩): al centro, in (dx/2, dy/2);
  - due stringhe (⟨111⟩ di Si, Ge, diamante e W): in (dx/4, dy/4) e (3dx/4, 3dy/4);
  - le coppie di stringhe ravvicinate (dumbbell) del ⟨110⟩ del diamante, Si e Ge: centrate negli stessi due punti.

  `channeling.fastsim.axial_fastsim` (funzione `_geant4_origin`) sceglie l'origine con questa regola, partendo solo
  dalla struttura cristallina: non legge i file ufficiali. In precedenza mettevamo l'origine nel minimo del potenziale,
  e per confrontare le mappe serviva una traslazione, a volte con una specchiatura in y.

**Verifica** (`tests/test_fastsim.py`). Per i 10 file assiali, la migliore sovrapposizione cercata con la correlazione
incrociata, ammettendo traslazioni e specchiatura in y, coincide con il confronto diretto. Il residuo è lo stesso, e i
residui di E_x ed E_y, segno compreso, sono 0,07–0,2 % del massimo del campo per Si, Ge e W. I test controllano il
confronto diretto per Si ⟨110⟩ e la posizione delle stringhe per Si ⟨100⟩, Si ⟨111⟩ e W ⟨111⟩.

Il caso W ⟨111⟩ ha richiesto attenzione. Nel reticolo bcc i due atomi della cella cubica stanno sulla stessa colonna
⟨111⟩, mentre la cella rettangolare ne contiene due. La regola usa quindi le colonne generate anche dalle traslazioni
del reticolo.

**Nelle simulazioni l'origine non conta.** Il fascio (σ = 0,3 mm) copre milioni di celle, quindi Geant4 campiona
uniformemente il punto d'ingresso nella cella. Le simulazioni Geant4 di questa cartella sono state fatte con i nostri
file nella convenzione precedente, con l'origine nel minimo del potenziale, e restano valide.

L'unica differenza geometrica residua è il parametro reticolare: la nostra cella W è più piccola dello 0,05 % di quella
ufficiale, e quella di Si e Ge differisce di meno dello 0,05 %.

### Figura 01: `01_planes_Si.png`

Si, piani (100), (110), (111). Nelle colonne ci sono le quattro grandezze; in grigio tratteggiato i dati ufficiali,
in arancione i nostri. Dove le curve coincidono l'arancione copre il grigio. Sotto ogni pannello c'è il residuo:

- T_min e densità elettronica: (Nostri − Geant4) / Geant4, in %;
- campo e densità nucleare: (Nostri − Geant4) / max(Geant4), in %. Il campo cambia segno e la densità nucleare è
  praticamente zero al centro del canale, quindi il residuo punto per punto non avrebbe senso.

### Figura 02: `02_axes_Si_Tmin_maps.png`

Si, assi ⟨100⟩, ⟨110⟩, ⟨111⟩: mappe di T_min su una cella. Per ogni asse ci sono tre pannelli:

1. ufficiale;
2. nostro;
3. **(Nostri − Geant4) / Geant4**, in %.

Nei primi due pannelli T_min è in scala logaritmica, perché varia di più di tre ordini di grandezza sulla cella: da
~0,3 eV nei canali a ~800 eV sulle stringhe di atomi. Nel terzo la scala è saturata a ±5 %: blu dove i nostri valori
sono più bassi, rosso dove sono più alti; i colori pieni indicano differenze di almeno il 5 %. Restano deboli anelli
concentrici attorno alle stringhe atomiche, di qualche per mille, dove T_min cambia più rapidamente.

### Figura 03: `03_Tmin_planes_all.png`

T_min su tutti e 9 i file planari (Si, Ge, diamante), nostri contro ufficiali, in scala logaritmica. Nel titolo di
ogni pannello c'è lo scarto quadratico medio di (Nostri − Geant4) / Geant4 sul periodo.

### Figura 04: `04_Tmin_agreement_all_files.png`

Una barra per ciascuno dei 19 file ufficiali: lo scarto quadratico medio di (Nostri − Geant4) / Geant4 su T_min,
calcolato sulla cella, in %.

### `agreement_table.csv`

Una riga per file. Le colonne `T_*` e `ne_rel_rms` sono (Nostri − Geant4) / Geant4, come frazione (0,05 = 5 %):

| colonna | significato |
|---|---|
| `T_mean` | media di (N − G)/G su T_min nella cella (0 = nessuno scarto sistematico) |
| `T_rms` | scarto quadratico medio di (N − G)/G su T_min |
| `T_channel_median` | mediana di (N − G)/G su T_min nei canali (T_min ufficiale < 20 eV) |
| `T_atom_median` | idem vicino agli atomi (T_min ufficiale > 100 eV) |
| `ne_rel_rms` | scarto quadratico medio di (N − G)/G sulla densità elettronica |
| `ne_rel_max` | rms di (N − G) / max(G) sulla densità elettronica |
| `nuc_rel_max` | idem per la densità nucleare |
| `E_rel_max` | idem per il modulo del campo |

**Che cosa si vede.**

| | T_min: rms di (N − G)/G | T_min: media | campo, densità elettronica e nucleare: rms di (N − G)/max G |
|---|---|---|---|
| Si, 6 file | 0,5–0,7 % | +0,3 / +0,4 % | ≤ 0,03 % |
| Ge, 6 file | 0,8–0,9 % | +0,5 / +0,6 % | ≤ 0,26 % |
| diamante, 6 file | 0,8–1,6 % | +0,1 / +0,5 % | ≤ 0,04 % |
| W ⟨111⟩ | 0,7 % | +0,3 % | ≤ 0,11 % |

Per Ge e W il residuo su campo e densità segue quello della densità nucleare, che dipende solo dalla geometria: il
nostro parametro reticolare differisce fino allo 0,05 % da quello dei file ufficiali. Vmax coincide entro lo 0,06 %
per i piani (100), (110) e gli assi ⟨100⟩; è più alto dello 0,2–0,5 % per i piani (111) e W ⟨111⟩, dello 0,6–1,3 %
per gli assi ⟨111⟩ e del 2–6 % per gli assi ⟨110⟩ di Si, Ge e diamante, cioè dove nella cella ci sono due buche o
stringhe non equivalenti: il valore ufficiale è probabilmente calcolato in altro modo (il campo coincide).

### Il modello atomico: Dudarev, Peng e Whelan (1995)

Il README dei dati ufficiali non dice con quale modello di atomo sono calcolati. Con i fattori di forma usati prima
(Peng 1996; Doyle-Turner 1968 per Ge; per il diamante Peng con densità da raggi X) T_min differiva del 4–9 % (Si),
3–7 % (Ge), 10,6 % (W) e 9–27 % (diamante). Il modello si è trovato in due passi.

**1. Che cosa c'è nei file ufficiali.** Per un piano, la serie di Fourier del campo ha un'armonica per ogni multiplo n
del vettore reciproco, proporzionale a f_e(s) · exp(−B s²) · (fattore di struttura), con s = n / (2 d). Dividendo le
armoniche di un file ufficiale per quelle di un file calcolato con un f_e noto e la stessa ampiezza termica, il
fattore di struttura si semplifica e resta f_e ufficiale / f_e noto a ogni s (31–36 valori fino a 3,4–6 Å⁻¹ dai 9
file planari; 99 valori dalla FFT bidimensionale di W ⟨111⟩). Risultati:

| | risultato |
|---|---|
| densità nucleare | rapporto 1,0000 a tutti gli s: stessa ampiezza termica u, nucleo gaussiano |
| densità elettronica | è quella di Mott-Bethe dello stesso f_e, n(q) = Z − ½ a₀ q² f_e (entro lo 0,3 %), anche per il diamante |
| f_e(s) | entro l'1–2 % da Doyle-Turner 1968 fino a s ≈ 1,9 Å⁻¹; oltre, una coda che Doyle-Turner (fit fino a s = 2) non ha. Non è Peng 1996, Kirkland 2010 né Lobato 2014 (scarti fino al 5–10 % a s = 2–4 Å⁻¹) |

**2. La fonte pubblicata.** Il codice CRYSTAL di Sytov (Sytov e Tikhomirov, NIM B 355, 383 (2015),
DOI 10.1016/j.nimb.2015.02.042) usa il potenziale "Doyle-Turner" della referenza Dudarev, Peng, Whelan, Surf. Sci.
330, 86 (1995), DOI 10.1016/0039-6028(95)00464-5. Le Tabelle 1–2 di quel lavoro danno, per 50 elementi, 5 gaussiane
per la parte reale di f_e, adattate ai valori Hartree-Fock relativistici fino a s = 5 Å⁻¹. Una somma di 5 gaussiane
adattata ai valori estratti al punto 1 dà gli stessi coefficienti entro qualche per mille (per C: a = 0,670 1,186
0,489 0,131 0,031 Å, b = 38,5 12,2 3,31 0,632 0,076 Å², contro 0,662 1,193 0,490 0,132 0,031 e 38,9 12,3 3,31 0,633
0,076 della tabella; lo stesso per Si e W). I file ufficiali sono quindi costruiti con Dudarev-Peng-Whelan e la
densità di Mott-Bethe.

Il generatore usa ora la tabella pubblicata (`channeling/data/dudarev_peng_whelan_1995.json`, modello `dpw` in
`channeling/form_factors.py`) per i 50 elementi che contiene, con la densità di Mott-Bethe; gli altri restano con
Peng 1996. La trascrizione è controllata: b decrescente in ogni riga, e f_e(0) = Σ a_i entro l'1 % dai valori
tabulati da Doyle e Turner (1968) per 30 dei 32 elementi in comune (Al −1,8 %, Zn −4,9 %, come stampato).

| | T_min: rms di (N − G)/G, prima → con Dudarev-Peng-Whelan | campo: rms / max G, prima → ora |
|---|---|---|
| Si | 4–9 % → 0,5–0,7 % | 0,10–0,28 % → ≤ 0,03 % |
| Ge | 3–7 % → 0,8–0,9 % | 0,21–0,35 % → ≤ 0,26 % |
| diamante | 9–27 % → 0,8–1,6 % | 0,52–1,19 % → ≤ 0,04 % |
| W ⟨111⟩ | 10,6 % → 0,7 % | 0,14 % → 0,11 % |

Quasi tutta la differenza su T_min veniva quindi dal fattore di forma, non dalla definizione di T_min.

---

## Parte 2: rocking curve assiali in Si (figura 05)

### Configurazione

- **Codice**: Geant4 11.4.3, esempio `ch1` modificato (`~/chtest/ch2` in WSL), con il modello FastSim di channeling
  attivo per e±.
- **Fascio**: elettroni (e⁻) da 855 MeV, σ = 0,3 mm in posizione, divergenza 1 µrad.
- **Cristallo**: Si dritto, spessore 30 µm, assi ⟨100⟩, ⟨110⟩, ⟨111⟩.
- **Inclinazione**: in x, 13 valori (0, ±200, ±300, ±400, ±500, ±600, ±700 µrad), per 39 punti in tutto. I primi
  7 valori (0, ±200, ±500, ±700) sono stati simulati per primi; ±300, ±400, ±600 sono stati aggiunti dopo, con gli
  stessi 14 seed, per riempire la curva.
- **Statistica**: 20 000 eventi per punto e per seed, 14 seed (1–14). Lo stesso seed è usato per i file ufficiali e
  per i nostri.
- **Osservabile**: l'istogramma dell'angolo di uscita in x dell'elettrone primario (`*_h1_thx.csv`, 2400 bin da
  25 µrad su ±30 mrad).

### La grandezza confrontata

La **frazione di elettroni che esce nella migliore finestra di ±100 µrad**, cioè la finestra di 8 bin con più
conteggi. Misura quanto il fascio resta collimato attraversando il cristallo. Dipende dal campo (channeling) e dallo
scattering (che fa uscire le particelle dal canale), quindi da tutti gli ingredienti del file.

### Come si stimano gli errori

Con lo stesso seed le due simulazioni condividono parte della sequenza casuale, e i loro risultati sono correlati.
L'errore binomiale di ciascun punto non va bene per la differenza. Si procede così:

1. per ogni seed si calcola la differenza Nostri − Geant4;
2. si prende la media sui 14 seed;
3. l'errore è la deviazione standard fra seed divisa per √14.

È anche il modo per stabilire quale accordo ci si può aspettare (`seeds_analysis.py`):

| quantità | dispersione fra seed della differenza |
|---|---|
| singolo punto, 20 000 eventi | ±0,0036 (circa ±1,7 %) |
| media dei 39 punti, 1 seed | ±0,0007 (circa ±0,33 %) |
| media dei 39 punti, 14 seed | ±0,0002 (circa ±0,09 %) |

### Figura 05: `05_geant4_rocking_axes.png`

- **Pannelli in alto**: la frazione nella finestra di ±100 µrad in funzione dell'inclinazione, un pannello per asse.
  Grigio tratteggiato: ufficiali; arancione: nostri. Media sui 14 seed.
- **Pannelli in basso**: (Nostri − Geant4) / Geant4 in %, con errore dalla dispersione fra seed.

Le curve non sono monotone: a ±500 µrad la frazione ha un minimo, poi risale a ±700 µrad. L'andamento è identico per
i due insiemi di file.

**Risultato.**

- Differenza media +0,0002 ± 0,0002, cioè +0,08 % ± 0,09 %.
- χ² rispetto a zero: 36,6 su 39 punti. Un solo punto supera 2σ (2,5σ, ⟨111⟩ a +300 µrad); per puro caso se ne
  aspettano circa 2.
- Un eventuale effetto sistematico è sotto lo 0,4 % (limite a 2σ).

Con i primi 4 seed la differenza sembrava +0,5 %: era dovuta al seed 2 (+1,3 %) ed è sparita con 14 seed.

### Test sulla forma completa: `shape_test.py`

Per ogni punto (39) si sommano i 14 seed (280 000 eventi per lato) e si confrontano gli istogrammi interi.

- **χ² a due campioni**, bin per bin: 7847 su 7921, p = 0,72. Il test tratta i due insiemi come indipendenti; con gli
  stessi seed sono in parte correlati, quindi il test è prudente.
- **Frazioni in finestre centrate sul picco**, da ±25 a ±1000 µrad: differenze medie fra 0,0 % e 0,13 %. La più
  lontana da zero è quella a ±400 µrad, +0,05 % ± 0,02 % (2,3σ); su sei finestre, per giunta correlate, non è
  significativa. In ogni caso è cento volte sotto il 5 %.

### `rocking_fraction.csv`

Una riga per punto (asse, inclinazione): frazione ufficiale e nostra con errore, differenza con errore,
(Nostri − Geant4) / Geant4 in % con errore, numero di seed, eventi per seed.

---

## Parte 3: radiazione in W ⟨111⟩ (figura 06)

### Perché questo test

La radiazione di channeling è molto più sensibile al potenziale vicino alle stringhe di atomi rispetto all'angolo di
uscita. Il tungsteno, con Z alto e asse ⟨111⟩ allineato, è il caso con il campo più intenso.

### Configurazione

- **Fascio**: e⁻ da 855 MeV, come sopra.
- **Cristallo**: W, spessore 100 µm, asse ⟨111⟩ allineato (inclinazione 0).
- **Radiazione**: modello di Baier-Katkov del FastSim attivo.
- **Statistica**: 20 seed × 2000 eventi = 40 000 elettroni per lato, con gli stessi seed per i due insiemi.
- **Tempi**: circa 0,74 s per evento su un core, cioè circa 1 ora per 2000 eventi con 10 processi in parallelo su
  questo portatile.

### Le grandezze confrontate

Tutte normalizzate per **elettrone incidente**:

- **spettro dei fotoni E·dN/dE**: istogramma da 1 MeV fino a 860 MeV, raggruppato in 28 bin logaritmici;
- **energia irraggiata totale**, e quella fra 178 e 438 MeV (la parte dura dello spettro);
- **numero di fotoni contati**;
- **spettro di energia dell'elettrone in uscita** e sua media.

Circa il 4 % degli elettroni primari non arriva al rivelatore, con la stessa frazione nei due insiemi
(3,9 % e 4,1 %). I loro fotoni sono comunque contati; per questo si normalizza per elettrone incidente e non per
elettrone rivelato.

### Come si stimano gli errori

Come per il rocking: deviazione standard fra i 20 seed divisa per √20, sulle differenze appaiate. Per gli spettri,
(Nostri − Geant4) / Geant4 si calcola dagli spettri sommati sui seed, con l'errore dalla dispersione fra seed. Fare la
media dei rapporti seed per seed sposta il risultato verso l'alto nei bin con pochi conteggi.

### Figura 06: `06_geant4_radiation_W111.png`

- **A sinistra**: lo spettro E·dN/dE (in alto) e (Nostri − Geant4) / Geant4 in % (in basso), in funzione dell'energia
  del fotone in scala logaritmica.
- **A destra**: la distribuzione dell'energia dell'elettrone in uscita, come frazione per MeV (in alto, scala
  logaritmica), e (Nostri − Geant4) / Geant4 in % (in basso). I bin hanno larghezza variabile: si costruiscono
  dall'alto, ciascuno con almeno 400 elettroni Geant4 (sommando i seed) e largo almeno 10 MeV, così che ogni residuo
  abbia un errore di qualche percento. In basso a sinistra l'intervallo 0–480 MeV, con pochi elettroni, è un solo bin.
- L'asse dell'energia dei fotoni è lineare sotto 1 MeV e logaritmico sopra, per mostrare anche il bin 0–1 MeV.

**Risultato.**

| per elettrone incidente | Geant4 | Nostri | differenza |
|---|---|---|---|
| energia irraggiata | 46,72 MeV | 47,37 MeV | +1,4 % ± 1,1 % (1,3σ) |
| energia irraggiata 178–438 MeV | 11,47 MeV | 11,80 MeV | +2,9 % ± 3,0 % (1,0σ) |
| fotoni contati | 1,134 | 1,150 | +1,4 % ± 0,7 % (1,9σ) |
| energia media dell'e⁻ in uscita | 813,4 MeV | 813,2 MeV | −0,03 % ± 0,07 % |

Sullo spettro dei fotoni, (Nostri − Geant4) / Geant4 ha χ² = 35,3 su 28 bin rispetto a zero (p = 0,16), con 17 bin
positivi e 11 negativi. Sullo spettro dell'elettrone in uscita, χ² = 20,7 su 21 bin (p = 0,48).
Un eventuale effetto sull'energia irraggiata è sotto il 3,5 % (limite a 2σ).

### Le mappe W: `maps_W.py`

Confronto diretto dei due file W ⟨111⟩; il nostro è generato al momento con `channeling.fastsim` (modello
atomico attuale, Dudarev-Peng-Whelan):

| grandezza | scarto rms (% del massimo) | massimo nostro / ufficiale − 1 |
|---|---|---|
| campo E_x | 0,14 % | −0,05 % |
| campo E_y | 0,11 % | +0,33 % |
| densità elettronica | 0,07 % | +0,49 % |
| densità nucleare | 0,08 % | +0,69 % |

Gli scarti residui vengono dalla geometria: la nostra cella W è più piccola dello 0,05 % e il confronto diretto
accumula uno spostamento fino a 0,002 Å sul bordo della cella.

Altri valori:

- profondità della buca Vmax: 886,8 eV per noi contro 884,8 eV ufficiale;
- T_min: (Nostri − Geant4) / Geant4 con rms 0,67 % e media +0,29 %.

Le simulazioni di radiazione di questa parte sono state fatte con il fattore di forma usato allora (Peng 1996), con
cui T_min differiva del 10,5 % rms e il campo dello 0,16 %: già allora la radiazione non mostrava differenze.

### `radiation_W111.csv`

- **Prima tabella**: le grandezze integrate (valore ufficiale, nostro, differenza, ciascuno con errore).
- **Seconda tabella**: lo spettro per bin (estremi in MeV, E·dN/dE ufficiale e nostro,
  (Nostri − Geant4) / Geant4 in % con errore).

---

## Parte 4: il test fra seed (figura 07)

### L'idea

Due esecuzioni di Geant4 con gli stessi file ufficiali e seed diversi danno risultati diversi solo per la statistica.
Questa differenza "Geant4 contro Geant4" è il riferimento naturale: è il massimo accordo che si può chiedere a
qualunque altro insieme di file con la statistica raccolta. Se "Nostri contro Geant4" ha la stessa distribuzione,
le nostre simulazioni non si distinguono da un'altra esecuzione di Geant4 ufficiale.

Script: `seed_test.py`. Usa gli stessi run delle figure 05 e 06, senza simulazioni aggiuntive.

### Figura 07: `07_seed_test.png`

**(a) Rocking, tutti i seed.** Per ogni punto (asse e inclinazione) ci sono 14 punti grigi (Geant4, un seed ciascuno) e
14 arancioni (Nostri), come scarto percentuale dalla media di Geant4 sui seed. Le due nuvole hanno la stessa
larghezza e lo stesso centro.

**(b) Rocking, differenza fra due esecuzioni.** Per ogni punto e ogni seed s si confrontano due distribuzioni:

- **grigio tratteggiato**: Geant4 con il seed s meno Geant4 con il seed s+1 (riferimento);
- **arancione**: Nostri con il seed s meno Geant4 con il seed s+1.

Entrambe sono divise per la media di Geant4. Il test di Kolmogorov-Smirnov fra le due dà p = 0,90: non si distinguono.
Larghezze: 1,91 % per Geant4 contro Geant4, 1,82 % per noi contro Geant4. Con lo stesso seed (Nostri s contro
Geant4 s) la larghezza è 1,80 %: la correlazione fra esecuzioni con lo stesso seed è debole.

Un altro controllo, punto per punto: il t-test di Welch fra i 14 valori Geant4 e i 14 nostri dà p < 0,05 in 2 punti
su 39, esattamente quanto atteso per puro caso (2,0).

**(c, d) Radiazione W ⟨111⟩.** I 20 valori per seed dell'energia irraggiata per elettrone incidente: in totale (c) e
fra 178 e 438 MeV (d). Grigio: Geant4; arancione: Nostri. La barra sottile è la deviazione standard fra seed, quella
spessa l'errore sulla media.

| grandezza | Geant4: media (dispersione fra seed) | Nostri: media (dispersione) | KS p | t-test p |
|---|---|---|---|---|
| energia irraggiata | 46,7 (2,2) MeV | 47,4 (2,2) MeV | 0,34 | 0,36 |
| energia irraggiata 178–438 MeV | 11,5 (1,4) MeV | 11,8 (1,0) MeV | 0,57 | 0,38 |

Nella radiazione le traiettorie si separano subito, per cui i run con lo stesso seed sono di fatto indipendenti e i
test a due campioni si applicano direttamente.

### `seed_test.csv`

Una riga per test: media e dispersione fra seed per Geant4 e per i nostri, p-value di Kolmogorov-Smirnov e del
t-test.

---

## Parte 5: rocking curve assiali in Ge e diamante (figure 08–09)

Script: `rocking_GeC.py`. Produce le figure 08 e 09 e `rocking_fraction_GeC.csv`.

### Configurazione

Come per il Si (Parte 2): e⁻ da 855 MeV, cristallo da 30 µm, assi ⟨100⟩, ⟨110⟩, ⟨111⟩, 13 inclinazioni in x
(0, ±200, ±300, ±400, ±500, ±600, ±700 µrad), 20 000 eventi per run, stessi seed per i due insiemi di file.
Grandezza confrontata: la frazione dell'istogramma dell'angolo di uscita nella migliore finestra di ±100 µrad.

- **Materiali**: `G4_Ge` (u = 0,085 Å) e un materiale `G4_Diamond` definito nell'esempio (carbonio, 3,515 g/cm³).
  Il nome del file di dati che Geant4 cerca è il nome del materiale senza `G4_`, quindi il diamante ha bisogno di un
  materiale suo: `G4_C` è grafite (2,0 g/cm³) e cercherebbe un file `C`.
- **I nostri file** hanno l'origine della cella come nei file ufficiali e il fattore di forma usato allora:
  Doyle-Turner 1968 per Ge; Peng 1996 con densità da raggi X (Waasmaier-Kirfel) per il diamante.
- **Seed**: 4 per il Ge (312 run); 12 per il diamante (936 run), perché con 4 seed lo scarto del diamante era al
  limite della significatività.

### Come si stimano gli errori

Con pochi seed la dispersione di un singolo punto è mal determinata (3 gradi di libertà con 4 seed): stimandola punto
per punto, alcuni punti risultano con errori quasi nulli e il χ² esplode. Si usa allora la **varianza relativa
mediata sui 39 punti**: la dispersione relativa di una frazione da 20 000 eventi è quasi la stessa ovunque, e così
si hanno 39 × (n − 1) gradi di libertà. Lo script stampa come controllo l'atteso binomiale:

| | dispersione relativa di una run, Geant4 / Nostri | atteso binomiale | dispersione della differenza |
|---|---|---|---|
| Ge | 1,57 % / 1,45 % | 1,88 % | 2,37 % |
| diamante | 1,24 % / 1,28 % | 1,36 % | 1,83 % |

Le dispersioni coincidono con l'atteso: i residui di ±1–2 % dei singoli punti sono solo statistica. L'errore sulla
media dei 39 punti assume i punti indipendenti. Lo script stampa anche l'errore ricavato dalla dispersione fra le
medie dei seed, che è più piccolo (±0,08 % per Ge, ±0,05 % per il diamante) ma ha solo n − 1 gradi di libertà; nel
testo si usa il primo, più prudente.

### Figura 08: `08_geant4_rocking_Ge.png`

In alto le tre rocking curve (frazione entro ±100 µrad contro inclinazione), Geant4 in grigio e Nostri in arancione;
in basso (Nostri − Geant4)/Geant4 in %.

- Differenza media sui 39 punti +0,06 % ± 0,19 %; χ² rispetto a zero 36,8 su 39; un punto oltre 2σ (atteso 2).
- Per asse: ⟨100⟩ +0,32 ± 0,33 %, ⟨110⟩ −0,41 ± 0,33 %, ⟨111⟩ +0,28 ± 0,33 %.

**Nessuna differenza misurabile**, come per il Si.

### Figura 09: `09_geant4_rocking_Diamond.png`

- Differenza media sui 39 punti **+0,22 % ± 0,09 %** (2,5σ); χ² rispetto a zero 54,5 su 39 (p = 0,05); 5 punti
  oltre 2σ (attesi 2).
- Per asse: ⟨100⟩ +0,36 ± 0,15 % (χ² 28,8 / 13), ⟨110⟩ +0,05 ± 0,15 %, ⟨111⟩ +0,24 ± 0,15 %.
- Per inclinazione: entro ±200 µrad −0,31 % ± 0,18 %; oltre ±200 µrad **+0,37 % ± 0,10 %** (3,7σ).

Con 4 seed lo scarto era +0,22 % ± 0,15 %, con 9 seed +0,22 % ± 0,10 %, con 12 seed +0,22 % ± 0,09 %: la media è
stabile mentre l'errore scende, quindi è un effetto vero. La forma si vede nel pannello ⟨100⟩: residui leggermente
negativi vicino all'asse e positivi (+1–1,5 %) a ±500–700 µrad, cioè con i nostri file un po' più elettroni
fuori asse escono entro ±100 µrad.

**Interpretazione.** Con il fattore di forma usato in queste simulazioni il diamante era il materiale riprodotto
peggio anche nei dati: densità elettronica all'1,3–1,7 % del massimo dai file ufficiali (Ge ≤ 0,6 %), campo allo
0,5–1,2 %, T_min con rms 21–27 % sugli assi. In pratica +0,2–0,4 % è molto sotto la precisione tipica delle misure di
channeling (qualche %) e più di dieci volte sotto il 5 %, ma con quei file non si poteva dire che le simulazioni nel
diamante fossero indistinguibili. Con Dudarev-Peng-Whelan (Parte 1) i dati del diamante coincidono entro lo 0,04 %
(campo, densità) e l'1,6 % (T_min).

### `rocking_fraction_GeC.csv`

Una riga per materiale, asse e inclinazione: frazione media ed errore per Geant4 e per i nostri, differenza ed
errore, differenza relativa in %, numero di seed, eventi per seed.

---

## Parte 6: rocking curve in W ⟨111⟩ (figura 10)

Script: `rocking_W.py`. Produce la figura 10 e `rocking_W111.csv`.

### Perché un protocollo diverso

Il test della radiazione (Parte 3) era a una sola inclinazione, con il file W della versione precedente. Qui si
ripete per il W lo scan in inclinazione fatto per gli altri materiali, con il file della versione attuale
(`~/chtest/nowW2/T293K/`). Il protocollo del Si però non funziona: nel W l'angolo di Lindhard a 855 MeV è
1,44 mrad (circa tre volte quello del Si) e già a 10 µm lo scattering multiplo allarga la distribuzione in uscita a
circa 1,2 mrad rms. La frazione entro ±100 µrad cambia appena con l'inclinazione (0,069 a 0 µrad, 0,064 a
1500 µrad) e non dice nulla sul potenziale.

### Configurazione

- e⁻ da 855 MeV, W ⟨111⟩ da **10 µm**, senza radiazione.
- 13 inclinazioni in x: 0, ±500, ±1000, ±1500, ±2000, ±2500, ±3000 µrad.
- 12 seed × 20 000 eventi per punto, stessi seed per i due insiemi di file (312 run).
- Tre grandezze dall'istogramma dell'angolo di uscita in x:
  - **frazione nella migliore finestra di ±500 µrad**;
  - **angolo medio di uscita**: il potenziale attira il fascio verso l'asse (a 1500 µrad di inclinazione l'angolo
    medio di uscita è circa 820 µrad), quindi è la grandezza più direttamente legata al campo;
  - **larghezza rms** della distribuzione.

### Come si stimano gli errori

Come per Ge e diamante (Parte 5): dispersione fra seed / √12, varianza mediata sulle 13 inclinazioni (relativa per
frazione e larghezza, assoluta per l'angolo medio), appaiata per la differenza. Dispersione di una singola run:
frazione 0,93 %, angolo medio 8–9 µrad, larghezza 0,6–0,7 %.

### Figura 10: `10_geant4_rocking_W111.png`

In alto le tre grandezze contro l'inclinazione (al centro la retta grigia è "angolo di uscita = inclinazione", cioè
nessuna deflessione); in basso i residui: (Nostri − Geant4)/Geant4 in % per frazione e larghezza, Nostri − Geant4
in µrad per l'angolo medio, che passa per zero a inclinazione nulla.

| grandezza | (Nostri − Geant4)/Geant4, media sulle inclinazioni | χ² rispetto a zero |
|---|---|---|
| frazione entro ±500 µrad | −0,02 % ± 0,11 % | 12,2 / 13 |
| angolo medio di uscita (inclinazione ≠ 0) | −0,17 % ± 0,18 % | 19,1 / 12 (p = 0,09) |
| larghezza rms | +0,13 % ± 0,07 % | 13,7 / 13 |

Per l'angolo medio, il χ² su tutti i 13 punti è 23,2: il contributo in più viene dal punto a inclinazione nulla,
−6,9 ± 3,4 µrad, dove per simmetria entrambe le simulazioni hanno media zero e la differenza è solo rumore. La
parte antisimmetrica della differenza, quella che misurerebbe una deflessione diversa, è −1,1 ± 1,0 µrad su
deflessioni di 400–800 µrad.

**Nessuna differenza misurabile**: anche fuori allineamento il W con i nostri file si comporta come con quello ufficiale.

### `rocking_W111.csv`

Una riga per grandezza e inclinazione: valore Geant4 e nostro, differenza ed errore, residuo ed errore (in % o in
µrad, colonna `residual_unit`), numero di seed, eventi per seed.

---

## Come rifare i conti

### Geant4 (in WSL Ubuntu-24.04)

- **Installazione**: Geant4 11.4.3 in `~/g4/install`.
- **Esempio**: `~/chtest/ch2`, eseguibile `~/chtest/ch2/build/ch1`. Rispetto all'esempio ufficiale `ch1` legge i
  parametri da variabili d'ambiente.

| variabile | significato |
|---|---|
| `G4CHANNELINGDATA` | cartella dei file FastSim da usare |
| `CH_MAT` | materiale (`G4_Si`, `G4_Ge`, `G4_W`, `G4_Diamond`, ...) |
| `CH_LATTICE` | piano `(hkl)` o asse `<uvw>` |
| `CH_THICK` | spessore in mm |
| `CH_ANGLEX`, `CH_ANGLEY` | inclinazione in µrad |
| `CH_BEND` | angolo di curvatura in mrad |
| `CH_RAD` | se definita, attiva il modello di radiazione |
| `CH_OUT` | nome dei file di uscita |

**Istogrammi in uscita**:

| file | contenuto |
|---|---|
| `_h1_thx` / `_h1_thy` | angolo di uscita del primario |
| `_h1_Eprim` | energia del primario all'uscita |
| `_h1_Spectrum` | energia dei fotoni |
| `_h1_x_out` | posizione |

**Script di lancio** (in `~/chtest`):

- `jobs.py`: esecuzione in parallelo;
- `run_now.py` e `run_seeds.py`: rocking, 7 inclinazioni;
- `run_fill.py`: rocking, inclinazioni aggiuntive con gli stessi seed (salta i run già fatti);
- `run_W10k.py` e `run_W_more.py`: radiazione;
- `run_GeC.py <seed> [processi] [materiali]`: rocking di Ge e diamante, ad esempio
  `run_GeC.py 1,2,3,4 10` oppure `run_GeC.py 5,6,7,8 10 Diamond` (salta i run già fatti);
- `run_W_rock.py <seed> [processi]`: rocking di W ⟨111⟩, 10 µm.

I nostri file usati nelle simulazioni sono in `~/chtest/now/Si/T293K/` e `~/chtest/nowW/W/T293K/` (versione
precedente, con l'origine della cella nel minimo del potenziale: per Geant4 è solo una traslazione, irrilevante) e in
`~/chtest/nowGC/T293K/` (Ge e diamante, versione attuale) e `~/chtest/nowW2/T293K/` (W, versione attuale, usato
per le rocking curve del W). Il materiale `G4_Diamond` è definito in
`~/chtest/ch2/src/DetectorConstruction.cc`.

### Analisi (in questa cartella)

Tutti gli script si lanciano senza argomenti dalla radice del progetto, ad esempio
`python output/g4_comparison/seed_test.py`. Leggono i file ufficiali, i nostri e gli istogrammi direttamente da WSL
(`\\wsl.localhost\Ubuntu-24.04\home\tacab\...`), quindi WSL deve essere avviato.

| script | che cosa fa | uscite |
|---|---|---|
| `make_comparison_plots.py` | confronto dei dati, tutti i 19 file | figure 01–04, `agreement_table.csv` |
| `rocking_comparison.py` | rocking curve, 14 seed, tutte le inclinazioni disponibili | figura 05, `rocking_fraction.csv` |
| `seeds_analysis.py` | dispersione fra seed, χ² per punto | solo testo |
| `shape_test.py` | confronto della forma degli istogrammi | solo testo |
| `radiation_comparison.py` | radiazione W, seed 1–20 (opzionale: lista dei seed) | figura 06, `radiation_W111.csv` |
| `maps_W.py` | confronto delle mappe W ⟨111⟩ | solo testo |
| `seed_test.py` | test fra seed, rocking e radiazione | figura 07, `seed_test.csv` |
| `rocking_GeC.py` | rocking curve di Ge e diamante, tutti i seed completi | figure 08–09, `rocking_fraction_GeC.csv` |
| `rocking_W.py` | rocking curve di W ⟨111⟩, tutti i seed completi | figura 10, `rocking_W111.csv` |

`rocking_comparison.py`, `seeds_analysis.py`, `shape_test.py` e `seed_test.py` usano ogni inclinazione per cui
esistono tutti i 14 seed, i 3 assi e i due insiemi di file. Le inclinazioni aggiunte entrano da sole.
