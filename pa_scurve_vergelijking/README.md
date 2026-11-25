# Gemeten S-curves vs. het fononbandmodel

Vergelijkt de S-curves van 24-11-2026 (BdJ1; Cobolt op 1060 nm en SuperK op 1064 nm) met het populatiemodel in [`fonon_model`](../fonon_model/README.md). Het model zelf staat niet in deze map: de scripts importeren `pa_model.py` en `pa_fononband.py` uit `../fonon_model`.

## Starten

Gebruik Python 3.10 of nieuwer met NumPy, SciPy en Matplotlib (`python -m pip install -r requirements.txt` in de hoofdmap). Vanuit deze map:

```text
python run_all.py        # hele vergelijking, ~4 min (SNEL = True: ~2 min, zonder parameterscan)
python fit_s_lee.py      # alleen de helling s van de metingen, ~5 s, zonder model
```

Figuren en JSON-resultaten komen in `results/` (niet in git). Losse scripts kun je apart draaien, maar draai eerst `make_figures.py`: dat schrijft `results/summary.json`, die de andere scripts lezen. Draai je in Spyder of een Jupyter-console opnieuw na een wijziging in het model of in `pa_fit_lib.py`, herstart dan eerst de kernel.

## Welke modelversie?

`pa_fit_lib.py` zoekt `pa_model.py` + `pa_fononband.py` in deze volgorde en meldt bij het starten welke map en welke Huang–Rhys-factor S(³H₆→³H₅) het gebruikt (ook in `results/summary.json` onder `model_info`):

1. de omgevingsvariabele `PA_MODEL_DIR`;
2. `../Script` (de lokale BONZ-map);
3. `../fonon_model` (deze repo);
4. `model/` (oude losse kopie, alleen als niets anders bestaat).

De scripts gebruiken interne functies (`_stationary_population`, `_make_rhs`, `_lineshape`, `_stark_sigma`, `rise_time_95_fast`); een sterk afwijkende modelversie kan daardoor breken.

## Wat er gebeurt

Gedeelde basis, [`pa_fit_lib.py`](pa_fit_lib.py): GSA/ESA-snelheden uit de fononbandspectra, de stationaire toestand vanuit de grondtoestand (zoals een oplopende sweep) en de 800-nm-emissie. Omdat in de focus veel deeltjes zitten die elk een andere intensiteit zien, wordt de emissie gesommeerd over het gemeten spotprofiel (elliptische Gauss-kern + ronde halo).

Fitmodel ([`fit_scurves.py`](fit_scurves.py)): `counts = A · M(k · I_gerapporteerd) + b_FW`. `k` schaalt de intensiteit en vangt zowel een kalibratiefout als verschillen tussen deze deeltjes en die van Lee et al. op; `A` en één achtergrond per filterwielstand volgen per `k` uit een gewogen kleinste-kwadratenfit. Een goede χ²ᵣ betekent dus dat de vorm van de S-curve klopt, niet de absolute drempel.

| Stap ([`run_all.py`](run_all.py)) | Wat | Uitvoer |
|---|---|---|
| [`make_figures.py`](make_figures.py) | Fit van beide S-curves (Lee SI sample 5: 8 % Tm, geen niet-radiatief verlies); I_th(λ)/I_th(1060 nm) | `fig1_scurves_vs_model.png`, `fig2_threshold_vs_wavelength.png`, `summary.json` |
| [`scan_shape.py`](scan_shape.py) | Scan van σ_GSA (×0,1–10), Tm-concentratie (4–20 %) en niet-radiatief verlies; past de knik beter? | `scan_shape.json` |
| [`rise_variants.py`](rise_variants.py) | 95 %-stijgtijd voor drie verklaringen van `k`: kalibratie, σ_ESA × k, w₂ / k | `rise_variants.json` |
| [`protocol_sim.py`](protocol_sim.py) | Meetprotocol nagebootst (2 s per punt, geen wachttijd): hoeveel valt het gemiddelde bij de knik te laag uit? | `protocol_sim.json` |
| [`variants_fit.py`](variants_fit.py) | Refit met σ_ESA × k en w₂ / k, en het 2-s-protocol opnieuw | `variants_fit.json` |
| [`emission_spec.py`](emission_spec.py) | Emissiespectrum (785–862 nm) vs. de ³H₄→³H₆-Stark-lijnen; fit van verschuiving, breedte, amplitude en achtergrond | `emission_spec.json` |
| [`make_figures2.py`](make_figures2.py) | Fig. 3: emissiespectrum (met de fit van `emission_spec.py`), stijgtijd, meetprotocol, de twee Cobolt-sweeps | `fig3_overige_scriptdelen.png` |

### Helling s zoals Lee et al.

[`fit_s_lee.py`](fit_s_lee.py) staat los van het model en van `run_all.py`. Lee et al. (2021) definiëren s via I_em ∝ I_pomp^s (helling in log-log); de drempel is het snijpunt van rechte lijnen door de punten onder en boven de drempel (SI, Fig. S3). Het script gebruikt alleen punten boven 3× de ruis, verdeelt die in 2 of 3 rechte stukken in log-log (knikpunten bij minimale χ², BIC kiest 2 of 3) en neemt s = helling van het steilste stuk. De onzekerheid komt uit 1000 bootstraps. Uitvoer: `fig_s_fit_lee.pdf/.png` en `s_fit_lee.json`.

Onder de knik ligt het signaal van deze metingen in de ruis; I_th volgens Lee is dan niet te bepalen en het script meldt dat. Omdat de curve geleidelijk buigt, hangt s van het fitvenster af en is het eerder een ondergrens.

## Data (`data/`, niet in de repo)

De meetdata staan **niet** in deze openbare repo (`data/` staat in `.gitignore`). Zet deze drie bestanden zelf in `pa_scurve_vergelijking/data/`; zonder die bestanden stoppen de scripts met een melding welk bestand ontbreekt.

| Bestand | Inhoud |
|---|---|
| `cobolt_1060_scurve.csv`, `superk_1064_scurve.csv` | Kolommen uit `scurve.csv` op de Drive (5 Metingen S-curve), gesorteerd op intensiteit: `fw6, gray, I_kWcm2, net_cps, noise_cps, net_cps_gauss` |
| `emission_check_spectrum_superk_785_862nm.csv` | Deel van `emission_check_spectrum.csv` (BdJ1, SuperK) rond de emissieband: `wavelength_nm, cts_per_s_per_pixel` |

## Aandachtspunten

- De SuperK wordt als monochromatisch behandeld (`fwhm=0.0` in `fit_scurves.py`); `rates()` kan over een laserspectrum middelen.
- De achtergrond per filterwielstand in `summary.json` komt uit de modelfit; `fit_s_lee.py` trekt die standaard af (`BACKGROUND = "summary"`).
- Het gemeten emissiemaximum ligt ~20 nm (~300 cm⁻¹) roder dan het model met de Stark-niveaus van "Table 4". Of dat aan de spectrometerkalibratie of aan de niveaus ligt, is nog open.
