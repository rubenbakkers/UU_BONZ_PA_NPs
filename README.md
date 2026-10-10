# UU BONZ — photon-avalanche nanoparticles

Python-scripts voor Tm³⁺:NaYF₄ photon-avalanche (PA), Stark-absorptiespectra, fononzijbanden en PASSI-beeldvorming. De 29 scripts zijn gegroepeerd op hun functie en onderlinge afhankelijkheden. Iedere scriptmap heeft een eigen README met uitleg over de berekening, instellingen, uitvoer en toepassingen.

## Wegwijzer

| Map | Wat staat erin? | Wanneer gebruiken? |
| --- | --- | --- |
| [pa_simulations](pa_simulations/README.md) | Gedeeld populatiemodel, figuurreproducties, resonantiescans en verschillende Stark-/golflengtemodellen | Emissie, drempel, niet-lineariteit en stijgtijd berekenen; modelvarianten vergelijken |
| [spectra](spectra/README.md) | Zelfstandige scripts voor energieniveaus, overgangsbanden en elektronische/discrete-fononabsorptiefactoren | Spectra onderzoeken zonder volledige populatieberekeningen |
| [fonon_model](fonon_model/README.md) | Zelfstandig PA-model met een continue fononband of discrete modi | Fononbijdragen aan zowel GSA als ESA en de gevolgen voor de kinetiek onderzoeken |
| [standalone_simulations](standalone_simulations/README.md) | Zelfstandige RK4-demonstraties en een fit aan ingebouwde experimentele drempels | Eenvoudige PA-dynamiek begrijpen of eerdere modelaannames vergelijken |
| [passi](passi/README.md) | Fenomenologische beeldvorming en resolutie, met historische Fig4-gegevens/toelichting | Het effect van een aangenomen niet-lineaire respons op beeldvorming verkennen |
| [spectra/reference_results](spectra/reference_results/README.md) | Bewaarde CSV-, SVG-, PNG- en JSON-exports uit de oorspronkelijke hoofdmap | Eerdere resultaten inspecteren en hun herkomst terugvinden |

GSA = ground-state absorption; ESA = excited-state absorption. Spectrale factoren beschrijven relatieve absorptie. De PA-simulaties voegen daar pompsnelheden, populaties en emissie aan toe.

## Installatie en eerste berekeningen

Gebruik Python 3.10 of nieuwer. Voer vanuit de hoofdmap uit:

```text
python -m pip install -r requirements.txt
```

Snelle elektronische absorptieberekening, zonder grafiekvenster:

```text
python spectra/gsa_esa_electronic_factors.py --data-only --output-dir spectra/results/electronic
```

Discrete fononzijbanden voor GSA, alleen gegevens:

```text
python spectra/gsa_esa_phonon_factors.py --data-only --output-dir spectra/results/discrete_phonons
```

Continue fononband voor GSA en ESA, alleen gegevens:

```text
python fonon_model/pa_fononband.py --data-only --output-dir fonon_model/results/phonon_band
```

Een snelle resonantiefiguur met het gedeelde PA-model:

```text
python pa_simulations/gsa_resonance_scan.py
```

De map-README's bevatten de overige opdrachten. De meeste figuur- en RK4-scripts hebben geen opdrachtregelopties: wijzig hun instellingen in het script. Grote concentratiescans, golflengtescans en stijgtijdberekeningen kunnen aanzienlijk langer duren dan spectrale exports.

## Samenhang en uitvoerlocaties

- Alle scripts die het gedeelde `pa_model` importeren, staan samen in `pa_simulations`. Houd deze bestanden bij elkaar; hun gewone lokale imports blijven zo werken bij direct uitvoeren.
- `pa_spectra.py` en `pa_curve_analysis.py` zijn afzonderlijke bibliotheken. Ze worden nog niet automatisch gebruikt door de bestaande simulatiescripts.
- `fonon_model` behoudt zijn eigen `pa_model.py`, zodat de map zelfstandig uitvoerbaar blijft. Bij de reorganisatie verschilden de twee modelmodules inhoudelijk alleen in een afsluitende lege regel.
- Zelfstandige simulaties bevatten hun eigen modelcode. Vergelijkbare bestandsnamen of figuren betekenen niet dat hun modelaannames gelijk zijn.
- Exporteurs met `--output-dir` schrijven naar die gekozen map; zonder deze optie meestal naast het script. Veel figuurscripts schrijven naar de huidige werkmap. Verschillende varianten gebruiken dezelfde uitvoerbestandsnamen: gebruik aparte uitvoermappen om resultaten te bewaren.
- De twee scripts met een vast persoonlijk Windows-pad schrijven nu naar `results` in hun eigen scriptmap: `spectra/stark_multiplet_distributions.py` en `pa_simulations/wavelength_detuning_stark_gsa_esa.py`.
- De bewaarde bestanden in `spectra/reference_results` zijn exports, geen vereiste invoer. De PASSI-code leest `Fig4_data.npz` niet. De historische Fig4-toelichting beschrijft een andere implementatie; zie de waarschuwing in de PASSI-map.

## Oude namen terugvinden

| Oorspronkelijke naam | Huidige locatie |
| --- | --- |
| `pa_model.py` | [pa_simulations/pa_model.py](pa_simulations/pa_model.py) |
| `pa_spectra.py` | [pa_simulations/pa_spectra.py](pa_simulations/pa_spectra.py) |
| `pa_curve_analysis.py` | [pa_simulations/pa_curve_analysis.py](pa_simulations/pa_curve_analysis.py) |
| `pa_script_1_fig2a.py` | [pa_simulations/figure2a_power_dependence.py](pa_simulations/figure2a_power_dependence.py) |
| `pa_script_2_fig2b_risetime.py` | [pa_simulations/figure2b_rise_time.py](pa_simulations/figure2b_rise_time.py) |
| `pa_script_3_fig3a.py` | [pa_simulations/figure3a_architecture_concentration.py](pa_simulations/figure3a_architecture_concentration.py) |
| `pa_script_4_fig3b.py` | [pa_simulations/figure3b_threshold_relaxation.py](pa_simulations/figure3b_threshold_relaxation.py) |
| `pa_script_5_wavelength_detuning.py` | [pa_simulations/wavelength_detuning_gaussian_cauchy.py](pa_simulations/wavelength_detuning_gaussian_cauchy.py) |
| `pa_script_5_wavelength_detuning_double.py` | [pa_simulations/wavelength_detuning_stark_gsa_esa.py](pa_simulations/wavelength_detuning_stark_gsa_esa.py) |
| `pa_script_5_wavelength_detuning_double2.py` | [pa_simulations/wavelength_detuning_stark_esa_background.py](pa_simulations/wavelength_detuning_stark_esa_background.py) |
| `pa_script_6_resonance_gsa.py` | [pa_simulations/gsa_resonance_scan.py](pa_simulations/gsa_resonance_scan.py) |
| `pa_script_7_resonance_esa.py` | [pa_simulations/esa_resonance_scan.py](pa_simulations/esa_resonance_scan.py) |
| `pa_script_8_emission_vs_wavelength.py` | [pa_simulations/emission_wavelength_scan.py](pa_simulations/emission_wavelength_scan.py) |
| `fig2a.py` | [pa_simulations/figure2a_sample_specific_grids.py](pa_simulations/figure2a_sample_specific_grids.py) |
| `gaussian detuning specifieke overgang.py` | [pa_simulations/stark_detuning_analysis.py](pa_simulations/stark_detuning_analysis.py) |
| `wavelength detuning backup.py` | [pa_simulations/stark_detuning_esa_background.py](pa_simulations/stark_detuning_esa_background.py) |
| `Energie niveas gaussian 3h4 3f2 3f3.py` | [spectra/stark_multiplet_distributions.py](spectra/stark_multiplet_distributions.py) |
| `Stark Energy per overgang.py` | [spectra/stark_transition_spectra.py](spectra/stark_transition_spectra.py) |
| `script f_factor.py` | [spectra/gsa_esa_electronic_factors.py](spectra/gsa_esa_electronic_factors.py) |
| `script f_factor fononen.py` | [spectra/gsa_esa_phonon_factors.py](spectra/gsa_esa_phonon_factors.py) |
| `untitled0.py` | [spectra/gsa_esa_phonon_factors_legacy.py](spectra/gsa_esa_phonon_factors_legacy.py) |
| `Simulatie PA RK4.py` | [standalone_simulations/rk4_minimal_three_level.py](standalone_simulations/rk4_minimal_three_level.py) |
| `photon_avalanche_rk4_corrected_v2.py` | [standalone_simulations/rk4_full_gaussian_demo_v2.py](standalone_simulations/rk4_full_gaussian_demo_v2.py) |
| `Simulatie PA 2.py` | [standalone_simulations/rk4_full_wavelength_demo.py](standalone_simulations/rk4_full_wavelength_demo.py) |
| `untitled1.py` | [standalone_simulations/rk4_full_wavelength_dense_fig3b.py](standalone_simulations/rk4_full_wavelength_dense_fig3b.py) |
| `fig3b.py` | [standalone_simulations/figure3b_experimental_linear_fits.py](standalone_simulations/figure3b_experimental_linear_fits.py) |
| `Fig4.py` | [passi/passi_phenomenological_simulation.py](passi/passi_phenomenological_simulation.py) |
| `fonon_model/pa_model.py` | Ongewijzigde locatie: [fonon_model/pa_model.py](fonon_model/pa_model.py) |
| `fonon_model/pa_fononband.py` | Ongewijzigde locatie: [fonon_model/pa_fononband.py](fonon_model/pa_fononband.py) |

`Fig4_data.npz` staat nu in `passi/`; `Fig4_toelichting.md` heet daar `historical_fig4_notes.md`. Alle oorspronkelijke `f_gsa_f_esa*`-exports staan in `spectra/reference_results/` met dezelfde bestandsnamen en inhoud
