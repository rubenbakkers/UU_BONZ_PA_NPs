# Bewaarde spectrale resultaten

Deze map bewaart de `f_gsa_f_esa*`-bestanden uit de oorspronkelijke hoofdmap van de repository. Hun inhoud is bij de herindeling niet gewijzigd. Ze zijn handig als referentie bij het vergelijken van modellen en instellingen. De scripts lezen deze bestanden niet in; ze kunnen hun spectra uit de ingebouwde niveaus opnieuw berekenen.

## Welke bestanden horen bij elkaar?

| Bestandsbasis | Aanwezige formaten | Model en herkomst | CSV-gegevensrijen |
|---|---|---|---|
| `f_gsa_f_esa_900_2000nm_zonder_achtergrond` | CSV, SVG | Zelfstandige elektronische factoren: H5-GSA σ = 300 cm⁻¹, overige σ = 8,5 cm⁻¹, geen achtergronden. Zie [gsa_esa_electronic_factors.py](../gsa_esa_electronic_factors.py). | 22.001 |
| `f_gsa_f_esa_fononzijbanden` | CSV, PNG, SVG, `_instellingen.json` | Discrete GSA-fononzijbanden bij 300 K; ESA elektronisch. Zie [gsa_esa_phonon_factors.py](../gsa_esa_phonon_factors.py) en [legacyvariant](../gsa_esa_phonon_factors_legacy.py). | 22.001 |
| `f_gsa_f_esa_900_1500nm_broad_gsa` | CSV | Elektronische Starkfactoren van het model met verbrede H5-GSA. Zie [stark_detuning_analysis.py](../../pa_simulations/stark_detuning_analysis.py). | 12.001 |
| `f_gsa_f_esa_1040_1100nm_broad_gsa` | CSV, PNG, SVG | Zoom rond het 1064-nm-anker van hetzelfde `broad_gsa`-model. | 1.201 |
| `f_gsa_f_esa_900_1500nm_hybrid` | CSV, SVG | Historische export van een eerdere hybridvariant met GSA-achtergrondbijdrage. | 12.001 |

De golflengtestap is in alle vijf CSV's 0,05 nm. Ze hebben de kopregel:

```text
wavelength_nm,f_gsa,f_esa
```

De factoren zijn relatieve absorptiedoorsneden, elk genormaliseerd op 1064 nm. Ze bevatten geen extra fotonfluxfactor. De SVG- en PNG-bestanden visualiseren die factoren; de JSON bewaart de fononinstellingen en numerieke diagnostiek.

## Herkomst van de discrete fononexport

`f_gsa_f_esa_fononzijbanden_instellingen.json` vermeldt drie modi van 250, 305 en 365 cm⁻¹, gelijke modefracties, T = 300 K en Huang–Rhys-factor 0,05 voor beide GSA-kanalen. De elektronische kernbreedte is 8,5 cm⁻¹. De drie ESA-kanalen hebben koppeling 0.

Deze export hoort daarom bij het zelfstandige **discrete GSA-fononmodel**, niet bij de standaard continue band van [fonon_model](../../fonon_model/README.md). Die fonondriver gebruikt standaard zijbanden in zowel GSA als ESA en schrijft `*_phonon_band.csv` met twee extra elektronische-componentkolommen. Die resultaten staan niet in deze bewaarde map.

## De hybridexport is historisch

De SVG-metadata vermeldt 2 oktober 2026. De figuurtekst noemt een `GSA-achtergrondcentrum: 1219.7 nm` en een Starkbreedte van 8,5 cm⁻¹. De huidige `stark_detuning_analysis.py` heeft die achtergrond verwijderd. Het huidige `hybrid`-model kan met de standaard smalle GSA-lijnen bovendien geen voldoende sterk 1064-nm-anker vormen en wijst die instelling af.

Gebruik deze bestanden dus als voorbeeld van de eerdere modelversie. Hun aanwezigheid bevestigt geen reproduceerbaarheid met de huidige standaardinstellingen. De precieze historische achtergrondgewichten zijn niet in een apart instellingenbestand vastgelegd.

## Nieuwe resultaten bewaren

Schrijf nieuwe runs bij voorkeur naar `spectra/results/` met een aparte submap per model of parameterkeuze, bijvoorbeeld:

```text
python spectra/gsa_esa_electronic_factors.py --no-show --output-dir spectra/results/electronic
python spectra/gsa_esa_phonon_factors_legacy.py --no-show --output-dir spectra/results/phonon_300K
python pa_simulations/stark_detuning_analysis.py --spectra-only --output-dir spectra/results/broad_gsa
```

De bewaarde bestanden geven voorbeelden van eerdere uitvoer. De herindeling op zichzelf valideert geen gemeten spectrum of volledige kinetische standaardrun.
