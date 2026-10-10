# PASSI: niet-lineaire beeldvorming

[`passi_phenomenological_simulation.py`](passi_phenomenological_simulation.py)
onderzoekt hoe een steile photon-avalanche-respons de breedte van een gescand
beeld kan verkleinen. Het maakt figuur-4-achtige beelden, een breedtescan en
profielen van twee deeltjes. De berekening gebruikt een aangenomen
emissiecurve; het is geen zelfstandige exacte reproductie van de publicatie.

## Gebruik

Vereist: Python, NumPy, SciPy en Matplotlib. Vanuit de hoofdmap:

```text
python -m pip install -r requirements.txt
python passi/passi_phenomenological_simulation.py
```

Het script heeft geen opdrachtregelopties. De invoer staat in het script:
bundelbreedte, deeltjesafmetingen, pompintensiteiten, emissiecurve en de
meetpunten voor de vergelijking. Het gebruikt geen `pa_model.py` en leest
geen externe bestanden. De berekening loopt ook wanneer je het script
importeert.

## Hoe de berekening werkt

1. Een Gaussische pomp met FWHM 357 nm wordt over een beeldveld van
   500 × 500 nm gescand, met een rasterstap van 1 nm.
2. Een deeltje wordt voorgesteld als een ellips van ongeveer 19,5 × 16 nm.
3. De lokale pompintensiteit wordt omgezet in emissie met een
   fenomenologische S-curve: drempelparameter 6 kW/cm², exponent 26, plus een
   zwakke bijdrage onder de lawine. Die parameters worden in dit script
   niet aan meetdata gefit of uit rate equations berekend.
4. FFT-convolutie sommeert de emissiebijdragen van de deeltjespixels.
   Elk beeld wordt afzonderlijk op zijn maximum genormaliseerd.
5. De FWHM volgt uit de pixels boven de halve piekhoogte van het centrale
   horizontale profiel.

De simulatie is nuttig om de invloed van niet-lineariteit, pompintensiteit,
deeltjesafmetingen en onderlinge afstand op ruimtelijke resolutie te verkennen.

## Resultaten en opslaan

Het script maakt twee Matplotlib-figuren:

- Een enkel deeltje bij 9,9 en 7,1 kW/cm², plus een FWHM-scan over 60
  intensiteiten tussen 6,2 en 11 kW/cm². De zwarte punten en foutbalken zijn
  ingebouwde, als gedigitaliseerde experimentele waarden aangeduide gegevens.
- Twee deeltjes op 300 nm afstand bij 13,8, 10,9, 9,9, 8,9 en 7,6 kW/cm²,
  met beelden, lijnprofielen en een lineaire diffractiereferentie.

De console geeft de breedtes en een eenvoudige herkenbaarheidscontrole: een
dip tussen de twee pieken telt als zichtbaar wanneer de centrale waarde
kleiner is dan 95% van de hoogste piek.

**Het huidige script schrijft geen afbeeldingen naar schijf.** De twee
`savefig`-regels zijn uitgecommentarieerd. De melding dat
`fig4def_passi_simulation.png` is opgeslagen klopt daarom niet. Ook ontbreekt
een expliciete `plt.show()`; een interactieve Python-omgeving kan de gemaakte
figuren wel tonen. Om zonder bronwijziging bestanden op te slaan, voer vanuit
de hoofdmap bijvoorbeeld deze opdracht uit:

```text
python -c "import runpy; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from pathlib import Path; runpy.run_path('passi/passi_phenomenological_simulation.py'); out=Path('passi/results'); out.mkdir(parents=True, exist_ok=True); names=['fig4def_passi_simulation.png', 'fig4h_passi_simulation.png']; [plt.figure(n).savefig(out/name, dpi=200, bbox_inches='tight') for n,name in zip(plt.get_fignums(),names)]"
```

Deze opdracht bewaart de twee figuren in `passi/results/`.

## Grenzen van de interpretatie

De afzonderlijke normalisatie laat beeldvormen zien; daarmee kun je geen
absolute helderheid tussen pompintensiteiten vergelijken. De FWHM is bepaald
op het pixelrooster, zonder interpolatie of Gauss-fit. De aangenomen
emissiecurve en de eenvoudige dipcontrole zijn geen onafhankelijke
experimentele validatie. De bundelparameters en het eindige beeldveld bepalen
mede de uitkomst.

## Referentiedata en historische toelichting

[`Fig4_data.npz`](Fig4_data.npz) is behouden als apart referentiebestand.
Het huidige simulatiescript opent dit bestand niet. De herkomst wordt alleen
in de historische toelichting beschreven: daarin worden de brondata van
figuur 3 en 4 van Lee et al. genoemd. Het NPZ-bestand levert geen invoer of
validatie voor de hier beschreven fenomenologische simulatie.

[`historical_fig4_notes.md`](historical_fig4_notes.md), oorspronkelijk
`Fig4_toelichting.md`, beschrijft een andere DRE-gebaseerde implementatie die
niet in het huidige script staat. De daarin genoemde opties `--mode`,
`--beam`, `--show` en `--output-dir`, CSV-uitvoer en verificatieresultaten
horen dus niet bij `passi_phenomenological_simulation.py`. Gebruik deze README
voor de werking van de aanwezige code; beschouw die oudere toelichting als
historische context.

Oorspronkelijke scriptnaam: `Fig4.py`.
