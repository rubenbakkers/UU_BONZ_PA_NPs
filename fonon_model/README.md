# PA-model met fononband

Deze map bevat een uitvoerbare versie van het GSA/ESA-kinetiekscript voor Tm3+:NaYF4, met een continue fononband. `pa_fononband.py` gebruikt de meegeleverde `pa_model.py`. De bestanden in de hoofdmap van de repository worden niet aangepast.

## Uitvoeren

Installeer de benodigde pakketten vanuit de hoofdmap van de repository:

```text
python -m pip install -r fonon_model/requirements.txt
```

Volledige berekening, inclusief golflengtescan, rise-time en sigma-sweep:

```text
python fonon_model/pa_fononband.py
```

Alleen absorptiegrafieken en CSV-gegevens:

```text
python fonon_model/pa_fononband.py --spectra-only --no-show --output-dir fonon_model/resultaten
```

Alleen CSV-gegevens en fononinstellingen:

```text
python fonon_model/pa_fononband.py --data-only --output-dir fonon_model/resultaten
```

Zonder `--output-dir` worden resultaten naast het script opgeslagen. De meegeleverde modelmodule moet naast `pa_fononband.py` blijven staan. Grafieken vereisen matplotlib en populatiesimulaties vereisen scipy.

## Absorptiefactoren

De standaard is `ABSORPTION_MODEL = "phonon_band"`. Alle vijf kanalen gebruiken Boltzmanngewogen Stark-lijnen, een elektronische Gauss-breedte van 8.5 cm^-1 en een eindige-T Huang-Rhys-kernel. De fononzijbanden vervangen de effectieve H5-GSA-breedte van 300 cm^-1.

De elektronische en fononbijdragen worden samen berekend in een spectrum A_i. De relatieve absorptiefactor is:

```text
f_i(lambda) = A_i(lambda) / A_i(1064 nm)
sigma_i(lambda) = sigma_i(1064 nm) * f_i(lambda)
```

`f_gsa` en `f_esa` bevatten dus al de volledige Stark- en fononbijdrage. Vermenigvuldig ze niet opnieuw met een apart Stark-spectrum of een extra fononfactor. Beide zijn 1 bij 1064 nm. Ze worden eenmaal toegepast in de pompsnelheden; de populatievergelijkingen gebruiken daarna de totale grond- en aangeslagen populaties. Er wordt geen absorptievloer toegevoegd.

De bandvorm, koppelingen en elektronische breedte zijn modelaannames, geen nieuwe experimentele kalibratie.

## Instellingen

De temperatuur en elektronische breedte staan bij `TEMPERATURE_K` en `STARK_SIGMA_CM1`. De Huang-Rhys-factoren zijn standaard 0.05 per kanaal. Het standaardprofiel is een afgeknotte Gauss-band tussen 225 en 450 cm^-1, met gewogen gemiddelde 320 cm^-1.

Voorbeelden:

```text
python fonon_model/pa_fononband.py --spectra-only --coupling-h5 0.08 --coupling-h4 0.10
python fonon_model/pa_fononband.py --spectra-only --phonon-model discrete
python fonon_model/pa_fononband.py --spectra-only --band-file eigen_band.csv
```

Een eigen bandbestand bevat twee CSV-kolommen: golfgetal in cm^-1 en intensiteit. Een kopregel is toegestaan. De fononroute gebruikt een elektronische Gauss-kern en vereist `LORENTZ_HWHM_CM1 = 0`.

De sigma-sweep bouwt de spectra opnieuw op voor tien breedtes en twee pompgolflengtes. Deze stap kan meerdere minuten duren. Met het 1064-nm-anker blijven de pompsnelheden bij 1064 nm gelijk tijdens die sweep.

## Validatie

De absorptiefactoren en nul-fononcomponenten zijn vergeleken met het oorspronkelijke fononscript op tien golflengten tussen 900 en 2000 nm. Beide referentieankers, kernelnormalisatie, detailed balance, populatienormalisatie en de eenmalige toepassing van de factoren in de pompsnelheden zijn gecontroleerd.

De volledige standaardberekening is succesvol uitgevoerd met `--no-show`, inclusief de golflengtescan, rise-time-berekeningen, beide sigma-sweeps en grafiekexport. De diagnostische eigenschap `PhononSpectrum.background_weight` is beschikbaar met waarde `0.0`.
