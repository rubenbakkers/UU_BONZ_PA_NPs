# PA-model met een fononband

Deze map koppelt een continue fononband aan de GSA/ESA-absorptie en photon-avalanche-kinetiek van Tm³⁺:NaYF₄. Je kunt alleen spectra exporteren of ook emissie, drempels en 95%-stijgtijden berekenen. Het is de uitgebreidere fononroute naast de [zelfstandige spectrumscripts](../spectra/README.md) en de [andere PA-simulaties](../pa_simulations/README.md).

## Bestanden en samenhang

| Bestand | Functie |
|---|---|
| [pa_fononband.py](pa_fononband.py) | Bouwt de elektronische en fononspectra, exporteert factoren en voert de kinetische analyses uit |
| [pa_model.py](pa_model.py) | Lokale modelmodule met SI-parameters, populatievergelijkingen, RK4 en emissie-/analysefuncties |
| [requirements.txt](requirements.txt) | NumPy, Matplotlib en SciPy voor de volledige berekening |

De lokale `pa_model.py` blijft naast `pa_fononband.py` staan, zodat de commandoregelberekening in deze map zelfstandig werkt. Deze kopie gebruikt hetzelfde model als `pa_simulations/pa_model.py`; bij de herindeling verschilden ze uitsluitend in een afsluitende lege regel. De zelfstandige fononmap hoeft de andere scriptmap daardoor niet aan zijn importpad toe te voegen.

Het script voor de discrete GSA-fononfactoren staat nu in `spectra/gsa_esa_phonon_factors.py`. De reeds bewaarde `f_gsa_f_esa_fononzijbanden.*`-bestanden zijn naar [spectra/reference_results](../spectra/reference_results/README.md) verplaatst. Ze horen bij dat zelfstandige discrete model, niet bij de standaard continue fononband in deze map.

## Uitvoeren

Voer deze voorbeelden uit vanuit de hoofdmap van de repository met Python 3.10 of nieuwer. Installeer de pakketten:

```text
python -m pip install -r fonon_model/requirements.txt
```

Alleen factor-CSV's en fononinstellingen, zonder grafieken of kinetiek:

```text
python fonon_model/pa_fononband.py --data-only --output-dir fonon_model/resultaten
```

Absorptiegrafieken, CSV's en instellingen:

```text
python fonon_model/pa_fononband.py --spectra-only --no-show --output-dir fonon_model/resultaten
```

Volledige berekening, inclusief golflengtescan, stijgtijden en breedtesweeps:

```text
python fonon_model/pa_fononband.py --no-show --output-dir fonon_model/resultaten
```

Je kunt kiezen welke onderdelen een volledige run maakt, met `--plots` (alleen deze) of `--skip-plots` (alles behalve deze), of met de dictionary `PLOTS` bovenin het script:

| Naam | Onderdeel | Rekentijd (indicatie) |
|---|---|---|
| `spectra` | Absorptiefactoren f_gsa/f_esa (900–1500 nm en zoom) | ~1 s |
| `detuning` | Emissie tegen intensiteit per pompgolflengte | ~8 s, samen met `smax` en `ith` |
| `smax` | Helling van het avalanchedeel tegen golflengte | gebruikt de detuning-berekening |
| `ith` | Drempel tegen golflengte | gebruikt de detuning-berekening |
| `rise` | 95%-stijgtijd tegen intensiteit | ~2 s |
| `sigma` | Piek-stijgtijd tegen elektronische breedte | ~2 min, verreweg het traagst |

```text
python fonon_model/pa_fononband.py --no-show --skip-plots sigma --output-dir fonon_model/resultaten
python fonon_model/pa_fononband.py --no-show --plots ith rise --output-dir fonon_model/resultaten
```

`smax` en `ith` hebben de detuning-berekening nodig; die draait dan ook als `detuning` uit staat, alleen zonder de detuning-figuur. Staan alle kinetische onderdelen uit, dan worden alleen de spectra berekend. Per onderdeel wordt de rekentijd afgedrukt.

Zonder `--output-dir` komen resultaten naast het script. `--no-show` slaat grafieken op en onderdrukt vensters. Voor het standaard fononmodel heeft `--data-only` alleen NumPy nodig en kan de export zonder `pa_model.py` of Matplotlib werken. Grafieken vereisen Matplotlib. De volledige populatieanalyse vereist de lokale modelmodule en SciPy; de snelle stationaire solver gebruikt LSODA en de stijgtijdzoeker gebruikt `brentq`.

Een volledige run bevat veel integraties. De `sigma`-sweep bouwt spectra opnieuw op voor tien elektronische breedtes en twee pompgolflengtes en kost het grootste deel van de tijd; sla die over met `--skip-plots sigma`. Begin met `--data-only` of `--spectra-only` wanneer je alleen absorptie wilt vergelijken.

## Hoe het model werkt

De standaard is `ABSORPTION_MODEL = "phonon_band"`. GSA bevat ³H₆ → ³F₄/³H₅; ESA bevat ³F₄ → ³H₄/³F₃/³F₂. **Alle vijf kanalen** gebruiken Boltzmanngewogen Starklijnen, een elektronische Gaussbreedte van 8,5 cm⁻¹ en eindige-temperatuur Huang–Rhys-zijbanden. De fononbijdrage vervangt de effectieve H5-GSA-breedte van 300 cm⁻¹; die breedte wordt niet nogmaals op het fononspectrum toegepast.

De band wordt benaderd met gewogen integratieknopen. Standaard (`--band-profile four_gaussian`) is het profiel de som van vier Gaussians die de gestippelde Yb/Tm-Ramancurve van Wilhelm et al. (2013, fig. 3A, [doi:10.7150/thno.5113](https://doi.org/10.7150/thno.5113)) benaderen, dezelfde parameters als in [FOUR_GAUSSIAN_PHONONS.md](FOUR_GAUSSIAN_PHONONS.md):

| Component | Centrum (cm⁻¹) | σ (cm⁻¹) | Hoogte |
|---|---:|---:|---:|
| 1 | 253 | 14 | 0.68 |
| 2 | 306 | 24 | 0.95 |
| 3 | 364 | 15 | 0.66 |
| 4 | 408 | 24 | 0.36 |

Het profiel wordt bemonsterd op knopen van 5 cm⁻¹ tussen 200 en 480 cm⁻¹ (57 knopen, gewogen gemiddelde ≈ 325 cm⁻¹). Elke knoop is een scherpe mode; de breedte van de zijband komt uit de spreiding over de knopen plus de elektronische breedte. De oude enkele Gauss-band (225–450 cm⁻¹, gewogen gemiddelde 320 cm⁻¹, breedte 45 cm⁻¹) blijft beschikbaar met `--band-profile gaussian --band-min 225 --band-max 450 --band-step 15`. De Huang–Rhys-factor is standaard 0,05 per kanaal. Met `--phonon-model discrete` worden de drie modi van 250, 305 en 365 cm⁻¹ gebruikt. Ook bij die keuze houden de ESA-kanalen hier standaard fononkoppeling; dit verschilt van het zelfstandige GSA-fononscript.

De relatieve absorptiefactoren zijn:

```text
f_i(lambda) = A_i(lambda) / A_i(1064 nm)
sigma_i(lambda) = sigma_i(1064 nm) * f_i(lambda)
w_i(lambda) = sigma_i(lambda) * fotonflux_per_intensiteit(lambda)
```

`f_gsa` en `f_esa` bevatten al de volledige Stark- en fononbijdrage. Beide zijn 1 bij 1064 nm en worden eenmaal in de pompsnelheden toegepast. De populatievergelijkingen gebruiken de totale grond- en aangeslagen populaties: `R_gsa*N0` en `R_esa*N1`. Er hoort geen tweede Boltzmann-, Stark- of fononfactor bij. De drie populaties zijn ³H₆, ³F₄ en ³H₄; de hogere aangeslagen multiplets worden via de bestaande drie-populatiebenadering verwerkt.

De componenten zonder fononevents worden met dezelfde normalisatie als het volledige spectrum bijgehouden. Dat is de bijdrage zonder enig fononevent, niet automatisch alle processen met netto energieverschuiving nul. Er wordt geen absorptievloer toegevoegd.

### Kanaalgewichten (Judd–Ofelt)

De relatieve sterkte van de vijf kanalen volgt uit Judd–Ofelt in plaats van een vast gewicht 1:

```text
S_c = Omega_2 [U(2)]^2 + Omega_4 [U(4)]^2 + Omega_6 [U(6)]^2
w_c = S_c * nu_c / sum(S_c' * nu_c')   (som over de kanalen van hetzelfde spectrum)
```

met Ω₂, Ω₄, Ω₆ = 2,37; 3,05; 0,41 × 10⁻²⁰ cm² (SI tabel S4 van Lee et al. 2021) en ν_c de afstand tussen de multiplet-zwaartepunten. De matrixelementen komen uit SI tabel S5; die voor ³F₄ → ³F₃ en ³F₄ → ³F₂ ontbreken daar en komen uit dezelfde bron, Carnall, Crosswhite & Crosswhite (1978, ANL-78-XX-95), appendix XIII tabel 2.

| Kanaal | S (10⁻²⁰ cm²) | Aandeel |
|---|---:|---:|
| ³H₆ → ³F₄ | 3,593 | 0,666 (GSA) |
| ³H₆ → ³H₅ | 1,222 | 0,334 (GSA) |
| ³F₄ → ³H₄ | 0,789 | 0,376 (ESA) |
| ³F₄ → ³F₃ | 0,077 | 0,045 (ESA) |
| ³F₄ → ³F₂ | 0,907 | 0,579 (ESA) |

Door het 1064-nm-anker tellen alleen de verhoudingen binnen één spectrum. `USE_JUDD_OFELT_WEIGHTS = False` bovenin het script herstelt de oude gewichten 1,0. De matrixelementen gelden voor Tm³⁺:LaF₃ (intermediate coupling) en zijn een benadering voor NaYF₄.

## Instellingen en eigen bandbestanden

Temperatuur en elektronische breedte staan bovenin bij `TEMPERATURE_K` en `STARK_SIGMA_CM1`. De temperatuur heeft in dit script geen commandoregeloptie. Voor de fononroute moet `LORENTZ_HWHM_CM1 = 0` blijven; een niet-nul Lorentzbreedte wordt afgewezen.

| Optie | Betekenis |
|---|---|
| `--phonon-model band` of `discrete` | Continue bandbenadering of drie discrete modi |
| `--band-min`, `--band-max`, `--band-step` | Bandgrenzen en knoopafstand in cm⁻¹ |
| `--band-profile four_gaussian`, `gaussian`, `uniform` of `file` | Vorm van de band; standaard `four_gaussian` |
| `--band-mean`, `--band-sigma` | Alleen voor `gaussian`: gewogen gemiddelde en aangenomen profielbreedte in cm⁻¹ |
| `--band-file` | Eigen CSV met bandprofiel; selecteert de bestandsroute |
| `--coupling-h5`, `--coupling-f4` | Totale koppelingen voor de twee GSA-kanalen |
| `--coupling-h4`, `--coupling-f3`, `--coupling-f2` | Totale koppelingen voor de drie ESA-kanalen |
| `--kernel-tolerance` | Grens aan weggelaten kernelmassa; standaard 10⁻³⁵ |

Voorbeelden:

```text
python fonon_model/pa_fononband.py --spectra-only --no-show --coupling-h5 0.08 --coupling-h4 0.10 --output-dir fonon_model/resultaten/koppeling
python fonon_model/pa_fononband.py --data-only --phonon-model discrete --output-dir fonon_model/resultaten/discrete
python fonon_model/pa_fononband.py --spectra-only --no-show --band-file eigen_band.csv --output-dir fonon_model/resultaten/eigen_band
```

Een eigen bestand bevat minstens twee gegevensrijen en twee kommagescheiden kolommen: golfgetal in cm⁻¹ en intensiteit. Een kopregel is toegestaan. Bijvoorbeeld:

```csv
wavenumber_cm1,intensity
225,0.1
300,1.0
375,0.7
450,0.1
```

Dit voorbeeldbestand wordt niet meegeleverd; sla het zelf op als `eigen_band.csv` voor de derde opdracht. Intensiteiten worden op de bandknopen geïnterpoleerd en de gewichten worden genormaliseerd. Knoopposities moeten op het verschuivingsrooster van 5 cm⁻¹ passen en de bandbreedte moet een geheel aantal knoopstappen bevatten. De Gaussinstelling stuurt het gewogen gemiddelde van de afgeknotte band, niet alleen het ongewogen profielcentrum.

De oudere absorptiemodi `broad_gsa`, `stark`, `si_1064` en `hybrid` blijven beschikbaar via de constante `ABSORPTION_MODEL` in het script. Daar is geen commandoregeloptie voor. Alleen `broad_gsa` gebruikt nog de effectieve H5-breedte van 300 cm⁻¹. Met de standaard smalle GSA-lijnen kan het huidige `hybrid`-model zijn 1064-nm-anker niet betrouwbaar vormen; die combinatie wordt afgewezen.

## Uitvoer

Bij het standaardmodel worden geschreven:

| Bestandsbasis | Inhoud |
|---|---|
| `f_gsa_f_esa_900_1500nm_phonon_band` | Factoren over 900–1500 nm, CSV en bij grafieken ook PNG/SVG |
| `f_gsa_f_esa_1040_1100nm_phonon_band` | Zoom rond 1064 nm, dezelfde formaten |
| `fononband.csv` | Bandknopen, modefracties en koppeling per knoop en kanaal |
| `fonon_instellingen.json` | Modelinstellingen, kerneldiagnostiek en nauwkeurigheidsgrenzen |

De factor-CSV's hebben vijf kolommen:

```text
wavelength_nm,f_gsa,f_esa,f_gsa_zonder_fononen,f_esa_zonder_fononen
```

De laatste twee kolommen geven de bijdrage zonder fononevents met dezelfde volledige-spectrum-normalisatie. Een volledige kinetische run maakt daarnaast:

- `pa_wavelength_detuning_phonon_band.pdf`: emissie tegen pompintensiteit per golflengte;
- `pa_smax_vs_wavelength_phonon_band.pdf`: helling van het gefitte avalanchedeel;
- `pa_ith_vs_wavelength_phonon_band.pdf`: numerieke knie en analytische drempel;
- `pa_rise_time_vs_intensity_phonon_band.pdf`: 95%-stijgtijd tegen intensiteit, wanneer geldige stijgtijdgegevens gevonden worden;
- `pa_rise_peak_vs_sigma_phonon_band.pdf`: maximale stijgtijd tegen elektronische breedte.

Gebruik aparte uitvoermappen voor parametervergelijkingen: nieuwe runs gebruiken dezelfde bestandsnamen. Het 1064-nm-anker houdt bij de standaard breedtesweep de pompsnelheden op die golflengte gelijk.

## Aannamen en controles

Bandvorm, koppelingen, elektronische breedtes en relatieve Stark-lijnsterktes zijn ongekalibreerde modelaannamen. Een anker op 1064 nm is geen onafhankelijke kalibratie van alle andere golflengten. De lange-golflengteanalyse is verkennend. De bandbenadering beschrijft geen volledige gemeten fonon-DOS.

De code bevat controles op de referentiefactoren, standaard stickgewichten, behoud van populatietotalen, overeenkomst tussen de snelle en de gedeelde populatievergelijkingen en de begrensde weggelaten kernelmassa. De stationaire solver controleert zowel tijdstabiliteit als de restterm van de populatievergelijkingen. Die ingebouwde controles vervangen geen vergelijking met meetgegevens. De volledige standaardscan is voor deze herindeling niet opnieuw uitgevoerd; dit README claimt daarom geen nieuwe validatie daarvan.
