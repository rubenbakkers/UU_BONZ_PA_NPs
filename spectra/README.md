# Starkniveaus en GSA/ESA-spectra

Nieuw: [gsa_esa_fononband.py](gsa_esa_fononband.py) gebruikt een continue fononband voor GSA én ESA, met standaard het bij benadering overgenomen Yb/Tm Raman-profiel en 46 knopen. Zie [RAMAN_PROFILE.md](RAMAN_PROFILE.md) voor bron, aannamen, profielplot en gebruik.

Deze map bevat zes zelfstandige scripts voor de energieniveaus en relatieve absorptie van Tm³⁺ in NaYF₄. Ze gebruiken ingebouwde Starkenergieniveaus en berekenen geen photon-avalanche-populatiedynamiek. Voor de koppeling aan emissie, drempels en stijgtijden zie [pa_simulations](../pa_simulations/README.md) en het [model met een fononband](../fonon_model/README.md).

## Welk script gebruik je?

| Script | Wat berekent het? | Nuttig voor |
|---|---|---|
| [gsa_esa_fononband.py](gsa_esa_fononband.py) | Continue fononband voor GSA én ESA, standaard met overgenomen Yb/Tm Raman-profiel | Multipeak-profiel, modegewichten en absorptiefactoren bekijken |
| [stark_multiplet_distributions.py](stark_multiplet_distributions.py) | Gaussiaanse verdelingen rond de Starkniveaus van zes multiplets | Niveauposities en de invloed van Boltzmannweging bekijken |
| [stark_transition_spectra.py](stark_transition_spectra.py) | Verschillen tussen begin- en eindniveaus, verbrede overgangsbanden en energetische kandidaten rond een pompgolflengte | Van losse niveaus naar mogelijke absorptieovergangen gaan |
| [gsa_esa_electronic_factors.py](gsa_esa_electronic_factors.py) | GSA/ESA-factoren met elektronische lijnverbreding, zonder achtergrond | De relatieve absorptie over 900–2000 nm bekijken zonder kinetieksimulatie |
| [gsa_esa_phonon_factors.py](gsa_esa_phonon_factors.py) | Dezelfde factoren met drie discrete fononmodi voor GSA | De invloed van temperatuur en fononkoppeling op GSA vergelijken |
| [gsa_esa_phonon_factors_legacy.py](gsa_esa_phonon_factors_legacy.py) | Dezelfde discrete fononberekening, met de oorspronkelijke afhandeling van het grafiekvenster | De berekening uitvoeren met een werkende `--no-show`-optie |

GSA betekent *ground-state absorption*; ESA betekent *excited-state absorption*. De GSA-kanalen zijn ³H₆ → ³F₄ en ³H₆ → ³H₅. De ESA-kanalen zijn ³F₄ → ³H₄, ³F₃ en ³F₂.

## Installeren en uitvoeren

Voer de voorbeelden uit vanuit de hoofdmap van de repository. Gebruik Python 3.10 of nieuwer en installeer de pakketten:

```text
python -m pip install numpy matplotlib
```

De vier factorscripts kunnen met alleen NumPy CSV-gegevens berekenen via `--data-only`. De scripts voor niveauverdelingen en overgangsspectra gebruiken Matplotlib voor hun figuren. Geen van deze zes scripts importeert de kinetische module `pa_model.py`.

### 1. Verdelingen van de Starkniveaus

```text
python spectra/stark_multiplet_distributions.py
```

Het script sommeert Gaussians rond de niveaus van ³H₆, ³F₄, ³H₅, ³H₄, ³F₃ en ³F₂. Standaard zijn `SIGMA_CM1 = 8.5`, `TEMPERATURE_K = 300`, `BOLTZMANN_MODE = "multiplet"` en `NORMALIZE_EACH = True`. Deze instellingen staan bovenin het script; er zijn geen commandoregelopties. Met `"off"` krijgen alle niveaus hetzelfde gewicht. Met `"global"` worden de energieën ten opzichte van het laagste ³H₆-niveau gewogen.

Uitvoer: `spectra/results/Gaussian_Stark_functies.pdf` en een grafiekvenster. De uitvoermap wordt automatisch aangemaakt. De vroegere persoonlijke Windows-map is vervangen door deze repositorymap.

De curves tonen **energieposities van multiplets**, geen absorptie- of emissiespectra. De extra golflengte-as geeft de equivalente golflengte ten opzichte van E = 0. Voor een optische overgang moet je juist het verschil tussen het eind- en beginniveau gebruiken. Iedere curve krijgt standaard een eigen maximum van 1; de hoogten vergelijken dus geen absolute multipletpopulaties of lijnsterktes.

### 2. Spectra van afzonderlijke overgangen

```text
python spectra/stark_transition_spectra.py
```

Dit script berekent voor elk paar multiplets de Starkverschillen `E_upper - E_lower`. Een overgangslijn heeft gewicht `p_lower * r_ij`: Boltzmannweging geldt standaard alleen wanneer het beginmultiplet ³H₆ of ³F₄ is. Het eindmultiplet krijgt geen Boltzmannfactor. Zonder eigen lijnsterktes krijgen de eindniveaus gelijke relatieve gewichten.

De sticks worden geconvolueerd met een Gauss- en eventueel Lorentzprofiel. Pas bovenin `PUMP_NM`, `TEMPERATURE_K`, `LINE_WIDTH_CM1`, `LORENTZ_HWHM_CM1` en `INSTRUMENT_SIGMA_CM1` aan. De daadwerkelijke standaardbreedte voor overgangslijnen is **8,5 cm⁻¹**. De aparte gladde multipletfiguren gebruiken `MULTIPLET_PLOT_SIGMA_CM1 = 150` cm⁻¹.

Het script schrijft drie figuren naast zichzelf:

- `tm_multiplets_3H6.png`: multipletverdelingen met ³H₆ als energiereferentie;
- `tm_multiplets_3F4.png`: multipletverdelingen met ³F₄ als energiereferentie;
- `tm_overgangsspectra.png`: kandidaat-overgangen en de geschaalde ³F₄ → ³F₂-band.

De console rangschikt kandidaten rond de ingestelde pomp op energiemismatch. Dit is geen rangschikking op absorptiesterkte. Alleen ³F₄ → ³F₂ heeft standaard een kruisdoorsnede-anker: 6,4 × 10⁻²¹ cm² bij 1064 nm. Voor banden zonder anker of eigen bandsterkte blijft de absolute kruisdoorsnede onbekend (`NaN`). Het GSA-fitpunt bij 1064 nm wordt niet gebruikt om alle elektronische banden te schalen.

### 3. Elektronische GSA/ESA-factoren

```text
python spectra/gsa_esa_electronic_factors.py --no-show --output-dir spectra/results/electronic
python spectra/gsa_esa_electronic_factors.py --data-only --output-dir spectra/results/electronic
```

De scan loopt standaard van 900 tot 2000 nm met stap 0,05 nm. De ³H₆ → ³H₅-GSA-lijnen krijgen een effectieve Gaussbreedte van 300 cm⁻¹. De andere GSA-band en alle ESA-banden gebruiken 8,5 cm⁻¹. Beide achtergronden zijn uitgeschakeld. De onderste Starkniveaus krijgen Boltzmanngewichten bij 300 K. Berekeningen zonder Lorentzverbreding gebruiken een analytische Gaussiaanse som, zodat geen kunstmatige absorptievloer nodig is.

De relatieve factoren zijn:

```text
f_i(lambda) = sigma_i(lambda) / sigma_i(1064 nm)
```

Daarom zijn beide factoren 1 bij 1064 nm en kunnen ze bij andere golflengten groter dan 1 worden. Ze bevatten **geen fotonfluxfactor**. Voor een pompsnelheid moet de kruisdoorsnede nog met de fotonflux worden gecombineerd, zoals de kinetische scripts doen.

Uitvoer: `f_gsa_f_esa_900_2000nm_zonder_achtergrond.csv`, plus `.png` en `.svg` tenzij `--data-only` is gekozen. De CSV-kolommen zijn `wavelength_nm,f_gsa,f_esa`. Zonder `--output-dir` komt nieuwe uitvoer naast het script.

### 4. Discrete GSA-fononzijbanden

```text
python spectra/gsa_esa_phonon_factors.py --data-only --output-dir spectra/results/phonon
python spectra/gsa_esa_phonon_factors.py --temperature 350 --coupling-h5 0.08 --data-only --output-dir spectra/results/phonon_350K
python spectra/gsa_esa_phonon_factors_legacy.py --no-show --output-dir spectra/results/phonon_no_show
```

Beide fononscripts gebruiken drie effectieve modi van 250, 305 en 365 cm⁻¹ met gelijke verdeling. De eindige-temperatuur Huang–Rhys-kernel verschuift de elektronische overgangslijnen. Positieve verschuivingen betekenen fononcreatie bij absorptie; negatieve verschuivingen verbruiken fononen. Alleen **GSA** krijgt zijbanden; ESA blijft het elektronische model. De elektronische kernbreedte is 8,5 cm⁻¹: de eerdere effectieve H5-breedte van 300 cm⁻¹ wordt hierdoor vervangen.

Standaard zijn `S(H5) = S(F4) = 0.05` en T = 300 K. De commandoregelopties zijn:

| Optie | Betekenis |
|---|---|
| `--temperature` | Temperatuur in kelvin voor Starkbezettingen en fononkernel |
| `--coupling-h5` | Totale Huang–Rhys-factor voor ³H₆ → ³H₅ |
| `--coupling-f4` | Totale Huang–Rhys-factor voor ³H₆ → ³F₄ |
| `--kernel-tolerance` | Grens aan weggelaten kernelmassa; standaard 10⁻³⁵ |
| `--data-only` | Alleen CSV en JSON, zonder grafieken |
| `--no-show` | Venster onderdrukken in de legacyvariant; zie hieronder |
| `--output-dir` | Uitvoermap; standaard naast het script |

Uitvoer: `f_gsa_f_esa_fononzijbanden.csv`, `f_gsa_f_esa_fononzijbanden_instellingen.json` en, bij grafieken, `.png` en `.svg`. De CSV heeft dezelfde drie kolommen als het elektronische script. De JSON bevat instellingen, genormaliseerde kernelmassa, weggelaten-massa-grenzen en de nauwkeurigheidsgrens bij het 1064-nm-anker.

**Verschil tussen de varianten:** `gsa_esa_phonon_factors.py` opent zijn grafiek niet-blokkerend, maar negeert in de bestaande grafiekfunctie de optie `--no-show`. `gsa_esa_phonon_factors_legacy.py` respecteert die optie en sluit de figuur wanneer geen venster gewenst is; zonder die optie gebruikt deze variant een blokkerend venster. De modelberekeningen zijn gelijk. Beide gebruiken dezelfde uitvoerbestandsnamen; gebruik aparte uitvoermappen als je meerdere runs wilt bewaren. Met `--data-only` opent geen van beide een venster.

Voor een **continue fononband én fononen in ESA** gebruik je [fonon_model/pa_fononband.py](../fonon_model/pa_fononband.py). Dat model kan ook de populatiedynamiek berekenen en heeft andere uitvoerbestanden.

## Aannamen en bewaarde resultaten

De Starkenergieniveaus zijn overgenomen uit de oorspronkelijke scripts. Lijnbreedtes, relatieve lijn- en kanaalsterktes, fononkoppelingen en modegewichten zijn modelaannamen. Een spectrum dat op 1064 nm is genormaliseerd bepaalt geen absolute GSA/ESA-verhouding. De scripts vormen geen nieuwe experimentele kalibratie.

[reference_results](reference_results/README.md) bevat de ongewijzigde CSV-, SVG-, PNG- en JSON-exportbestanden uit de voormalige hoofdmap. Het zijn referentie-uitkomsten, geen invoerbestanden voor deze scripts. In het bijzonder hoort de bewaarde `hybrid`-export bij een eerdere versie met achtergrondbijdrage. Een nieuwe run met de huidige instellingen hoeft die historische uitvoer niet te reproduceren.

## Oude en nieuwe namen

| Oude naam | Nieuwe naam |
|---|---|
| `Energie niveas gaussian 3h4 3f2 3f3.py` | `stark_multiplet_distributions.py` |
| `Stark Energy per overgang.py` | `stark_transition_spectra.py` |
| `script f_factor.py` | `gsa_esa_electronic_factors.py` |
| `script f_factor fononen.py` | `gsa_esa_phonon_factors.py` |
| `untitled0.py` | `gsa_esa_phonon_factors_legacy.py` |
