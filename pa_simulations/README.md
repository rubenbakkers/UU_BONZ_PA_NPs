# Photon-avalanche-simulaties

Deze map bevat de scripts die het gedeelde driepopulatiemodel in
`pa_model.py` gebruiken. Ze berekenen populaties en relatieve 800-nm-emissie
van Tm³⁺-gedoteerde NaYF₄-deeltjes, maken figuren in de stijl van Lee et al.
(2021), of onderzoeken de invloed van de pompgolflengte. De verschillende
golflengtemodellen zijn afzonderlijke experimenten: hun aannames en
drempeldefinities zijn niet onderling uitwisselbaar.

## Installatie en starten

Voer de onderstaande opdrachten uit vanuit de hoofdmap van de repository.
Gebruik Python 3.10 of nieuwer met NumPy, Matplotlib en SciPy:

```bash
python -m pip install numpy matplotlib scipy
python pa_simulations/gsa_resonance_scan.py
python pa_simulations/figure2a_power_dependence.py
```

`gsa_resonance_scan.py` en `esa_resonance_scan.py` zijn de snelste voorbeelden:
ze evalueren alleen de absorptiefactoren. Emissiecurven, stijgtijden en vooral
`figure3b_threshold_relaxation.py` vragen veel meer rekenwerk.

De scripts gebruiken `import pa_model` of `from pa_model import ...`.
Daarom staan de uitvoerbare scripts en hun gedeelde model in dezelfde map.
Start een script via zijn pad zoals hierboven; zo kan Python `pa_model.py`
naast het script vinden. Bij gebruik vanuit een notebook of interactieve
sessie moet deze map ook op het Python-importpad staan.

De hieronder beschreven scripts hebben, tenzij bij het specifieke script
anders aangegeven, geen opdrachtregelopties en lezen geen meetbestanden.
De standaardwaarden staan in `main()` en in de constanten bovenaan het
bestand. Wijzig bijvoorbeeld daar de sample-ID, pompgolflengte, intensiteiten
of scanresolutie. GSA betekent *ground-state absorption* en ESA
*excited-state absorption*. Intensiteiten zijn in kW/cm², golflengten in nm,
relaxatiesnelheden in s⁻¹ en Stark-energieën in cm⁻¹.

### Waar verschijnen de resultaten?

Veel scripts geven een relatief bestandspad aan Matplotlib mee. Hun figuren
verschijnen daardoor in de **huidige werkmap**. Bij de opdrachten in deze
README is dat de hoofdmap van de repository. Starten vanuit een andere
werkmap verandert de uitvoerlocatie.

`wavelength_detuning_stark_gsa_esa.py` gebruikt de map `results/` naast zijn
eigen script voor zijn drie PDF-figuren. `stark_detuning_analysis.py` en
`stark_detuning_esa_background.py` schrijven standaard naast hun script;
hun optie `--output-dir` kiest een andere map. Een bestaand bestand met
dezelfde naam kan bij opnieuw uitvoeren worden overschreven. Console-uitvoer bevat
berekende parameters, slopes, drempels of samenvattingen van de scan.
Sommige scripts openen ook figuurvensters; andere schrijven alleen een
bestand. Raadpleeg de afzonderlijke beschrijvingen hieronder.

## Gedeelde bibliotheken

### `pa_model.py`: populatiedynamica en RK4

Dit bestand is een bibliotheek, geen zelfstandig figuurscript. Het bevat:

- Fysische constanten en de SI-samples met concentraties, relaxatiesnelheden
  en beschikbare gemeten drempels uit Lee et al.
- `PAParameters`, `build_parameters()` en `parameters_for_sample()` voor
  het samenstellen van de modelparameters.
- De gekoppelde differentiaalvergelijkingen voor de genormaliseerde
  populaties N0 = ³H₆, N1 = ³F₄ en N2 = ³H₄, inclusief GSA, ESA,
  relaxatie, cross-relaxation en upconversion.
- Een expliciete RK4-integrator, tijdsevolutie en benadering van de
  stationaire toestand vanaf de grondtoestand `(1, 0, 0)`.
- Relatieve 800-nm-emissie, power-dependence-curven, verschillende
  log-log-slopeberekeningen, een broken-line-drempel, 95%-stijgtijd en
  scans van de pompgolflengte.

Invoer is een intensiteit of array van intensiteiten, een parameterset en
optioneel een pompgolflengte en integratie-instellingen. Uitvoer bestaat uit
getallen, NumPy-arrays of een `WavelengthScanResult`; de bibliotheek schrijft
zelf geen bestanden. Het standaardmodel gebruikt sample 3, 8% Tm, en
1064 nm. De Gaussische bandcentra zijn circa 1219,7 nm voor GSA en 1060 nm
voor ESA; hun breedteparameters zijn respectievelijk 120 en 150 nm.
Deze breedten zijn illustratieve modelaannames.

Nuttig wanneer je een nieuwe figuur wilt maken zonder de modelvergelijkingen
te kopiëren. Bijvoorbeeld vanuit een Python-sessie in deze map:

```python
import numpy as np
from pa_model import parameters_for_sample, power_dependence

p = parameters_for_sample(3)
intensity = np.logspace(0, 2, 20)
emission = power_dependence(intensity, p, pump_wavelength_nm=1064.0)
```

Let op: het standaardmaximum van 30.000 stappen voor een stationaire
berekening kan worden bereikt voordat convergentie is bereikt. Vooral bij
hoge Tm-concentraties is een controle van het residu en een groter
stappenbudget nodig. `steady_state_solution(..., warn_if_unconverged=True)`
kan een waarschuwing afdrukken. De tijdintegrator heeft daarnaast een
stappenlimiet van 300.000; een lange aangevraagde tijd garandeert dus niet
dat de volledige tijd wordt doorgerekend. De 1%-parameters zijn een
expliciete extrapolatie van de concentratierelaties.

### `pa_spectra.py`: onafhankelijk, bij 1064 nm verankerd absorptiemodel

`StarkAbsorptionModel` bouwt GSA- en ESA-spectra uit de zes Stark-manifolds
van Conrad et al. (2025), Table 4(B), DOI
[10.1021/acs.jpcc.5c06209](https://doi.org/10.1021/acs.jpcc.5c06209).
De beginpopulaties binnen een multiplet volgen een Boltzmannverdeling.
Standaard zijn de temperatuur 300 K, de smalle Gaussische breedte
8,5 cm⁻¹, de brede GSA-breedte 1000 cm⁻¹ en de brede GSA-fractie 0,5.
De thermische asymmetrie onderdrukt de rode zijde van deze brede component.
ESA gebruikt de expliciete F4→H4/F3/F2-overgangen en heeft hier geen
willekeurig toegevoegde brede band.

Invoer: een golflengte of array van golflengten, een `PAParameters`-object
voor de doorsneden, en optioneel temperatuur, bandgewichten of
oscillatorsterktematrices. Uitvoer:

- `factors()` geeft relatieve GSA- en ESA-doorsneden, elk exact 1 bij
  1064 nm.
- `cross_sections()` geeft de doorsneden in m².
- `rates()` geeft twee absorptiesnelheidscoëfficiënten per kW/cm².
- `diagnostics()` geeft aannames, provenance en kalibratiestatus terug.

Deze bibliotheek schrijft geen bestanden. Ze is nuttig voor expliciete
absorptie-aannames en controle van de 1064-nm-ankerwaarden. Alleen die
golflengte is aan de Lee-SI verankerd; waarden daarbuiten zijn verkennend.
Dit bestand wordt momenteel **niet door de figuur- of detuningscripts
geïmporteerd**. Het is een afzonderlijke API; het toevoegen van deze
bibliotheek verandert de bestaande simulaties niet automatisch.

```python
from pa_model import parameters_for_sample
from pa_spectra import StarkAbsorptionModel

p = parameters_for_sample(3)
spectrum = StarkAbsorptionModel()
gsa, esa = spectrum.rates(1064.0, p)
audit = spectrum.diagnostics(1064.0, p)
```

### `pa_curve_analysis.py`: onafhankelijke analyse van log-log-curven

`analyze_curve(intensity, emission, fit_windows=None)` analyseert matching
positieve arrays met minstens vijf punten. De functie sorteert de
intensiteiten, vereist unieke waarden en past geen clipping of smoothing
toe. Optionele fysieke fitvensters hebben de vorm
`{"low": (a, b), "rising": (c, d)}`. Zonder vensters gebruikt de functie
gedocumenteerde automatische kwaliteitscriteria.

Uitvoer is een dictionary met onder andere de maximale differentiële slope
`s_peak`/`s_max`, de bijbehorende intensiteit `Ith`/`Ith_star`, de afzonderlijke
regressieslope `s_fit`, de low/rising-fits, hun kwaliteit en de
line-intersection-`threshold`. Die laatste blijft NaN als beide regimes
onvoldoende zijn opgelost of hun kruising niet plausibel is. Een gewone
power law kan een regressieslope hebben zonder een drempel te hebben.

Nuttig om een gesimuleerde of gemeten curve te onderzoeken en onderscheid
te maken tussen een differentiële slope, een lijnfit en een onset-drempel.
De curve alleen bevestigt geen photon avalanche. Ook dit bestand wordt
momenteel **niet door de andere scripts geïmporteerd** en schrijft geen
bestanden.

```python
from pa_curve_analysis import analyze_curve

result = analyze_curve(intensity, emission)
print(result["s_peak"], result["s_fit"], result["threshold"])
```

## Figuren in de stijl van Lee et al.

### `figure2a_power_dependence.py`

```bash
python pa_simulations/figure2a_power_dependence.py
```

Vergelijkt 800-nm-emissie voor samples 1, 2 en 3: 1%, 4% en 8% Tm bij
1064 nm. De standaardscan bevat 100 logaritmisch verdeelde intensiteiten
van 1 tot 150 kW/cm². Een extra lokale scan rond de knie bepaalt een
regressieslope over ±0,10 log decade.

Uitvoer: `pa_fig2a_style_v2.png`, modelparameters en fitinformatie in de
console, en een figuurvenster. Het venster blokkeert de uitvoering totdat
het wordt gesloten. Nuttig voor vergelijking van power dependence en
concentratie-effecten.

De geprinte slope gebruikt de verfijnde lokale scan. De plotfunctie fit
opnieuw de oorspronkelijke, grovere punten binnen dat venster. De slope
van de getekende fit kan daardoor afwijken van de geprinte waarde.

### `figure2a_sample_specific_grids.py`

```bash
python pa_simulations/figure2a_sample_specific_grids.py
```

Alternatieve Figure-2a-uitwerking voor samples 1–3 met respectievelijk
100, 1000 en 5000 logaritmisch verdeelde intensiteiten van 1 tot
150 kW/cm². De pompgolflengte is 1064 nm. De console rapporteert de
maximale differentiële slopes op deze rasters; dit zijn geen lokale
regressieslopes. Gebruik dit bestand om het effect van fijnere sampling
bij hogere concentratie te bekijken. De 8%-curve met 5000 punten is zwaar.

Uitvoer: `pa_fig2a_style_v2.png` in de huidige werkmap en een figuurvenster
via `plt.show()`. Deze bestandsnaam is hetzelfde als die van
`figure2a_power_dependence.py`; uitvoeren in dezelfde werkmap overschrijft
de eerdere figuur. Dat andere script gebruikt één gezamenlijk raster en
een lokale windowed fit. Beide gebruiken het gedeelde `pa_model.py`.

### `figure2b_rise_time.py`

```bash
python pa_simulations/figure2b_rise_time.py
```

Berekent eerst een verfijnde broken-line-drempel voor sample 3, 8% Tm bij
1064 nm. Daarna volgt een tijdscan op 44 unieke intensiteiten tussen één
en vier keer deze drempel. De stijgtijd is de tijd tot 95% van de
geschatte stationaire 800-nm-emissie. De aangevraagde tijdvensters beginnen
bij 5 s en worden zo nodig verdubbeld, met een ingestelde bovengrens van
200 s.

Uitvoer: `pa_fig2b_rise_time_v2.png`, drempel/fit-slopes in de console en
een figuurvenster. Nuttig om kritische vertraging rond de avalanche-onset
te onderzoeken. De schatting hangt af van de stationaire oplossing en de
integratiestappenlimieten in `pa_model.py`.

### `figure3a_architecture_concentration.py`

```bash
python pa_simulations/figure3a_architecture_concentration.py
```

Berekent 100 intensiteiten van 1 tot 150 kW/cm² voor samples 3–8. De
bovenste subplot vergelijkt samples 3/4/5, alle 8% Tm maar met
verschillende verliesparameters. De onderste subplot vergelijkt samples
3/7/8, met 8%, 20% en 100% Tm. Sample 6 wordt ook doorgerekend, maar staat
niet in de twee subplots.

Uitvoer: `pa_fig3a_style_v2.png` en sampleparameters in de console; er is
geen expliciete `show()`-aanroep. Nuttig voor vergelijking van
architectuur- en concentratie-effecten. De legenda rapporteert het
maximum van de differentiële slope op het gebruikte raster.

### `figure3b_threshold_relaxation.py`

```bash
python pa_simulations/figure3b_threshold_relaxation.py
```

Onderzoekt de PA-drempel als functie van de totale relaxatiesnelheid W2
voor 8%, 20% en 100% Tm. De scan gebruikt 18 W2-waarden van 162,6 tot
1500 s⁻¹ en power-dependence-curven op dichte pompintensiteitsrasters.
De interpolatie neemt W3_NR = 2 × W2_NR aan. De modeldrempels worden
lineair gefit; gemeten SI-drempels staan als afzonderlijke punten.

Uitvoer: `pa_fig3b_style_v2.png` en model-/experimentwaarden plus
fitcoëfficiënten in de console. De figuren worden gesloten. Nuttig om
het verband tussen relaxatieverliezen en benodigde pompintensiteit te
onderzoeken. Dit is een zeer zware berekening, vooral bij 100% Tm.
Een onopgeloste knie blijft NaN. De coarse scan krijgt een verhoogd
stappenbudget, maar de extra drempelverfijning gebruikt de standaard
stationaire integratie-instellingen van `pa_model.py`.

## Golflengte-detuning

### `wavelength_detuning_gaussian_cauchy.py`

```bash
python pa_simulations/wavelength_detuning_gaussian_cauchy.py
```

Vergelijkt in de code een Gaussische en Cauchy/Lorentzische lijnvorm door
de resonantiefactor van `pa_model` tijdelijk te vervangen. De actieve
`main()` berekent alleen de Gaussische variant: sample 3, vijftien
golflengten tussen 850 en 1180 nm en 100 intensiteiten van 10 tot
150 kW/cm².

Uitvoer bij normaal starten: `pa_wavelength_detuning_gaussian.pdf` en
curve-/drempelinformatie in de console. De Cauchy-aanroepen in `main()`
staan uitgecommentarieerd. Nuttig om de invloed van een aangenomen
lijnvorm op de response te verkennen.

Gaussian gebruikt een broken-line-drempel. De Cauchy-helper gebruikt de
geconvergeerde intensiteit bij maximale log-log-slope, aangeduid als
`Ith*`; deze is een andere grootheid. De functie voor het opslaan schrijft
PDF-formaat, ook als een aangepaste bestandsnaam een andere extensie
zou hebben. De oude koptekst die twee PNG-uitvoerbestanden noemt,
beschrijft niet de huidige actieve `main()`.

### `wavelength_detuning_stark_gsa_esa.py`

```bash
python pa_simulations/wavelength_detuning_stark_gsa_esa.py
```

Bouwt zowel GSA als ESA uit expliciete Stark-to-Stark-overgangen:
H6→F4/H5 voor GSA en F4→H4/F3/F2 voor ESA. De smalle breedte is
8,5 cm⁻¹; de toegevoegde brede achtergrond heeft voor beide een gewicht
van 0,00001. De spectrale factoren worden aan 1064 nm verankerd. Dit is
een ander spectrum dan `pa_spectra.py` en gebruikt die bibliotheek niet.

Standaard: sample 3, vijftien golflengten 850–1180 nm, een
intensiteitsbereik van 10–1000 kW/cm² en adaptieve rasters met 30 coarse,
40 knee- en 40 polish-punten. SciPy/LSODA integreert vanuit de grondtoestand
en verlengt het tijdvenster totdat relatieve stationariteitscriteria zijn
gehaald of de limiet van 1280 s wordt bereikt. Het script controleert of
zijn lokale modelvergelijkingen overeenkomen met `pa_model`.

Uitvoer in `pa_simulations/results/`:

- `pa_wavelength_detuning.pdf`: emissiecurven met lokale lijnfits.
- `pa_smax_vs_wavelength.pdf`: regressieslope versus pompgolflengte.
- `pa_ith_vs_wavelength.pdf`: numerieke `Ith*` en analytische
  drempelformule versus pompgolflengte.

Er verschijnen ook consolewaarden en figuurvensters. Nuttig om
Stark-structuur en een analytische drempelschatting te vergelijken.
SciPy is voor de actieve simulatie nodig; de afgevangen importfout
implementeert hier geen werkende RK4-fallback. `Ith*` is de intensiteit
bij maximale slope. De in dit script gebruikte aanduiding `s_max`
kan een regressieslope beschrijven; dat is niet automatisch het maximum
van de differentiële slope.

### `wavelength_detuning_stark_esa_background.py`

```bash
python pa_simulations/wavelength_detuning_stark_esa_background.py
```

Vervangt alleen ESA door 45 F4→F2- en 63 F4→F3-Stark-overgangen met
gelijke totale manifoldgewichten. De breedte is 8,5 cm⁻¹. Een brede
Gaussische ESA-achtergrond met gewicht 0,5 wordt toegevoegd; GSA behoudt
de oorspronkelijke Gaussische factor. Het samengestelde ESA-spectrum
wordt op zijn globale piek genormaliseerd, niet exact op 1064 nm.

Standaard: sample 3, dezelfde vijftien golflengten 850–1180 nm,
20–200 kW/cm² en de adaptieve rasters 30+40+40. Met SciPy gebruikt het
script een vast LSODA-tijdvenster van 20 s. Zonder SciPy, of met
`USE_FAST_SOLVER=False`, gebruikt het de RK4-bibliotheek.

Uitvoer: `pa_wavelength_detuning_stark_gaussian.pdf` in de werkmap.
Twee extra vensters tonen de gefitte slope en `Ith*` versus golflengte;
die samenvattingsfiguren worden niet opgeslagen. Nuttig om een
fenomenologische brede ESA-band met smalle Stark-structuur te combineren.
De import berekent grote normalisatierasters; ook importeren kan daardoor
tijd en geheugen kosten. Het vaste 20-s-venster voert geen afzonderlijke
stationariteitscontrole uit.

### `stark_detuning_analysis.py`

```bash
python pa_simulations/stark_detuning_analysis.py --spectra-only --output-dir pa_simulations/results/stark_no_background
python pa_simulations/stark_detuning_analysis.py --output-dir pa_simulations/results/stark_no_background
```

Dit uitgebreidere script koppelt Boltzmannpopulaties van de Stark-niveaus
aan het driepopulatiemodel. De standaardmodus `ABSORPTION_MODEL="broad_gsa"`
verbreedt alleen H6→H5-GSA-lijnen tot sigma = 300 cm⁻¹; de overige
lijnen hebben sigma = 8,5 cm⁻¹. GSA en ESA hebben geen extra brede
achtergrond. De factoren zijn bij 1064 nm verankerd. Alle invoer staat
in het script; bestaande CSV-exportbestanden worden niet ingelezen.

De standaardkinetiek gebruikt sample 3, een intensiteitsbereik van
1–150 kW/cm² en 21 pompgolflengten in twee gebieden: 1055–1100 en
1390–1490 nm. De intensiteitspunten worden rond de knie verfijnd. SciPy
is nodig voor de actieve LSODA-kinetiek en voor aanvullende
root-finding. De berekening controleert stationariteit, geeft numerieke
`Ith*`, een afzonderlijke analytische drempel en lijnfit-slopes, en voert
95%-stijgtijd- en Stark-breedtescans uit. Het volledige programma is zwaar.

`--spectra-only` exporteert alleen de twee factorplots en hun data:

- `f_gsa_f_esa_900_1500nm_broad_gsa.csv`, `.png` en `.svg`.
- `f_gsa_f_esa_1040_1100nm_broad_gsa.csv`, `.png` en `.svg`.

Een volledige run maakt daarnaast:

- `pa_wavelength_detuning_broad_gsa.pdf`.
- `pa_smax_vs_wavelength_broad_gsa.pdf`.
- `pa_ith_vs_wavelength_broad_gsa.pdf`.
- `pa_rise_time_vs_intensity_broad_gsa.pdf`, als geldige stijgtijddata
  worden gevonden.
- `pa_rise_peak_vs_sigma_broad_gsa.pdf` voor de Stark-breedtescan.

Zonder `--output-dir` is de uitvoermap de map naast het script. De
voorbeeldopdrachten kiezen een afzonderlijke resultatenmap. Nuttig om de
invloed van thermische Stark-bezetting, overgangsspecifieke verbreding en
stijgtijden samen te onderzoeken. De 1064-nm-verankering kalibreert de
langgolvige resultaten niet.

De alternatieve `hybrid`-modus wordt met haar huidige standaardwaarden
afgewezen wegens te zwakke smalle GSA bij 1064 nm. Historische bestanden
met `_hybrid` in `../spectra/reference_results/` komen uit andere
achtergrondinstellingen en kunnen dus niet met de huidige defaults van
dit script worden gereproduceerd.

### `stark_detuning_esa_background.py`

```bash
python pa_simulations/stark_detuning_esa_background.py --spectra-only --output-dir pa_simulations/results/stark_esa_background
python pa_simulations/stark_detuning_esa_background.py --output-dir pa_simulations/results/stark_esa_background
```

Deze behouden variant gebruikt dezelfde brede H5-GSA van 300 cm⁻¹,
dezelfde overige breedte van 8,5 cm⁻¹, thermische Stark-populaties en
hetzelfde 1064-nm-anker. Het verschil met `stark_detuning_analysis.py`
is de kleine ESA-achtergrond met gewicht 0,00001; de GSA-achtergrond
blijft uit. Ook de standaardintensiteiten, actieve golflengtegebieden,
adaptieve scan, analytische/numerieke drempels, stijgtijdberekeningen en
SciPy-afhankelijkheid komen overeen.

De opdrachtregelopties en alle standaarduitvoerbestandsnamen zijn dezelfde
als bij `stark_detuning_analysis.py`. Gebruik daarom verschillende
`--output-dir`-mappen, zoals in de voorbeelden, als je de modellen wilt
vergelijken. Anders kunnen de resultaten elkaar overschrijven.

Nuttig als historische vergelijking van een model met een kleine
ESA-achtergrond en de variant zonder achtergrond. De standaard
`SLOPE_FRACTION=0.25` gebruikt een vaste fitvensterkeuze. De strengere
automatische fitcriteria worden pas toegepast als deze instelling `None`
is; de standaardfits moeten daarom niet allemaal als strikt gevalideerde
lineaire segmenten worden geïnterpreteerd. De grote langgolvige response
blijft afhankelijk van de absorptie-aannames.

## Resonantie en emissie versus golflengte

### `gsa_resonance_scan.py`

```bash
python pa_simulations/gsa_resonance_scan.py
```

Evalueert voor sample 3 alleen de GSA-resonantiefactor op 31 golflengten
van 950 tot 1250 nm. Uitvoer: `pa_resonance_gsa_v2.png` en het gesamplede
maximum in de console. Nuttig voor snelle controle van GSA-bandcentrum
en breedte, zonder stationaire populaties door te rekenen.

### `esa_resonance_scan.py`

```bash
python pa_simulations/esa_resonance_scan.py
```

Dezelfde snelle scan voor de ESA-factor. Uitvoer:
`pa_resonance_esa_v2.png` en het gesamplede maximum in de console.
Nuttig om het ESA-profiel en de afwijking van resonantie te visualiseren.

### `emission_wavelength_scan.py`

```bash
python pa_simulations/emission_wavelength_scan.py
```

Berekent de volledige RK4-stationaire populaties en 800-nm-emissie voor
sample 3 op 31 golflengten van 950 tot 1250 nm, bij vaste intensiteiten
van 20 en 60 kW/cm². Uitvoer: `pa_emission_vs_wavelength_v2.png` en
emissiebereik/pieklocatie in de console. Nuttig om te zien hoe de
pompgolflengte bij een vast vermogen doorwerkt in de emissie. Dit kost
meer tijd dan alleen de GSA-/ESA-factorplots.

## Wetenschappelijke grenzen en definities

De gebruikte Stark-energieën komen uit een andere bron dan de SI-parameters
van de Lee-nanodeeltjes. Posities van Stark-niveaus leveren op zichzelf geen
gemeten lijnbreedten, oscillatorsterkten of fononzijbandsterkten voor die
nanodeeltjes. Gelijke band-/lijngewichten en achtergrondbanden zijn expliciete
modelkeuzes. Resultaten bij andere pompgolflengten moeten daarom als
verkennend worden gelezen, tenzij onafhankelijke absorptiedata worden gebruikt
om de aannames te toetsen.

Onderscheid bij vergelijking van de scripts:

- Een **differentiële slope** is d(log E)/d(log I), bepaald op een raster.
- Een **regressieslope** is de helling van een rechte lijn door een gekozen
  venster in de log-log-curve.
- **`Ith*`** is in de detuningvarianten de intensiteit bij maximale slope.
- Een **broken-line-drempel** is de kruising van afzonderlijke lijnen onder
  en in/boven de onset. De verschillende implementaties selecteren niet
  noodzakelijk dezelfde vensters.

De heuristische slopegrens van 10 in enkele detuningvarianten is een
numerieke classificatie binnen die scripts. Een steile curve alleen bewijst
geen photon avalanche. Voor een kwantitatieve vergelijking zijn ook
ESA/GSA-contrast, stijgtijd, passende fitvensters en convergentie relevant.
De rootmodelwaarde voor het GSA-bandcentrum is circa 1219,7 nm; enkele oude
commentaren in de scripts noemen nog circa 943,5 nm.

De README beschrijft de gelezen broncode en haar standaardinstellingen.
Een succesvolle import of een korte controle verifieert geen volledige
golflengtescan, integratieconvergentie of overeenkomst met een experimentele
figuur.

## Oude namen

| Oude naam | Huidige naam |
|---|---|
| `pa_script_1_fig2a.py` | `figure2a_power_dependence.py` |
| `pa_script_2_fig2b_risetime.py` | `figure2b_rise_time.py` |
| `pa_script_3_fig3a.py` | `figure3a_architecture_concentration.py` |
| `pa_script_4_fig3b.py` | `figure3b_threshold_relaxation.py` |
| `pa_script_5_wavelength_detuning.py` | `wavelength_detuning_gaussian_cauchy.py` |
| `pa_script_5_wavelength_detuning_double.py` | `wavelength_detuning_stark_gsa_esa.py` |
| `pa_script_5_wavelength_detuning_double2.py` | `wavelength_detuning_stark_esa_background.py` |
| `pa_script_6_resonance_gsa.py` | `gsa_resonance_scan.py` |
| `pa_script_7_resonance_esa.py` | `esa_resonance_scan.py` |
| `pa_script_8_emission_vs_wavelength.py` | `emission_wavelength_scan.py` |
| `fig2a.py` | `figure2a_sample_specific_grids.py` |
| `gaussian detuning specifieke overgang.py` | `stark_detuning_analysis.py` |
| `wavelength detuning backup.py` | `stark_detuning_esa_background.py` |

De gedeelde bibliotheeknamen `pa_model.py`, `pa_spectra.py` en
`pa_curve_analysis.py` zijn behouden.
