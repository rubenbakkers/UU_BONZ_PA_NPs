# Zelfstandige simulaties en experimentele fits

Deze scripts bevatten hun eigen model of meetwaarden. Ze importeren geen
`pa_model.py`. Ze blijven apart beschikbaar omdat hun aannames en berekeningen
verschillen van de gedeelde workflow in [`pa_simulations`](../pa_simulations/README.md).

## Installatie en starten

Vereist: Python 3.10 of hoger, NumPy en Matplotlib. SciPy is voor deze scripts
niet nodig. Installeer vanuit de hoofdmap de afhankelijkheden met:

```text
python -m pip install -r requirements.txt
```

Voorbeelden, uitgevoerd vanuit de hoofdmap van de repository:

```text
python standalone_simulations/rk4_minimal_three_level.py
python standalone_simulations/figure3b_experimental_linear_fits.py
python standalone_simulations/rk4_full_gaussian_demo_v2.py
python standalone_simulations/rk4_full_wavelength_demo.py
python standalone_simulations/rk4_full_wavelength_dense_fig3b.py
```

Er zijn geen opdrachtregelopties of externe invoerbestanden. Pas de parameters
en roosters in het betreffende script aan, of gebruik de daarin gedefinieerde
functies vanuit Python. De PNG-bestanden worden in de **huidige werkmap**
geschreven. De drie volledige demonstraties gebruiken dezelfde uitvoernamen;
start ze in verschillende werkmappen als je hun resultaten wilt bewaren.

## Welk script gebruik je waarvoor?

| Script | Werking en nut | Uitvoer |
|---|---|---|
| [`rk4_minimal_three_level.py`](rk4_minimal_three_level.py) | Kwalitatief drie-niveaumodel met GSA, ESA, cross-relaxatie en verval. Handmatige RK4-integratie laat zien hoe positieve terugkoppeling een steile emissietoename kan veroorzaken. Geschikt om het mechanisme en de populatiedynamica te verkennen. | `pa_power_dependence.png`, `pa_time_domain.png`; grafiekvensters en consolewaarden voor de maximale helling en geschatte drempel. |
| [`rk4_full_gaussian_demo_v2.py`](rk4_full_gaussian_demo_v2.py) | Zelfstandig DRE/RK4-model met Q22/Q23, relaxatiekanalen en samples uit het supplement van Lee et al. Maakt figuur-2/3-achtige emissiecurves, 95%-stijgtijden, drempels en een golflengtedemonstratie. Beide resonantiecentra staan standaard op 1064 nm, met aangenomen Gaussbreedtes van 50 en 80 nm. | De vijf PNG-bestanden hieronder; parameters, drempels en emissieversterking in de console. |
| [`rk4_full_wavelength_demo.py`](rk4_full_wavelength_demo.py) | Uitgebreide variant met afzonderlijke GSA-/ESA-factoren, een golflengtescan bij vaste pompintensiteiten en controles van het aangenomen Gaussmodel. De GSA-resonantie is hier ongeveer 943,5 nm, ESA 1064 nm; breedtes zijn 120 en 150 nm. Figuur 3b gebruikt een grof rooster. | De vijf algemene PNG-bestanden plus drie golflengtegrafieken; diagnostiek in de console. |
| [`rk4_full_wavelength_dense_fig3b.py`](rk4_full_wavelength_dense_fig3b.py) | Variant van de voorgaande demonstratie met een dichter figuur-3b-rooster: 18 W2-waarden per concentratie, een 80-punts intensiteitsrooster en lokale drempelverfijning. Tekent lineaire fits met vrije helling en intercept door de berekende drempels. Nuttig om de invloed van W2 en roosterkeuze te onderzoeken. | Dezelfde acht PNG-bestanden als de golflengtedemonstratie; fitcoëfficiënten in de console. Sluit de grafiekvensters na afloop. |
| [`figure3b_experimental_linear_fits.py`](figure3b_experimental_linear_fits.py) | Fit `I_th = m × W2` door de oorsprong aan ingebouwde experimentele drempels voor 8%, 20% en 100% Tm. Berekent geen DRE-populaties of numerieke lawinedrempels. Nuttig als overzicht van de experimentele trend. | `Figure_3b_corrected.png`, een grafiekvenster en de drie fithellingen in de console. |

De vijf algemene uitvoerbestanden van de volledige demonstraties zijn:

- `pa_fig2a_style_v2.png`: emissie versus pompintensiteit voor 1%, 4% en 8% Tm.
- `pa_fig2b_rise_time_v2.png`: tijd tot 95% van de berekende stationaire emissie.
- `pa_fig3a_style_v2.png`: emissiecurves voor verschillende samples en concentraties.
- `pa_fig3b_style_v2.png`: drempel versus totale W2-relaxatiesnelheid.
- `pa_wavelength_detuning_v2.png`: emissie versus intensiteit bij verschillende pompgolflengtes.

De twee uitgebreide golflengtevarianten maken daarnaast
`pa_resonance_gsa_v2.png`, `pa_resonance_esa_v2.png` en
`pa_emission_vs_wavelength_v2.png`.

## Modelverschillen en grenzen

De minimale simulatie gebruikt illustratieve snelheidsconstanten, geen fit aan
het artikel. De drempel is daar de positie van de maximale logaritmische
helling. De volledige demonstraties schatten de drempel met snijdende rechte
lijnen op log-log-assen, met lokale roosterverfijning. Deze routine begrenst het
snijpunt tot de omgeving van de knie en kan bij vrijwel gelijke hellingen
terugvallen op het hellingsmaximum.

De volledige modellen zijn geen onderling uitwisselbare versies. In
`rk4_full_gaussian_demo_v2.py` is de Q22/Q23-noemer daadwerkelijk
`c² + 4.32`. In beide uitgebreide golflengtevarianten staat in de berekening
`c² + 4.3²`; de oude functietoelichting in `rk4_full_wavelength_demo.py`
beschrijft die noemer onjuist. De verschillende resonantiecentra en breedtes
zijn eveneens behouden. De afleiding van het GSA-centrum rond 943,5 nm is een
historische modelkeuze; andere scripts in deze repository gebruiken een
centrum rond 1220 nm. De Gaussprofielen zijn aannames, geen gemeten spectra.
De golflengtevarianten vermenigvuldigen de doorsneden bovendien met die
profielen en verankeren ze niet opnieuw op 1064 nm.

Expliciete RK4-integratie kan veel tijd kosten bij hoge Tm-concentraties. De
staplimieten kunnen een nog niet stationaire populatie opleveren, standaard
zonder waarschuwing. Grotere gevraagde simulatietijden doorbreken de vaste
staplimiet niet. In de uitgebreide figuur-3b-berekeningen krijgt de grove scan
een grotere staplimiet, maar de drempelverfijning gebruikt weer de standaard.
Controleer convergentie voordat je drempels of stijgtijden kwantitatief gebruikt.

`figure3b_experimental_linear_fits.py` heeft bij 100% Tm slechts één meetpunt:
de bijbehorende helling is dus de verhouding van dat punt. Het script voert
de plot ook uit wanneer het wordt geïmporteerd. De twee oudere volledige
demonstraties eindigen met de behouden aanroep `plt.plot("all")`; die toont
niet automatisch alle figuren en kan een extra plot opleveren.

## Oorspronkelijke namen

| Oorspronkelijk | Huidig |
|---|---|
| `Simulatie PA RK4.py` | `rk4_minimal_three_level.py` |
| `photon_avalanche_rk4_corrected_v2.py` | `rk4_full_gaussian_demo_v2.py` |
| `Simulatie PA 2.py` | `rk4_full_wavelength_demo.py` |
| `untitled1.py` | `rk4_full_wavelength_dense_fig3b.py` |
| `fig3b.py` | `figure3b_experimental_linear_fits.py` |
