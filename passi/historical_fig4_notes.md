> **Historische toelichting:** deze tekst beschrijft een andere, DRE-gebaseerde implementatie die niet in deze repository staat. De actuele `passi_phenomenological_simulation.py` gebruikt een fenomenologische sigmoid, heeft geen CLI en leest `Fig4_data.npz` niet. De hieronder beschreven opties, exports en validatieresultaten gelden daarom niet voor het actuele script. Zie [README.md](README.md).

# Fig4.py - lokale DRE-respons met uitsluitend gepubliceerde parameters

Deze versie vervangt de vorige `recomputed`-berekening. De interpolatie van
ruwe filmmetingen, intensiteitsbins, monotone regressie en aangenomen
kwadratische staart zijn verwijderd. Het model gebruikt nu de lokale
stationaire oplossing van de rate equations uit het supplement.

**De resterende afwijking van figuur 4f is niet opgelost door parameters te
veranderen.** De DRE-implementatie is gecontroleerd en reproduceert het
gepubliceerde film-model uit figuur 3a, maar geeft bij hoge intensiteiten
andere beeldbreedtes dan de gepubliceerde PASSI-curve. Deze uitkomsten zijn
geen claim dat figuur 4 exact is gereproduceerd door een onafhankelijk model.

## Gebruik

Houd `Fig4.py` en `Fig4_data.npz` in dezelfde map. Vereist: NumPy, SciPy en
Matplotlib. De berekening heeft geen internetverbinding nodig.

```text
python Fig4.py
python Fig4.py --mode simulation --show
python Fig4.py --mode article
python Fig4.py --mode simulation --beam si-s7 --output-dir controle_S7
```

Standaard worden de eigen simulatie en de artikelreproductie gemaakt.
Wanneer het script in `BONZ/Script` staat, komen de afbeeldingen in
`BONZ/Script_afbeeldingen`. Elders worden ze naast het script opgeslagen.

- `fig4def_passi_simulation.png`: nieuwe rasterberekening met de lokale DRE-
  respons. Blauw: directe FWHM. Groen gestreept: Gauss-fit. Rood: de door de
  auteurs aangeleverde simulatiecurve als referentie. Zwart: experiment.
- `fig4def_artikel.png`: oorspronkelijke gepubliceerde beeldmatrices en
  curve. Dit is uitsluitend een weergave van brondata.
- `fig4_emissiecurve.png`: lokale DRE-respons, ruimtelijk geïntegreerde
  filmrespons en de gepubliceerde DRE-curve uit figuur 3a.
- `fig4f_breedtes.csv`: berekende breedtes over het intensiteitsbereik.

## Waarom de vorige berekening ongeschikt was

Figuur 3a betreft films met veel deeltjes, gemeten met een Gaussische pomp.
De openbare code van de auteurs integreert de lokale bevolking van het
emitterende niveau over die pomp om de filmrespons te berekenen:

`F_film(P) evenredig met integral_0^P n3(I) / I dI`.

De filmrespons is dus niet dezelfde functie als de lokale respons van één
deeltje. Mijn vorige versie gebruikte de ruwe filmmetingen rechtstreeks als
lokale respons. Bovendien maakte de interpolatie de berekende breedte gevoelig
voor lokale meetruis. Dat was geen goede correctie van het oorspronkelijke script.

De nieuwe versie gebruikt `f(I) = (1-b32) * W3_R * n3(I)` uit de DREs als
lokale emissie. De filmmetingen en de figuur-4-breedtes bepalen geen enkele
parameter van deze berekening. Het supplement vermeldt wel gebruik van de
gemeten S-curve voor PASSI; de nieuwe DRE-berekening is daarom een fysische
controle met de gepubliceerde parameters, niet een bewering dat de onbekende
PASSI-implementatie van de auteurs exact is nagebouwd.

## Parameters en vergelijkingen

SI Eq. S1-S3 worden stationair opgelost met behoud van `n1+n2+n3=1`.
De emissie wordt vervolgens over het hele deeltje gesommeerd bij het scannen
van een Gaussische pomp. Het deeltje is de 19,5 × 16 nm ellips uit de
PASSI-sectie van het supplement, met een rasterstap van 1 nm.

| Grootheid | Waarde | Bron |
|---|---:|---|
| Tm-concentratie | 8% (numerieke c = 8) | Sample 4 |
| W2 radiatief | 162,60 s^-1 | Tabel S4 |
| W3 radiatief | 636,01 s^-1 | Tabel S4 |
| b32 | 0,144 | Tabel S4 |
| W2 niet-radiatief | 40,7 s^-1 | Tabel S7, sample 4 |
| W3 niet-radiatief | 87,3 s^-1 | Tabel S7, sample 4 |
| sigma GSA | 6,0 × 10^-29 m² | Tabel S6 |
| sigma ESA | 6,4 × 10^-25 m² | Tabel S6 |
| a_cr | 160 s^-1 | Tabel S6 |
| a_uc | 25,6 s^-1 | Tabel S6 |
| a_inv | 9,00 s^-1 | Tabel S6 |
| Pompgolflengte | 1064 nm | Hoofdtekst en supplement |

Eq. S4-S6 geven `s31 = a_cr*c²`, `Q22 = a_uc*c³/(c²+4.3²)` en
`Q23 = a_inv*c³/(c²+4.3²)`. Absorptiesnelheden zijn `sigma*I/(h*nu)`,
met `1 kW/cm² = 10^7 W/m²`. De doorsneden gelden al bij 1064 nm; er wordt
geen extra aangenomen golflengtefactor toegepast.

De stationaire oplossing wordt algebraïsch berekend en daarna op een dicht
rooster opgeslagen voor snelle beeldberekening. Deze numerieke interpolatie
bevat uitsluitend berekende DRE-waarden. Buiten dat rooster worden dezelfde
vergelijkingen direct opgelost. Er wordt geen emissiekromme gefit.

## Resultaten en resterende afwijking

De standaardpomp heeft de expliciet genoemde FWHM van 357 nm uit de hoofdtekst.
De beeldannotaties tonen directe FWHM; Gauss-fitbreedtes zijn apart vermeld.

| Berekening | 4d: 9,9 kW/cm² | 4e: 7,1 kW/cm² |
|---|---:|---:|
| Lokale DRE, directe FWHM | 151,7 nm | 72,5 nm |
| Lokale DRE, Gauss-fit | 140,6 nm | 66,9 nm |
| Gepubliceerde beeldmatrix, directe FWHM | 211,8 nm | 83,3 nm |
| Experiment S10, gemiddelde van de twee afgeronde breedtes | 216,5 nm | 73,0 nm |

De vorm van de film-DRE-curve uit figuur 3a wordt, na uitsluitend verticale
normalisatie en integratie op het gepubliceerde intensiteitsrooster, tot
binnen 0,6% gereproduceerd. Dit ondersteunt de parameterkeuze en implementatie.
De fijn geïntegreerde curve in de controleafbeelding wijkt lokaal rond de
steile overgang meer af van de grof geïntegreerde broncurve.

De bronmatrices van 4d en 4e zijn numeriek exact Gaussisch. Een Gauss-fit
laat afwijkingen op het niveau van floating-point-afronding zien. Dat beschrijft
een eigenschap van de aangeleverde matrices; de code waarmee de auteurs ze
hebben gemaakt is niet meegeleverd. Het bewijst niet welke nabewerking of
benadering zij hebben gebruikt.

Een tweede onduidelijkheid zit in de pompdefinitie. SI Eq. S7 schrijft
`exp(-r²/omega²)` met `omega=lambda/(2*NA)`. Letterlijk geeft dat een FWHM
van circa 595 nm, terwijl de hoofdtekst 357 nm noemt. De optie `--beam si-s7`
maakt precies die letterlijke tweede keuze, zonder parameters te fitten.
Dan worden de directe breedtes 252,4 nm bij 9,9 en 120,9 nm bij 7,1 kW/cm².
Ook die keuze reproduceert beide panelen niet tegelijk.

De openbare tijdcode stopt standaard na 0,5 s. Een onafhankelijke controle
laat zien dat de emitterende populatie bij 7,1 kW/cm² dan nog ongeveer 8%
onder de stationaire waarde ligt. Die afkaptijd is daarom niet als
stationaire oplossing gebruikt. De minimaal verschillende afrondingen van
relaxatiesnelheden in de auteurscode verklaren de breedteafwijking evenmin.

Om de oorspronkelijke rode PASSI-curve zonder fit exact te reconstrueren is
meer informatie nodig over de daadwerkelijke PASSI-invoer en implementatie.
De huidige versie laat het verschil zien en past de gepubliceerde parameters
niet aan om dat verschil weg te werken.

## Uitgevoerde controles

- Stationaire DRE-uitkomsten vergeleken met onafhankelijke tijdintegratie
  (Radau) van 0 tot 5 seconden, bij meerdere intensiteiten inclusief de drempel.
- Behoud van totale populatie, niet-negatieve populaties en verwaarloosbaar
  residu in alle drie rate equations.
- De puntsource-limiet `357/sqrt(s)` gecontroleerd voor `s=1, 2, 26`.
- FFT-convolutie vergeleken met rechtstreeks optellen over de deeltjespixels,
  ook aan de rand van het beeld.
- Verfijning van 1 naar 0,5 nm pixels verandert de breedtes bij 7,1 en 9,9
  kW/cm² met minder dan 0,003 nm.
- Verdubbeling van de DRE-tabeldichtheid verandert de breedtes met minder dan
  0,001 nm en het genormaliseerde beeld met minder dan een miljoenste van de piek.
- Beide pompconventies gecontroleerd. Figuren visueel nagekeken.

## Bronnen en referentiegegevens

- [Lee et al., Nature 589, 230-235 (2021)](https://doi.org/10.1038/s41586-020-03092-9).
- Aangeleverd supplement: Eq. S1-S7, tabellen S4/S6/S7 en figuur S10.
- [Openbare DRE-code van Changhwan Lee](https://github.com/nawhgnahc/Photon_Avalanche_DRE_calculation/blob/master/Time-resolved%20photoluminescence%20calculation.py):
  rate matrices en de integraal `population_3H4_gaussian_array`.
- [Source Data Fig. 3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-03092-9/MediaObjects/41586_2020_3092_MOESM4_ESM.xlsx):
  tab `Figure 3a`, kolommen P/Q zijn de DRE-filmcurve voor 17/5,6 nm.
- [Source Data Fig. 4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-03092-9/MediaObjects/41586_2020_3092_MOESM5_ESM.xlsx):
  tabnamen vermelden ten onrechte `Figure 3`. Het brede beeld in `Figure 3e`
  hoort bij artikelpaneel 4d; het smalle in `Figure 3d` bij 4e. De oorspronkelijke
  pixelwaarden, schaal van 5 nm en maximum 0,8 zijn behouden.
  `Figure 3f` bevat de experimentele punten in B2:C7 en de rode curve in D/E.
  De RMS-foutbalken komen uit de twee standaarddeviaties in S10.

De eerder gedownloade bronwaarden blijven in `Fig4_data.npz` bewaard voor de
referentiepanelen. Ze zijn geen invoer voor het lokale DRE-model.
