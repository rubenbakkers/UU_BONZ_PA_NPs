# GSA en ESA met vier Gaussian fononbanden

`gsa_esa_four_gaussian_phonons.py` berekent de relatieve GSA- en ESA-spectra van 900 tot 2000 nm met vier eenvoudige fononbanden. Elke elektronische overgang bevat een bijdrage zonder fonon en bijdragen waarbij **één fonon wordt gemaakt of geabsorbeerd**. Er worden geen processen met twee of meer fononen en geen combinaties tussen de vier banden berekend.

Dit is een nieuw, afzonderlijk script. De bestaande scripts behouden hun eigen instellingen en normalisatie, inclusief hun eventuele 1064-nm-anker. Alleen deze nieuwe fononversie gebruikt de hieronder beschreven pieknormalisatie zonder 1064-nm-anker.

Dit is een genormaliseerd fenomenologisch eersteordemodel. De fononparameters illustreren hoe de aangeleverde Ramanfiguur een spectrale vorm kan geven; ze vormen geen experimenteel bepaalde elektronen-fononkoppeling.

De vier fononbanden kunnen afzonderlijk aan of uit. Standaard staan ze allemaal aan. Met `--phonon-bands 1 3` worden alleen banden 1 en 3 gebruikt; `--no-phonons` schakelt alle vier uit. De elektronische overgangen blijven in beide gevallen aanwezig.

De bestaande GSA/ESA-figuur toont standaard de zichtbare lokale pieken op hoogte één. Dat is een aanvullende bewerking van alleen de grafische weergave. De berekende f-factoren, lijngewichten, fononkansen en CSV-gegevens behouden hun hieronder beschreven modelnormalisatie. Met `--plot-normalization global` toont dezelfde figuur de berekende f-factoren rechtstreeks.

## De figuur en de vier banden

De bron is Wilhelm et al., *Multicolor Upconversion Nanoparticles for Protein Conjugation*, Theranostics **3**, 239–248 (2013), [doi:10.7150/thno.5113](https://doi.org/10.7150/thno.5113). [Figuur 3, gedrukte pagina 244 / PDF-pagina 6](https://www.thno.org/v03p0239.pdf#page=6) toont in paneel A de Ramanbanden van NaYF₄ rond 225–450 cm⁻¹. De gestippelde curve hoort bij NaYF₄(Yb/Tm); de groene doorgetrokken curve bij NaYF₄(Yb/Er). Paneel B toont CH₂-strektrillingen van het ligand oleïnezuur. Die worden hier niet aan het roosterfononmodel toegevoegd.

De parameters zijn een afgeronde benadering van de gestippelde Yb/Tm-curve, mede gebaseerd op de eerder handmatig afgelezen punten in `spectra/data/raman_ybtm_fig3a_approx.csv`. Het betreft geen fit aan oorspronkelijke numerieke meetgegevens. Overlappende Gaussiancomponenten hoeven hun centrum niet precies bij een zichtbaar maximum van de som te hebben.

| Band | Centrum Ωₖ (cm⁻¹) | Gaussian σₖ (cm⁻¹) | Relatieve componenthoogte Aₖ | Relatief oppervlaktegewicht wₖ |
| --- | ---: | ---: | ---: | ---: |
| 1 | 253 | 14 | 0.68 | 0.187 |
| 2 | 306 | 24 | 0.95 | 0.448 |
| 3 | 364 | 15 | 0.66 | 0.195 |
| 4 | 408 | 24 | 0.36 | 0.170 |

De breedtes zijn standaardafwijkingen, met FWHM = 2.355 σ. Het benaderde Ramanprofiel is

$$
R(\Omega)=\sum_{k=1}^{4}A_k\exp\!\left[-\frac{(\Omega-\Omega_k)^2}{2\sigma_k^2}\right].
$$

Een brede component bevat bij gelijke hoogte meer oppervlakte dan een smalle. Daarom gebruiken de fononbijdragen de oppervlaktegewichten

$$
w_k=\frac{A_k\sigma_k}{\sum_\ell A_\ell\sigma_\ell},\qquad \sum_k w_k=1.
$$

Deze vier oorspronkelijke oppervlaktegewichten blijven de vaste referentie wanneer een band wordt uitgezet. Definieer eₖ = 1 voor een ingeschakelde band en eₖ = 0 voor een uitgeschakelde band. Het actieve profiel is dan

$$
R_{\mathrm{actief}}(\Omega)=\sum_{k=1}^{4}e_kA_k\exp\!\left[-\frac{(\Omega-\Omega_k)^2}{2\sigma_k^2}\right].
$$

De actieve koppelingsgewichten zijn eₖwₖ. Ze worden niet opnieuw op som één gezet: het uitschakelen van een band verwijdert haar koppeling in plaats van die over de overblijvende banden te verdelen. Zo behouden bijvoorbeeld banden 1 en 3 hun oorspronkelijke gewichten van ongeveer 0.187 en 0.195 wanneer alleen die twee aanstaan.

De Ramanintensiteit wordt hiermee uitsluitend als **vormproxy** gebruikt. Ramanverstrooiing weegt trillingen met hun Ramanactiviteit en de meetgeometrie; het is geen rechtstreekse fonontoestandsdichtheid en bepaalt de koppeling aan een Tm³⁺-overgang niet. Ook zijn de deeltjes, doperingen en oppervlaktelaag van de bron niet automatisch gelijk aan een ander monster.

Een ongecorrigeerd Stokes-Ramanspectrum bevat bovendien al een thermische factor n(Ω, T_R) + 1 bij de meettemperatuur T_R. Voor een kwantitatieve reconstructie van een temperatuuronafhankelijk profiel moet die factor, naast eventuele instrument- en frequentiecorrecties, eerst worden verwijderd. Dit eenvoudige script doet die reconstructie niet: de vier componenthoogtes worden rechtstreeks als gekozen modelvorm gebruikt. De Bosefactoren hieronder bepalen vervolgens de temperatuurafhankelijkheid van het absorptiemodel. De figuur alleen onderbouwt die koppeling of haar absolute sterkte niet.

## Elektronische lijnen en temperatuur

Het script behoudt de aangeleverde Stark-energieniveaus en vijf kanalen:

| Soort | Overgang |
| --- | --- |
| GSA | ³H₆ → ³F₄ |
| GSA | ³H₆ → ³H₅ |
| ESA | ³F₄ → ³H₄ |
| ESA | ³F₄ → ³F₃ |
| ESA | ³F₄ → ³F₂ |

Voor een overgang van lager Stark-niveau i naar hoger niveau j is de elektronische resonantie νᵢⱼ = Eⱼ − Eᵢ, uitgedrukt in cm⁻¹. Het gewicht is pᵢrᵢⱼ. Hierin is pᵢ de Boltzmannkans binnen het onderste multiplet:

$$
p_i(T)=\frac{\exp[-(E_i-E_{\min})/(k_BT)]}
{\sum_m\exp[-(E_m-E_{\min})/(k_BT)]},
\qquad k_B=0.6950348005\ \mathrm{cm^{-1}K^{-1}}.
$$

De relatieve lijnsterktes rᵢⱼ zijn standaard gelijk over de hogere subniveaus, met rᵢⱼ = 1/N_upper. Bij precies 0 K gebruikt het script de limiet waarin alleen de laagste niveaus van het onderste multiplet bezet zijn. Het hogere multiplet krijgt geen extra Boltzmanngewicht. De ESA-vorm is daarmee voorwaardelijk gegeven dat het ion al in ³F₄ zit; dit script berekent de populatie van dat multiplet niet.

Alle vijf elektronische kanalen gebruiken een Gaussian lijnbreedte σ_el = 8.5 cm⁻¹. De eerdere effectieve breedte van 300 cm⁻¹ voor ³H₆ → ³H₅ wordt in deze fononversie vervangen door dezelfde smalle elektronische lijn plus de expliciete fononbijdragen. Beide achtergrondgewichten blijven nul.

De gemiddelde thermische fononbezetting van band k wordt bij het bandcentrum geëvalueerd:

$$
n_k(T)=\frac{1}{\exp[\Omega_k/(k_BT)]-1}.
$$

Fononcreatie heeft een factor nₖ + 1: ook bij nul temperatuur kan een fonon worden gemaakt. Fononabsorptie heeft een factor nₖ en verdwijnt als T naar nul gaat. Het gebruik van harmonische trillingen en Huang–Rhys-koppeling wordt bijvoorbeeld beschreven door [Betzholz, Torres en Bienert, Phys. Rev. A 90, 063818 (2014), secties II en IV](https://arxiv.org/html/1411.0864). De [officiële AbiPy-theorie, vergelijkingen 35–36](https://abinit.github.io/abipy_book/lumabi/theory/lesson_theory.html#multi-phonon-mode-model-and-generating-function), geeft de volledige eindige-temperatuur-genererende functie. Het onderstaande script gebruikt bewust alleen de eerste orde hiervan als eenvoudig model.

## Koppeling en genormaliseerde éénfononkernel

Elk elektronisch kanaal a heeft een eigen nominale dimensieloze koppeling Sₐ voor het volledige vierbandprofiel. De standaardwaarde is 0.05 voor alle vijf kanalen; dat is een illustratieve keuze die niet uit de figuur is gemeten. De afzonderlijke nominale bandkoppelingen zijn Sₐ,ₖ = Sₐwₖ; alleen ingeschakelde banden dragen bij. De werkelijk actieve som van de bandkoppelingen is dus Sₐ∑ₖeₖwₖ en kan kleiner zijn dan Sₐ. In de code geldt Sₐ = `--coupling` × `CHANNEL_COUPLING_MULTIPLIERS[a]`. De multipliers zijn standaard allemaal 1 en zijn per kanaal aanpasbaar; de waarde 0 schakelt de zijbanden van dat kanaal uit. Met de standaardmultipliers zet de opdrachtregeloptie `--coupling` alle vijf nominale koppelingen op dezelfde opgegeven waarde. De bandselectie geldt voor alle vijf kanalen.

Schrijf G(Δ; μ, σ) voor een Gaussian met centrum μ, standaardafwijking σ en oppervlakte één. De energieverschuiving Δ is het verschil tussen de geabsorbeerde fotonenergie en de elektronische resonantie, beide in cm⁻¹. De kernel is

$$
K_a(\Delta,T)=\frac{1}{Z_a}
\left[\delta(\Delta)+\sum_k e_kS_{a,k}\left\{
(n_k+1)G(\Delta;+\Omega_k,\sigma_k)
+n_kG(\Delta;-\Omega_k,\sigma_k)\right\}\right],
$$

met

$$
Z_a=1+\sum_ke_kS_{a,k}(2n_k+1).
$$

De deltapiek is het proces zonder fonon. Bij Δ ≈ +Ωₖ levert het foton extra energie om een fonon te maken. Bij Δ ≈ −Ωₖ levert het rooster via fononabsorptie een deel van de energie. In een absorptiespectrum ligt de creatiebijdrage dus aan de kant van hogere fotonenergie, oftewel kortere golflengte. De absorptiebijdrage ligt aan de kant van lagere fotonenergie, oftewel langere golflengte.

Omdat iedere Gaussian oppervlakte één heeft, geldt ∫Kₐ dΔ = 1. De gewichten zijn

$$
P_{0,a}=1/Z_a,\qquad
P_{+,a,k}=e_kS_{a,k}(n_k+1)/Z_a,\qquad
P_{-,a,k}=e_kS_{a,k}n_k/Z_a.
$$

Ze zijn niet-negatief en tellen op tot één. Voor Sₐ = 0 of wanneer alle banden uitstaan, blijft alleen de elektronische lijn over. Bij 300 K, Sₐ = 0.05 en alle vier banden aan heeft het model circa 7.3% geïntegreerd gewicht in de éénfononbijdragen; dit is een modeluitkomst, geen gemeten percentage.

Wanneer een band wordt uitgezet, wordt Zₐ opnieuw berekend. Het totale éénfonongewicht neemt daardoor af en het aandeel zonder fonon neemt toe. Hoewel de nominale koppelingen Sₐ,ₖ van de overgebleven banden gelijk blijven, veranderen hun uiteindelijke kansen P door de gewijzigde gemeenschappelijke noemer. Dat is de normalisatie van de toegestane processen en geen herverdeling van de verwijderde bandkoppeling. De latere pieknormalisatie kan de zichtbare relatieve hoogtes verder veranderen.

De factor Zₐ herverdeelt het gewicht over de toegestane processen. Dit is geen volledige Huang–Rhys-verdeling: bijdragen van twee of meer fononen ontbreken, ook bijvoorbeeld een gelijktijdige creatie en absorptie. De parameter Sₐ functioneert hier als koppeling op eerste orde. Voor kleine ΣeₖSₐ,ₖ(2nₖ + 1) sluit de kernel aan op de eersteorde-uitbreiding van het volledige model; bij sterke koppeling of een energiemismatch die meerdere fononen vereist, is deze benadering ontoereikend. Dat geldt bijvoorbeeld voor GSA bij 1064 nm met de aangeleverde Stark-niveaus: het verschil tot de elektronische GSA-lijnen is groter dan één van de gekozen roosterfononenergieën. De zeer kleine Gaussianstaart daar vervangt geen berekening van de ontbrekende meerfononprocessen.

Het evalueren van nₖ alleen bij Ωₖ is een effectieve-modusbenadering voor iedere band met eindige breedte. De verhouding van de geïntegreerde absorptie- en creatiegewichten van één band is nₖ/(nₖ + 1) = exp[−Ωₖ/(k_BT)]. Gedetailleerd evenwicht geldt zo bij de bandcentra. Een volledig frequentieafhankelijke verhouding binnen iedere verbrede band wordt niet afgedwongen. Daarvoor zou n(Ω) per fijne roosterbin moeten worden gebruikt.

## Convolutie en f-factoren

De kernel wordt geconvolveerd met iedere elektronische lijn. Omdat beide profielen Gaussian zijn, kan dat rechtstreeks: de zijbandcentra zijn νᵢⱼ ± Ωₖ en hun breedtes zijn

$$
\sigma_{\mathrm{side},k}=\sqrt{\sigma_{\mathrm{el}}^2+\sigma_k^2}.
$$

Voor één kanaal volgt daarmee

$$
\mathcal A_a(\nu)=\sum_{ij}p_ir_{ij}\left[
P_{0,a}G(\nu;\nu_{ij},\sigma_{\mathrm{el}})
+\sum_k\left\{P_{+,a,k}G(\nu;\nu_{ij}+\Omega_k,\sigma_{\mathrm{side},k})
+P_{-,a,k}G(\nu;\nu_{ij}-\Omega_k,\sigma_{\mathrm{side},k})\right\}\right].
$$

De geïntegreerde kanaalsterkte over golfgetal blijft zo gelijk wanneer alleen de koppeling verandert. De piekhoogte kan wel veranderen. Elektronische Stark-populaties en thermische fononbezettingen beschrijven verschillende vrijheidsgraden en worden ieder eenmaal verwerkt.

Na het combineren van de kanalen worden GSA en ESA elk afzonderlijk op hun globale spectrale maximum genormaliseerd:

$$
f_{\mathrm{GSA}}(\lambda)=\frac{\mathcal A_{\mathrm{GSA}}(10^7/\lambda)}
{\max_\nu\mathcal A_{\mathrm{GSA}}(\nu)},\qquad
f_{\mathrm{ESA}}(\lambda)=\frac{\mathcal A_{\mathrm{ESA}}(10^7/\lambda)}
{\max_\nu\mathcal A_{\mathrm{ESA}}(\nu)},
$$

waar λ in nm staat. Er is geen normalisatie op 1064 nm; de waarden daar volgen uit de berekening. Dit zijn relatieve absorptievormen, zonder fotonfluxfactor. De afzonderlijke pieknormalisatie levert geen absolute GSA/ESA-doorsnedes en geen absolute verhouding tussen GSA en ESA. Om een modeldoorsnede te maken is een afzonderlijk bepaalde amplitudeschaal nodig. Bij gebruik van een doorsnede die juist bij 1064 nm is gekalibreerd, moet die schaal consequent opnieuw worden geïnterpreteerd.

De horizontale as wordt alleen van golfgetal naar golflengte omgerekend. De f-factor is de waarde van een doorsnede bij die golflengte, geen kansdichtheid per nm; daarom wordt geen Jacobiaan toegevoegd. Gaussianstaarten worden niet fysisch afgeknipt. Extreem kleine staartwaarden kunnen in drijvendekommagetallen wel nul worden. De ondergrens van de logaritmische plot verandert de geëxporteerde gegevens niet.

## Pieken op één in dezelfde spectrafiguur

De standaardoptie `--plot-normalization peaks` zet de gedetecteerde lokale maxima van iedere volledige GSA- en ESA-curve op hoogte één. Dit gebeurt na de berekening en uitsluitend voor de getekende coördinaten. Er wordt geen extra GSA/ESA-figuur gemaakt: dezelfde spectrafiguur en dezelfde PNG/SVG-bestandsnamen worden gebruikt.

Voor GSA en ESA afzonderlijk worden de lokale maxima boven de relatieve plotondergrens in de bemonsterde golflengtescan gevonden. Numerieke schommelingen bij vrijwel nul tellen niet mee. Tussen twee naburige maxima zoekt het script het dal. Iedere dalende en stijgende flank krijgt een lineaire transformatie van de logaritmische intensiteit. De maxima eindigen op één en beide flanken ontmoeten elkaar op de gedeelde dalhoogte f_dal / sqrt(f_links f_rechts). Buiten de eerste en laatste gevonden piek gebruikt het script een constante deling door de betreffende randpiek. De weergegeven volledige curve is

$$
y_{\mathrm{plot}}(\lambda)=\frac{f(\lambda)}{E(\lambda)}.
$$

Hierin is E de positieve schaal die uit de flanktransformatie volgt. Op ieder gevonden maximum geldt E(λ) = f(λ), zodat die piek op één staat. Er wordt niets op één afgeknipt: de flanken blijven dalen en stijgen en krijgen geen kunstmatige vlakke bovenkant. Als een niet-nulcurve geen lokaal maximum binnen de scan heeft, wordt voor de weergave haar hoogste bemonsterde waarde gebruikt; een volledig nulcurve blijft nul.

Ook zwakke lokale maxima die door de fononbijdragen ontstaan, krijgen zo hoogte één. Dit zijn maxima van het samengestelde spectrum. Afzonderlijke Stark-overgangen die door overlap geen apart maximum vormen, worden niet los gedetecteerd of los op één gezet. Het resultaat hangt mede af van het getoonde scanbereik en de bemonstering.

De bijdrage zonder fononevent wordt met **dezelfde schaal van de volledige curve** gedeeld. Zij krijgt geen eigen normalisatie of afkapping. Omdat deze bijdrage nergens groter is dan de volledige berekende curve, blijft dat ook na dezelfde positieve deling gelden. Als alle fononbanden uitstaan, vallen beide curves samen.

Deze weergave maakt de piekstructuur zichtbaar, maar verwijdert informatie over de relatieve hoogtes van verschillende maxima. Omdat E(λ) over de golflengte varieert, veranderen ook dalhoogtes en schijnbare breedtes. De piekgenormaliseerde curve is daarom geen nieuwe absorptiedoorsnede, kansverdeling of gekalibreerde f-factor. Haar oppervlakte en hoogtes zijn niet geschikt om lijnsterktes of koppelingen te vergelijken. De y-as heet daarom *Piekgenormaliseerde weergave* en de figuur vermeldt dat de CSV de berekende f-factoren bevat.

Gebruik `--plot-normalization global` voor een kwantitatieve vergelijking van relatieve hoogtes en vormen binnen het berekende GSA- of ESA-spectrum, of gebruik de CSV. In die stand worden f(λ) en de gewogen bijdrage zonder fononevent rechtstreeks weergegeven op de modelnormalisatie met één globaal maximum per volledig spectrum. Die modelnormalisatie blijft GSA en ESA elk afzonderlijk schalen en levert dus nog steeds geen absolute GSA/ESA-verhouding.

## Uitvoeren en vergelijken

Voer onderstaande opdrachten uit vanuit de repositoryhoofdmap. NumPy is vereist; Matplotlib is alleen nodig voor de afbeeldingen.

```text
python fonon_model/gsa_esa_four_gaussian_phonons.py --coupling 0.05 --temperature 300 --no-show --output-dir fonon_model/results/four_gaussian_single_phonon
python fonon_model/gsa_esa_four_gaussian_phonons.py --no-phonons --temperature 300 --no-show --output-dir fonon_model/results/four_gaussian_no_phonon
python fonon_model/gsa_esa_four_gaussian_phonons.py --phonon-bands 1 3 --coupling 0.05 --temperature 300 --no-show --output-dir fonon_model/results/four_gaussian_bands_1_3
python fonon_model/gsa_esa_four_gaussian_phonons.py --plot-normalization global --coupling 0.05 --temperature 300 --no-show --output-dir fonon_model/results/four_gaussian_global
python fonon_model/gsa_esa_four_gaussian_phonons.py --coupling 0.05 --temperature 300 --data-only --output-dir fonon_model/results/four_gaussian_data
```

`--no-show` slaat de grafieken op zonder een venster te openen. `--data-only` maakt alleen tabellen en instellingen. Zonder `--output-dir` wordt `fonon_model/results/four_gaussians` gebruikt.

`--plot-normalization peaks` is de standaard voor de bestaande spectrafiguur; `--plot-normalization global` kiest de berekende f-factoren als verticale coördinaten. Beide opties leveren dezelfde CSV-waarden en modelkansen op bij gelijke overige instellingen. De gekozen weergavestand staat in het instellingenbestand. Wie dezelfde uitvoermap opnieuw gebruikt, overschrijft de vorige uitvoerbestanden met de nieuwe uitvoering.

De bandnummers 1 tot en met 4 staan in de parametertabel. `--phonon-bands 1 3` betekent dat uitsluitend die twee banden aanstaan; de overige staan uit. `--no-phonons` en `--phonon-bands` kunnen niet in dezelfde opdracht worden gebruikt. Zonder beide opties volgt het script de lijst `PHONON_BANDS_ENABLED` bovenin het bestand. Bijvoorbeeld `[True, False, True, False]` maakt banden 1 en 3 de standaardselectie; `[False, False, False, False]` maakt alle banden standaard inactief. Een expliciete opdrachtregeloptie overschrijft die lijst voor de betreffende uitvoering. `--coupling 0` schakelt de fononbijdragen eveneens uit, ongeacht de bandselectie.

De uitvoer bestaat uit:

- `f_gsa_f_esa_four_gaussian_single_phonon.csv`: berekende f-factoren met de modelnormalisatie, onafhankelijk van de gekozen grafische weergavestand.
- `f_gsa_f_esa_four_gaussian_single_phonon.png` en `.svg`: dezelfde GSA/ESA-figuur in de gekozen weergavestand `peaks` of `global`.
- `phonon_profile_four_gaussians.csv`, `.png` en `.svg`: de vier gekozen fononbanden en hun som.
- `four_gaussian_single_phonon_settings.json`: gebruikte parameters en normalisatie-instellingen.

In de spectrafiguur geven de stippellijnen alleen de gewogen bijdrage zonder fononevent weer. Zij delen de modelnormalisatie van het volledige spectrum en, bij `peaks`, ook de aanvullende grafische envelop. Ze zijn niet afzonderlijk op hun eigen maximum gezet en ook niet identiek aan een nieuwe berekening met `--coupling 0`. Wanneer alle fononbanden uitstaan, vallen de volledige curve en de bijdrage zonder fononevent samen.

De profielafbeelding toont ingeschakelde banden en de som van alleen die actieve banden. Uitgeschakelde banden worden voor de herkenbaarheid grijs gestippeld getoond en tellen niet mee in de som. Alle profielcurves worden gedeeld door het maximum van het volledige oorspronkelijke vierbandprofiel, ook als sommige banden uitstaan. Het verlies van een band blijft daardoor zichtbaar als verlies van profielhoogte; de actieve som wordt niet opnieuw naar één geschaald. Wanneer alle banden uitstaan, is die actieve som overal nul; dit wordt zonder deling door nul weergegeven. De profiel-CSV bewaart alle vier oorspronkelijke componenten in de kolommen `gaussian_1` tot en met `gaussian_4`; de kolom `sum` bevat uitsluitend hun actieve som en kan dus afwijken van de som van die vier referentiekolommen. De actieve selectie staat in het instellingenbestand.

Deze grafische herschaling van het profiel verandert de vaste oppervlaktegewichten wₖ niet. Het profiel toont de geselecteerde vorm onafhankelijk van `--coupling`: bij `--coupling 0` kunnen daarom wel geselecteerde profielbanden zichtbaar zijn terwijl hun bijdrage aan het absorptiespectrum nul is. `--plot-normalization` beïnvloedt alleen de GSA/ESA-figuur en verandert de profielafbeelding niet.

De vergelijking tussen `--no-phonons` en de standaardberekening is binnen ditzelfde model. Alle banden uit geeft exact dezelfde spectra als `--coupling 0`, bij dezelfde overige instellingen. Beide gebruiken σ_el = 8.5 cm⁻¹ voor alle elektronische lijnen. De nul-koppelingversie reproduceert daarom niet de eerdere elektronische variant met σ = 300 cm⁻¹ voor ³H₆ → ³H₅. Dat onderscheid is nodig om een verandering door expliciete fononen niet te verwarren met een verandering van de elektronische lijnbreedte.

## Numerieke controle van deze versie

De standaardberekening is uitgevoerd met 22 001 golflengten en met zowel CSV/JSON als PNG/SVG-uitvoer. De nul-koppelingberekening is daarnaast via `--data-only` bij 0 K uitgevoerd. De kansmassa en de numeriek geïntegreerde ruwe GSA- en ESA-spectra zijn één tot op ongeveer 10⁻¹⁵. Bij nul koppeling komen de ruwe spectra overeen met het oorspronkelijke elektronische rekenmodel wanneer daarin voor alle kanalen de smalle breedte van 8.5 cm⁻¹ wordt gebruikt. Een losse elektronische lijn krijgt uitsluitend de centra 0, ±253, ±306, ±364 en ±408 cm⁻¹ ten opzichte van haar resonantie.

Met de standaardinstellingen volgen f_GSA(1064 nm) ≈ 2.79 × 10⁻⁷² en f_ESA(1064 nm) ≈ 0.4873. De eerste waarde is slechts een verre Gaussianstaart, geen voorspelling van de werkelijk via meerdere fononen geassisteerde GSA. De figuren zijn visueel gecontroleerd. Dit zijn controles van de implementatie; de gekozen koppeling en vier Gaussiancomponenten zijn hiermee niet experimenteel gevalideerd.
