# GSA/ESA met het overgenomen Yb/Tm Raman-profiel

`gsa_esa_fononband.py` is de aangepaste versie van het door de gebruiker aangeleverde continue-band-script. Het standaardprofiel is geen Gaussian: het volgt bij benadering de **gestippelde NaYF₄(Yb/Tm)-curve in paneel A van Fig. 3** van Wilhelm et al. (2013), [doi:10.7150/thno.5113](https://doi.org/10.7150/thno.5113). Paneel B bevat ligandtrillingen en wordt niet gebruikt.

De punten zijn handmatig bij benadering afgelezen uit de aangeleverde afbeelding. Het betreft **Raman-intensiteit als DOS-/koppelingsproxy**, geen gemeten fonon-toestandsdichtheid en geen oorspronkelijke numerieke meetdata. `data/raman_ybtm_fig3a_approx.csv` bevat de reproduceerbare invoerpunten. Het zelfstandige script bevat dezelfde tabel zodat het ook zonder bijkomende bestanden werkt. Met `--band-file` kan een nauwkeuriger gedigitaliseerde of gecorrigeerde DOS worden geladen.

De kernel gebruikt standaard 46 effectieve modi: 225, 230, ..., 450 cm⁻¹. De intensiteit wordt lineair geïnterpoleerd en met trapeziumgewichten over dit bereik genormaliseerd. Het gewogen gemiddelde volgt uit het profiel en wordt niet meer naar 320 cm⁻¹ geforceerd. `BAND_MEAN_CM1` en `BAND_SIGMA_CM1` gelden alleen voor het optionele Gaussian-profiel. De elektronische breedtes, Stark-lijnen, temperatuur, 1064-nm-ankers en totale Huang–Rhys-factoren van het aangeleverde script zijn behouden; de hier aangeleverde koppelingen zijn **1e-6** voor alle vijf kanalen.

Uitvoeren vanaf de repository-hoofdmap (NumPy en Matplotlib):

```text
python spectra/gsa_esa_fononband.py --profile-only --no-show --output-dir spectra/results/raman_profile
python spectra/gsa_esa_fononband.py --no-show --output-dir spectra/results/raman_spectra
python spectra/gsa_esa_fononband.py --data-only --output-dir spectra/results/raman_spectra
```

De eerste opdracht maakt alleen `phonon_band_profile.png`, `.svg` en `.csv`, zonder spectrale berekening. De tweede berekent zowel GSA als ESA met fononen en schrijft ook de profielplot. De profielplot toont de genormaliseerde curve en de knopen die de kernel gebruikt. De factorplots bevatten daarnaast de nul-fononbijdragen. CSV/JSON-export werkt zonder Matplotlib via `--data-only`.

Andere profielen blijven beschikbaar:

```text
python spectra/gsa_esa_fononband.py --band-profile gaussian --profile-only --no-show
python spectra/gsa_esa_fononband.py --band-profile uniform --profile-only --no-show
python spectra/gsa_esa_fononband.py --band-file spectra/data/raman_ybtm_fig3a_approx.csv --profile-only --no-show
python spectra/gsa_esa_fononband.py --phonon-model discrete --data-only
```

Een stap kleiner dan 5 cm⁻¹ vereist ook aanpassing van het energieverschuivingsrooster. Fijner samplen dan de afbeelding ondersteunt voegt geen experimentele informatie toe.

