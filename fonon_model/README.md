# PA model with a phonon band

This folder couples a continuous phonon band to the GSA/ESA absorption and photon-avalanche kinetics of Tm³⁺:NaYF₄. You can export only the spectra, or also compute emission, thresholds and 95% rise times. It is the more extensive phonon route next to the [standalone spectrum scripts](../spectra/README.md) and the [other PA simulations](../pa_simulations/README.md).

## Files and how they fit together

| File | Purpose |
|---|---|
| [pa_fononband.py](pa_fononband.py) | Builds the electronic and phonon spectra, exports the factors and runs the kinetic analyses |
| [1 fonon zijband.py](<1 fonon zijband.py>) | Standalone GSA/ESA spectra with the same four Gaussian phonon bands, evaluated analytically with up to N phonons per transition |
| [pa_model.py](pa_model.py) | Local model module with the SI parameters, population equations, RK4 and emission/analysis functions |
| [requirements.txt](requirements.txt) | NumPy, Matplotlib and SciPy for the full calculation |

The local `pa_model.py` stays next to `pa_fononband.py`, so the command-line calculation works on its own in this folder. This copy uses the same model as `pa_simulations/pa_model.py`; at the reorganization they differed only in a trailing empty line. The standalone phonon folder therefore does not need to add the other script folder to its import path.

The script for the discrete GSA phonon factors now lives in `spectra/gsa_esa_phonon_factors.py`. The previously saved `f_gsa_f_esa_fononzijbanden.*` files were moved to [spectra/reference_results](../spectra/reference_results/README.md). They belong to that standalone discrete model, not to the default continuous phonon band in this folder.

## Running

Run these examples from the repository root with Python 3.10 or newer. Install the packages:

```text
python -m pip install -r fonon_model/requirements.txt
```

Factor CSVs and phonon settings only, without plots or kinetics:

```text
python fonon_model/pa_fononband.py --data-only --output-dir fonon_model/results
```

Absorption plots, CSVs and settings:

```text
python fonon_model/pa_fononband.py --spectra-only --no-show --output-dir fonon_model/results
```

Full calculation, including the wavelength scan, rise times and width sweeps:

```text
python fonon_model/pa_fononband.py --no-show --output-dir fonon_model/results
```

You can choose which parts a full run makes with `--plots` (only these) or `--skip-plots` (all except these), or with the `PLOTS` dictionary at the top of the script:

| Name | Part | Run time (indication) |
|---|---|---|
| `spectra` | Absorption factors f_gsa/f_esa (900–1500 nm and zoom), linear and log y-axis | ~2 s |
| `detuning` | Emission vs intensity per pump wavelength | ~15 s, together with `smax` and `ith` |
| `smax` | Slope of the avalanche segment vs wavelength | uses the detuning calculation |
| `ith` | Threshold vs wavelength | uses the detuning calculation |
| `rise` | 95% rise time vs intensity | ~3 s |
| `sigma` | Peak rise time vs electronic width | ~2.5 min, by far the slowest |

```text
python fonon_model/pa_fononband.py --no-show --skip-plots sigma --output-dir fonon_model/results
python fonon_model/pa_fononband.py --no-show --plots ith rise --output-dir fonon_model/results
```

`smax` and `ith` need the detuning calculation; it also runs when `detuning` is off, just without the detuning figure. When all kinetic parts are off, only the spectra are computed. The run time of each part is printed.

Without `--output-dir` the results are written next to the script. `--no-show` saves the plots and suppresses windows. For the default phonon model `--data-only` needs only NumPy and can export without `pa_model.py` or Matplotlib. Plots require Matplotlib. The full population analysis requires the local model module and SciPy; the fast steady-state solver uses LSODA and the rise-time finder uses `brentq`.

A full run contains many integrations. The `sigma` sweep rebuilds the spectra for ten electronic widths and two pump wavelengths and takes most of the time; skip it with `--skip-plots sigma`. Start with `--data-only` or `--spectra-only` when you only want to compare absorption.

## How the model works

The default is `ABSORPTION_MODEL = "phonon_band"`. GSA contains ³H₆ → ³F₄/³H₅; ESA contains ³F₄ → ³H₄/³F₃/³F₂. **All five channels** use Boltzmann-weighted Stark lines, an electronic Gaussian width of 8.5 cm⁻¹ and finite-temperature Huang–Rhys sidebands. The phonon contribution replaces the effective H5 GSA width of 300 cm⁻¹; that width is not applied a second time to the phonon spectrum.

The band is approximated with weighted integration nodes. By default (`--band-profile four_gaussian`) the profile is the sum of four Gaussians that approximate the dotted Yb/Tm Raman curve of Wilhelm et al. (2013, Fig. 3A, [doi:10.7150/thno.5113](https://doi.org/10.7150/thno.5113)), with the same parameters as `PHONON_*_CM1` in `1 fonon zijband.py`:

| Component | Centre (cm⁻¹) | σ (cm⁻¹) | Height |
|---|---:|---:|---:|
| 1 | 250 | 14 | 0.68 |
| 2 | 306 | 24 | 0.95 |
| 3 | 364 | 15 | 0.66 |
| 4 | 408 | 24 | 0.36 |

The profile is sampled on 5 cm⁻¹ nodes between 200 and 480 cm⁻¹ (57 nodes, weighted mean ≈ 324 cm⁻¹). Each node is a sharp mode; the sideband width comes from the spread over the nodes plus the electronic width. The old single Gaussian band (225–450 cm⁻¹, weighted mean 320 cm⁻¹, width 45 cm⁻¹) remains available with `--band-profile gaussian --band-min 225 --band-max 450 --band-step 15`. The Huang–Rhys factor is 0.05 per channel by default. With `--phonon-model discrete` the three modes of 250, 305 and 365 cm⁻¹ are used. With that choice the ESA channels here still keep phonon coupling by default; this differs from the standalone GSA phonon script.

The relative absorption factors are:

```text
f_i(lambda) = A_i(lambda) / A_i(1064 nm)
sigma_i(lambda) = sigma_i(1064 nm) * f_i(lambda)
w_i(lambda) = sigma_i(lambda) * photon_flux_per_intensity(lambda)
```

`f_gsa` and `f_esa` already contain the full Stark and phonon contribution. Both are 1 at 1064 nm and are applied once in the pump rates. The population equations use the total ground and excited populations: `R_gsa*N0` and `R_esa*N1`. No second Boltzmann, Stark or phonon factor belongs there. The three populations are ³H₆, ³F₄ and ³H₄; the higher excited multiplets are handled by the existing three-population approximation.

The components without phonon events are tracked with the same normalization as the full spectrum. That is the contribution without any phonon event, not automatically all processes with zero net energy shift. No absorption floor is added.

### Channel weights (Judd–Ofelt)

The relative strength of the five channels follows from Judd–Ofelt instead of a fixed weight of 1:

```text
S_c = Omega_2 [U(2)]^2 + Omega_4 [U(4)]^2 + Omega_6 [U(6)]^2
w_c = S_c * nu_c / sum(S_c' * nu_c')   (sum over the channels of the same spectrum)
```

with Ω₂, Ω₄, Ω₆ = 2.37, 3.05, 0.41 × 10⁻²⁰ cm² (SI Table S4 of Lee et al. 2021) and ν_c the gap between the multiplet barycentres. The matrix elements come from SI Table S5; those for ³F₄ → ³F₃ and ³F₄ → ³F₂ are missing there and come from the same source, Carnall, Crosswhite & Crosswhite (1978, ANL-78-XX-95), Appendix XIII Table 2.

| Channel | S (10⁻²⁰ cm²) | Share |
|---|---:|---:|
| ³H₆ → ³F₄ | 3.593 | 0.666 (GSA) |
| ³H₆ → ³H₅ | 1.222 | 0.334 (GSA) |
| ³F₄ → ³H₄ | 0.789 | 0.376 (ESA) |
| ³F₄ → ³F₃ | 0.077 | 0.045 (ESA) |
| ³F₄ → ³F₂ | 0.907 | 0.579 (ESA) |

Because of the 1064-nm anchor only the ratios within one spectrum matter. `USE_JUDD_OFELT_WEIGHTS = False` at the top of the script restores the old weights of 1.0. The matrix elements are for Tm³⁺:LaF₃ (intermediate coupling) and are an approximation for NaYF₄.

## Settings and custom band files

Temperature and electronic width are set at the top with `TEMPERATURE_K` and `STARK_SIGMA_CM1`. The temperature has no command-line option in this script. For the phonon route `LORENTZ_HWHM_CM1 = 0` must stay; a nonzero Lorentz width is rejected.

| Option | Meaning |
|---|---|
| `--phonon-model band` or `discrete` | Continuous band approximation or three discrete modes |
| `--band-min`, `--band-max`, `--band-step` | Band limits and node spacing in cm⁻¹ |
| `--band-profile four_gaussian`, `gaussian`, `uniform` or `file` | Shape of the band; default `four_gaussian` |
| `--band-mean`, `--band-sigma` | Only for `gaussian`: weighted mean and assumed profile width in cm⁻¹ |
| `--band-file` | Custom CSV with a band profile; selects the file route |
| `--coupling-h5`, `--coupling-f4` | Total couplings for the two GSA channels |
| `--coupling-h4`, `--coupling-f3`, `--coupling-f2` | Total couplings for the three ESA channels |
| `--kernel-tolerance` | Limit on the omitted kernel mass; default 10⁻³⁵ |

Examples:

```text
python fonon_model/pa_fononband.py --spectra-only --no-show --coupling-h5 0.08 --coupling-h4 0.10 --output-dir fonon_model/results/coupling
python fonon_model/pa_fononband.py --data-only --phonon-model discrete --output-dir fonon_model/results/discrete
python fonon_model/pa_fononband.py --spectra-only --no-show --band-file my_band.csv --output-dir fonon_model/results/my_band
```

A custom file contains at least two data rows and two comma-separated columns: wavenumber in cm⁻¹ and intensity. A header line is allowed. For example:

```csv
wavenumber_cm1,intensity
225,0.1
300,1.0
375,0.7
450,0.1
```

This example file is not included; save it yourself as `my_band.csv` for the third command. Intensities are interpolated onto the band nodes and the weights are normalized. Node positions must fit on the 5 cm⁻¹ shift lattice and the band width must contain a whole number of node steps. The Gaussian setting controls the weighted mean of the truncated band, not only the unweighted profile centre.

The older absorption modes `broad_gsa`, `stark`, `si_1064` and `hybrid` remain available through the constant `ABSORPTION_MODEL` in the script. There is no command-line option for it. Only `broad_gsa` still uses the effective H5 width of 300 cm⁻¹. With the default narrow GSA lines the current `hybrid` model cannot form its 1064-nm anchor reliably; that combination is rejected.

## Output

With the default model the following files are written:

| File base | Contents |
|---|---|
| `f_gsa_f_esa_900_1500nm_phonon_band` | Factors over 900–1500 nm: `.csv`, and with plots `_linear.png/.svg` and `_log.png/.svg` |
| `f_gsa_f_esa_1040_1100nm_phonon_band` | Zoom around 1064 nm, same formats |
| `phonon_band.csv` | Band nodes, mode fractions and coupling per node and channel |
| `phonon_settings.json` | Model settings, kernel diagnostics and accuracy bounds |

The spectrum is always plotted twice with the same data: once with a linear and once with a logarithmic y-axis. On the linear axis the ³H₆ → ³H₅ GSA peak (f ≈ 5.6·10⁴ after the 1064-nm normalization) dominates; the log axis shows the weak phonon tails and the ESA structure.

The factor CSVs have five columns:

```text
wavelength_nm,f_gsa,f_esa,f_gsa_zero_phonon,f_esa_zero_phonon
```

The last two columns give the contribution without phonon events, with the same full-spectrum normalization. Before October 2026 these files were called `fononband.csv` and `fonon_instellingen.json`, and the columns `f_gsa_zonder_fononen` and `f_esa_zonder_fononen`. A full kinetic run also makes:

- `pa_wavelength_detuning_phonon_band.pdf`: emission vs pump intensity per wavelength;
- `pa_smax_vs_wavelength_phonon_band.pdf`: slope of the fitted avalanche segment;
- `pa_ith_vs_wavelength_phonon_band.pdf`: numerical knee and analytical threshold;
- `pa_rise_time_vs_intensity_phonon_band.pdf`: 95% rise time vs intensity, when valid rise-time data are found;
- `pa_rise_peak_vs_sigma_phonon_band.pdf`: maximum rise time vs electronic width.

Use separate output folders for parameter comparisons: new runs reuse the same file names. With the default width sweep the 1064-nm anchor keeps the pump rates at that wavelength unchanged.

## Assumptions and checks

Band shape, couplings, electronic widths and relative Stark line strengths are uncalibrated model assumptions. An anchor at 1064 nm is no independent calibration of all other wavelengths. The long-wavelength analysis is exploratory. The band approximation does not describe a full measured phonon DOS.

The code checks the reference factors, default stick weights, conservation of population totals, agreement between the fast and the shared population equations and the bounded omitted kernel mass. The steady-state solver checks both time stability and the residual of the population equations. These built-in checks do not replace a comparison with measured data.

### Last full run (October 2026, first band centre 250 cm⁻¹)

The full default run (`--no-show`, about 3 minutes) passed all built-in checks. Findings:

- f_GSA(1064 nm) = f_ESA(1064 nm) = 1; σ_GSA = 6.0·10⁻²⁹ m² and σ_ESA = 6.4·10⁻²⁵ m² at 1064 nm.
- The spectra agree with `1 fonon zijband.py` (N = 6 and N = 10) within 1.4% (GSA) and 2.4% (ESA) wherever f exceeds 10⁻³ of its maximum, and within about 7% down to 10⁻⁶ of the maximum.
- Moving the first band centre from 253 to 250 cm⁻¹ changes the factors by at most about 2%. At 1064 nm the threshold (24.17 kW cm⁻², numerical and analytical), the fitted slope (42.5) and the rise time are unchanged because of the anchor; elsewhere Ith shifts by at most 0.05 kW cm⁻².
- Numerical and analytical thresholds agree within 0.3%. Between 1400 and 1470 nm the lowest threshold is at 1450 nm (20.8 kW cm⁻²), as in Lee et al. (2021); the measured value there is about 40 kW cm⁻². At 1064 nm Lee et al. measured about 20 kW cm⁻².
- The peak rise times in the `sigma` sweep are limited by its coarse intensity grid (14 + 9 points). At 1450 nm and σ = 16 cm⁻¹ the sweep gives 487 ms, whereas a finer grid gives 693 ms; at 1064 nm the main rise-time scan gives 437 ms and the sweep 367 ms for identical pump rates. Treat the peak heights in `pa_rise_peak_vs_sigma_phonon_band.pdf` as indicative.
