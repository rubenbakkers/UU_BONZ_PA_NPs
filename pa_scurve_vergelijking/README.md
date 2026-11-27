# Measured S-curves vs. the phonon-band model

Compares measured S-curves with the population model in [`fonon_model`](../fonon_model/README.md). The model itself is not in this folder: the scripts import `pa_model.py` and `pa_fononband.py` from `../fonon_model` (in the repo) or `../Script` (local BONZ folder). The measurements are listed in [`measurements.py`](measurements.py); at present the S-curves of 24-11-2026 (BdJ1; Cobolt at 1060 nm and SuperK at 1064 nm).

## Adding a measurement

All measurements are listed in [`measurements.py`](measurements.py), and every script loops over that list, so a new S-curve needs no changes anywhere else.

1. Put the CSV in `data/` (columns `fw6, I_kWcm2, net_cps, noise_cps`, as in the `scurve.csv` export; other columns are ignored).
2. In `measurements.py`, copy the template at the bottom of `MEASUREMENTS` and fill in at least `file`, `label` and `wavelength_nm`, preferably also `sample` and `date`.
3. Run `run_all.py`, or `make_figures.py` first and then the scripts you need: `make_figures.py` writes `results/summary.json`, which the other scripts read.

`ACTIVE = [...]` runs only some of the measurements; `REFERENCE` is the measurement that fig. 2 normalises the thresholds to. The optional fields (laser linewidth, the sweep through the knee, two sweeps to compare, an emission spectrum) are explained at the top of `measurements.py`; a typo in a field name or a missing file stops the scripts with a message. The file is read again on every run, so editing it does not need a kernel restart. The S-curve figures get one panel per measurement (at most 3 per row). The run time grows about linearly with the number of measurements, mainly in `scan_shape.py` and `fit_all_params.py`.

## Running

Use Python 3.10 or newer with NumPy, SciPy and Matplotlib (`python -m pip install -r requirements.txt` in the repository root). From this folder:

```text
python run_all.py        # whole comparison, ~4 min + ~6 min for fit_all_params.py (FAST = True: without the parameter scan)
python fit_all_params.py # only the fit with all model parameters free, ~6 min
python fit_s_lee.py      # only the slope s of the measurements, ~5 s, without the model
```

Figures and JSON results go to `results/` (not in git); results per measurement are stored under its name in `measurements.py` (e.g. `cobolt_1060`). The fitting scripts print the parameters they used for the (best) fit: all model parameters, the GSA/ESA rates at the pump wavelength, and the fitted k, A and background per filter-wheel setting (`make_figures.py` at the end of its output, `fit_all_params.py` after each fit, `scan_shape.py` for the best combination). You can run single scripts, but run `make_figures.py` first: it writes `results/summary.json`, which the other scripts read. If you rerun in Spyder or a Jupyter console after changing the model or `pa_fit_lib.py`, restart the kernel first.

## Which model version?

`pa_fit_lib.py` looks for `pa_model.py` + `pa_fononband.py` in this order and reports at start-up which folder and which Huang–Rhys factor S(³H₆→³H₅) it uses (also stored in `results/summary.json` under `model_info`):

1. the environment variable `PA_MODEL_DIR`;
2. `../Script` (the local BONZ folder);
3. `../fonon_model` (this repository);
4. `model/` (old separate copy, only if nothing else exists).

The scripts use internal functions (`_stationary_population`, `_make_rhs`, `_lineshape`, `_stark_sigma`, `rise_time_95_fast`), so a very different model version can break them. This folder comes before the model folder on `sys.path`, so the older copies of these scripts in `../Script` (`fit_scurves.py`, `emission_spec.py`, ...) are never imported instead of the ones here.

## What happens

Shared basis, [`pa_fit_lib.py`](pa_fit_lib.py): GSA/ESA rates from the phonon-band spectra, the steady state starting from the ground state (as in an upward sweep), and the 800-nm emission. Because the focus contains many particles that each see a different intensity, the emission is summed over the measured spot profile (elliptical Gaussian core + round halo).

Fit model ([`fit_scurves.py`](fit_scurves.py)): `counts = A · M(k · I_reported) + b_FW`. `k` scales the intensity and absorbs both a calibration error and differences between these particles and those of Lee et al.; `A` and one background per filter-wheel setting follow for each `k` from a weighted least-squares fit. A good χ²ᵣ therefore means that the shape of the S-curve agrees, not the absolute threshold.

| Step ([`run_all.py`](run_all.py)) | What | Output |
|---|---|---|
| [`make_figures.py`](make_figures.py) | Fit of every S-curve (Lee SI sample 5: 8 % Tm, no non-radiative loss); I_th(λ) relative to the `REFERENCE` measurement | `fig1_scurves_vs_model.png`, `fig2_threshold_vs_wavelength.png`, `summary.json` |
| [`scan_shape.py`](scan_shape.py) | Scan of σ_GSA (×0.1–10), Tm concentration (4–20 %) and non-radiative loss: does the knee fit better? | `scan_shape.json` |
| [`rise_variants.py`](rise_variants.py) | 95 % rise time for three explanations of `k`: calibration, σ_ESA × k, w₂ / k (per measurement) | `rise_variants.json` |
| [`protocol_sim.py`](protocol_sim.py) | Measurement protocol simulated (2 s per point, no settling time): how much too low is the average near the knee? For the sweep through the knee of every measurement (`knee_fw`, chosen automatically) and for `compare_sweeps` | `protocol_sim.json` |
| [`variants_fit.py`](variants_fit.py) | Refit with σ_ESA × k and w₂ / k, and the 2-s protocol again | `variants_fit.json` |
| [`fit_all_params.py`](fit_all_params.py) | Fit with all model parameters free at once (see below) | `fig4_all_params_fit.png`, `fig5_all_params_values.png`, `all_params_fit.json` |
| [`fit_concentration.py`](fit_concentration.py) | Asks the Tm concentration of every sample (blank = fit it); s₃₁, Q₂₂, Q₂₃ follow from it, σ_GSA, w₂, w₃ and b₃₂ are fitted (see below) | `fig6_concentration_fit.png`, `fig7_concentration_values.png`, `concentration_fit.json` |
| [`emission_spec.py`](emission_spec.py) | Emission spectra (every measurement with a `spectrum_file`) vs. the ³H₄→³H₆ Stark lines; fit of shift, width, amplitude and background | `emission_spec.json` |
| [`make_figures2.py`](make_figures2.py) | Fig. 3: emission spectrum (first measurement with a `spectrum_file`, fit of `emission_spec.py`), rise time (`REFERENCE`), measurement protocol (all measurements), two sweeps of one measurement (`compare_sweeps`; now the Cobolt) | `fig3_other_script_parts.png` |

### Fit with all model parameters free

`make_figures.py` fits only `k`, which scales σ_GSA and σ_ESA together. [`fit_all_params.py`](fit_all_params.py) also fits multipliers on Lee's values for σ_GSA (the GSA/ESA ratio), w₂, s₃₁, Q₂₂, Q₂₃ and b₃₂, at the same time, with `scipy.optimize.least_squares` from several starting points. w₃ and σ_ESA stay fixed: the steady state does not change when all rates and `k` are scaled together, so without a fixed rate and cross section the fit is degenerate. Three fits: `k` only (as `make_figures.py`), all parameters shared by the curves of the same sample (`sample` in `measurements.py`: same particles; `k`, `A`, `b_FW` per curve; only for samples with two or more measurements), and all parameters per curve. A parameter counts as determined when both its 1σ (Jacobian) and its spread over starting points that fit equally well are below a factor 3; the others are open markers in fig. 5. Choose the free parameters with `FREE_PARAMS`.

[`fit_concentration.py`](fit_concentration.py) answers a narrower question: if the Tm concentration of the particles is known, which values of the *other* parameters fit best? At the start it asks the concentration of every sample in `measurements.py` (e.g. `8`, `8.5` or `8,5`). In the model the concentration sets only s₃₁, Q₂₂ and Q₂₃ (`pa_model.transfer_coefficients`); these follow from the given value, and σ_GSA, w₂, w₃ and b₃₂ are fitted with the same machinery (w₃ may be free here, because s₃₁, Q₂₂ and Q₂₃ are tied to the concentration, which removes the degeneracy). Leave the answer blank (or type `none`) and the concentration is fitted as well, between 4 and 100 %: at 4 % Lee's relations jump to the coefficients of the 4 % sample, and below 4 % they are an extrapolation. Without a console (no input possible) every concentration is fitted. The printout compares χ² and BIC with `fit_all_params.py` and gives the concentration and the fitted lifetimes τ(³F₄) = 1/w₂ and τ(³H₄) = 1/w₃ next to Lee's; in fig. 7 a given concentration is a ×.

The fit uses `emission_curve_fast` from `pa_fit_lib.py`: the same steady state, solved algebraically (with 2F₁ + F₂ the cross-relaxation term drops out and N₂ is linear in N₁, which leaves a quartic in N₁). It is ~250× faster than LSODA and agrees to ~1e-13 (checked at the end of every fit); only where several physical steady states exist (bistability) does it fall back to LSODA.

### Slope s of the measurements

[`fit_s_lee.py`](fit_s_lee.py) is not part of `run_all.py` (run `make_figures.py` first for the background per filter-wheel setting). It uses only points above 3× the noise and determines s in two ways; the figure `fig_s_fit_lee.pdf/.png` shows the data with both fitted lines, and `s_fit_lee.json` holds the numbers.

1. **As in Lee et al. (2021)** (orange, s_Lee): s is defined through I_em ∝ I_pump^s (slope in log-log) and the threshold is the intersection of straight lines through the points below and above the threshold (SI, Fig. S3). The script splits the points into 2 or 3 straight segments in log-log (breakpoints at minimal χ², the BIC chooses 2 or 3) and takes s = slope of the steepest segment.
2. **As in `pa_fononband`** (black, s_fit): the model's own `fit_avalanche_line` with `SLOPE_FRACTION` = 0.25, i.e. a straight line through the contiguous window where the secant slope is ≥ 0.25 × its maximum s_peak — the black lines of the `pa_fononband` detuning figure. Secant slopes between single noisy points are meaningless, so the data are first averaged in bins of 0.05 decades of intensity (grey squares).

The uncertainty of both comes from bootstrap resampling. Below the knee the signal of these measurements is in the noise, so I_th as in Lee cannot be determined and the script says so. Because the curve bends gradually, s depends on the fit window: method 1 uses the steepest stretch, method 2 averages over the wide window above 0.25 × s_peak.

## Data (`data/`)

The measurement data are **not** in the public GitHub repository (`data/` is in `.gitignore`). Put the files named in `measurements.py` in `data/` yourself (for the current measurements the three below); without them the scripts stop with a message naming the missing file.

| File | Contents |
|---|---|
| `cobolt_1060_scurve.csv`, `superk_1064_scurve.csv` | Columns from `scurve.csv` on the Drive (5 Metingen S-curve), sorted by intensity: `fw6, gray, I_kWcm2, net_cps, noise_cps, net_cps_gauss` |
| `emission_check_spectrum_superk_785_862nm.csv` | Part of `emission_check_spectrum.csv` (BdJ1, SuperK) around the emission band: `wavelength_nm, cts_per_s_per_pixel` |

## Points to keep in mind

- The SuperK is treated as monochromatic (`fwhm_nm=0.0` in `measurements.py`); with `fwhm_nm > 0` the rates are averaged over a Gaussian laser spectrum.
- The background per filter-wheel setting in `summary.json` comes from the model fit; `fit_s_lee.py` subtracts it by default (`BACKGROUND = "summary"`).
- The measured emission peak lies ~20 nm (~300 cm⁻¹) to the red of the model with the "Table 4" Stark levels. Whether this is due to the spectrometer calibration or to the levels is still open.
