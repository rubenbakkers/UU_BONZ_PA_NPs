# Measured S-curves vs. the phonon-band model

Compares the S-curves of 24-11-2026 (BdJ1; Cobolt at 1060 nm and SuperK at 1064 nm) with the population model in [`fonon_model`](../fonon_model/README.md). The model itself is not in this folder: the scripts import `pa_model.py` and `pa_fononband.py` from `../fonon_model` (in the repo) or `../Script` (local BONZ folder).

## Running

Use Python 3.10 or newer with NumPy, SciPy and Matplotlib (`python -m pip install -r requirements.txt` in the repository root). From this folder:

```text
python run_all.py        # whole comparison, ~4 min (FAST = True: ~2 min, without the parameter scan)
python fit_s_lee.py      # only the slope s of the measurements, ~5 s, without the model
```

Figures and JSON results go to `results/` (not in git). You can run single scripts, but run `make_figures.py` first: it writes `results/summary.json`, which the other scripts read. If you rerun in Spyder or a Jupyter console after changing the model or `pa_fit_lib.py`, restart the kernel first.

## Which model version?

`pa_fit_lib.py` looks for `pa_model.py` + `pa_fononband.py` in this order and reports at start-up which folder and which Huang–Rhys factor S(³H₆→³H₅) it uses (also stored in `results/summary.json` under `model_info`):

1. the environment variable `PA_MODEL_DIR`;
2. `../Script` (the local BONZ folder);
3. `../fonon_model` (this repository);
4. `model/` (old separate copy, only if nothing else exists).

The scripts use internal functions (`_stationary_population`, `_make_rhs`, `_lineshape`, `_stark_sigma`, `rise_time_95_fast`), so a very different model version can break them.

## What happens

Shared basis, [`pa_fit_lib.py`](pa_fit_lib.py): GSA/ESA rates from the phonon-band spectra, the steady state starting from the ground state (as in an upward sweep), and the 800-nm emission. Because the focus contains many particles that each see a different intensity, the emission is summed over the measured spot profile (elliptical Gaussian core + round halo).

Fit model ([`fit_scurves.py`](fit_scurves.py)): `counts = A · M(k · I_reported) + b_FW`. `k` scales the intensity and absorbs both a calibration error and differences between these particles and those of Lee et al.; `A` and one background per filter-wheel setting follow for each `k` from a weighted least-squares fit. A good χ²ᵣ therefore means that the shape of the S-curve agrees, not the absolute threshold.

| Step ([`run_all.py`](run_all.py)) | What | Output |
|---|---|---|
| [`make_figures.py`](make_figures.py) | Fit of both S-curves (Lee SI sample 5: 8 % Tm, no non-radiative loss); I_th(λ)/I_th(1060 nm) | `fig1_scurves_vs_model.png`, `fig2_threshold_vs_wavelength.png`, `summary.json` |
| [`scan_shape.py`](scan_shape.py) | Scan of σ_GSA (×0.1–10), Tm concentration (4–20 %) and non-radiative loss: does the knee fit better? | `scan_shape.json` |
| [`rise_variants.py`](rise_variants.py) | 95 % rise time for three explanations of `k`: calibration, σ_ESA × k, w₂ / k | `rise_variants.json` |
| [`protocol_sim.py`](protocol_sim.py) | Measurement protocol simulated (2 s per point, no settling time): how much too low is the average near the knee? | `protocol_sim.json` |
| [`variants_fit.py`](variants_fit.py) | Refit with σ_ESA × k and w₂ / k, and the 2-s protocol again | `variants_fit.json` |
| [`emission_spec.py`](emission_spec.py) | Emission spectrum (785–862 nm) vs. the ³H₄→³H₆ Stark lines; fit of shift, width, amplitude and background | `emission_spec.json` |
| [`make_figures2.py`](make_figures2.py) | Fig. 3: emission spectrum (with the fit of `emission_spec.py`), rise time, measurement protocol, the two Cobolt sweeps | `fig3_other_script_parts.png` |

### Slope s of the measurements

[`fit_s_lee.py`](fit_s_lee.py) is not part of `run_all.py` (run `make_figures.py` first for the background per filter-wheel setting). It uses only points above 3× the noise and determines s in two ways; the figure `fig_s_fit_lee.pdf/.png` shows the data with both fitted lines, and `s_fit_lee.json` holds the numbers.

1. **As in Lee et al. (2021)** (orange, s_Lee): s is defined through I_em ∝ I_pump^s (slope in log-log) and the threshold is the intersection of straight lines through the points below and above the threshold (SI, Fig. S3). The script splits the points into 2 or 3 straight segments in log-log (breakpoints at minimal χ², the BIC chooses 2 or 3) and takes s = slope of the steepest segment.
2. **As in `pa_fononband`** (black, s_fit): the model's own `fit_avalanche_line` with `SLOPE_FRACTION` = 0.25, i.e. a straight line through the contiguous window where the secant slope is ≥ 0.25 × its maximum s_peak — the black lines of the `pa_fononband` detuning figure. Secant slopes between single noisy points are meaningless, so the data are first averaged in bins of 0.05 decades of intensity (grey squares).

The uncertainty of both comes from bootstrap resampling. Below the knee the signal of these measurements is in the noise, so I_th as in Lee cannot be determined and the script says so. Because the curve bends gradually, s depends on the fit window: method 1 uses the steepest stretch, method 2 averages over the wide window above 0.25 × s_peak.

## Data (`data/`)

The measurement data are **not** in the public GitHub repository (`data/` is in `.gitignore`). Put these three files in `data/` yourself; without them the scripts stop with a message naming the missing file.

| File | Contents |
|---|---|
| `cobolt_1060_scurve.csv`, `superk_1064_scurve.csv` | Columns from `scurve.csv` on the Drive (5 Metingen S-curve), sorted by intensity: `fw6, gray, I_kWcm2, net_cps, noise_cps, net_cps_gauss` |
| `emission_check_spectrum_superk_785_862nm.csv` | Part of `emission_check_spectrum.csv` (BdJ1, SuperK) around the emission band: `wavelength_nm, cts_per_s_per_pixel` |

## Points to keep in mind

- The SuperK is treated as monochromatic (`fwhm=0.0` in `fit_scurves.py`); `rates()` can average over a laser spectrum.
- The background per filter-wheel setting in `summary.json` comes from the model fit; `fit_s_lee.py` subtracts it by default (`BACKGROUND = "summary"`).
- The measured emission peak lies ~20 nm (~300 cm⁻¹) to the red of the model with the "Table 4" Stark levels. Whether this is due to the spectrometer calibration or to the levels is still open.
