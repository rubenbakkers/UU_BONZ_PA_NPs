# -*- coding: utf-8 -*-
"""All measured S-curves. This is the only file to edit when you add a measurement.

Adding a measurement
--------------------
1. Put the CSV in data/ (the scurve.csv export, sorted or not, with at least the
   columns fw6, I_kWcm2, net_cps, noise_cps; see README.md).
2. Copy the template at the bottom of MEASUREMENTS, fill it in and remove the #.
3. Run run_all.py, or make_figures.py first and then the scripts you need
   (make_figures.py writes results/summary.json, which the others read).

Every script loops over the measurements below; nothing else has to change.
The S-curve figures get one panel per measurement (at most 3 per row). This file
is read again every time a script runs, so you do not need to restart the
console (kernel) after editing it.

Fields
------
file           CSV in data/                                                (required)
label          name in the figures and printed output                      (required)
wavelength_nm  pump wavelength [nm]                                        (required)
fwhm_nm        laser linewidth [nm]; 0 = single frequency (default). With
               fwhm_nm > 0 the GSA/ESA rates are averaged over a Gaussian
               laser spectrum.
sample, date   particles and measurement day (text). fit_all_params.py fits
               the measurements of the same sample with shared particle
               parameters ("joint" fit); the figure titles show them.
knee_fw        filter-wheel setting (fw6) of the sweep through the knee; the
               2-s measurement protocol is simulated for that sweep
               (protocol_sim.py, variants_fit.py, fig. 3c). Default None:
               make_figures.py picks the sweep that contains I_th/k.
compare_sweeps (fw_a, fw_b): two sweeps over the same intensities to compare in
               fig. 3d (make_figures2.py); protocol_sim.py also simulates
               sweep fw_a starting right after full intensity. Optional.
compare_window_kWcm2  (I_min, I_max) of fig. 3d; default: where both sweeps overlap.
spectrum_file  emission spectrum in data/ (columns wavelength_nm,
               cts_per_s_per_pixel), fitted by emission_spec.py. Optional.
spectrum_noise_cts  noise per pixel of that spectrum (default 180).

The key of each entry (e.g. "cobolt_1060") is its name in results/*.json; use
letters, digits and underscores.
"""

MEASUREMENTS = {
    "cobolt_1060": dict(
        file="cobolt_1060_scurve.csv",
        label="Cobolt, 1060 nm",
        wavelength_nm=1060.0,
        fwhm_nm=0.0,
        sample="BdJ1", date="24-11-2026",
        compare_sweeps=(4, 5),               # FW6=4 (sweep 1) vs. FW6=5 (sweep 2)
        compare_window_kWcm2=(0.85, 2.3),
    ),
    "superk_1064": dict(
        file="superk_1064_scurve.csv",
        label="SuperK, 1064 nm",
        wavelength_nm=1064.0,
        fwhm_nm=0.0,                         # SuperK treated as monochromatic
        sample="BdJ1", date="24-11-2026",
        spectrum_file="emission_check_spectrum_superk_785_862nm.csv",
        spectrum_noise_cts=180.0,
    ),
    # Template for a new measurement (copy, fill in, remove the #):
    # "superk_1070": dict(
    #     file="superk_1070_scurve.csv",
    #     label="SuperK, 1070 nm",
    #     wavelength_nm=1070.0,
    #     sample="BdJ1", date="01-12-2026",
    # ),
}

# Run only these measurements, e.g. ["cobolt_1060"]; None = all of the above.
ACTIVE = None

# Fig. 2 (make_figures.py) shows every threshold relative to this measurement,
# and fig. 3b its rise time. None = the first active measurement.
REFERENCE = "cobolt_1060"
