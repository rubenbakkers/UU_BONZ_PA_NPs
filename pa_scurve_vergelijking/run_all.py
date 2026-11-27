# -*- coding: utf-8 -*-
"""Runs the whole comparison in the right order (~4 min + ~6 min, for two measurements).

Usage: open this folder in Spyder and run this file, or in a terminal:
    python run_all.py
The measurements are listed in measurements.py (add new ones there).
Results (figures + json) go to the results/ folder.
Set FAST = True to skip the parameter scan (~2 min per measurement).
"""
import os, sys, runpy, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pa_fit_lib import measurements

FAST = False
STEPS = [
    ("make_figures.py",  "fit S-curves + fig 1 and 2 (writes summary.json)"),
    ("scan_shape.py",    "parameter scan (GSA, concentration, non-radiative)"),
    ("rise_variants.py", "rise time for the three explanations of k"),
    ("protocol_sim.py",  "simulation of the measurement protocol (2 s per point)"),
    ("variants_fit.py",  "refit with sigma_ESA x k and w2 / k"),
    ("fit_all_params.py", "fit with all model parameters free (fig 4 and 5)"),
    ("fit_concentration.py", "fit with the Tm concentration asked per sample, or fitted (fig 6 and 7)"),
    ("emission_spec.py", "emission spectrum vs. Stark lines"),
    ("make_figures2.py", "fig 3"),
]
print("Measurements (measurements.py): "
      + ", ".join(f"{n} ({m['label']})" for n, m in measurements().items()), flush=True)
for script, what in STEPS:
    if FAST and script == "scan_shape.py":
        continue
    t0 = time.time()
    print(f"\n=== {script}: {what} ===", flush=True)
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    print(f"--- done in {time.time() - t0:.0f} s", flush=True)
print("\nAll done; see the results/ folder.")
