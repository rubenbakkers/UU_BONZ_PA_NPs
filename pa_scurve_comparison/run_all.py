# -*- coding: utf-8 -*-
"""Runs the whole comparison in the right order (~4 min).

Usage: open this folder in Spyder and run this file, or in a terminal:
    python run_all.py
Results (figures + json) go to the results/ folder.
Set FAST = True to skip the parameter scan (~2 min).
"""
import os, sys, runpy, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAST = False
STEPS = [
    ("make_figures.py",  "fit S-curves + fig 1 and 2 (writes summary.json)"),
    ("scan_shape.py",    "parameter scan (GSA, concentration, non-radiative)"),
    ("rise_variants.py", "rise time for the three explanations of k"),
    ("protocol_sim.py",  "simulation of the measurement protocol (2 s per point)"),
    ("variants_fit.py",  "refit with sigma_ESA x k and w2 / k"),
    ("emission_spec.py", "emission spectrum vs. Stark lines"),
    ("make_figures2.py", "fig 3"),
]
for script, what in STEPS:
    if FAST and script == "scan_shape.py":
        continue
    t0 = time.time()
    print(f"\n=== {script}: {what} ===", flush=True)
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    print(f"--- done in {time.time() - t0:.0f} s", flush=True)
print("\nAll done; see the results/ folder.")
