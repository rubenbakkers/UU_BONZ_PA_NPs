# -*- coding: utf-8 -*-
"""Draait de hele vergelijking in de juiste volgorde (~4 min).

Gebruik: open deze map in Spyder en run dit bestand, of in een terminal:
    python run_all.py
Resultaten (figuren + json) komen in de map results/.
Zet SNEL = True om de parameterscan (~2 min) over te slaan.
"""
import os, sys, runpy, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SNEL = False
STAPPEN = [
    ("make_figures.py",  "fit S-curves + fig 1 en 2 (schrijft summary.json)"),
    ("scan_shape.py",    "parameterscan (GSA, concentratie, niet-radiatief)"),
    ("rise_variants.py", "stijgtijd voor de drie verklaringen van k"),
    ("protocol_sim.py",  "simulatie meetprotocol (2 s per punt)"),
    ("variants_fit.py",  "refit met sigma_ESA x k en w2 / k"),
    ("emission_spec.py", "emissiespectrum vs. Stark-lijnen"),
    ("make_figures2.py", "fig 3"),
]
for script, wat in STAPPEN:
    if SNEL and script == "scan_shape.py":
        continue
    t0 = time.time()
    print(f"\n=== {script}: {wat} ===", flush=True)
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    print(f"--- klaar in {time.time() - t0:.0f} s", flush=True)
print("\nAlles klaar; zie de map results/.")
