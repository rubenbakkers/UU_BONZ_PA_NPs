# -*- coding: utf-8 -*-
"""Shared figure style and the panel layout for any number of measurements."""
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#257de0", "#f47230", "#00b894", "#9a9a9a", "#191919", "#666666"
PALETTE = (BLUE, ORANGE, AQUA, "#a05195", "#d4a017", "#e84393", "#2d3436")
FW_MARKERS = ("o", "s", "^", "D", "v", "P")
LINESTYLES = ("-", ":", "--", "-.")
MAX_COLUMNS = 3          # S-curve figures: at most this many measurements per row


def apply_style(font_size=10):
    plt.rcParams.update({"font.size": font_size, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 2})


def fw_markers(fws):
    """Marker per filter-wheel setting of one curve: o, s, ^, ... in increasing order."""
    return {f: FW_MARKERS[i % len(FW_MARKERS)] for i, f in enumerate(sorted(set(fws)))}


def scurve_axes(n, max_columns=MAX_COLUMNS):
    """Figure with, per measurement, an S-curve panel above a slope panel.

    Returns (fig, [(ax_curve, ax_slope), ...]) in the order of the measurements.
    Two measurements give the original 10.5 x 7.6 inch layout.
    """
    ncols = min(n, max_columns)
    nrows = -(-n // ncols)
    fig = plt.figure(figsize=(5.25 * ncols, 7.6 * nrows))
    gs = fig.add_gridspec(2 * nrows, ncols, height_ratios=[1.6, 1] * nrows)
    pairs = []
    for i in range(n):
        r, c = divmod(i, ncols)
        top = fig.add_subplot(gs[2 * r, c])
        bottom = fig.add_subplot(gs[2 * r + 1, c], sharex=top)
        top.tick_params(which="both", labelbottom=False)   # also the minor (log) labels
        pairs.append((top, bottom))
    return fig, pairs
