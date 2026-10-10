"""Transparent log-log curve diagnostics; these do not establish photon avalanche.

``Ith`` / ``Ith_star`` retain the original slope-maximum definition:
the intensity at maximum d(log emission)/d(log intensity). ``s_max`` is that
maximum differential slope. These scan diagnostics remain numeric even when
there is no resolved knee; they do not establish a physical PA threshold.

The separate ``threshold`` follows Lee et al., Supplementary Fig. S3:
the intersection of separate straight fits below and above the onset. This is
different from the intensity at the maximum differential log-log slope.

Explicit windows are preferable for a paper comparison. Pass physical intensity
bounds as ``fit_windows={"low": (a, b), "rising": (c, d)}``. Automatic windows are
only reproducible heuristics, not windows prescribed by the paper. They use the
widest adequately linear rising segment around the slope maximum and a separate
linear prefix of the low-intensity data. A reliable threshold is withheld if the
scan does not resolve both regimes or their intersection is implausible.

No intensities, emissions, derivatives, or slopes are clipped or smoothed. Fits
are weighted by log-intensity cell width, so dense adaptive sampling does not
silently dominate a regression. The reported maximum slope and fitted slope are
separate quantities; no constraint forces the fitted slope toward the maximum.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np


# Numerical fitting criteria, not physical parameters or PA classification rules.
MIN_FIT_POINTS = 5
MIN_RISING_EMISSION_SPAN_DEX = 0.5
MIN_LOW_INTENSITY_SPAN_DEX = 0.2
MIN_R_SQUARED = 0.999
MAX_FIT_ERROR_DEX = 0.03
AUTO_PEAK_FRACTION = 0.5
MAX_AUTO_BOUNDARIES = 121


def _validated_curve(intensity, emission):
    intensity = np.asarray(intensity, dtype=float)
    emission = np.asarray(emission, dtype=float)
    if (intensity.ndim != 1 or emission.ndim != 1
            or intensity.shape != emission.shape or len(intensity) < 5):
        raise ValueError("Intensity and emission must be matching 1-D arrays with at least five points.")
    if (not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(emission))
            or np.any(intensity <= 0) or np.any(emission <= 0)):
        raise ValueError("Log-log analysis requires finite, strictly positive intensity and emission; no clipping is applied.")
    order = np.argsort(intensity)
    intensity, emission = intensity[order], emission[order]
    if np.any(np.diff(intensity) <= 0):
        raise ValueError("Intensity values must be distinct.")
    x, y = np.log10(intensity), np.log10(emission)
    if np.any(np.diff(x) <= 0):
        raise ValueError("Intensity points are too close to resolve on a logarithmic numerical grid.")
    return intensity, emission, x, y


def _line_fit(x, y, start, stop):
    """Fit inclusive indices with weights proportional to log-I cell width."""
    xx, yy = x[start:stop + 1], y[start:stop + 1]
    if len(xx) < MIN_FIT_POINTS or xx[-1] <= xx[0]:
        return None
    weight = np.empty(len(xx))
    weight[0], weight[-1] = (xx[1] - xx[0]) / 2, (xx[-1] - xx[-2]) / 2
    weight[1:-1] = (xx[2:] - xx[:-2]) / 2
    weight /= weight.sum()
    xm, ym = float(weight @ xx), float(weight @ yy)
    dx, dy = xx - xm, yy - ym
    slope = float(np.sum(weight * dx * dy) / np.sum(weight * dx * dx))
    intercept = ym - slope * xm
    residual = yy - (slope * xx + intercept)
    variance = float(np.sum(weight * dy * dy))
    squared_error = float(np.sum(weight * residual * residual))
    r_squared = 1.0 - squared_error / variance if variance > 0 else float("nan")
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "max_error_dex": float(np.max(np.abs(residual))),
        "n_points": int(len(xx)),
        "intensity_range": (float(10 ** xx[0]), float(10 ** xx[-1])),
        "emission_range": (float(10 ** yy.min()), float(10 ** yy.max())),
        "x_range": (float(xx[0]), float(xx[-1])),
        "y_range": (float(yy.min()), float(yy.max())),
        "span_intensity_dex": float(xx[-1] - xx[0]),
        "span_emission_dex": float(yy.max() - yy.min()),
        "weighting": "log_intensity_cell_width",
    }


def _quality(fit, regime):
    if fit is None:
        return False
    common = (fit["slope"] > 0 and fit["r_squared"] >= MIN_R_SQUARED
              and fit["max_error_dex"] <= MAX_FIT_ERROR_DEX)
    if regime == "rising":
        return common and fit["span_emission_dex"] >= MIN_RISING_EMISSION_SPAN_DEX
    return common and fit["span_intensity_dex"] >= MIN_LOW_INTENSITY_SPAN_DEX


def _manual_fit(x, y, bounds, regime):
    try:
        lo, hi = map(float, bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The {regime} window must contain two intensity bounds.") from exc
    if not np.all(np.isfinite([lo, hi])) or not 0 < lo < hi:
        raise ValueError(f"The {regime} window requires finite bounds with 0 < lower < upper.")
    selected = np.flatnonzero((x >= np.log10(lo)) & (x <= np.log10(hi)))
    fit = _line_fit(x, y, int(selected[0]), int(selected[-1])) if len(selected) else None
    if fit is not None:
        fit.update(method="manual", requested_intensity_range=(lo, hi),
                   quality_passed=bool(_quality(fit, regime)))
    return fit


def _boundary_indices(x, first, last):
    """Bound search cost while sampling candidate boundaries uniformly in log-I."""
    if last - first + 1 <= MAX_AUTO_BOUNDARIES:
        return np.arange(first, last + 1)
    targets = np.linspace(x[first], x[last], MAX_AUTO_BOUNDARIES)
    return np.unique(np.r_[first, np.searchsorted(x, targets), last])


def _automatic_rising_fit(x, y, derivative, peak_index, baseline):
    peak = float(derivative[peak_index])
    # This only bounds the search. It does not impose a fitted-slope target.
    cutoff = max(AUTO_PEAK_FRACTION * peak, baseline + 0.5 * (peak - baseline))
    first = last = peak_index
    while first > 0 and derivative[first - 1] >= cutoff:
        first -= 1
    while last + 1 < len(x) and derivative[last + 1] >= cutoff:
        last += 1
    starts = _boundary_indices(x, first, peak_index)
    stops = _boundary_indices(x, peak_index, last)
    candidates = [(float(x[b] - x[a]), int(a), int(b))
                  for a in starts for b in stops if b - a + 1 >= MIN_FIT_POINTS]
    candidates.sort(reverse=True)
    for _span, a, b in candidates:
        fit = _line_fit(x, y, a, b)
        if _quality(fit, "rising"):
            fit.update(method="automatic", quality_passed=True,
                       candidate_slope_cutoff=float(cutoff))
            return fit
    return None


def _automatic_low_fit(x, y, before_x):
    # Every candidate begins at the lowest measured intensity. This prevents
    # calling a tiny segment next to the maximum slope the low-power regime.
    last = int(np.searchsorted(x, before_x, side="left")) - 1
    for stop in range(last, MIN_FIT_POINTS - 2, -1):
        if x[stop] - x[0] < MIN_LOW_INTENSITY_SPAN_DEX:
            break
        fit = _line_fit(x, y, 0, stop)
        if _quality(fit, "low"):
            fit.update(method="automatic", quality_passed=True)
            return fit
    return None


def analyze_curve(intensity, emission, fit_windows=None):
    """Return separate slopes, line-intersection threshold, and diagnostics.

    ``fit_windows`` may contain ``low`` and/or ``rising`` bounds in the same
    physical intensity units as the input. Missing windows are selected
    automatically. A supplied window that fails quality checks remains in the
    diagnostics, but cannot produce a reliable threshold.

    ``Ith`` and ``Ith_star`` alias ``peak_intensity``; ``s_max`` aliases
    ``s_peak``. These values do not depend on a successful regression.
    ``ith_method`` documents their meaning. ``peak_is_flat`` flags a nearly
    constant slope across the scan, where its location is not a unique onset.
    Other keys include ``fit`` (rising fit), ``low_fit``, ``threshold``,
    ``s_fit``, ``status``, and ``message``.
    Slopes use base-10 logs, which give the same exponents as natural logs.
    ``s_fit`` reports a qualified ordinary power-law fit even when no knee is
    present. ``fit_kind`` and ``curve_fit`` identify the reported fit and its
    window. Only ``fit`` (a rising regime) and ``low_fit`` can define a threshold;
    an ordinary fallback never creates one. ``s_rising_fit`` stays unavailable
    if no qualifying rising regime is found. Missing numerical results are NaN.
    ``steep_knee_candidate`` is strictly a
    curve-shape description: ESA/GSA contrast and rise-time evidence are needed
    to diagnose photon avalanche. No uncertainty interval is inferred from
    deterministic curve samples.
    """
    intensity, emission, x, y = _validated_curve(intensity, emission)
    if fit_windows is None:
        fit_windows = {}
    if not isinstance(fit_windows, Mapping) or set(fit_windows) - {"low", "rising"}:
        raise ValueError("fit_windows must map 'low' and/or 'rising' to intensity-bound pairs.")
    derivative = np.gradient(y, x, edge_order=2)
    if not np.all(np.isfinite(derivative)):
        raise ValueError("The supplied intensity spacing does not permit finite log-log derivatives.")
    peak_value = float(np.max(derivative))
    # For a true power-law plateau, a central point is more reproducible than
    # whichever endpoint happens to win through floating-point roundoff.
    near_peak = np.flatnonzero(np.isclose(derivative, peak_value, rtol=1e-9, atol=1e-10))
    peak_index = int(near_peak[len(near_peak) // 2])
    baseline_last = max(MIN_FIT_POINTS, int(np.searchsorted(x, x[0] + 0.1 * (x[-1] - x[0]))))
    baseline = float(np.median(derivative[:baseline_last]))
    result = {
        "fit": None, "low_fit": None, "threshold": float("nan"),
        "peak_intensity": float(intensity[peak_index]),
        "s_peak": peak_value, "s_fit": float("nan"),
        "Ith": float(intensity[peak_index]),
        "Ith_star": float(intensity[peak_index]), "s_max": peak_value,
        "ith_method": "maximum_differential_log_log_slope",
        "peak_is_flat": bool(np.ptp(derivative) <= 1e-3 * max(abs(peak_value), 1.)),
        "s_rising_fit": float("nan"), "curve_fit": None,
        "fit_kind": "unavailable",
        "peak_at_boundary": bool(peak_index in (0, len(x) - 1)),
        "status": "no_resolved_knee", "message": "",
        "threshold_method": "intersection_of_low_and_rising_log_log_fits",
        "criteria": {
            "min_points": MIN_FIT_POINTS,
            "min_r_squared": MIN_R_SQUARED,
            "max_error_dex": MAX_FIT_ERROR_DEX,
            "min_rising_emission_span_dex": MIN_RISING_EMISSION_SPAN_DEX,
            "min_low_intensity_span_dex": MIN_LOW_INTENSITY_SPAN_DEX,
            "auto_peak_fraction": AUTO_PEAK_FRACTION,
        },
    }
    if "rising" in fit_windows:
        rising = _manual_fit(x, y, fit_windows["rising"], "rising")
    elif peak_value - baseline >= max(1.0, 0.5 * abs(baseline)):
        rising = _automatic_rising_fit(x, y, derivative, peak_index, baseline)
    else:
        rising = None
    result["fit"] = rising
    if "low" in fit_windows:
        low = _manual_fit(x, y, fit_windows["low"], "low")
    else:
        before_x = rising["x_range"][0] if rising is not None else np.nextafter(x[-1], np.inf)
        low = _automatic_low_fit(x, y, before_x)
    result["low_fit"] = low
    if _quality(rising, "rising"):
        result.update(s_fit=rising["slope"], s_rising_fit=rising["slope"],
                      curve_fit=rising, fit_kind="rising_segment")
    elif "rising" not in fit_windows:
        # A slope exists for an ordinary power law too. Do not confuse a
        # missing avalanche segment with an impossible regression. Prefer the
        # full curve, otherwise report the separately qualified low-power fit.
        ordinary = _line_fit(x, y, 0, len(x) - 1)
        if _quality(ordinary, "low"):
            ordinary.update(method="ordinary_full_curve", quality_passed=True)
            result.update(s_fit=ordinary["slope"], curve_fit=ordinary,
                          fit_kind="ordinary_power_law")
        elif _quality(low, "low"):
            result.update(s_fit=low["slope"], curve_fit=low,
                          fit_kind="low_power_segment")
    if not _quality(rising, "rising"):
        result["message"] = "No resolved rising regime in the scanned intensity range; no reliable line-intersection threshold."
        if result["curve_fit"] is not None:
            result["message"] += " The reported s_fit describes ordinary power dependence, not an avalanche segment."
        if result["peak_at_boundary"]:
            result["message"] += " The differential-slope maximum lies at the scan boundary."
        return result
    result["status"] = "threshold_unreliable"
    if not _quality(low, "low"):
        result["message"] = "The low-intensity regime is not adequately resolved; extend the scan below the knee or provide justified fit windows."
        return result
    if low["x_range"][1] >= rising["x_range"][0]:
        result["message"] = "Low and rising fit windows overlap or are reversed; no reliable line-intersection threshold."
        return result
    slope_difference = rising["slope"] - low["slope"]
    if slope_difference < max(1.0, 0.5 * abs(low["slope"])):
        result["status"] = "no_resolved_knee"
        result["message"] = "The fitted slopes do not resolve a steepening knee; a pure power law has no line-intersection threshold."
        return result
    cross_x = (low["intercept"] - rising["intercept"]) / slope_difference
    # The low branch must end before the inferred onset. The crossing may fall
    # within the early rising fit, but never beyond its last measured point.
    if not low["x_range"][1] - 1e-10 <= cross_x <= rising["x_range"][1] + 1e-10:
        result["message"] = "The fitted lines intersect outside the resolved onset region; no reliable line-intersection threshold."
        return result
    result.update(threshold=float(10 ** cross_x), status="steep_knee_candidate",
                  message="Separate low and rising regimes resolve a steep knee; this curve alone does not confirm photon avalanche.")
    return result


__all__ = ["analyze_curve"]
