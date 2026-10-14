# -*- coding: utf-8 -*-
"""GSA/ESA voor Tm3+:NaYF4, met fononzijdenbanden uit een CONTINUE fononband.

Model: Boltzmanngewogen Stark-lijnen, een elektronische Gauss-kern en een
genormaliseerde, eindige-T Huang-Rhys-kernel. In plaats van drie vaste modi
wordt over een fononband [BAND_MIN, BAND_MAX] in golfgetal GEINTEGREERD:

    S(w) = S_tot * g(w),      Integraal g(w) dw = 1

De integraal wordt numeriek benaderd met de trapeziumregel op knopen
w_k = BAND_MIN + k*BAND_STEP. Elke knoop is een onafhankelijke harmonische
modus met Huang-Rhys-factor S_k = S_tot * g(w_k) * t_k / som(g*t), waarbij t_k
het trapeziumgewicht is. Alle S_k samen sommeren dus exact tot S_tot, en de
rest van het model (Poisson/Skellam-kernel per modus, detailed balance,
anker op 1064 nm) is ongewijzigd.

NIEUW t.o.v. de vorige versie:
  * Ook de ESA-kanalen (3F4 -> 3H4 / 3F3 / 3F2) krijgen fononzijdenbanden,
    met dezelfde fononband en per kanaal een eigen totale S (hier 1e-6).
  * Het plot toont extra (andere kleuren, gestippeld) de bijdrage van de
    zuiver elektronische overgangen ZONDER fononen: de nul-fonon-lijnen
    (geen enkel fononevent) van het volledige model, met exact dezelfde
    normalisatie (piek en 1064 nm-anker) als de volledige curves. Het
    verschil tussen volle en gestippelde curve is dus de zijbandbijdrage.

Voor optische absorptie betekent een POSITIEVE energieverschuiving creatie
van roosterfononen; een NEGATIEVE verschuiving verbruikt roosterfononen.

Bron voor de band (Raman van beta-NaYF4:Yb,Tm nanokristallen):
  Wilhelm et al., Theranostics 3(4), 239-248 (2013),
  https://doi.org/10.7150/thno.5113
  -> fononbanden 'duidelijk en onderscheidbaar' tussen 225 en 450 cm^-1;
     gewogen gemiddelde 320 cm^-1 (Yb/Tm), 304 cm^-1 (Yb/Er); bulk 360 cm^-1.
Eerdere (discrete) modi 250/305/365 cm^-1 en het vibronische model:
  https://doi.org/10.1016/j.omx.2021.100102
  https://doi.org/10.1080/0026897021000046889 (Liu et al., 2003)

BELANGRIJK: standaard gebruiken we een handmatig en bij benadering afgelezen
profiel van de gestippelde NaYF4(Yb/Tm)-curve in Fig. 3A van Wilhelm et al.
(de door de gebruiker aangeleverde afbeelding). Dit is Raman-intensiteit,
gebruikt als DOS-/koppelingsproxy, geen gemeten fonon-DOS of ruwe brondata.
Lineaire interpolatie bewaart de meerdere pieken; alleen de oppervlakte
wordt genormaliseerd. Het gemiddelde wordt NIET naar 320 cm^-1 geforceerd.
De band blijft standaard [225, 450], nu met 5 cm^-1 knoopafstand (46 modi).
Gebruik --band-file voor betere brondata.
Het Gaussian-, uniform- en discrete-model blijven optioneel beschikbaar.
De Huang-Rhys-factoren, elektronische breedte en lijnsterktes zijn eveneens
modelaannames. De koppelingen uit het aangeleverde script blijven behouden.
Normalisatie op 1064 nm bepaalt geen absolute GSA/ESA-verhouding.

Gebruik:
  python gsa_esa_fononband.py --profile-only --no-show
  python gsa_esa_fononband.py --no-show
  python gsa_esa_fononband.py --data-only
  python gsa_esa_fononband.py --band-min 225 --band-max 450 --band-step 15
  python gsa_esa_fononband.py --band-profile gaussian --band-mean 304
  python gsa_esa_fononband.py --band-profile uniform
  python gsa_esa_fononband.py --band-file raman_fig3A.csv
  python gsa_esa_fononband.py --phonon-model discrete    # oude 3 modi
  python gsa_esa_fononband.py --temperature 350 --coupling-h5 0.08
  python gsa_esa_fononband.py --coupling-h4 0.1 --coupling-f3 0.05 --coupling-f2 0.05

Benodigd: numpy; voor PNG/SVG-grafieken ook matplotlib.
CSV, instellingen/diagnostiek (JSON), de gebruikte band (CSV) en grafieken
komen naast dit script, of in de map opgegeven met --output-dir.
Er wordt geen vloer toegevoegd.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np

# Aanpasbare spectrale instellingen.
WAVELENGTH_MIN_NM = 900.0
WAVELENGTH_MAX_NM = 2000.0
WAVELENGTH_STEP_NM = 0.05
REFERENCE_WAVELENGTH_NM = 1064.0
TEMPERATURE_K = 300.0
STARK_SIGMA_CM1 = 8.5
GSA_H5_SIGMA_CM1 = 8.5  # Elektronische kern; oude effectieve 300 vervalt.
INSTRUMENT_SIGMA_CM1 = 0.0
PLOT_MIN_FACTOR = 1e-20  # Alleen de y-as, nooit een datavloer.
K_B_CM1_PER_K = 0.6950348005
BOLTZMANN_MODE = "multiplet"
SI_REFERENCE_ANCHOR = True

# Fononmodel: "band" (integratie over golfgetal) of "discrete" (oude 3 modi).
PHONON_MODEL = "band"
BAND_MIN_CM1 = 225.0     # Wilhelm et al. 2013: onderrand fononbanden.
BAND_MAX_CM1 = 450.0     # Wilhelm et al. 2013: bovenrand fononbanden.
BAND_STEP_CM1 = 5.0      # Fijnere bemonstering: 46 knopen, veelvoud van 5.
BAND_MEAN_CM1 = 320.0    # Gewogen gemiddelde, Yb/Tm-monster (304 voor Yb/Er).
BAND_SIGMA_CM1 = 45.0    # Alleen gebruikt bij optioneel Gaussian-profiel.
BAND_PROFILE = "raman_tm"  # "raman_tm", "gaussian", "uniform" of "file"
# Approximate manual trace of the dotted Yb/Tm Raman curve, panel A.
# Right tail refined using the visible dot centers in the supplied image.
# Source: Wilhelm et al. (2013), https://doi.org/10.7150/thno.5113, Fig. 3A.
# Wavenumber (cm^-1), relative Raman intensity; not an experimental DOS table.
RAMAN_TM_PROFILE_POINTS = np.array([
    (200.0, 0.000),
    (205.0, 0.005),
    (210.0, 0.015),
    (215.0, 0.030),
    (220.0, 0.055),
    (225.0, 0.100),
    (230.0, 0.170),
    (235.0, 0.300),
    (240.0, 0.450),
    (245.0, 0.600),
    (250.0, 0.720),
    (255.0, 0.770),
    (260.0, 0.730),
    (265.0, 0.670),
    (270.0, 0.600),
    (275.0, 0.570),
    (280.0, 0.590),
    (285.0, 0.640),
    (290.0, 0.750),
    (295.0, 0.880),
    (300.0, 0.980),
    (305.0, 0.960),
    (310.0, 0.900),
    (315.0, 0.840),
    (320.0, 0.800),
    (325.0, 0.730),
    (330.0, 0.670),
    (335.0, 0.610),
    (340.0, 0.540),
    (345.0, 0.550),
    (350.0, 0.610),
    (355.0, 0.700),
    (360.0, 0.790),
    (365.0, 0.800),
    (370.0, 0.730),
    (373.99, 0.6724),
    (375.35, 0.6273),
    (378.89, 0.5871),
    (381.08, 0.5414),
    (383.57, 0.5000),
    (385.95, 0.4607),
    (390.37, 0.4258),
    (393.90, 0.3858),
    (398.13, 0.3690),
    (402.74, 0.3385),
    (408.24, 0.3602),
    (413.75, 0.3759),
    (418.86, 0.3534),
    (422.61, 0.3121),
    (425.20, 0.2672),
    (430.12, 0.2274),
    (433.16, 0.1828),
    (437.19, 0.1442),
    (442.50, 0.1280),
    (448.06, 0.1189),
    (450.00, 0.1160),
], dtype=float)

DISCRETE_ENERGIES_CM1 = (250.0, 305.0, 365.0)
DISCRETE_FRACTIONS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

# Totale Huang-Rhys-factoren per kanaal (ongekalibreerd).
GSA_H5_HUANG_RHYS = 0.000001  # Te fitten totale S voor 3H6 -> 3H5.

GSA_F4_HUANG_RHYS = GSA_H5_HUANG_RHYS
GSA_F4_HUANG_RHYS = GSA_H5_HUANG_RHYS  # Afzonderlijk te fitten voor 3H6 -> 3F4.
ESA_H4_HUANG_RHYS = GSA_H5_HUANG_RHYS  # 3F4 -> 3H4.
ESA_F3_HUANG_RHYS = GSA_H5_HUANG_RHYS  # 3F4 -> 3F3.
ESA_F2_HUANG_RHYS = GSA_H5_HUANG_RHYS


# GSA_F4_HUANG_RHYS = 0.001  # Afzonderlijk te fitten voor 3H6 -> 3F4.
# ESA_H4_HUANG_RHYS = 0.001  # 3F4 -> 3H4.
# ESA_F3_HUANG_RHYS = 0.001  # 3F4 -> 3F3.
# ESA_F2_HUANG_RHYS = 0.001  # 3F4 -> 3F2.
PHONON_SHIFT_GRID_STEP_CM1 = 5.0  # Modi moeten exact op dit rooster liggen.
PHONON_TAIL_TOLERANCE = 1e-35  # Bovengrens ontbrekende totale kernelmassa.
PHONON_MAX_EVENTS_PER_MODE = 256  # Veiligheidslimiet; overschrijding is een fout.
REFERENCE_MAX_RELATIVE_KERNEL_ERROR = 1e-8  # Weiger onvoldoende opgelost anker.

# Wordt gevuld door set_phonon_model(); alle kernelfuncties lezen deze globals.
PHONON_ENERGIES_CM1 = DISCRETE_ENERGIES_CM1
PHONON_MODE_FRACTIONS = DISCRETE_FRACTIONS
PHONON_DESCRIPTION = ""
PHONON_INFO = {}

# Stark-niveaus en kanaalgewichten uit het aangeleverde script.
E_3H6_CM1 = np.array([0., 9., 54., 80., 122., 149., 187., 223., 246.,
                       296., 314., 405., 417.])
E_3F4_CM1 = np.array([5634., 5661., 5689., 5717., 5730., 5750., 5761.,
                       5782., 5823.])
E_3H5_CM1 = np.array([8207., 8218., 8267., 8288., 8317., 8348., 8377.,
                       8428., 8440., 8528., 8539.])
E_3H4_CM1 = np.array([12546., 12566., 12588., 12613., 12642., 12673.,
                       12721., 12766., 12776.])
E_3F3_CM1 = np.array([14279., 14317., 14341., 14361., 14380., 14398., 14420.])
E_3F2_CM1 = np.array([14936., 14951., 15018., 15061., 15084.])
GSA_WEIGHT_3F4 = GSA_WEIGHT_3H5 = 1.0
ESA_WEIGHT_3H4 = ESA_WEIGHT_3F3 = ESA_WEIGHT_3F2 = 1.0
GSA_CHANNELS = {
    "3H6 -> 3F4": (E_3H6_CM1, E_3F4_CM1, GSA_WEIGHT_3F4),
    "3H6 -> 3H5": (E_3H6_CM1, E_3H5_CM1, GSA_WEIGHT_3H5),
}
ESA_CHANNELS = {
    "3F4 -> 3H4": (E_3F4_CM1, E_3H4_CM1, ESA_WEIGHT_3H4),
    "3F4 -> 3F3": (E_3F4_CM1, E_3F3_CM1, ESA_WEIGHT_3F3),
    "3F4 -> 3F2": (E_3F4_CM1, E_3F2_CM1, ESA_WEIGHT_3F2),
}


# ---------------------------------------------------------------------------
# Fononband: kwadratuur over golfgetal
# ---------------------------------------------------------------------------
def band_nodes(band_min, band_max, band_step):
    """Knopen w_k en trapeziumgewichten t_k voor de integraal over de band."""
    values = np.array([band_min, band_max, band_step], dtype=float)
    if not np.all(np.isfinite(values)) or not 0 < band_min < band_max or band_step <= 0:
        raise ValueError("Band needs finite 0 < min < max and a positive step.")
    count = int(round((band_max - band_min) / band_step))
    if count < 1 or not np.isclose(band_min + count * band_step, band_max,
                                   rtol=0, atol=1e-9):
        raise ValueError("band-max minus band-min must be a multiple of band-step.")
    nodes = band_min + band_step * np.arange(count + 1)
    lattice = nodes / PHONON_SHIFT_GRID_STEP_CM1
    if not np.allclose(lattice, np.rint(lattice), rtol=0, atol=1e-9):
        raise ValueError("Band nodes must be multiples of PHONON_SHIFT_GRID_STEP_CM1 "
                         f"({PHONON_SHIFT_GRID_STEP_CM1:g} cm^-1).")
    trapezoid = np.ones(count + 1)
    trapezoid[[0, -1]] = 0.5
    return nodes, trapezoid


def _gaussian_log_weights(nodes, trapezoid, center, sigma):
    return -0.5 * ((nodes - center) / sigma) ** 2 + np.log(trapezoid)


def _weighted_mean(nodes, weights):
    return float(np.sum(nodes * weights) / np.sum(weights))


def _read_profile_file(path):
    path = Path(path)
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    for skip in (0, 1):
        try:
            data = np.loadtxt(lines[skip:], delimiter=",", ndmin=2)
            break
        except ValueError:
            data = None
    if data is None or data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError("band-file needs >= 2 rows and 2 comma-separated columns "
                         "(wavenumber_cm1,intensity).")
    x, y = data[:, 0], data[:, 1]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("band-file contains non-finite values.")
    order = np.argsort(x)
    return x[order], y[order]


def band_fractions(nodes, trapezoid, profile, mean_cm1, sigma_cm1, band_file=None):
    """Genormaliseerde gewichten f_k (som = 1) van g(w) op de knopen."""
    if profile == "raman_tm":
        x, y = RAMAN_TM_PROFILE_POINTS.T
        if nodes[0] < x[0] or nodes[-1] > x[-1]:
            raise ValueError("Built-in Raman trace covers 200-450 cm^-1; "
                             "use --band-file to extend that range.")
        density = np.interp(nodes, x, y)
        weights = density * trapezoid
    elif profile == "uniform":
        weights = trapezoid.copy()
    elif profile == "gaussian":
        if not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
            raise ValueError("band-sigma must be positive.")
        if not nodes[0] < mean_cm1 < nodes[-1]:
            raise ValueError("band-mean must lie strictly inside the band.")

        def mean_for(center):
            logw = _gaussian_log_weights(nodes, trapezoid, center, sigma_cm1)
            return _weighted_mean(nodes, np.exp(logw - logw.max()))

        # mean_for is monotoon stijgend in het centrum: bisectie.
        low, high = nodes[0] - 8 * sigma_cm1, nodes[-1] + 8 * sigma_cm1
        for _ in range(200):
            middle = 0.5 * (low + high)
            if mean_for(middle) < mean_cm1:
                low = middle
            else:
                high = middle
        logw = _gaussian_log_weights(nodes, trapezoid, 0.5 * (low + high), sigma_cm1)
        weights = np.exp(logw - logw.max())
    elif profile == "file":
        if band_file is None:
            raise ValueError("Profile 'file' needs --band-file.")
        x, y = _read_profile_file(band_file)
        density = np.clip(np.interp(nodes, x, y, left=0.0, right=0.0), 0.0, None)
        weights = density * trapezoid
    else:
        raise ValueError("profile must be 'raman_tm', 'gaussian', 'uniform' or 'file'.")
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("Band profile has no positive weight on the nodes.")
    return weights / weights.sum()


def set_phonon_model(model=PHONON_MODEL, band_min=BAND_MIN_CM1, band_max=BAND_MAX_CM1,
                     band_step=BAND_STEP_CM1, profile=BAND_PROFILE,
                     mean_cm1=BAND_MEAN_CM1, sigma_cm1=BAND_SIGMA_CM1, band_file=None):
    """Stel PHONON_ENERGIES_CM1 / PHONON_MODE_FRACTIONS in (vervangt de vaste modi)."""
    global PHONON_ENERGIES_CM1, PHONON_MODE_FRACTIONS, PHONON_DESCRIPTION, PHONON_INFO
    if model == "discrete":
        energies = np.asarray(DISCRETE_ENERGIES_CM1, dtype=float)
        fractions = np.asarray(DISCRETE_FRACTIONS, dtype=float)
        description = "discreet " + "/".join(f"{e:g}" for e in energies) + " cm⁻¹"
    elif model == "band":
        energies, trapezoid = band_nodes(band_min, band_max, band_step)
        fractions = band_fractions(energies, trapezoid, profile, mean_cm1,
                                   sigma_cm1, band_file)
        keep = fractions > 0  # Knopen zonder gewicht dragen niets bij.
        energies, fractions = energies[keep], fractions[keep]
        fractions = fractions / fractions.sum()
        description = (f"band {band_min:g}-{band_max:g} cm⁻¹ ({profile}, "
                       f"{len(energies)} knopen, gem. {_weighted_mean(energies, fractions):.1f} cm⁻¹)")
    else:
        raise ValueError("phonon model must be 'band' or 'discrete'.")
    PHONON_ENERGIES_CM1 = tuple(float(e) for e in energies)
    PHONON_MODE_FRACTIONS = tuple(float(f) for f in fractions)
    PHONON_DESCRIPTION = description
    PHONON_INFO = {
        "phonon_model": model,
        "description": description,
        "profile": profile if model == "band" else "discrete",
        "band_min_cm1": float(band_min) if model == "band" else None,
        "band_max_cm1": float(band_max) if model == "band" else None,
        "band_step_cm1": float(band_step) if model == "band" else None,
        "number_of_modes": len(energies),
        "profile_provenance": ("Approximate manual trace of supplied Fig. 3A, "
                               "Wilhelm et al. 2013, dotted NaYF4(Yb/Tm) Raman curve"
                               if model == "band" and profile == "raman_tm" else None),
        "raman_as_dos_proxy": bool(model == "band" and profile in ("raman_tm", "file")),
        "mean_forced_to_target": bool(model == "band" and profile == "gaussian"),
        "weighted_mean_cm1": _weighted_mean(energies, fractions),
        "weighted_std_cm1": float(np.sqrt(
            np.sum(fractions * (energies - _weighted_mean(energies, fractions)) ** 2))),
    }
    return PHONON_INFO


set_phonon_model()  # Standaard: overgenomen Yb/Tm Raman-profiel op 225-450 cm^-1.


# ---------------------------------------------------------------------------
# Elektronische structuur en fononkernel (ongewijzigd)
# ---------------------------------------------------------------------------
def boltzmann_within(levels_cm1, temperature_k=TEMPERATURE_K):
    """Conditional populations WITHIN one multiplet, normalized to one."""
    levels = np.asarray(levels_cm1, dtype=float)
    if levels.ndim != 1 or not levels.size or not np.all(np.isfinite(levels)):
        raise ValueError("levels_cm1 must be a finite, nonempty 1D vector.")
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperature must be finite and positive.")
    weights = np.exp(-(levels - levels.min()) / (K_B_CM1_PER_K * temperature_k))
    return weights / weights.sum()


def level_populations(levels_cm1, temperature_k=TEMPERATURE_K):
    if BOLTZMANN_MODE == "multiplet":
        return boltzmann_within(levels_cm1, temperature_k)
    if BOLTZMANN_MODE == "off":
        levels = np.asarray(levels_cm1, dtype=float)
        if levels.ndim != 1 or not levels.size or not np.all(np.isfinite(levels)):
            raise ValueError("levels_cm1 must be a finite, nonempty 1D vector.")
        return np.full(levels.size, 1.0 / levels.size)
    raise ValueError("BOLTZMANN_MODE must be 'multiplet' or 'off'.")


def transition_gaps(lower_cm1, upper_cm1):
    gaps = np.asarray(upper_cm1)[None, :] - np.asarray(lower_cm1)[:, None]
    if not np.all(np.isfinite(gaps)) or np.any(gaps <= 0):
        raise ValueError("All electronic transition gaps must be positive and finite.")
    return gaps


def transition_sticks(lower_cm1, upper_cm1, temperature_k=TEMPERATURE_K,
                      relative_lines=None):
    gaps = transition_gaps(lower_cm1, upper_cm1)
    populations = level_populations(lower_cm1, temperature_k)
    if relative_lines is None:
        relative = np.full(gaps.shape, 1.0 / gaps.shape[1])
    else:
        relative = np.asarray(relative_lines, dtype=float)
        if (relative.shape != gaps.shape or not np.all(np.isfinite(relative))
                or np.any(relative < 0) or not np.any(relative > 0)):
            raise ValueError("Invalid relative line strengths.")
    return gaps, populations[:, None] * relative


def phonon_occupation(energy_cm1, temperature_k):
    """Bose occupation; T=0 is supported for the phonon kernel itself."""
    if (not np.isfinite(temperature_k) or temperature_k < 0
            or not np.isfinite(energy_cm1) or energy_cm1 <= 0):
        raise ValueError("Phonon energy must be > 0 and temperature >= 0.")
    if temperature_k == 0:
        return 0.0
    exponent = energy_cm1 / (K_B_CM1_PER_K * temperature_k)
    return 0.0 if exponent > 700 else float(1.0 / np.expm1(exponent))


def _poisson_tail_bound(mean, order, probability):
    """Upper bound on P(N > order), avoiding cancellation of 1-CDF."""
    if mean == 0:
        return 0.0
    ratio = mean / (order + 2)
    if ratio >= 1:
        return float("inf")
    return probability * mean / (order + 1) / (1.0 - ratio)


def _poisson_pair(mean_emit, mean_absorb, tolerance):
    """Use the SAME event cutoff for both signs, preserving detailed balance."""
    if max(mean_emit, mean_absorb) > 100:
        raise ValueError("Phonon event mean exceeds the supported range (100).")
    emit = [float(np.exp(-mean_emit))]
    absorb = [float(np.exp(-mean_absorb))]
    for order in range(PHONON_MAX_EVENTS_PER_MODE + 1):
        missing = (_poisson_tail_bound(mean_emit, order, emit[-1])
                   + _poisson_tail_bound(mean_absorb, order, absorb[-1]))
        if missing <= tolerance:
            return np.asarray(emit), np.asarray(absorb), missing
        emit.append(emit[-1] * mean_emit / (order + 1))
        absorb.append(absorb[-1] * mean_absorb / (order + 1))
    raise ValueError("Phonon expansion did not converge; revise cutoff or parameters.")


@lru_cache(maxsize=32)
def _cached_phonon_kernel(energies, strengths, temperature_k, step_cm1, tolerance):
    if (len(energies) != len(strengths) or not energies
            or not np.all(np.isfinite(energies)) or min(energies) <= 0
            or not np.all(np.isfinite(strengths)) or min(strengths) < 0
            or not np.isfinite(step_cm1) or step_cm1 <= 0
            or not np.isfinite(temperature_k) or temperature_k < 0
            or not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("Invalid phonon kernel parameters.")
    lattice_positions = np.asarray(energies) / step_cm1
    if not np.allclose(lattice_positions, np.rint(lattice_positions), rtol=0, atol=1e-9):
        raise ValueError("Phonon energies must be exact multiples of the shift-grid step.")
    kernel = np.array([1.0])
    min_index = 0
    missing_bound = 0.0
    orders = []
    for energy, strength, stride in zip(energies, strengths, np.rint(lattice_positions).astype(int)):
        n = phonon_occupation(energy, temperature_k)
        emit, absorb, missing = _poisson_pair(strength * (n + 1), strength * n,
                                             tolerance / len(energies))
        # Emitted minus absorbed phonons: q=-N,...,+N (Skellam distribution).
        mode = np.convolve(emit, absorb[::-1])
        order = len(emit) - 1
        expanded = np.zeros(len(kernel) + 2 * order * stride)
        for position, probability in enumerate(mode):
            start = position * stride
            expanded[start:start + len(kernel)] += probability * kernel
        kernel = expanded
        min_index -= order * stride
        missing_bound += missing  # Union bound for all omitted event tails.
        orders.append(order)
    kernel /= kernel.sum()  # Conditional normalization; correction bounded above.
    shifts = (min_index + np.arange(len(kernel))) * step_cm1
    positive = kernel > 0  # Remove exact zeros only; no amplitude threshold/floor.
    shifts, kernel = shifts[positive], kernel[positive]
    shifts.setflags(write=False)
    kernel.setflags(write=False)
    return shifts, kernel, float(missing_bound), tuple(orders)


def phonon_kernel(total_coupling, temperature_k=TEMPERATURE_K,
                  tolerance=PHONON_TAIL_TOLERANCE):
    fractions = np.asarray(PHONON_MODE_FRACTIONS, dtype=float)
    if (fractions.shape != (len(PHONON_ENERGIES_CM1),)
            or not np.all(np.isfinite(fractions)) or np.any(fractions < 0)
            or not np.isclose(fractions.sum(), 1.0)
            or not np.isfinite(total_coupling) or total_coupling < 0):
        raise ValueError("Mode fractions must sum to one and coupling must be >= 0.")
    return _cached_phonon_kernel(tuple(PHONON_ENERGIES_CM1),
                                 tuple(total_coupling * fractions),
                                 float(temperature_k), float(PHONON_SHIFT_GRID_STEP_CM1),
                                 float(tolerance))


def broadened_transition_sticks(lower, upper, temperature_k, coupling, tolerance):
    gaps, line_weights = transition_sticks(lower, upper, temperature_k)
    shifts, probabilities, missing, orders = phonon_kernel(coupling, temperature_k, tolerance)
    positions = (gaps.ravel()[:, None] + shifts[None, :]).ravel()
    weights = (line_weights.ravel()[:, None] * probabilities[None, :]).ravel()
    # Exact merging of coincident lines; no binning of electronic energies.
    unique, inverse = np.unique(positions, return_inverse=True)
    combined = np.bincount(inverse, weights=weights, minlength=len(unique))
    positive = combined > 0
    return unique[positive], combined[positive], missing, orders


def gaussian_density(energy_cm1, sticks_cm1, weights, sigma_cm1):
    """Analytic unit-area Gaussian mixture; no FFT noise or numerical floor.

    A distance >40 sigma has exp(-800), which underflows to zero in float64.
    Restricting each sorted query chunk to that interval changes no nonzero
    float64 Gaussian contribution and keeps memory bounded.
    """
    energy = np.asarray(energy_cm1, dtype=float)
    if not np.all(np.isfinite(energy)) or not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
        raise ValueError("Energy must be finite and sigma must be positive.")
    flat = energy.ravel()
    order = np.argsort(flat)
    sorted_energy = flat[order]
    result = np.empty_like(flat)
    for start in range(0, len(flat), 256):
        values = sorted_energy[start:start + 256]
        left = np.searchsorted(sticks_cm1, values[0] - 40 * sigma_cm1, side="left")
        right = np.searchsorted(sticks_cm1, values[-1] + 40 * sigma_cm1, side="right")
        delta = (values[:, None] - sticks_cm1[None, left:right]) / sigma_cm1
        result[order[start:start + 256]] = np.sum(
            np.exp(-0.5 * delta**2) * weights[None, left:right], axis=1)
    result /= np.sqrt(2 * np.pi) * sigma_cm1
    return float(result[0]) if energy.ndim == 0 else result.reshape(energy.shape)


def merge_sticks(sticks, weights):
    unique, inverse = np.unique(sticks, return_inverse=True)
    return unique, np.bincount(inverse, weights=weights, minlength=len(unique))


class StarkSpectrum:
    """Peak-normalized spectrum with per-channel phonon coupling and width.

    Naast het volledige spectrum (met fononzijbanden) wordt per kanaal ook de
    nul-fonon-lijn (zuiver elektronische overgangen) bijgehouden, zodat die
    met dezelfde normalisatie kan worden geplot (zie electronic_only).
    """
    def __init__(self, channels, temperature_k=TEMPERATURE_K,
                 channel_couplings=None, channel_sigmas_cm1=None,
                 tolerance=PHONON_TAIL_TOLERANCE):
        self.channels = channels
        self.temperature_k = temperature_k
        self.channel_couplings = dict(channel_couplings or {})
        self.channel_sigmas_cm1 = dict(channel_sigmas_cm1 or {})
        if (set(self.channel_couplings) - set(channels)
                or set(self.channel_sigmas_cm1) - set(channels)):
            raise ValueError("Unknown channel in width/coupling settings.")
        self.weight_sum = sum(weight for _, _, weight in channels.values())
        if (not np.isfinite(self.weight_sum) or self.weight_sum <= 0
                or any(not np.isfinite(w) or w < 0 for _, _, w in channels.values())
                or not np.isfinite(INSTRUMENT_SIGMA_CM1) or INSTRUMENT_SIGMA_CM1 < 0):
            raise ValueError("Invalid channel weights or instrument width.")
        self.prepared = {}
        self.electronic = {}  # Nul-fonon-lijnen: (sticks, gewichten, sigma, kanaalgewicht)
        self.diagnostics = {}
        for name, (lower, upper, weight) in channels.items():
            if weight == 0:
                continue
            coupling = self.channel_couplings.get(name, 0.0)
            base_sigma = self.channel_sigmas_cm1.get(name, STARK_SIGMA_CM1)
            if not np.isfinite(base_sigma) or base_sigma <= 0:
                raise ValueError("Electronic Gaussian sigma must be positive.")
            sigma = float(np.hypot(base_sigma, INSTRUMENT_SIGMA_CM1))
            sticks, amplitudes, missing, orders = broadened_transition_sticks(
                lower, upper, temperature_k, coupling, tolerance)
            channel_weight = weight / self.weight_sum
            self.prepared[name] = (sticks, amplitudes, sigma, channel_weight)
            shifts, kernel, _, _ = phonon_kernel(coupling, temperature_k, tolerance)
            no_event_weight = float(np.exp(-sum(
                coupling * fraction * (2 * phonon_occupation(e, temperature_k) + 1)
                for e, fraction in zip(PHONON_ENERGIES_CM1, PHONON_MODE_FRACTIONS))))
            self.diagnostics[name] = {
                "total_huang_rhys_S": float(coupling),
                "electronic_sigma_cm1": float(base_sigma),
                "kernel_mass": float(kernel.sum()),
                "omitted_kernel_mass_upper_bound": missing,
                "max_events_per_mode_each_sign": list(orders),
                "no_phonon_event_weight": no_event_weight,
                "net_zero_energy_shift_weight": float(kernel[shifts == 0].sum()),
                "integrated_stick_weight": float(amplitudes.sum()),
            }
            # Zuiver elektronische overgangen: de kale Stark-lijnen, geschaald met
            # het gewicht van "geen enkel fononevent" in het volledige model.
            gaps, line_weights = transition_sticks(lower, upper, temperature_k)
            e_sticks, e_weights = merge_sticks(gaps.ravel(), line_weights.ravel())
            self.electronic[name] = (e_sticks, no_event_weight * e_weights,
                                     sigma, channel_weight)
        # Global peak normalization includes the full retained sideband support.
        lo = max(1.0, min(s[0] - 10 * sig for s, _, sig, _ in self.prepared.values()))
        hi = max(s[-1] + 10 * sig for s, _, sig, _ in self.prepared.values())
        min_sigma = min(sig for _, _, sig, _ in self.prepared.values())
        energy = np.linspace(lo, hi, max(20001, int((hi - lo) / min_sigma * 20) + 1))
        self.stark_norm = float(self.raw_stark(energy).max())
        self.norm = 1.0
        if not np.isfinite(self.stark_norm) or self.stark_norm <= 0:
            raise ValueError("Invalid spectrum normalization.")

    def raw_stark(self, energy):
        result = np.zeros_like(np.asarray(energy, dtype=float))
        for sticks, weights, sigma, channel_weight in self.prepared.values():
            result += channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return result

    def raw_electronic(self, energy):
        """Som van de nul-fonon-lijnen over alle kanalen (zelfde eenheid als raw_stark)."""
        result = np.zeros_like(np.asarray(energy, dtype=float))
        for sticks, weights, sigma, channel_weight in self.electronic.values():
            result += channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return result

    def mixed(self, wavelength_m):
        wavelength = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Wavelength must be finite and positive.")
        return self.raw_stark(1e-2 / wavelength) / self.stark_norm

    def factor(self, wavelength_m):
        result = self.mixed(wavelength_m)
        return float(result) if np.ndim(result) == 0 else result

    def electronic_only(self, wavelength_m):
        """Zuiver elektronische bijdrage (zonder fononevents), met dezelfde
        piek-normalisatie als factor(); som met de zijbanden = factor()."""
        wavelength = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Wavelength must be finite and positive.")
        result = self.raw_electronic(1e-2 / wavelength) / self.stark_norm
        return float(result) if np.ndim(result) == 0 else result

    def channel_components(self, name, energy):
        """No-phonon-event component and remaining vibronic contributions.

        The remainder includes phonon processes with zero NET energy shift;
        it must not be equated to simply everything outside the central line.
        """
        sticks, weights, sigma, channel_weight = self.electronic[name]
        no_event = channel_weight * gaussian_density(energy, sticks, weights, sigma)
        sticks, weights, sigma, channel_weight = self.prepared[name]
        total = channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return no_event, np.maximum(total - no_event, 0.0)


def _reference_error_bound(spectrum):
    """Relatieve fout op het 1064 nm-anker door weggelaten kernelmassa."""
    raw_reference = float(spectrum.raw_stark(1e7 / REFERENCE_WAVELENGTH_NM))
    error_bound = 0.0
    for name, (_, _, sigma, channel_weight) in spectrum.prepared.items():
        missing = spectrum.diagnostics[name]["omitted_kernel_mass_upper_bound"]
        error_bound += channel_weight * 2 * missing / (np.sqrt(2 * np.pi) * sigma)
    return error_bound / raw_reference


def _build_spectra(temperature_k=TEMPERATURE_K, coupling_h5=GSA_H5_HUANG_RHYS,
                   coupling_f4=GSA_F4_HUANG_RHYS, tolerance=PHONON_TAIL_TOLERANCE,
                   coupling_h4=ESA_H4_HUANG_RHYS, coupling_f3=ESA_F3_HUANG_RHYS,
                   coupling_f2=ESA_F2_HUANG_RHYS):
    gsa = StarkSpectrum(GSA_CHANNELS, temperature_k,
                        {"3H6 -> 3H5": coupling_h5, "3H6 -> 3F4": coupling_f4},
                        {"3H6 -> 3H5": GSA_H5_SIGMA_CM1}, tolerance)
    esa = StarkSpectrum(ESA_CHANNELS, temperature_k,
                        {"3F4 -> 3H4": coupling_h4, "3F4 -> 3F3": coupling_f3,
                         "3F4 -> 3F2": coupling_f2},
                        tolerance=tolerance)
    g_ref = gsa.factor(REFERENCE_WAVELENGTH_NM * 1e-9) if SI_REFERENCE_ANCHOR else 1.0
    e_ref = esa.factor(REFERENCE_WAVELENGTH_NM * 1e-9) if SI_REFERENCE_ANCHOR else 1.0
    if min(g_ref, e_ref) <= 0 or not np.all(np.isfinite([g_ref, e_ref])):
        raise ValueError("Reference absorption is zero/non-finite; revise coupling or reference. "
                         "A purely narrow electronic H5 band cannot be anchored at 1064 nm.")
    # A tiny integrated omission can dominate an extremely weak reference.
    # Bound the density error using the maximum of a unit-area Gaussian.
    if SI_REFERENCE_ANCHOR:
        for label, spectrum in (("GSA", gsa), ("ESA", esa)):
            relative_bound = _reference_error_bound(spectrum)
            if relative_bound > REFERENCE_MAX_RELATIVE_KERNEL_ERROR:
                raise ValueError(f"{label} reference is not resolved accurately by this "
                                 "phonon cutoff. Reduce --kernel-tolerance or revise the "
                                 f"reference/model. Relative error bound: {relative_bound:.3g}.")
            spectrum.reference_relative_kernel_error_bound = float(relative_bound)
    return gsa, esa, g_ref, e_ref


GSA_SPECTRUM = ESA_SPECTRUM = None
_GSA_REFERENCE_FACTOR = _ESA_REFERENCE_FACTOR = None


def configure_spectra(temperature_k=TEMPERATURE_K, coupling_h5=GSA_H5_HUANG_RHYS,
                       coupling_f4=GSA_F4_HUANG_RHYS, tolerance=PHONON_TAIL_TOLERANCE,
                       coupling_h4=ESA_H4_HUANG_RHYS, coupling_f3=ESA_F3_HUANG_RHYS,
                       coupling_f2=ESA_F2_HUANG_RHYS):
    global GSA_SPECTRUM, ESA_SPECTRUM, _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR
    (GSA_SPECTRUM, ESA_SPECTRUM,
     _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR) = _build_spectra(
        temperature_k, coupling_h5, coupling_f4, tolerance,
        coupling_h4, coupling_f3, coupling_f2)


def _stark_resonance_factors(pump_wavelength_nm, p=None):
    if GSA_SPECTRUM is None:
        configure_spectra()
    wavelength_m = np.asarray(pump_wavelength_nm, dtype=float) * 1e-9
    gsa = GSA_SPECTRUM.factor(wavelength_m)
    esa = ESA_SPECTRUM.factor(wavelength_m)
    if SI_REFERENCE_ANCHOR:
        gsa /= _GSA_REFERENCE_FACTOR
        esa /= _ESA_REFERENCE_FACTOR
    return gsa, esa


def _electronic_resonance_factors(pump_wavelength_nm):
    """Zuiver elektronische bijdrage, met DEZELFDE normalisatie en 1064 nm-anker
    als _stark_resonance_factors (dus direct vergelijkbaar met de volle curves)."""
    if GSA_SPECTRUM is None:
        configure_spectra()
    wavelength_m = np.asarray(pump_wavelength_nm, dtype=float) * 1e-9
    gsa = GSA_SPECTRUM.electronic_only(wavelength_m)
    esa = ESA_SPECTRUM.electronic_only(wavelength_m)
    if SI_REFERENCE_ANCHOR:
        gsa = gsa / _GSA_REFERENCE_FACTOR
        esa = esa / _ESA_REFERENCE_FACTOR
    return gsa, esa


def _check_wavelengths(wavelengths_nm):
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if (wavelengths.ndim != 1 or not wavelengths.size
            or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0)):
        raise ValueError("Provide a nonempty 1D array of positive finite wavelengths.")
    return wavelengths


def compute_factors(wavelengths_nm):
    wavelengths = _check_wavelengths(wavelengths_nm)
    return tuple(np.asarray(x) for x in _stark_resonance_factors(wavelengths))


def compute_electronic_factors(wavelengths_nm):
    """(f_gsa_elec, f_esa_elec): bijdrage van overgangen zonder fononen."""
    wavelengths = _check_wavelengths(wavelengths_nm)
    return tuple(np.asarray(x) for x in _electronic_resonance_factors(wavelengths))


def plot_factors(wavelengths, f_gsa, f_esa, f_gsa_el, f_esa_el, basename, show=True):
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(11, 6.8))
        fig.subplots_adjust(left=.11, right=.97, bottom=.19, top=.76)
        fig.suptitle("GSA en ESA met fononzijdenbanden | Tm³⁺:NaYF₄", x=.11, y=.96,
                     ha="left", fontsize=19, fontweight="bold")
        subtitle = ("Relatieve doorsneden, elk genormaliseerd op 1064 nm"
                    if SI_REFERENCE_ANCHOR else "Relatieve doorsneden, elk genormaliseerd op eigen piek")
        fig.text(.11, .885, subtitle, color="#526171")
        ax.plot(wavelengths, np.where(f_gsa > 0, f_gsa, np.nan), color="#ba4a18",
                lw=2, label=r"$f_{\mathrm{GSA}}$ met fononen", zorder=3)
        ax.plot(wavelengths, np.where(f_esa > 0, f_esa, np.nan), color="#1468b1",
                lw=1.8, label=r"$f_{\mathrm{ESA}}$ met fononen", zorder=3)
        ax.plot(wavelengths, np.where(f_gsa_el > 0, f_gsa_el, np.nan), color="#2e9e5b",
                lw=1.6, ls="--", label=r"$f_{\mathrm{GSA}}$ zonder fononen (alleen overgangen)",
                zorder=4)
        ax.plot(wavelengths, np.where(f_esa_el > 0, f_esa_el, np.nan), color="#8a3ffc",
                lw=1.6, ls="--", label=r"$f_{\mathrm{ESA}}$ zonder fononen (alleen overgangen)",
                zorder=4)
        ax.set_yscale("log")
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_xticks(np.arange(900, 2001, 100))
        positive = np.concatenate((f_gsa[f_gsa > 0], f_esa[f_esa > 0]))
        ax.set_ylim(PLOT_MIN_FACTOR, 10 ** np.ceil(np.log10(positive.max())))
        ax.set_xlabel("Pompgolflengte (nm)")
        ax.set_ylabel("Relatieve absorptiedoorsnede f")
        ax.grid(which="major", color="#dce3eb", lw=.7)
        ax.axvline(REFERENCE_WAVELENGTH_NM, color="#687684", ls="--", lw=1)
        ax.axhline(1., color="#9ba6b1", ls=":", lw=1)
        if SI_REFERENCE_ANCHOR:
            ax.plot([REFERENCE_WAVELENGTH_NM], [1.], "o", color="#465360", ms=4)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
                  frameon=False, fontsize=9.5, borderaxespad=0)
        gc, ec = GSA_SPECTRUM.channel_couplings, ESA_SPECTRUM.channel_couplings
        fig.text(.11, .095, f"Fononen: {PHONON_DESCRIPTION} | "
                 f"T={GSA_SPECTRUM.temperature_k:g} K", fontsize=9, color="#526171")
        fig.text(.11, .065, f"S GSA: H5={gc['3H6 -> 3H5']:g}, F4={gc['3H6 -> 3F4']:g} | "
                 f"S ESA: H4={ec['3F4 -> 3H4']:g}, F3={ec['3F4 -> 3F3']:g}, "
                 f"F2={ec['3F4 -> 3F2']:g}", fontsize=9, color="#526171")
        fig.text(.11, .035, "Koppelingen, bandvorm en kernbreedtes zijn ongekalibreerde modelaannames.",
                 fontsize=9, color="#526171")
        fig.savefig(str(basename) + ".png", dpi=200, facecolor="white")
        fig.savefig(str(basename) + ".svg", facecolor="white")

        # Display all figures together.
        # block=False prevents Spyder from remaining busy until the plot windows
        # are manually closed.
        if show:
            plt.show(block=False)
            plt.pause(0.1)


def plot_band_profile(args, profile, show=True):
    """Plot the profile separately from its trapezoidal integration weights."""
    import matplotlib.pyplot as plt
    if args.phonon_model != "band":
        return
    nodes, trapezoid = band_nodes(args.band_min, args.band_max, args.band_step)
    fractions = band_fractions(nodes, trapezoid, profile, args.band_mean,
                               args.band_sigma, args.band_file)
    density = fractions / (trapezoid * args.band_step)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if profile == "raman_tm":
        x, y = RAMAN_TM_PROFILE_POINTS.T
        dense = np.linspace(args.band_min, args.band_max, 901)
        normalization = np.sum(np.interp(nodes, x, y) * trapezoid) * args.band_step
        ax.plot(dense, np.interp(dense, x, y) / normalization,
                color="#1468b1", label="Approximate Yb/Tm Raman trace")
    ax.plot(nodes, density, "o", color="#ba4a18", label="Kernel integration nodes")
    if profile != "raman_tm":
        ax.plot(nodes, density, color="#1468b1")
    ax.set(xlabel=r"Phonon wavenumber $\omega$ (cm$^{-1}$)",
           ylabel=r"Normalized profile $g(\omega)$ [(cm$^{-1}$)$^{-1}$]",
           xlim=(args.band_min, args.band_max))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    basename = args.output_dir / "phonon_band_profile"
    fig.savefig(str(basename) + ".png", dpi=200)
    fig.savefig(str(basename) + ".svg")
    np.savetxt(str(basename) + ".csv",
               np.column_stack((nodes, density, fractions)), delimiter=",",
               header="wavenumber_cm1,normalized_profile,mode_fraction", comments="")
    if show:
        plt.show(block=False)
        plt.pause(.1)
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GSA/ESA met NaYF4-fononband (integratie over golfgetal)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K)
    parser.add_argument("--coupling-h5", type=float, default=GSA_H5_HUANG_RHYS,
                        help="S voor GSA 3H6 -> 3H5")
    parser.add_argument("--coupling-f4", type=float, default=GSA_F4_HUANG_RHYS,
                        help="S voor GSA 3H6 -> 3F4")
    parser.add_argument("--coupling-h4", type=float, default=ESA_H4_HUANG_RHYS,
                        help="S voor ESA 3F4 -> 3H4")
    parser.add_argument("--coupling-f3", type=float, default=ESA_F3_HUANG_RHYS,
                        help="S voor ESA 3F4 -> 3F3")
    parser.add_argument("--coupling-f2", type=float, default=ESA_F2_HUANG_RHYS,
                        help="S voor ESA 3F4 -> 3F2")
    parser.add_argument("--kernel-tolerance", type=float, default=PHONON_TAIL_TOLERANCE)
    parser.add_argument("--phonon-model", choices=("band", "discrete"), default=PHONON_MODEL)
    parser.add_argument("--band-min", type=float, default=BAND_MIN_CM1)
    parser.add_argument("--band-max", type=float, default=BAND_MAX_CM1)
    parser.add_argument("--band-step", type=float, default=BAND_STEP_CM1)
    parser.add_argument("--band-mean", type=float, default=BAND_MEAN_CM1)
    parser.add_argument("--band-sigma", type=float, default=BAND_SIGMA_CM1)
    parser.add_argument("--band-profile", choices=("raman_tm", "gaussian", "uniform", "file"),
                        default=BAND_PROFILE)
    parser.add_argument("--band-file", type=Path, default=None,
                        help="CSV golfgetal_cm1,intensiteit (zet profiel automatisch op 'file')")
    parser.add_argument("--profile-only", action="store_true",
                        help="Plot/export the band profile without calculating spectra")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    profile = "file" if args.band_file is not None else args.band_profile
    phonon_info = set_phonon_model(args.phonon_model, args.band_min, args.band_max,
                                   args.band_step, profile, args.band_mean,
                                   args.band_sigma, args.band_file)
    print(f"Fononen: {PHONON_DESCRIPTION}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.profile_only:
        if args.data_only or args.phonon_model != "band":
            parser.error("--profile-only needs band mode and cannot combine with --data-only")
        plot_band_profile(args, profile, show=not args.no_show)
        return
    configure_spectra(args.temperature, args.coupling_h5, args.coupling_f4,
                      args.kernel_tolerance, args.coupling_h4, args.coupling_f3,
                      args.coupling_f2)
    if WAVELENGTH_MAX_NM <= WAVELENGTH_MIN_NM or WAVELENGTH_STEP_NM <= 0:
        raise ValueError("Invalid wavelength range/step.")
    count = int(np.ceil((WAVELENGTH_MAX_NM - WAVELENGTH_MIN_NM) / WAVELENGTH_STEP_NM)) + 1
    wavelengths = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, count)
    f_gsa, f_esa = compute_factors(wavelengths)
    f_gsa_el, f_esa_el = compute_electronic_factors(wavelengths)
    if (np.any(~np.isfinite(f_gsa)) or np.any(~np.isfinite(f_esa))
            or np.any(~np.isfinite(f_gsa_el)) or np.any(~np.isfinite(f_esa_el))):
        raise RuntimeError("Non-finite spectrum factors.")
    reference = _stark_resonance_factors(REFERENCE_WAVELENGTH_NM)
    if SI_REFERENCE_ANCHOR:
        np.testing.assert_allclose(reference, (1., 1.), rtol=1e-12, atol=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.output_dir / "f_gsa_f_esa_fononband"
    np.savetxt(str(basename) + ".csv",
               np.column_stack((wavelengths, f_gsa, f_esa, f_gsa_el, f_esa_el)),
               delimiter=",", comments="", fmt="%.12g",
               header="wavelength_nm,f_gsa,f_esa,f_gsa_zonder_fononen,f_esa_zonder_fononen")
    # Gebruikte band: knopen, genormaliseerd gewicht en Huang-Rhys-factor per knoop.
    energies = np.asarray(PHONON_ENERGIES_CM1)
    fractions = np.asarray(PHONON_MODE_FRACTIONS)
    np.savetxt(str(basename) + "_fononband.csv",
               np.column_stack((energies, fractions,
                                args.coupling_h5 * fractions, args.coupling_f4 * fractions,
                                args.coupling_h4 * fractions, args.coupling_f3 * fractions,
                                args.coupling_f2 * fractions)),
               delimiter=",", comments="", fmt="%.12g",
               header="phonon_energy_cm1,fraction,S_k_GSA_3H5,S_k_GSA_3F4,"
                      "S_k_ESA_3H4,S_k_ESA_3F3,S_k_ESA_3F2")
    metadata = {
        "model": "Independent harmonic modes, finite-T Huang-Rhys (Poisson/Skellam); "
                 "phonon band integrated over wavenumber by trapezoid quadrature; "
                 "applied to both GSA and ESA channels",
        "temperature_K": args.temperature,
        "phonon": phonon_info,
        "phonon_energies_cm1": list(PHONON_ENERGIES_CM1),
        "mode_fractions": list(PHONON_MODE_FRACTIONS),
        "reference_wavelength_nm": REFERENCE_WAVELENGTH_NM,
        "reference_anchor_enabled": SI_REFERENCE_ANCHOR,
        "reference_peak_normalized_gsa": float(GSA_SPECTRUM.factor(REFERENCE_WAVELENGTH_NM * 1e-9)),
        "reference_peak_normalized_esa": float(ESA_SPECTRUM.factor(REFERENCE_WAVELENGTH_NM * 1e-9)),
        "reference_relative_kernel_error_upper_bound": {
            "GSA": getattr(GSA_SPECTRUM, "reference_relative_kernel_error_bound", None),
            "ESA": getattr(ESA_SPECTRUM, "reference_relative_kernel_error_bound", None)},
        "electronic_only_curves": "Nul-fonon-lijnen (geen fononevent) van het volledige model, "
                                  "zelfde piek- en 1064 nm-normalisatie als de volle curves.",
        "GSA": GSA_SPECTRUM.diagnostics,
        "ESA": ESA_SPECTRUM.diagnostics,
        "assumptions": "Default band is an approximate manually traced Yb/Tm Raman profile used "
                       "as a DOS/coupling proxy. Couplings (GSA and ESA), electronic widths and line/channel strengths are "
                       "uncalibrated; no measured phonon DOS or absolute cross-section fit.",
        "sources": ["https://doi.org/10.7150/thno.5113",
                    "https://doi.org/10.1016/j.omx.2021.100102",
                    "https://doi.org/10.1080/0026897021000046889"],
    }
    Path(str(basename) + "_instellingen.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"f_GSA(1064 nm)={reference[0]:.12g}; f_ESA(1064 nm)={reference[1]:.12g}")
    for label, spectrum in (("GSA", GSA_SPECTRUM), ("ESA", ESA_SPECTRUM)):
        for name, info in spectrum.diagnostics.items():
            print(f"{label} {name}: S={info['total_huang_rhys_S']:g}, "
                  f"omitted kernel mass <= {info['omitted_kernel_mass_upper_bound']:.3g}")
    print(f"Data saved: {basename}.csv")
    if not args.data_only:
        plot_band_profile(args, profile, show=not args.no_show)
        plot_factors(wavelengths, f_gsa, f_esa, f_gsa_el, f_esa_el, basename,
                     show=not args.no_show)
        print(f"Plot saved: {basename}.png / .svg")


if __name__ == "__main__":
    main()

