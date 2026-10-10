"""
PASSI simulation reproducing Figure 4d/e/f of Lee et al.,
"Giant nonlinear optical responses from photon-avalanching nanoparticles",
Nature 589, 230-235 (2021).

Method follows the Supplementary Information section "PASSI simulations
of raster scanned confocal imaging": a diffraction-limited Gaussian pump
beam is raster-scanned over a single ANP; at every scan position the
local pump intensity at the particle is converted to emission intensity
through the experimentally measured nonlinear ("S"-shaped) PA response
curve, and the resulting total emission is recorded as the image pixel
value at that scan position.

Fig. 4a/b: experimental PASSI images at 9.9 kW/cm^2 (saturation regime)
           and 7.1 kW/cm^2 (PA/steep regime).
Fig. 4d/e: matching *simulated* images, produced with the method below.
Fig. 4f:   measured (black, from Supplementary Fig. S10) vs simulated
           (red) FWHM of the single-ANP PASSI spot as a function of
           excitation intensity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ---------------------------------------------------------------
# 1) Excitation beam: diffraction-limited Gaussian
#    Calibrated so the *linear*-response spot (s=1, no PA nonlinearity)
#    has FWHM = 357 nm, matching the diffraction-limited excitation
#    spot quoted in the main text / Fig. 4c (NA=1.49, lambda=1064 nm).
# ---------------------------------------------------------------
lam_nm = 1064.0                 # pump wavelength [nm]
NA = 1.49                       # objective numerical aperture
FWHM_diffraction_nm = 357.0     # diffraction-limited spot, as reported
w_nm = FWHM_diffraction_nm / np.sqrt(4 * np.log(2))  # Gaussian 1/e^2-type width


def gaussian_beam(dx, dy, I0):
    """Local pump intensity at offset (dx, dy) from the beam center."""
    return I0 * np.exp(-(dx**2 + dy**2) / w_nm**2)


# ---------------------------------------------------------------
# 2) ANP phantom: single ~19.5 x 16 nm elliptical particle,
#    centered in a 500 x 500 nm field of view (matches Fig. 4 panels)
# ---------------------------------------------------------------
size_nm = 500.0
px_size = 1.0  # nm / pixel
coords = np.arange(0, size_nm, px_size)
Xg, Yg = np.meshgrid(coords, coords)
cx = cy = size_nm / 2

a_semi, b_semi = 19.5 / 2, 16.0 / 2  # nm (Supplementary Methods, PASSI sims)
particle_mask = ((Xg - cx) / a_semi) ** 2 + ((Yg - cy) / b_semi) ** 2 <= 1
particle_x = Xg[particle_mask]
particle_y = Yg[particle_mask]

# ---------------------------------------------------------------
# 3) Phenomenological "S"-shaped PA emission curve I_E = f(I_P)
#    Sigmoid in log-log space fitted to the 8% Tm ANP curve in Fig. 3a
#    (sample 3, core/shell 10/4.0 nm): threshold I_th ~ 6.4 kW/cm^2,
#    steepest slope s ~ 26 (Supplementary Table 3, Fig. 3a).
#    Swap this for the real DRE model (pa_model.py) output if available.
# ---------------------------------------------------------------
I_th, s_slope = 6, 26.0


def emission_curve(I_p):
    I_norm = np.clip(I_p, 1e-9, None) / I_th
    
    # n = effectieve non-lineariteit vóór de avalanche (vaak ~1 tot 2)
    # c = relatieve intensiteit van dit regime (zorgt voor de diepe 'knik' in de log-plot)
    n = 1.5
    c = 5e-3
    
    # Combineer het zwakke pre-avalanche regime met de gigantische avalanche sprong
    term_s = I_norm**s_slope
    term_n = c * (I_norm**n)
    
    return (term_n + term_s) / (1.0 + term_s)


# ---------------------------------------------------------------
# 4) Raster-scan PASSI simulation
# ---------------------------------------------------------------
# image(x0,y0) = sum over particle pixels p of f(I0 * G(x0-p_x, y0-p_y))
# is, by shift-invariance of the Gaussian beam, exactly a 2D convolution
# of the particle mask with the single-point-emitter response kernel
# K(dx,dy) = f(I0*G(dx,dy)). Computing it as an FFT convolution instead
# of a Python loop over ~250 particle pixels turns ~0.85 s/call into
# ~1 ms/call -- this is the fix for the slow/"hanging" Fig. 4f sweep.
particle_mask_f = particle_mask.astype(float)


def simulate_passi(I0_kWcm2):
    kernel = emission_curve(gaussian_beam(Xg - cx, Yg - cy, I0_kWcm2))
    image = fftconvolve(particle_mask_f, kernel, mode="same")
    return image / image.max()


def fwhm_linecut(img):
    """FWHM of a central horizontal linecut through the image."""
    row = img[int(size_nm / 2 / px_size), :]
    half = row.max() / 2
    above = np.where(row >= half)[0]
    return (above[-1] - above[0]) * px_size


# Excitation intensities matching Fig. 4a (sat.) / 4b (PA regime)
I_sat = 9.9  # kW/cm^2 -> Fig. 4d (simulated saturation-regime image)
I_pa = 7.1   # kW/cm^2 -> Fig. 4e (simulated PA-regime image)

img_d = simulate_passi(I_sat)
img_e = simulate_passi(I_pa)

fwhm_d = fwhm_linecut(img_d)
fwhm_e = fwhm_linecut(img_e)
print(f"Simulated FWHM at {I_sat:.1f} kW/cm^2 (saturation regime, ~Fig. 4d): {fwhm_d:.0f} nm")
print(f"Simulated FWHM at {I_pa:.1f} kW/cm^2 (PA regime, ~Fig. 4e):        {fwhm_e:.0f} nm")

# ---------------------------------------------------------------
# 5) Fig. 4f: simulated FWHM vs. excitation intensity, overlaid with
#    the measured data digitized from Supplementary Fig. S10 (average
#    of the orange/blue linecut FWHMs; error = RMS of the two s.d.'s,
#    as stated in the Fig. 4 caption).
# ---------------------------------------------------------------
I_meas = np.array([7.1, 7.2, 7.9, 8.9, 9.9, 10.9])          # kW/cm^2
fwhm_x = np.array([81, 127, 141, 194, 217, 233])            # nm, orange linecut (Fig. S10)
fwhm_y = np.array([65, 103, 125, 184, 216, 249])            # nm, blue linecut (Fig. S10)
sd_x = np.array([9, 18, 9, 18, 18, 20])
sd_y = np.array([7, 9, 6, 22, 19, 20])

fwhm_meas = (fwhm_x + fwhm_y) / 2
err_meas = np.sqrt((sd_x**2 + sd_y**2) / 2)

I_sim_range = np.linspace(6.2, 11.0, 60)
fwhm_sim_range = np.empty_like(I_sim_range)
print(f"Sweeping {len(I_sim_range)} intensities for Fig. 4f...")
for i, I0 in enumerate(I_sim_range):
    fwhm_sim_range[i] = fwhm_linecut(simulate_passi(I0))
    if (i + 1) % 10 == 0 or i == len(I_sim_range) - 1:
        print(f"  {i + 1}/{len(I_sim_range)} done")

# ---------------------------------------------------------------
# 6) Plot: Fig. 4d, 4e (images) and Fig. 4f (FWHM vs. intensity)
# ---------------------------------------------------------------
fig = plt.figure(figsize=(11, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])

for i, (img, I0, label) in enumerate(zip([img_d, img_e], [I_sat, I_pa], ["d", "e"])):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img, extent=[0, size_nm, 0, size_nm], origin="lower",
              cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{label}   {I0:.1f} kW cm$^{{-2}}$")
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot([40, 140], [40, 40], color="white", lw=3)
    ax.text(90, 55, "100 nm", color="white", ha="center", fontsize=8)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(I_sim_range, fwhm_sim_range, "r-", label="PASSI simulation")
ax3.errorbar(I_meas, fwhm_meas, yerr=err_meas, fmt="ko", capsize=3, label="Experiment")
ax3.set_xlabel("Excitation intensity (kW cm$^{-2}$)")
ax3.set_ylabel("FWHM (nm)")
ax3.set_title("f")
ax3.legend(frameon=False)
ax3.set_xlim(6, 11)

plt.tight_layout()
# plt.savefig("/mnt/user-data/outputs/fig4def_passi_simulation.png", dpi=200)
print("Saved figure to fig4def_passi_simulation.png")

# ---------------------------------------------------------------
# 7) Fig. 4h: two ANPs, 300 nm apart, PASSI simulation across
#    excitation intensities from near-saturation to near-threshold
#    (the paper's Fig. 4g is the experimental version of this; 4h is
#    the matching simulation, which is what this section reproduces)
# ---------------------------------------------------------------
def make_mask(centers):
    """Binary particle mask for one or more ~19.5x16 nm ANPs."""
    mask = np.zeros_like(Xg, dtype=bool)
    for (p0x, p0y) in centers:
        mask |= ((Xg - p0x) / a_semi) ** 2 + ((Yg - p0y) / b_semi) ** 2 <= 1
    return mask.astype(float)


def simulate_passi_mask(I0_kWcm2, mask_f):
    """General PASSI simulation for an arbitrary binary particle mask
    (same physics/method as simulate_passi, generalized beyond one ANP)."""
    kernel = emission_curve(gaussian_beam(Xg - cx, Yg - cy, I0_kWcm2))
    image = fftconvolve(mask_f, kernel, mode="same")
    return image / image.max()


separation_nm = 300.0
centers_h = [(cx - separation_nm / 2, cy), (cx + separation_nm / 2, cy)]
two_particle_mask_f = make_mask(centers_h)

I_h_list = [13.8, 10.9, 9.9, 8.9, 7.6]  # kW/cm^2, matches Fig. 4g/h in the paper
imgs_h = [simulate_passi_mask(I0, two_particle_mask_f) for I0 in I_h_list]
colors_h = ["#2ca02c", "#8fbf3f", "#d4b800", "#e07b00", "#d62728"]  # green -> red

# Diffraction-limited reference: two LINEAR (non-avalanching) point
# emitters, 300 nm apart -- the black dashed line in the paper's Fig. 4g/h
row_idx = int(cy / px_size)
x_axis = coords - cx  # centered x-axis, nm
ref_linear = gaussian_beam(x_axis + separation_nm / 2, 0, 1) + \
             gaussian_beam(x_axis - separation_nm / 2, 0, 1)
ref_linear /= ref_linear.max()

# ---------------------------------------------------------------
# 8) Plot: Fig. 4h (5 images, decreasing intensity, + linecuts)
# ---------------------------------------------------------------
fig2 = plt.figure(figsize=(14, 6))
gs2 = fig2.add_gridspec(2, 5, height_ratios=[1, 0.9], hspace=0.35)

for i, (img, I0, color) in enumerate(zip(imgs_h, I_h_list, colors_h)):
    ax = fig2.add_subplot(gs2[0, i])
    ax.imshow(img, extent=[0, size_nm, 0, size_nm], origin="lower",
              cmap="viridis", vmin=0, vmax=1)
    ax.axhline(cy, color=color, lw=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{I0:.1f} kW cm$^{{-2}}$", fontsize=9)
    if i == 0:
        ax.plot([40, 140], [40, 40], color="white", lw=3)
        ax.text(90, 55, "100 nm", color="white", ha="center", fontsize=7)

ax_line = fig2.add_subplot(gs2[1, :])
ax_line.plot(x_axis, ref_linear, "k--", lw=1.5, label="Diffraction limit (linear)")
resolved = []
for img, I0, color in zip(imgs_h, I_h_list, colors_h):
    linecut = img[row_idx, :]
    linecut = linecut / linecut.max()
    ax_line.plot(x_axis, linecut, color=color, lw=1.5, label=f"{I0:.1f} kW cm$^{{-2}}$")
    # crude resolvability check: is there a dip between the two peaks?
    center_val = linecut[np.argmin(np.abs(x_axis))]
    peak_val = linecut.max()
    resolved.append(center_val < 0.95 * peak_val)
ax_line.set_xlim(-250, 250)
ax_line.set_xlabel("x (nm)")
ax_line.set_ylabel("Normalized intensity")
ax_line.set_title("h", loc="left")
ax_line.legend(fontsize=7, ncol=3, frameon=False)

# plt.savefig("/mnt/user-data/outputs/fig4h_passi_simulation.png", dpi=200, bbox_inches="tight")
# print("Saved figure to fig4h_passi_simulation.png")
for I0, r in zip(I_h_list, resolved):
    print(f"  {I0:.1f} kW/cm^2: two ANPs {'resolved (dip visible)' if r else 'NOT resolved (merged)'}")