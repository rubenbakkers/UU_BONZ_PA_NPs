# Four Gaussian phonon bands with selectable multiphonon order

`gsa_esa_four_gaussian_multiphonon.py` extends the supplied standalone GSA/ESA spectrum model so that one electronic absorption can involve **up to 1, 2, 3, or 4 phonons**. It includes sums, differences, and repeated use of the same phonon band. The original supplied script is preserved in `backups/gsa_esa_four_gaussian_single_phonon_original.py`.

The number controls how many individual phonons participate in **one photon absorption**. It does not count phonon pairs, distinct bands, or absorbed photons. The electronic Stark levels, their original weighting, and the electronic width are retained.

## Run and choose the number of phonons

Install NumPy and Matplotlib using the accompanying `requirements.txt`, then run from this directory:

```powershell
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 1 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 2 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 3 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 4 --no-show
```

The default is **4**. To change the default in the script, edit:

```python
DEFAULT_MAX_PHONONS = 4
```

| Setting | Orders included in each absorption |
| --- | --- |
| `--max-phonons 1` | 0 and 1 phonon; reproduces the original single-phonon model |
| `--max-phonons 2` | 0, 1, and 2 phonons |
| `--max-phonons 3` | 0, 1, 2, and 3 phonons |
| `--max-phonons 4` | 0, 1, 2, 3, and 4 phonons |

This is a **maximum**, not an instruction to use exactly that number every time. Phonon participation is probabilistic; the coupling controls the weights of the different orders.

Other useful commands:

```powershell
# Use only bands 1 and 3, allowing repeated and mixed events up to order 3.
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 3 --phonon-bands 1 3 --no-show

# Disable every phonon contribution.
python gsa_esa_four_gaussian_multiphonon.py --no-phonons --no-show

# Equivalent electronic-only limit through zero coupling.
python gsa_esa_four_gaussian_multiphonon.py --coupling 0 --data-only

# Eliminate thermal phonon consumption; creation remains possible.
python gsa_esa_four_gaussian_multiphonon.py --temperature 0 --max-phonons 4 --no-show

# Inspect weak sidebands and choose a separate output location.
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 4 --log-y --no-show --output-dir results/N4
```

`--data-only` needs only NumPy and produces CSV/JSON files. `--no-show` still saves plots but closes the plot windows. `--phonon-bands` selects band numbers; `--max-phonons` selects the maximum total number of phonons. These controls serve different purposes.

## Physical idea: add phonon energies to an electronic transition

All energies in the calculation are written as wavenumbers in cm⁻¹. The photon wavenumber is

\[
\widetilde\nu_\gamma = \frac{10^7}{\lambda_{\rm nm}}.
\]

For a lower Stark level \(E_i\) and an upper Stark level \(E_j\), the electronic gap is \(\Delta_{ij}=E_j-E_i\). Energy conservation for phonon-assisted **absorption** is

\[
\widetilde\nu_\gamma
=\Delta_{ij}
+\sum_k c_k^+\omega_k
-\sum_k c_k^-\omega_k.
\]

Here \(c_k^+\) counts phonons **created** in the lattice and \(c_k^-\) counts phonons **consumed** from the lattice. Creation requires the photon to supply more energy than the electronic gap. Consumption supplies part of the electronic excitation energy from the lattice, so the photon needs less energy. Larger photon wavenumber corresponds to shorter wavelength.

The four Gaussian phonon bands are unchanged from the supplied script:

| Band \(k\) | Centre \(\omega_k\), cm⁻¹ | Standard deviation \(\sigma_k\), cm⁻¹ | Relative peak height \(h_k\) | Area weight \(w_k\) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 253 | 14 | 0.68 | 0.18718 |
| 2 | 306 | 24 | 0.95 | 0.44829 |
| 3 | 364 | 15 | 0.66 | 0.19465 |
| 4 | 408 | 24 | 0.36 | 0.16988 |

Examples of the shift relative to \(\Delta_{ij}\):

| Events | Total number \(q\) | Centre shift, cm⁻¹ |
| --- | ---: | ---: |
| Create one band-1 phonon | 1 | \(+253\) |
| Create band 1 and band 2 | 2 | \(253+306=559\) |
| Create two band-2 phonons | 2 | \(306+306=612\) |
| Create band 4 and consume band 1 | 2 | \(408-253=155\) |
| Consume bands 2 and 3 | 2 | \(-306-364=-670\) |
| Create band 2 and consume band 2 | 2 | \(306-306=0\) |
| Create bands 3, 4, and 4 | 3 | \(364+408+408=1180\) |
| Create four band-4 phonons | 4 | \(4\times408=1632\) |

The same band may appear more than once: phonons are not limited to one quantum per band. Creation and consumption can also occur together. The count is

\[
q=\sum_{k=1}^4(c_k^++c_k^-),\qquad 0\le q\le N,
\]

where \(N\) is `--max-phonons`. For example, two created phonons and one consumed phonon count as **three**, even when their net energy nearly cancels.

At 1064 nm, \(\widetilde\nu_\gamma\approx9398.50\) cm⁻¹. For the particular electronic transition from 0 to 8207 cm⁻¹, the excess photon energy is approximately 1191.50 cm⁻¹. The three-phonon creation combination \(364+408+408=1180\) cm⁻¹ places a broadened sideband near this energy. This is an example of a newly explicit combination; the full result also includes all other allowed Stark transitions. Finite Gaussian tails mean that a smaller cutoff does not impose an exact hard spectral boundary.

## How the model assigns probabilities

### Band areas, temperature, and coupling

The input Gaussian profile has the form

\[
R_k(\epsilon)=h_k\exp\!\left[-\frac{(\epsilon-\omega_k)^2}{2\sigma_k^2}\right].
\]

Its area is proportional to \(h_k\sigma_k\), so the normalized band weights are

\[
w_k=\frac{h_k\sigma_k}{\sum_jh_j\sigma_j}.
\]

Peak heights alone would overestimate narrow bands relative to broad bands. Switching a band off sets its event intensity to zero. The original four-band weights are retained, so its coupling is not redistributed to the remaining bands.

At temperature \(T\), the mean thermal occupation evaluated at band centre is

\[
\bar n_k(T)=\frac{1}{\exp[\omega_k/(k_BT)]-1},
\qquad k_B=0.6950348005\ {\rm cm^{-1}\,K^{-1}}.
\]

The occupation is an **average**, not the selected integer number of phonons in an absorption. At \(T=0\), the code sets all \(\bar n_k=0\).

For a channel with effective coupling \(S\), define the dimensionless creation and consumption event intensities

\[
\mu_k^+=Sw_k(\bar n_k+1),\qquad
\mu_k^-=Sw_k\bar n_k.
\]

The factor \(\bar n_k+1\) permits phonon creation even at zero temperature. Consumption requires an existing thermal phonon and vanishes at zero temperature. At a band centre,

\[
\frac{\mu_k^-}{\mu_k^+}
=\exp[-\omega_k/(k_BT)].
\]

`--coupling` sets the common base coupling; `CHANNEL_COUPLING_MULTIPLIERS` scales it for individual electronic channels. The default base value \(S=0.05\) is a **model assumption**, not a transition-specific measurement for Tm³⁺:NaYF₄. It controls sideband strength and is not an integer phonon count. When some bands are disabled, the total active coupling decreases.

### Independent counts and the finite cutoff

The multiphonon extension assumes independent Poisson counts for the eight event types: creation and consumption in each of the four bands. Write

\[
M=\sum_k(\mu_k^++\mu_k^-).
\]

Without a cutoff, the probability of a particular count vector \(\mathbf c\) would be

\[
P_\infty(\mathbf c)=e^{-M}
\prod_k\frac{(\mu_k^+)^{c_k^+}}{c_k^+!}
\frac{(\mu_k^-)^{c_k^-}}{c_k^-!}.
\]

The script retains only \(q\le N\) and **renormalizes those events**. Equivalently, it uses this Poisson model conditional on not exceeding the selected cutoff:

\[
D_N=\sum_{r=0}^N\frac{M^r}{r!},
\qquad
P_N(\mathbf c)=\frac{1}{D_N}
\prod_k\frac{(\mu_k^+)^{c_k^+}}{c_k^+!}
\frac{(\mu_k^-)^{c_k^-}}{c_k^-!},
\quad q\le N.
\]

The common factor \(e^{-M}\) cancels between numerator and retained probability. The total probability at order \(q\) is

\[
P_N(q)=\frac{M^q/q!}{D_N},
\qquad \sum_{q=0}^{N}P_N(q)=1.
\]

For repeated identical event types the factorial matters. Creating two band-1 phonons has weight \((\mu_1^+)^2/(2D_N)\); creating one band-1 and one band-2 phonon has weight \(\mu_1^+\mu_2^+/D_N\). The latter counts the two possible event orderings through the multinomial identity, without explicitly listing them twice. These are contributions to the energy of a single absorption, not a time sequence resolved by the script.

For \(N=1\),

\[
D_1=1+M,\qquad P_1(0)=\frac{1}{1+M},\qquad
P_1(k,+)=\frac{\mu_k^+}{1+M},\qquad
P_1(k,-)=\frac{\mu_k^-}{1+M}.
\]

These are exactly the probabilities in the original supplied script. Increasing \(N\) introduces higher orders and also changes the common normalizing denominator; it does not simply append peaks while holding all lower-order weights fixed.

The probability discarded from the underlying unbounded Poisson model is

\[
P_{\rm omitted}=1-e^{-M}D_N.
\]

The settings output reports this probability so that the cutoff can be assessed for each channel. At 300 K with coupling 0.05, all four bands active, and channel multiplier 1, \(M\approx0.07825379\):

| Maximum \(N\) | Underlying probability above the cutoff |
| --- | ---: |
| 1 | \(2.9067\times10^{-3}\) |
| 2 | \(7.5323\times10^{-5}\) |
| 3 | \(1.4678\times10^{-6}\) |
| 4 | \(2.2911\times10^{-8}\) |

These numbers measure omitted **integrated probability**, not relative error at every wavelength. A weak high-order sideband can dominate absorption far from the electronic lines even when its total weight is tiny. Larger coupling or temperature can increase the omitted probability; a maximum of four phonons is then a deliberate model cutoff rather than proof of convergence.

## Widths: convolution adds variances

The electronic Gaussian standard deviation is retained as \(\sigma_{\rm el}=8.5\) cm⁻¹. Each participating phonon is treated as an independent draw from its Gaussian band. Convolution of independent Gaussians adds their means and variances, giving

\[
\delta_{\mathbf c}=\sum_k(c_k^+-c_k^-)\omega_k,
\qquad
\sigma_{\mathbf c}^2
=\sigma_{\rm el}^2+\sum_k(c_k^++c_k^-)\sigma_k^2.
\]

A consumed phonon contributes a negative mean shift but a positive variance. For example, creating band 1 and consuming band 2 gives mean shift \(253-306=-53\) cm⁻¹ and width \(\sqrt{8.5^2+14^2+24^2}\approx29.06\) cm⁻¹. The 1180 cm⁻¹ three-phonon example above has standard deviation approximately 38.07 cm⁻¹.

Two events whose mean shifts cancel are **not** the no-phonon component. Creating and consuming in band 2 has zero mean shift but width \(\sqrt{8.5^2+2\times24^2}\approx34.99\) cm⁻¹. In this continuous-band approximation, the two phonons are independent samples from the same band; they are not forced to be the identical microscopic oscillator at exactly the same frequency. The plotted “without a phonon event” curve includes only \(q=0\).

There is no additional 300 cm⁻¹ effective electronic width on top of these sidebands. All quoted Gaussian widths are standard deviations; their FWHM is approximately \(2.355\sigma\).

## From Stark transitions to the plotted spectra

Within each lower multiplet the original conditional Boltzmann occupation is retained:

\[
p_i=\frac{\exp[-(E_i-E_{\min})/(k_BT)]}
{\sum_l\exp[-(E_l-E_{\min})/(k_BT)]}.
\]

At zero temperature only the lowest lower Stark level or degenerate lowest levels are occupied. The original script assigns equal strength to every upper Stark level: an electronic transition has relative weight \(p_i/N_{\rm upper}\). It does not apply a second upper-state Boltzmann factor. This is the source model's equal-transition-strength assumption, rather than measured individual oscillator strengths.

For a unit-area Gaussian \(G_\sigma(x)\), each channel contributes a sum

\[
A_{\rm ch}(\widetilde\nu)
=\sum_{i,j}\frac{p_i}{N_{\rm upper}}
\sum_{\mathbf c:q\le N}P_N(\mathbf c)
G_{\sigma_{\mathbf c}}
\!\left(\widetilde\nu-\Delta_{ij}-\delta_{\mathbf c}\right).
\]

The GSA channels remain `3H6 -> 3F4` and `3H6 -> 3H5`. The ESA channels remain `3F4 -> 3H4`, `3F4 -> 3F3`, and `3F4 -> 3F2`. Their original relative channel weights are normalized within their respective GSA or ESA spectrum.

Finally,

\[
f_{\rm GSA}(\lambda)=\frac{A_{\rm GSA}(10^7/\lambda)}{\max A_{\rm GSA}},
\qquad
f_{\rm ESA}(\lambda)=\frac{A_{\rm ESA}(10^7/\lambda)}{\max A_{\rm ESA}}.
\]

Each complete spectrum has its own global normalization, estimated on a fixed internal energy grid covering its electronic and phonon components, independently of the displayed wavelength interval. There is no normalization at 1064 nm and no equalization of separate local peaks. The line at 1064 nm is a visual marker. The no-event curves use the same denominators as the corresponding full spectra.

The underlying Gaussians are normalized by area in wavenumber. Plotting the absorption factor at the corresponding wavelength does not convert it into a probability density per nanometre, so no wavelength-density Jacobian is applied. The exported factors are dimensionless relative spectral shapes, not absolute cross sections or a measured absolute GSA/ESA ratio. The ESA shape is conditional on population already being in its lower multiplet; this script does not calculate that multiplet's population or photon-avalanche dynamics.

## Outputs

Without `--output-dir`, results go into `results/max_N_phonons` beside the script, where `N` is the selected maximum. Different cutoffs therefore use different default folders. Repeating a run with the same folder replaces that run's outputs.

| File | Contents |
| --- | --- |
| `f_gsa_f_esa_four_gaussian_multiphonon.csv` | Wavelength in nm, full GSA factor, full ESA factor |
| `f_gsa_f_esa_four_gaussian_multiphonon.png`, `.svg` | Full spectra and weighted no-event contributions |
| `phonon_profile_four_gaussians_multiphonon.csv`, `.png`, `.svg` | Original one-phonon input bands and active sum |
| `four_gaussian_multiphonon_settings.json` | Settings, per-order probabilities, omitted Poisson probability, every count combination and its shift/variance/weight, factors at 1064 nm |

`--data-only` skips the PNG/SVG plots. Four phonons and all four bands produce 495 count combinations per electronic channel, so this setting takes longer than the original single-phonon calculation. Only NumPy is used for the numerical calculation. Plotting reuses the full factors already calculated for the CSV.

Run `python verify_multiphonon.py` for independent numerical checks of the normalization, event combinatorics, signed shifts, widths, limiting cases, and full GSA/ESA agreement with the preserved original at `N=1`. These checks test the implementation, not experimental validity.

The phonon-profile figure still displays the **input one-phonon bands**. Its horizontal axis is the energy of one phonon. It is not the distribution of total multiphonon shifts. Sums and differences enter the absorption calculation through \(\delta_{\mathbf c}\).

All Gaussian tails are retained. No positive absorption floor is added to the calculations. A logarithmic plot's minimum axis limit affects display only. Very small Gaussian values may reach zero through ordinary floating-point underflow.

## Source and limits of the physical model

The Raman reference is [Wilhelm et al., *Multicolor Upconversion Nanoparticles for Protein Conjugation*, Theranostics 3 (2013), 239–248, Fig. 3A](https://www.thno.org/v03p0239.htm), DOI [10.7150/thno.5113](https://doi.org/10.7150/thno.5113). Its dotted blue trace is the Yb/Tm-doped NaYF₄ sample. The paper reports visible Raman bands in approximately the 225–450 cm⁻¹ region. The four Gaussian parameters were supplied with the original script as an approximation to that curve; they are not a new fit to raw experimental data.

Raman intensities are used here as a **phenomenological shape proxy for coupling**. They are not a measured phonon density of states, a measured optical sideband kernel, or a measurement of the coupling of each Tm³⁺ transition. The model inherits the original use of the Raman-shaped weights without a separate Raman Bose-factor correction. The independent-event distribution, coupling strengths, and extension to higher orders are model assumptions; the Raman paper does not validate them.

The thermal occupation is evaluated only at each Gaussian centre. Consequently the creation/consumption ratio has the thermal relation at the centre energies, but a broadened Gaussian band is not guaranteed to satisfy exact detailed balance at every energy. A frequency-resolved treatment would require a physically justified coupling spectral density and thermal weighting across the entire band.

Gaussian phonon bands also extend mathematically to negative frequencies, though the negative tails are negligible with the supplied positive centres and narrow widths. Retaining these tails preserves the original model and permits analytic Gaussian convolutions. They should not be interpreted as physical negative-energy phonons. Any future broadening or shift of the input bands should be reassessed rather than assuming that approximation remains adequate.

The script provides a controlled, selectable extension of the supplied spectral model. Comparing \(N=1,2,3,4\) shows how explicit multiphonon combinations alter the modeled absorption; experimental agreement and transition-specific coupling still require independent evidence.
