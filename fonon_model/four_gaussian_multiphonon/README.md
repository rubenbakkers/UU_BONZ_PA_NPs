# Four Gaussian phonon bands with selectable multiphonon order

`gsa_esa_four_gaussian_multiphonon.py` extends the supplied standalone GSA/ESA spectrum model so that one electronic absorption can involve **up to 1, 2, 3, 4, 5, or 6 phonons**. It includes sums, differences, and repeated use of the same phonon band, while retaining the four continuous Gaussian bands. The original supplied script is preserved in `backups/gsa_esa_four_gaussian_single_phonon_original.py`. The previous version limited to four phonons, together with its documentation and checks, is preserved in `backups/before_six_phonons_20261016/`.

The number controls how many individual phonons participate in **one photon absorption**. It does not count phonon pairs, distinct bands, or absorbed photons. The electronic Stark levels, their original weighting, and the electronic width are retained.

## Run and choose the number of phonons

Install NumPy and Matplotlib using the accompanying `requirements.txt`, then run from this directory:

```powershell
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 1 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 2 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 3 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 4 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 5 --no-show
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 6 --no-show
```

The default is **6**. To change the default in the script, edit:

```python
DEFAULT_MAX_PHONONS = 6
```

| Setting | Orders included in each absorption |
| --- | --- |
| `--max-phonons 1` | 0 and 1 phonon; reproduces the original single-phonon model |
| `--max-phonons 2` | 0, 1, and 2 phonons |
| `--max-phonons 3` | 0, 1, 2, and 3 phonons |
| `--max-phonons 4` | 0, 1, 2, 3, and 4 phonons |
| `--max-phonons 5` | 0, 1, 2, 3, 4, and 5 phonons |
| `--max-phonons 6` | 0, 1, 2, 3, 4, 5, and 6 phonons |

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
python gsa_esa_four_gaussian_multiphonon.py --temperature 0 --max-phonons 6 --no-show

# Inspect weak sidebands and choose a separate output location.
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 6 --log-y --no-show --output-dir results/N6

# Evaluate every Gaussian contribution that survives floating-point underflow.
python gsa_esa_four_gaussian_multiphonon.py --max-phonons 6 --sum-tolerance 0 --data-only
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
| Create five band-4 phonons | 5 | \(5\times408=2040\) |
| Create six band-4 phonons | 6 | \(6\times408=2448\) |

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
| 5 | \(2.9826\times10^{-10}\) |
| 6 | \(3.3296\times10^{-12}\) |

These numbers measure omitted **integrated probability**, not relative error at every wavelength. A weak high-order sideband can dominate absorption far from the electronic lines even when its total weight is tiny. Larger coupling or temperature can increase the omitted probability; even a maximum of six phonons is a deliberate model cutoff rather than proof of convergence. The reported tail is evaluated from a positive Poisson series at weak coupling, so tiny values are not lost by subtracting two nearly equal numbers.

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

## Number of combinations

With all four bands active at positive temperature there are eight event types: create or consume a phonon in each band. For exactly \(q\) events, the number of unordered count vectors is \(\binom{q+7}{7}\). Including all orders from zero to \(N\) gives

\[
\sum_{q=0}^N\binom{q+7}{7}=\binom{N+8}{8}.
\]

| Maximum \(N\) | Count combinations per electronic channel, including no events |
| --- | ---: |
| 1 | 9 |
| 2 | 45 |
| 3 | 165 |
| 4 | 495 |
| 5 | 1287 |
| 6 | 3003 |

These are possible event counts, not distinct peak positions. Different count vectors can have the same total energy shift, and some can also have the same variance. Disabling bands, setting coupling to zero, or setting temperature to zero reduces the number of nonzero combinations. In particular, at zero temperature the four consumption types have zero probability.

## How the faster calculation preserves the Gaussian bands

The calculation still constructs every allowed count combination with its original probability, signed energy shift, and summed Gaussian variance. It then reduces repeated numerical work in three ways:

1. **Combine identical Gaussians.** Once the electronic Stark gaps are included, terms with exactly the same centre and width are replaced by one Gaussian whose strength is the sum of their strengths. The no-event contribution is tracked separately. No energy bins, width bins, or rounding of peak positions are introduced. Equal centre alone is insufficient: a creation/consumption pair can cancel its mean shift while still broadening the line.
2. **Evaluate nearby lines in sorted blocks.** Requested energies and line centres are sorted. In each block of 32 requested energies, the calculation finds the relevant interval of line centres once and evaluates those Gaussians together. Very distant lines can be omitted only under the positive error bound described below. The results are then restored to the requested order.
3. **Reuse the phonon kernel.** Electronic channels with the same coupling, temperature, enabled bands, and maximum order share the already calculated count combinations. The cache also includes all four Gaussian centres, widths, and heights, so changing those parameters does not reuse an older kernel.

There is no FFT, interpolation of the spectra, or replacement of the continuous Gaussian bands by discrete phonon energies. Increasing the maximum order still increases the work, but repeated count combinations are no longer evaluated separately at every requested energy.

### Bound on omitted numerical contributions

`GAUSSIAN_RELATIVE_TOLERANCE = 1e-14` is the default numerical tolerance; `--sum-tolerance` overrides it. This is separate from the physical cutoff \(N\) and its omitted Poisson probability. It does not drop an event because its integrated probability is small. A high-order combination can remain important in a weak spectral region and must still be evaluated there.

For one group of equal-width Gaussians, write

\[
F_g(x)=\sum_l a_l\exp\!\left[-\frac{(x-c_l)^2}{2\sigma_g^2}\right],
\qquad A_g=\sum_l a_l,
\]

where \(a_l\) includes the unit-area prefactor \(1/(\sqrt{2\pi}\sigma_g)\). If every omitted line centre is at least \(r\sigma_g\) from a requested point, its total omitted contribution is bounded by \(A_g e^{-r^2/2}\).

The script divides the relative tolerance \(\epsilon\) between the \(G\) Gaussian groups. For each query block it sets

\[
B_g=\frac{\epsilon}{G}\min_{x\text{ in block}}F_{\rm accumulated}(x),
\qquad
r=\sqrt{2\ln(A_g/B_g)}.
\]

The radius is limited to 40 standard deviations, beyond which the Gaussian exponential has already underflowed to zero in the floating-point calculation. If \(B_g=0\), this full radius is used. If \(B_g\ge A_g\), the whole group's contribution in that block is already below its allowance.

All contributions are nonnegative. The accumulated result therefore cannot decrease as groups are added. Summing the allowances bounds the additional contribution omitted through this tolerance by \(\epsilon F_{\rm returned}(x)\) at each requested energy. This mathematical bound treats those positive sums in real arithmetic; ordinary floating-point rounding and underflow are separate. In particular, Gaussian tails beyond 40 standard deviations are already zero in the floating-point reference with `--sum-tolerance 0`, although their mathematical values are positive. Normalization uses the same evaluator, so relative changes in the final normalized factors caused by the tolerance are of order \(2\epsilon\), plus rounding; the pre-existing finite grid used to estimate each global maximum is unchanged.

Use `--sum-tolerance 0` to disable tolerance-based omissions and evaluate all Gaussian contributions that survive ordinary underflow. This is useful for direct numerical comparisons. Neither setting adds an absorption floor or changes the four input bands.

## Outputs

Without `--output-dir`, results go into `results/max_N_phonons` beside the script, where `N` is the selected maximum. Different cutoffs therefore use different default folders. Repeating a run with the same folder replaces that run's outputs.

| File | Contents |
| --- | --- |
| `f_gsa_f_esa_four_gaussian_multiphonon.csv` | Wavelength in nm, full GSA factor, full ESA factor |
| `f_gsa_f_esa_four_gaussian_multiphonon.png`, `.svg` | Full spectra and weighted no-event contributions |
| `phonon_profile_four_gaussians_multiphonon.csv`, `.png`, `.svg` | Original one-phonon input bands and active sum |
| `four_gaussian_multiphonon_settings.json` | Settings, per-order probabilities, omitted Poisson probability, every count combination and its shift/variance/weight, factors at 1064 nm, Gaussian counts before/after merging, numerical omission tolerance |

`--data-only` skips the PNG/SVG plots. Only NumPy is used for the numerical calculation. Plotting reuses the full factors already calculated for the CSV.

Run `python verify_multiphonon.py` for independent numerical checks of the normalization, event combinatorics through six phonons, signed shifts, widths, limiting cases, Gaussian merging, the numerical omission bound, and full GSA/ESA agreement with the preserved original at `N=1`. These checks test the implementation, not experimental validity.

All nine test cases passed for this version. The complete `--max-phonons 6 --no-show` execution also completed, including CSV/JSON export and both PNG/SVG figures.

The phonon-profile figure still displays the **input one-phonon bands**. Its horizontal axis is the energy of one phonon. It is not the distribution of total multiphonon shifts. Sums and differences enter the absorption calculation through \(\delta_{\mathbf c}\).

The physical model retains full Gaussian tails. The default evaluator can omit distant numerical contributions only within the stated relative bound; `--sum-tolerance 0` disables those omissions. No positive absorption floor is added to the calculations. A logarithmic plot's minimum axis limit affects display only. Very small Gaussian values may reach zero through ordinary floating-point underflow.

## Measured runtime and agreement with the previous version

Run the accompanying benchmark from this directory to repeat the comparison:

```powershell
python benchmark_multiphonon.py --output-dir results/performance
```

The benchmark executes the preserved four-phonon version and the optimized version sequentially in the same process. Each case constructs both GSA and ESA spectra, including their global normalization searches, and evaluates both at the same 22,001 wavelengths from 900 to 2000 nm in 0.05 nm steps. Temperature is 300 K, coupling is 0.05, and all four bands are active. The optimized cases start with a cleared phonon-kernel cache. Plotting and result-file export are excluded from the measured time.

The saved `results/performance/benchmark_report.json` records the following single-run measurements with Python 3.12.14 and NumPy 2.5.3 on Windows 11:

A preserved copy of the measured report is included in [validation/benchmark_report_20261016.json](validation/benchmark_report_20261016.json).

| Version and maximum order | Construct both spectra and find maxima | Evaluate both wavelength scans | Total |
| --- | ---: | ---: | ---: |
| Previous version, 4 phonons | 29.991 s | 31.279 s | 61.270 s |
| Optimized version, 4 phonons | 2.113 s | 2.504 s | 4.617 s |
| Optimized version, 6 phonons | 7.386 s | 8.962 s | 16.348 s |

For that run, the optimized four-phonon calculation was **13.27 times faster**. The six-phonon calculation also completed faster than the previous four-phonon calculation. These measurements describe this machine and these settings; another machine, coupling, temperature, wavelength range, or tolerance can produce different runtimes.

All 44,002 normalized GSA/ESA factors in the four-phonon comparison agreed within a maximum relative difference of \(3.63\times10^{-15}\) and a maximum absolute difference of \(2.00\times10^{-15}\). There were no zero/nonzero disagreements. In particular, the weak GSA tail down to approximately \(2.02\times10^{-92}\) remained present. This verifies numerical agreement for the tested scan, separately from the mathematical omission bound and from any experimental assessment of the physical model. The benchmark report includes the inputs, environment, comparison tolerances, component counts, and source hashes for the measured versions.

## Why the sidebands can look smaller than in the supplied Raman-band script

This comparison concerns the separately supplied script whose docstring begins `GSA/ESA voor Tm3+:NaYF4, met fononzijdenbanden uit een CONTINUE fononband` and whose example filename is `gsa_esa_fononband.py`. Its default settings differ from this four-Gaussian script. The most consequential difference for the plotted vertical values is the normalization.

### Different vertical normalization

The four-Gaussian script uses the global maximum of each complete spectrum:

\[
f_{\rm peak}(\lambda)=\frac{A(\lambda)}{\max A}.
\]

In the Raman-band script, `StarkSpectrum.factor()` first uses this same type of peak normalization. However, its default `SI_REFERENCE_ANCHOR = True` activates another division in `_stark_resonance_factors()`: both GSA and ESA are divided by their own value at 1064 nm. `compute_factors()` uses that path, and `plot_factors()` plots those returned values. Its final plotted factor is therefore

\[
f_{1064}(\lambda)
=\frac{A(\lambda)/\max A}{A(1064\,{\rm nm})/\max A}
=\frac{A(\lambda)}{A(1064\,{\rm nm})}.
\]

A weak 1064 nm reference can make every value on that curve much larger. As an illustration using **the same six-phonon four-Gaussian spectrum**, its peak-normalized values at 1064 nm are approximately \(1.67957548\times10^{-5}\) for GSA and \(0.48761959\) for ESA. Applying a 1064 nm anchor to that unchanged spectrum would multiply every GSA value by about **59,539** and every ESA value by about **2.051**. These factors illustrate normalization alone; they are not a claim that the two different phonon models produce identical spectra.

This uniform multiplication changes the absolute vertical numbers but **does not change a sideband's height relative to the main peak of its own spectrum**. If that ratio differs after matching normalizations, another model or display difference is responsible. Also, separately anchoring GSA and ESA does not establish an absolute physical cross-section ratio between them.

### Different display and spectral model

| Setting | This four-Gaussian script | Supplied Raman-band script |
| --- | --- | --- |
| Default plotted normalization | Each spectrum's global maximum | Each spectrum's value at 1064 nm |
| Default vertical scale | Linear; `--log-y` selects logarithmic | Logarithmic in `plot_factors()` |
| One-phonon input shape | Four Gaussian bands with the centres, widths, and areas listed above | Manually traced Raman profile over 225–450 cm⁻¹, sampled at 46 nodes spaced by 5 cm⁻¹ |
| Default coupling per channel | 0.05 | 0.04 |
| Thermal weighting | Bose occupation at each of four band centres | Bose occupation at each sampled node |
| Multiphonon treatment | Analytic convolution of the Gaussian bands, with total event count at most 6 by default | Convolution of sampled-node creation/consumption kernels, expanded to an omitted-mass tolerance of \(10^{-35}\) by default |

A logarithmic scale makes weak sidebands visible across many orders of magnitude. Use `--log-y` here for a comparable display. In the other script, setting `SI_REFERENCE_ANCHOR = False` gives peak-normalized factors and removes the 1064 nm rescaling. Matching these choices is the first step in comparing the actual spectral shapes.

The two input phonon profiles are also different approximations, rather than two numerical representations of an identical fitted function. In this script, one Gaussian-band event gives a line width \(\sqrt{8.5^2+\sigma_k^2}\), and additional independent events add more variance. In the other script, each individual shifted node is broadened by the electronic width of 8.5 cm⁻¹, while the spread of weighted nodes already supplies the phonon-band width. Its complete sidebands therefore do not have a total width of only 8.5 cm⁻¹. The different input shapes, thermal sampling, and convolutions can change local peak heights and tails; these facts alone do not establish which complete sideband must be higher at a particular wavelength.

### Peak height is different from total phonon probability

For an isolated unit-area Gaussian carrying probability \(P\), the peak is \(P/(\sqrt{2\pi}\sigma)\). Spreading the same probability over a wider energy interval lowers its peak without reducing its integrated weight. Several Stark transitions and phonon combinations then overlap in the complete spectrum, so a plotted peak is not a direct measure of the total phonon-event probability.

At the supplied defaults and 300 K, the four-Gaussian model with six phonons has \(M\approx0.07825379\) and an integrated probability of at least one phonon event of **7.5270%**. The Raman-band script has \(M\approx0.06274001\) and a corresponding probability of **6.0812%**. The latter was calculated from its actual 46 node weights and coupling 0.04, using its `no_phonon_event_weight = exp(-M)`. Thus the small vertical sideband values in the four-Gaussian plot do not indicate a smaller total phonon probability under these defaults.

The six-phonon cutoff discards approximately \(3.33\times10^{-12}\) of the underlying integrated probability at these Gaussian-model defaults. The other script's much tighter expansion can still differ in extremely weak spectral regions where higher orders dominate. This is a physical-model cutoff difference, separate from the optimized Gaussian summation tolerance. The speed optimization itself was checked against the previous four-phonon implementation and did not suppress its sidebands.

## Source and limits of the physical model

The Raman reference is [Wilhelm et al., *Multicolor Upconversion Nanoparticles for Protein Conjugation*, Theranostics 3 (2013), 239–248, Fig. 3A](https://www.thno.org/v03p0239.htm), DOI [10.7150/thno.5113](https://doi.org/10.7150/thno.5113). Its dotted blue trace is the Yb/Tm-doped NaYF₄ sample. The paper reports visible Raman bands in approximately the 225–450 cm⁻¹ region. The four Gaussian parameters were supplied with the original script as an approximation to that curve; they are not a new fit to raw experimental data.

Raman intensities are used here as a **phenomenological shape proxy for coupling**. They are not a measured phonon density of states, a measured optical sideband kernel, or a measurement of the coupling of each Tm³⁺ transition. The model inherits the original use of the Raman-shaped weights without a separate Raman Bose-factor correction. The independent-event distribution, coupling strengths, and extension to higher orders are model assumptions; the Raman paper does not validate them.

The thermal occupation is evaluated only at each Gaussian centre. Consequently the creation/consumption ratio has the thermal relation at the centre energies, but a broadened Gaussian band is not guaranteed to satisfy exact detailed balance at every energy. A frequency-resolved treatment would require a physically justified coupling spectral density and thermal weighting across the entire band.

Gaussian phonon bands also extend mathematically to negative frequencies, though the negative tails are negligible with the supplied positive centres and narrow widths. Retaining these tails preserves the original model and permits analytic Gaussian convolutions. They should not be interpreted as physical negative-energy phonons. Any future broadening or shift of the input bands should be reassessed rather than assuming that approximation remains adequate.

The script provides a controlled, selectable extension of the supplied spectral model. Comparing \(N=1,2,3,4,5,6\) shows how explicit multiphonon combinations alter the modeled absorption; experimental agreement and transition-specific coupling still require independent evidence.
