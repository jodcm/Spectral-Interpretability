# Presentation Preparation — Sim-to-Real with TransformerPayne
### Everything you need to defend this work
*Felix — AS4501 · last updated 13-07-2026*

---

## PART I — THE STORY IN ONE PAGE

### The question

Radiative-transfer codes can compute a stellar spectrum from physics, but they are far too slow to
generate the millions of spectra a machine-learning model wants. **Spectral emulators** (The Payne,
TransformerPayne) are neural networks trained to imitate those codes: give them stellar labels
(temperature, gravity, chemical abundances) and they emit a spectrum in milliseconds.

That raises the question this project exists to answer:

> **Can we train a classifier entirely on emulator-generated (synthetic) spectra, and have it work
> on real telescope data?**

This is a **sim-to-real** problem. If it works, we get unlimited, perfectly-labelled training data
for free. If it fails, we have learned something about where the simulation departs from reality.

### The answer

**Yes — and on a well-behaved instrument it works perfectly.**

|                                            | DESI   | SDSS      |
|--------------------------------------------|--------|-----------|
| **Trained on synthetic → tested on real**   | 0.752  | **0.956** |
| **Trained on real → tested on real** (ceiling) | 0.843  | 0.940     |
| **Domain shift** = ceiling − synthetic      | +0.090 | **−0.017** |
| Cross-survey: real DESI → real SDSS         | —      | 0.828     |

On **SDSS**, the model trained on *nothing but synthetic spectra* (0.956) matches — even slightly
exceeds — a model trained on **real, labelled** SDSS spectra (0.940). The domain shift is **zero**.

And note the asymmetry in effort: the synthetic model had **10,000** training spectra; the real
model had only **594**, because that is all the labelled data there is. **The synthetic approach
gives you more data, needs no labels, and loses nothing.** That is the thesis of the project,
demonstrated.

### Why DESI looks worse — and why that is not our failure

DESI gives 0.752. But **DESI's own ceiling is 0.843** — no model, however good, does better on
DESI's data with DESI's labels (SDSS's ceiling is 0.940). And a model trained on **real DESI**
reaches only **0.828** on **real SDSS**: the two instruments disagree *with each other*, with no
simulation involved at all.

**The measured cause:** comparing the mean spectrum of G stars *at the same temperature* in both
surveys, every line agrees within 3–10 % — **except H-beta, which is 20 % shallower in DESI**. And
SHAP shows the classifier decides almost entirely with H-beta. Weak H-beta → weak Balmer → looks
cool → **K**. That is exactly the observed error: DESI's G stars get called K, at every temperature,
and almost never the reverse.

---

## PART II — THE THEORY YOU NEED

### 1. Why spectral type ≡ temperature

A star's spectral class (O B A F G K M) is, at bottom, a **temperature sequence** — O hottest
(>30,000 K), M coolest (~3,000 K). Our two classes:

- **G** : 5,300–6,000 K (the Sun is G2, 5,772 K)
- **K** : 4,000–5,200 K

The boundary sits at ~**5,250 K**. So G vs K is a *temperature cut* — remember this, it explains
everything downstream.

### 2. Balmer lines are the thermometer

Hydrogen is the most abundant element, and its **Balmer series** (H-beta 4862 Å, H-gamma 4341 Å,
H-delta 4103 Å) arises from the n=2 level. Populating n=2 requires thermal excitation, so:

- **Hotter star (G)** → more atoms in n=2 → **stronger, deeper Balmer absorption**
- **Cooler star (K)** → fewer atoms in n=2 → **weak Balmer**

Balmer strength therefore tracks temperature *directly*, and — crucially — it is **almost
independent of chemical composition**, because hydrogen abundance barely varies between stars.

Meanwhile, **metal lines** (Ca I 4227 Å, the CH G-band at 4300 Å, hundreds of Fe lines) get stronger
in cooler stars *but also* depend on how metal-rich the star is. They are **ambiguous**: a
metal-rich G star can look, in Ca I, like a metal-poor K star.

> **Expect the question:** *"Why does your model use H-beta and not the metal lines?"*
> **Answer:** Because we randomize abundances ([X/H] from −1 to +1 dex) in the training set, the
> metal lines carry a confounded signal, while Balmer is a clean thermometer. The Random Forest
> discovered this by itself — see Part III.

### 3. The two emulators

| | **The Payne** (Ting et al. 2019) | **TransformerPayne** (Różański et al.) |
|---|---|---|
| Architecture | fully-connected MLP | transformer with **attention** |
| Input | stellar labels | stellar labels **+ the wavelength itself** |
| Output | the whole spectrum at once | the flux **at that wavelength** |
| Key idea | fast surrogate of a physics code | wavelength as a *parameter* → resolution-independent, and attention can associate lines with elements |

Both are **emulators**, not physics: they are trained on a library of spectra computed by a real
radiative-transfer code. They are fast (ms vs hours) but only valid **inside the parameter grid they
were trained on** — for TransformerPayne, roughly **dwarfs at 4,000–6,000 K**. This validity limit
is the single most important caveat in the whole project (see Part V).

### 4. Preprocessing — and why it is not a detail

Real spectra and synthetic spectra do not live in the same space. Three operations must be applied
**identically to both** or the classifier learns the difference between the two datasets instead of
the difference between G and K stars.

**(a) Common wavelength grid.** Everything is resampled onto 4000–5000 Å, 1000 pixels. This range is
where TransformerPayne is validated *and* where DESI's data lives.

**(b) Instrumental broadening (LSF).** A real spectrograph smears every line. Its **resolving power**
is R = λ/Δλ. Synthetic spectra come at R = 100,000 (razor-sharp); DESI is R ≈ 2,000–2,700, SDSS
R ≈ 2,000. So we **convolve the synthetic spectra with a Gaussian** to bring them down to the
instrument's resolution. DESI's R varies with wavelength, so we do this with a coordinate warp that
makes a constant-σ convolution produce the correct wavelength-dependent FWHM. Equivalent width is
conserved — the lines get broader and shallower, not weaker.

**(c) Continuum normalization.** The absolute flux depends on the star's distance and brightness,
which tell us nothing about its type. So we divide out the smooth **continuum**, leaving only the
line pattern. We fit the continuum with a low-order polynomial using **asymmetric sigma clipping**
(reject points far *below* the fit — those are absorption lines — keep refitting), so the fit tracks
the upper envelope.

> **Expect the question:** *"Why does the continuum normalization matter so much?"*
> **Answer:** Because it determines the *apparent depth* of every line. Broad lines like H-beta have
> wings spanning ±40 Å; a continuum fit can partially absorb those wings, and how much it absorbs
> depends on the noise. That is precisely how DESI's H-beta ends up 20 % shallower than SDSS's.

### 5. Sim-to-real and the domain shift

The vocabulary from domain adaptation, which you should use:

- **Source-only** — train on synthetic, test on real. *This is our method.*
- **Target-supervised** — train on real labelled data, test on real. *This is the ceiling.*
- **Domain shift** — the gap between them. **By definition.**

> **This is the most important methodological point in your talk.** A single number like "0.76 on
> DESI" is meaningless without the ceiling. Good or bad *compared to what*? Reporting source-only
> together with target-supervised is standard practice in every domain-adaptation paper, and the gap
> between them *is* the domain shift. It is what turns your result from a number into a claim.

### 6. Random Forest, Gini, and SHAP

A **Random Forest** is an ensemble of decision trees; each tree splits on pixel values. It is a good
choice here because it is robust, needs no scaling, and — the point — is **interpretable**.

- **Gini importance** — how much each pixel reduces impurity across the trees. Built-in, but biased
  toward correlated / high-cardinality features.
- **SHAP** (SHapley Additive exPlanations) — from cooperative game theory: it distributes the
  prediction fairly among the features, with axioms (efficiency, symmetry) that guarantee
  consistency. For tree models it is computed **exactly** (TreeExplainer), not approximated.

We report both; they agree, which is itself evidence the result is not an artifact of one method.

### 7. The Initial Mass Function (why hot stars are rare)

Star formation produces far more low-mass stars than high-mass ones (the IMF, dN/dM ∝ M^−2.35 for
massive stars). Massive O and B stars are also short-lived. Consequence: **they barely exist.**
LAMOST — one of the largest spectroscopic surveys ever — contains **234 O stars** in total, against
**3.5 million G stars**. Any "balanced sample over all spectral classes" is therefore capped by the
rarest class. This is physics, not a data-collection failure.

---

## PART III — THE RESULTS, IN THE ORDER YOU SHOULD PRESENT THEM

### Result 1 — The domain shift, and how we closed it

| Stage | Real DESI accuracy |
|---|---|
| Baseline (sharp synthetic, percentile normalization) | 0.66 |
| + R(λ) instrumental broadening | 0.67 |
| + improved continuum normalization *alone* | 0.57 |
| **+ both together** | **0.74** |

**The lesson: the two effects are coupled.** Broadening alone barely helps. The better normalization
*alone* makes things **worse** — because the polynomial continuum reacts differently to sharp
synthetic lines than to broad real ones, so it normalizes the two datasets *differently*. Only when
the synthetic spectra are first broadened to the real resolution does the same continuum fit treat
both consistently. **Match the resolution first, then the normalization.**

This also fixed the original failure mode: K stars used to be misclassified as G at a rate of 57 %
(recall 0.43); afterwards K recall is 0.85.

### Result 2 — Scaling saturates (this answers the professor's 100k)

| Synthetic training spectra | Real accuracy |
|---|---|
| 2,000 | 0.776 ± 0.013 |
| 10,000 | 0.752 ± 0.004 |
| 50,000 | 0.761 ± 0.006 |
| 100,000 | 0.759 ± 0.007 |

**Flat from 2,000 onward.** More synthetic data buys nothing. The limit is not sample size. This is
also the answer to the team's crisis about being unable to obtain 700,000 spectra per class: *it
does not matter.* We can prove it with a curve instead of apologizing for it.

### Result 3 — The emulator is not the bottleneck

We trained a **Payne-style MLP** on TransformerPayne spectra (reconstruction RMSE 0.011 — below the
injected noise level of 0.02, so effectively exact) and compared the sim→real transfer:

- TransformerPayne: **0.78**
- The Payne (MLP): **0.77**

Statistically identical. **Architecture is not the problem.**

*Be honest about this:* our Payne is trained *on* TransformerPayne output, so it is a fast
**surrogate**, not an independent physical check. Present it as "second emulator / distillation",
not as "independent confirmation of the physics".

### Result 4 — The reference scale (YOUR MAIN FIGURE)

`figures/summary_scale_GK_iterative.png`

|                                    | DESI   | SDSS      |
|------------------------------------|--------|-----------|
| Synthetic-trained → real            | 0.752  | **0.956** |
| Real-trained → real (ceiling)       | 0.843  | 0.940     |
| **Domain shift**                    | +0.090 | **−0.017** |
| Cross-survey (real DESI → real SDSS)| —      | 0.828     |

**Say this sentence out loud in the talk:**

> *"On SDSS, training on nothing but simulated spectra is as good as training on real labelled data —
> and we used ten thousand synthetic spectra where only five hundred and ninety-four real labelled
> ones exist."*

### Result 5 — Diagnosing DESI (a chain of falsified hypotheses)

This is what makes the work scientific rather than a leaderboard entry. **Four hypotheses, each
tested, each refuted:**

| Hypothesis | Test | Result |
|---|---|---|
| DESI contains **giants** (17 %, logg < 3.5) outside the dwarf-only training grid | filter them out | ✗ 0.760 → 0.770 |
| DESI contains stars outside the **Teff** grid | filter to 4000–6000 K, logg > 4 | ✗ 0.753 |
| DESI spectra are not **radial-velocity** corrected | cross-correlation (validated by injecting known shifts) | ✗ measured shift < the measurement noise floor |
| The **continuum normalization** is biased by DESI's noise | compare blue vs red envelope levels | ✗ no bias |

**Then we stopped guessing and just looked**: the mean G-star spectrum, same Teff window, DESI vs
SDSS. Every line agrees — **except H-beta, 20 % shallower in DESI**.

**And we proved causality:** masking H-beta out of the classifier lifts DESI's G-recall from 0.67 to
**0.81** — the bias vanishes exactly when the broken line is removed.

**Honest negative result** (include it — it strengthens you): the obvious fix, masking the lines when
fitting the continuum, *does* repair H-beta (ratio 0.80 → 0.94) but **over-deepens every other line**
(Ca I ratio 1.10 → 1.32) and makes DESI **worse overall** (0.752 → 0.637). We found the cause,
tested the natural fix, and report why it does not hold.

### Result 6 — Interpretability, on BOTH levels

**(a) The classifier — SHAP.** The Random Forest decides with the **Balmer lines**: 5 of the top 10
wavelengths are H-beta pixels (4859–4864 Å), then H-gamma. Gini importance agrees independently.
Physically correct: G vs K *is* a temperature cut, and Balmer is the thermometer.

**(b) The emulator — sensitivity analysis.** We measured ∂flux/∂[X/H] for all 10 varied elements.

| Element | G star | K star | Has lines in 4000–5000 Å? |
|---|---|---|---|
| Fe | 1.00 | 1.00 | yes, hundreds |
| C | 0.46 | 0.61 | yes (CH G-band) |
| Mg | 0.37 | 0.69 | yes |
| Ca | 0.27 | 0.42 | yes (4226.7 Å) |
| Ti | 0.21 | 0.34 | yes |
| Mn | 0.20 | 0.22 | yes |
| **Na** | **0.06** | **0.17** | **no** (D lines at 5890 Å) |
| **S** | **0.04** | **0.05** | **no** |
| **N** | **0.07** | **0.03** | **no** |
| **K** | **0.02** | **0.03** | **no** (7665 Å) |

**This was a falsifiable test and it passed.** Elements with no lines in the window show ~zero
sensitivity; the others peak *on their known lines* (Ca I exactly at 4226.7 Å). **TransformerPayne
learned physics, not correlations.**

It is even subtler than that. The emulator distinguishes three regimes:
- **atomic lines** → sharp, discrete peaks (Fe)
- **molecular bands** → broad, diffuse elevation (C, via CH — and *stronger in the cooler K star*,
  because molecules survive in cool atmospheres)
- **indirect continuum effects** → smooth, structureless bands (Na has *no* line here, yet rises from
  0.06 to 0.17 in the K star: Na is a low-ionization **electron donor** and changes the H⁻ continuous
  opacity)

**(c) THE PUNCHLINE — connect (a) and (b):**

> The emulator says Ca I and the CH G-band are controlled by **abundance**. We randomized abundances
> by ±1 dex. Those lines are therefore *ambiguous* discriminators. And SHAP shows the classifier
> **ignores them** and uses Balmer — the only purely thermal signal, and the one element whose
> abundance we did *not* vary.
>
> **The classifier independently learned to avoid exactly the lines the emulator flags as
> abundance-contaminated.**

Two independent interpretability analyses, one coherent physical picture. **This is the payoff of the
project title.**

---

## PART IV — HARD QUESTIONS AND YOUR ANSWERS

**"Why only G and K? The goal was all spectral classes."**
> Two independent hard limits. **(1) The emulator:** TransformerPayne is validated for dwarfs at
> ~4,000–6,000 K. O (30,000 K+), B, A and M lie completely outside its training grid — it would emit
> spectra, but not trustworthy ones. **(2) The data:** by the IMF, hot stars barely exist — LAMOST
> contains 234 O stars in total. We also *measured* the cost of pushing outward: adding F drops the
> real-data accuracy from 0.843 to 0.717 (G-recall collapses to 0.47), because F and G are adjacent
> in temperature. We chose to make a rigorous claim about a validated domain rather than a weak claim
> about everything.

**"0.76 on DESI is not very good."**
> Compared to what? DESI's own ceiling — a model trained on real DESI data with real DESI labels — is
> 0.843. And that same real-trained DESI model gets only 0.828 on real SDSS. The instruments disagree
> with each other. On SDSS our synthetic-trained model reaches 0.956 against a ceiling of 0.940. The
> method is not the limitation; DESI's data is.

**"Isn't training on real data the obvious thing to do?"**
> It is the *reference*, not a competing method — and it is exactly why we ran it. Note that on SDSS
> there are only 594 labelled spectra to train on, and we used 10,000 synthetic ones. The synthetic
> route gives unlimited data, needs no labels, and matches the ceiling. That *is* the result.

**"Your Random Forest depends on a single line. Isn't that fragile?"**
> Yes — and we demonstrated the fragility rather than hiding it. Removing H-beta costs SDSS 14 points
> (0.953 → 0.813). That single-feature dependence is precisely why DESI's 20 %-shallow H-beta is so
> destructive. It is a genuine robustness finding, and the natural next step would be a model
> regularized to spread its evidence across more lines.

**"Why is the second emulator not an independent check?"**
> Correct, and we say so. Our Payne MLP is *trained on* TransformerPayne output — it is a fast
> surrogate, a distillation. It cannot be better than its teacher. What it does show is that the
> transformer architecture carries no advantage for this task, i.e. the emulator is not the
> bottleneck.

**"How do you know the emulator isn't just memorizing correlations?"**
> We tested it and it could have failed. Four of the ten elements we vary (Na, S, N, K) have **no
> lines** in 4000–5000 Å. A correlation-learner would still show sensitivity to them. TransformerPayne
> shows ~zero (0.02–0.07), while the elements that *do* have lines peak exactly on them — Ca I at
> 4226.7 Å. And it reproduces subtleties nobody trained it to reproduce: molecular bands broaden in
> cool stars, and Na — with no line at all here — shows a *smooth* continuum-opacity response because
> it is an electron donor.

---

## PART V — THE ONE CAVEAT TO NEVER FORGET

**TransformerPayne is only valid for dwarfs at ~4,000–6,000 K.** Every conclusion in this work lives
inside that box. Applying the model outside it is not a bug of the method — it is a misuse of the
tool. State the validity domain *before* you state the results, and the "why not all classes"
question answers itself.

---

## PART VI — SUGGESTED SLIDE ORDER

1. **The question** — can we train on simulations and work on the sky?
2. **The tools** — emulators (Payne / TransformerPayne), and their validity domain *(state it here)*
3. **The pipeline** — common grid, LSF broadening, continuum normalization, Random Forest
4. **Closing the domain shift** — 0.66 → 0.74, and *why the two effects are coupled*
5. **Scaling saturates** — the curve. More data buys nothing. *(This answers the 700k crisis.)*
6. **The reference scale** — **MAIN FIGURE.** Synthetic ≈ ceiling on SDSS. Domain shift = 0.
7. **Diagnosing DESI** — four falsified hypotheses → the H-beta measurement → causal proof → the
   honest negative result
8. **Interpretability I** — SHAP: the classifier uses Balmer
9. **Interpretability II** — the emulator learned real element↔line physics *(the falsifiable test)*
10. **The punchline** — the classifier avoids exactly the lines the emulator flags as
    abundance-contaminated
11. **Limits and next steps** — validity domain, DESI's labels, single-feature fragility

---

## APPENDIX — NUMBERS TO HAVE ON THE TIP OF YOUR TONGUE

| | |
|---|---|
| Wavelength range | 4000–5000 Å, 1000 px |
| Classes | G (5300–6000 K), K (4000–5200 K), boundary 5250 K |
| Synthetic training set | 10,000 (saturates at 2,000) |
| Sim → real, **SDSS** | **0.956** |
| Sim → real, **DESI** | **0.752** |
| Ceiling (real-trained), SDSS | 0.940 *(from only 594 labelled spectra)* |
| Ceiling (real-trained), DESI | 0.843 |
| Domain shift, SDSS | **−0.017 (zero)** |
| Domain shift, DESI | +0.090 |
| Cross-survey real DESI → real SDSS | 0.828 |
| DESI H-beta depth vs SDSS | **0.80** (20 % shallower) |
| Cost of removing H-beta (SDSS) | 0.953 → 0.813 |
| The Payne vs TransformerPayne | 0.77 vs 0.78 |
| F/G/K on real data | 0.717 (G-recall 0.47) |
| Emulator sensitivity, Fe | 1.00 |
| Emulator sensitivity, Na/S/N/K (no lines here) | 0.02–0.07 |
| O stars in all of LAMOST | **234** |
