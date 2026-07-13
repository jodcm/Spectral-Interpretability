# Estado del Proyecto / Project Status
### Sim-to-Real con TransformerPayne — rama de Felix / Felix's branch
*Última actualización / last updated: 13-07-2026*

---

## 1. En una frase / In one sentence

**ES:** Entrenamos un clasificador espectral **solo con espectros sintéticos** (generados con
TransformerPayne) y lo probamos en espectros **reales** — y funciona: en SDSS alcanza el techo
de lo que logra un modelo entrenado con datos reales etiquetados, **sin usar ni una etiqueta**.

**EN:** We train a spectral classifier **purely on synthetic spectra** (generated with
TransformerPayne) and test it on **real** spectra — and it works: on SDSS it reaches the ceiling
of what a model trained on real labelled data achieves, **without using a single label**.

---

## 2. Resultado principal / Main result

|                                          | DESI  | SDSS  |
|------------------------------------------|-------|-------|
| **(1) Entrenado con sintéticos → real**   | 0.752 | **0.956** |
| **(2) Entrenado con reales → real** (techo) | 0.843 | 0.940 |
| **Domain shift** = (2) − (1)              | +0.090 | **−0.017** |
| Cross-survey: real DESI → real SDSS       | —     | 0.828 |

**ES — cómo leerlo:**
- En **SDSS el domain shift es CERO** (−0.017, dentro del error). Lo sintético **iguala** al
  entrenamiento con datos reales — y con 10.000 espectros de entrenamiento en vez de 594, sin
  necesitar etiquetas. **Ese es el punto del proyecto, y está demostrado.**
- En **DESI da 0.752, pero NO es culpa del método**: el techo de DESI es solo 0.843 (el de SDSS
  es 0.940). Ni un modelo entrenado con DESI real supera eso.
- **La prueba definitiva:** un modelo entrenado con **DESI real** solo llega a **0.828** en
  **SDSS real** — o sea, los dos instrumentos **no coinciden entre sí**, sin ninguna simulación
  de por medio. El problema es instrumental.

**EN — how to read it:**
- On **SDSS the domain shift is ZERO** (−0.017, within the error). Synthetic training **matches**
  training on real labelled data — with 10,000 training spectra instead of 594, and no labels.
  **That is the point of the project, and it is demonstrated.**
- On **DESI it gives 0.752, but this is NOT the method's fault**: DESI's own ceiling is only 0.843
  (SDSS's is 0.940). Not even a model trained on real DESI beats that.
- **The decisive test:** a model trained on **real DESI** reaches only **0.828** on **real SDSS** —
  the two instruments **disagree with each other**, with no simulation involved at all.

---

## 3. La causa del problema de DESI / The cause of the DESI problem

**ES:** Medimos el espectro medio de estrellas G a la **misma temperatura** en ambos surveys.
Todas las líneas coinciden (razón 0.97–1.10) **excepto una**:

| Línea | DESI | SDSS | razón |
|---|---|---|---|
| H-delta | 0.348 | 0.334 | 1.04 |
| Ca I | 0.334 | 0.303 | 1.10 |
| Banda G (CH) | 0.385 | 0.396 | 0.97 |
| H-gamma | 0.380 | 0.362 | 1.05 |
| **H-beta** | **0.298** | **0.374** | **0.80** |

**La H-beta de DESI es 20 % más plana.** Y con SHAP mostramos que el Random Forest decide **casi
solo con esa línea**. Una H-beta débil = Balmer débil = estrella fría = **K**. Por eso las G de
DESI se clasifican como K, en todo el rango de temperatura y nunca al revés.

**EN:** We measured the mean spectrum of G stars at the **same temperature** in both surveys. Every
line agrees (ratio 0.97–1.10) **except one**: DESI's **H-beta is 20 % shallower**. And SHAP showed
the Random Forest decides **almost entirely with that line**. Weak H-beta = weak Balmer = cool star
= **K**. Hence DESI's G stars get classified as K, at every temperature and never the other way.

**Hipótesis descartadas / hypotheses ruled out** (cada una probada y refutada / each tested and refuted):

| Hipótesis / Hypothesis | Resultado / Result |
|---|---|
| Gigantes (logg < 3.5) fuera del rango de entrenamiento | ✗ 0.760 → 0.770 |
| Estrellas fuera del dominio de Teff | ✗ 0.753 |
| Velocidad radial sin corregir | ✗ desplazamiento < ruido de medición |
| Sesgo de la normalización del continuo | ✗ sin sesgo azul/rojo |
| **Enmascarar las líneas al normalizar** (el "fix" de H-beta) | ⚠ arregla H-beta (0.80 → 0.94) pero **empeora DESI** (0.752 → 0.637) |

---

## 4. Otros resultados / Other results

**ES / EN:**

- **Saturación / Saturation.** La accuracy real se **satura ya en ~2.000 espectros sintéticos**
  (2k: 0.776 · 10k: 0.752 · 100k: 0.759). Los 100k del profe no aportan nada, y **el problema de
  conseguir 700.000 espectros nunca fue un problema**. / Real accuracy **plateaus at ~2,000
  synthetic spectra**. More data adds nothing.

- **Segundo emulador / Second emulator (The Payne).** Un MLP estilo Payne entrenado sobre espectros
  de TransformerPayne reproduce el espectro con RMSE 0.011 (por debajo del ruido, σ=0.02) y da la
  misma transferencia: **TransformerPayne 0.78 vs The Payne 0.77**. → **El emulador no es el cuello
  de botella.** / The emulator is not the bottleneck.

- **Normalización / Normalization.** Ensanchamiento R(λ) + normalización iterativa del continuo,
  aplicados **igual** a sintéticos y reales, subieron DESI de **0.66 → 0.76** y arreglaron la
  confusión K→G (recall de K: 0.43 → 0.85). Los dos efectos son **acoplados**: por separado no
  sirven. / The two effects are **coupled**: neither works alone.

- **F como experimento extra / F as an extra experiment.** Con F/G/K sobre datos reales la accuracy
  cae de 0.843 a **0.717** y el recall de G se hunde a 0.47 — F (6000–7500 K) y G (5200–6000 K) se
  confunden en el borde. **Confirma que presentar G/K fue lo correcto.** / Confirms that presenting
  G/K was the right call.

- **Interpretabilidad del clasificador / Classifier interpretability (SHAP).** El Random Forest
  decide con las **líneas de Balmer**: H-beta domina (5 de las 10 longitudes de onda más
  importantes son píxeles de H-beta), luego H-gamma. Físicamente correcto: G vs K **es** un corte
  en temperatura, y Balmer es el termómetro. / Physically correct.

- **Interpretabilidad del EMULADOR / EMULATOR interpretability.** Medimos ∂flujo/∂[X/H] para los 10
  elementos. Los que **no tienen líneas** en 4000–5000 Å (Na, S, N, K) dan sensibilidad **~cero**
  (0.02–0.07); los que sí tienen dominan (Fe 1.00, C, Mg, Ca, Ti, Mn) y **su máximo cae sobre sus
  líneas conocidas** (Ca I en 4226.7 Å, etc.). **TransformerPayne aprendió física, no
  correlaciones** — y es una prueba que podía fallar. / TransformerPayne learned physics, not
  correlations — and this was a test that could have failed.

- **La conexión entre ambas / The link between the two.** El emulador dice que Ca I y la banda G
  están controladas por la **abundancia** — y nosotros variamos las abundancias al azar (±1 dex).
  Esas líneas son, por lo tanto, discriminadores ambiguos. Y SHAP muestra que el clasificador
  **las ignora** y usa Balmer, la única señal puramente térmica. **El clasificador aprendió a
  evitar exactamente las líneas que el emulador marca como contaminadas por abundancia.** / The
  classifier learned to avoid exactly the lines the emulator flags as abundance-contaminated.

---

## 5. La pipeline / The pipeline

```
00_fetch_sdss.py     -> descarga espectros reales de SDSS (SkyServer)
                        download real SDSS spectra
       |
06_generate_large.py -> genera N espectros sintéticos con TransformerPayne -> .npz
                        generate N synthetic spectra -> .npz
       |
07_train_large.py    -> entrena el RF con los sintéticos, evalúa en los reales
                        train the RF on synthetic, evaluate on real
       |
14_train_real.py     -> entrena con REALES: el techo + F/G/K + cross-survey
                        train on REAL: the ceiling + F/G/K + cross-survey
       |
15_summary.py        -> la escala de referencia (FIGURA PRINCIPAL)
                        the reference scale (MAIN FIGURE)
```

### Scripts

| Script | ES | EN |
|---|---|---|
| `project_lib.py` | librería compartida: grilla, normalización, ensanchamiento, carga | shared library: grid, normalization, broadening, loading |
| `00_fetch_desi.py` | descarga DESI (SPARCL — actualmente caído) | fetch DESI (SPARCL — currently down) |
| `00_fetch_sdss.py` | descarga SDSS vía SkyServer (funciona) | fetch SDSS via SkyServer (works) |
| `01_generate_train.py` | pipeline chica: genera, entrena, figuras del milestone | small pipeline: generate, train, milestone figures |
| `02_evaluate_real_desi.py` | accuracy real sobre espectros etiquetados | real accuracy on labelled spectra |
| `03_broadening.py` | efecto del ensanchamiento R(λ) | effect of the R(λ) broadening |
| `04_check_broadening.py` | chequeo visual del ensanchamiento | visual check of the broadening |
| `05_normalization.py` | progreso: baseline → ensanch. → normalización → ambas | progression chart |
| `06_generate_large.py` | generación a gran escala (multiproceso, hasta 100k) | large-scale generation (multiprocessing) |
| `07_train_large.py` | RF desde el .npz + evaluación real (filtros logg/Teff/λ) | RF from .npz + real eval (logg/Teff/λ filters) |
| `08_shap.py` | **interpretabilidad del clasificador** (SHAP) | **classifier interpretability** |
| `09_payne_compare.py` | **segundo emulador** (The Payne) y comparación | **second emulator** and comparison |
| `10_saturation.py` | curva de aprendizaje: accuracy vs nº de espectros | learning curve |
| `11_compare_real_samples.py` | poblaciones estelares DESI vs SDSS (Teff/logg) | stellar populations DESI vs SDSS |
| `12_misclassified.py` | ¿dónde en Teff están los errores? | where in Teff are the errors? |
| `13_spectra_compare.py` | espectro medio DESI vs SDSS → **encontró la H-beta** | mean spectrum DESI vs SDSS → **found the H-beta** |
| `14_train_real.py` | **entrenamiento con reales**: techo, F/G/K, cross-survey | **real training**: ceiling, F/G/K, cross-survey |
| `15_summary.py` | **la escala de referencia** (figura principal) | **the reference scale** (main figure) |
| `16_attention.py` | **interpretabilidad del emulador** (sensibilidad por elemento) | **emulator interpretability** |
| `run_all.bat` | corre TODO de una / runs EVERYTHING in one go | |

**Cómo correr todo / How to run everything:**
```
conda activate astro-jax
run_all.bat > run_all_log.txt 2>&1
```

---

## 6. Figuras clave / Key figures

| Figura | ES | EN |
|---|---|---|
| `summary_scale_GK_iterative.png` | **la escala de referencia — figura principal** | **the reference scale — main figure** |
| `spectra_compare_*_G.png` | la H-beta de DESI, 20 % más plana | DESI's H-beta, 20 % shallower |
| `emulator_sensitivity_G.png` | qué elemento controla qué línea | which element controls which line |
| `shap_importance.png` | qué longitudes de onda usa el clasificador | which wavelengths the classifier uses |
| `saturation_curve.png` | la accuracy se satura en ~2.000 espectros | accuracy saturates at ~2,000 spectra |
| `emulator_comparison.png` | TransformerPayne vs The Payne | |
| `cross_survey_*.png` | DESI real → SDSS real | |
| `real_trained_*.png` | el techo / the ceiling | |

---

## 7. Qué falta / What is missing

**ES / EN:**

1. **Decisión con Steve: "todas las clases espectrales".** Con TransformerPayne **no es alcanzable**
   — solo está validado para enanas de ~4000–6000 K (G, K, y F apenas). Para O/B/A/M el emulador
   está fuera de su grilla. Además, por la IMF esas estrellas casi no existen (LAMOST tiene **234**
   estrellas O en total). Hay dos límites independientes: el **emulador** y los **datos**. /
   Not achievable with TransformerPayne; two independent limits: the emulator and the data.

2. **Las etiquetas Teff de DESI (MWS)** parecen menos precisas que las de SDSS (SSPP) — el techo bajo
   de DESI (0.843 vs 0.940) lo sugiere. Afecta a todo el equipo, no solo a esta rama. /
   DESI's MWS Teff labels look less precise than SDSS's SSPP. This affects the whole team.

3. **La presentación.** / The presentation.
