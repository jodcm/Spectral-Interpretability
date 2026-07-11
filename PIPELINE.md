# Pipeline / Flujo de trabajo

**EN:** All the logic lives in `project_lib.py`. Each pipeline step is a small,
self-contained script. The notebook `transformerpayne_y_rf.ipynb` is kept **only
for demonstration** (it shows the key figures + the conclusion) and imports
`project_lib` exactly like the scripts.

**ES:** Toda la logica esta en `project_lib.py`. Cada paso de la pipeline es un
script pequeno y autocontenido. El notebook `transformerpayne_y_rf.ipynb` se
mantiene **solo para demostracion** (muestra las figuras clave + la conclusion) e
importa `project_lib` igual que los scripts.

## Steps / Pasos

| Script | EN | ES | Needs TP? |
|---|---|---|---|
| `01_generate_train.py` | generate synthetic G/K, train RF, save figures + model | genera G/K sinteticos, entrena RF, guarda figuras + modelo | yes |
| `02_evaluate_real_desi.py` | REAL accuracy on Cata's labelled G/K spectra | accuracy REAL sobre los espectros G/K etiquetados de Cata | no |
| `03_broadening.py` | retrain with R(lambda) broadening, compare real accuracy before/after | reentrena con ensanchamiento R(lambda), compara accuracy real antes/despues | yes |
| `05_normalization.py` | improved continuum normalization, real-accuracy progression chart | normalizacion de continuo mejorada, grafico de progreso de accuracy real | yes |
| `06_generate_large.py` | scale-up: batched generation of ~100k synthetic spectra -> .npz (see SCALE_100K.md) | escalado: generacion por lotes de ~100k espectros -> .npz (ver SCALE_100K.md) | yes |
| `07_train_large.py` | train RF from the large .npz (fixed hyperparams) + real eval | entrena RF desde el .npz grande (hiperparams fijos) + eval real | no |
| `08_shap.py` | SHAP interpretability: which wavelengths/lines drive G/K | interpretabilidad SHAP: que longitudes de onda/lineas deciden G/K | no |
| `09_payne_compare.py` | 2nd emulator: train a Payne MLP, compare real accuracy vs TransformerPayne | 2do emulador: entrena un Payne MLP, compara accuracy real vs TransformerPayne | yes |
| `ingest_lamost.py` | convert LAMOST LRS FITS -> per-class CSVs (2nd real dataset) | convierte FITS LAMOST LRS -> CSV por clase (2do dataset real) | no |

## Run order / Orden de ejecucion

```bash
conda activate astro-jax
python 01_generate_train.py                                   # -> figures/ + rf_sim2real_model.joblib
python 02_evaluate_real_desi.py  <espectros_balanceados_desi> # -> real accuracy + confusion
python 03_broadening.py --data <espectros_balanceados_desi> --resolution desi   # DESI
# LAMOST:  python 03_broadening.py --data <espectros_lamost> --resolution 1800
python 05_normalization.py --data <espectros_balanceados_desi>  # -> figures/normalization_progress.png
```

`<espectros_balanceados_desi>` = **EN:** Cata's per-class folders (`G/`, `K/`, ...)
with labelled DESI CSVs (4000-5000 A). **ES:** las carpetas por clase de Cata
(`G/`, `K/`, ...) con CSV DESI etiquetados (4000-5000 A).

## Where to run / Donde ejecutar

**EN:** Steps 1 and 3 need TransformerPayne -> run locally in `astro-jax` (small N)
or on **Colab GPU** for large N (e.g. the professor's 100k). Step 2 has no heavy
dependency and runs anywhere in seconds.

**ES:** Los pasos 1 y 3 necesitan TransformerPayne -> ejecutar localmente en
`astro-jax` (N pequeno) o en **Colab GPU** para N grande (p.ej. los 100k del
profe). El paso 2 no tiene dependencia pesada y corre en segundos en cualquier lado.

## Current result / Resultado actual

**EN:** simulated G/K accuracy ~0.958 vs **REAL DESI ~0.64** -> the sim->real
domain shift is now quantified (many real K stars predicted as G). Step 3 tests
whether the R(lambda) broadening reduces this gap.

**ES:** accuracy G/K simulada ~0.958 vs **DESI REAL ~0.64** -> el domain shift
sim->real esta ahora cuantificado (muchas estrellas K reales predichas como G). El
paso 3 prueba si el ensanchamiento R(lambda) reduce esta brecha.

---
*EN: `project_lib.py` = shared library; scripts = runners; notebook = demo.
ES: `project_lib.py` = libreria compartida; scripts = ejecutores; notebook = demo.*

## Step 0 (optional) + LAMOST / Paso 0 (opcional) + LAMOST

**EN:** You can fetch the real spectra yourself instead of depending on shared files:
- `00_fetch_desi.py --classes G K --n 300` -> searches DESI x Gaia x MWS and downloads per-class CSVs (needs `astro-datalab` + `sparclclient`).
- `ingest_lamost.py --in lamost_fits --out proyecto_desi/espectros_lamost` -> converts LAMOST LRS FITS (per-class subfolders) into the same CSV format. LAMOST has more spectra but LOWER resolution, so train with `broaden_R=1800` (in `03_broadening.py`) instead of `"desi"`.

**ES:** Puedes conseguir los espectros reales tu mismo en vez de depender de archivos compartidos:
- `00_fetch_desi.py --classes G K --n 300` -> busca en DESI x Gaia x MWS y descarga CSV por clase (necesita `astro-datalab` + `sparclclient`).
- `ingest_lamost.py --in lamost_fits --out proyecto_desi/espectros_lamost` -> convierte FITS LAMOST LRS (subcarpetas por clase) al mismo formato CSV. LAMOST tiene mas espectros pero MENOR resolucion, asi que entrena con `broaden_R=1800` (en `03_broadening.py`) en vez de `"desi"`.

**Full chain / Cadena completa:** `00_fetch_desi.py` (or `ingest_lamost.py`) -> `01_generate_train.py` -> `02_evaluate_real_desi.py` -> `03_broadening.py`.

## Broadening sanity check / Chequeo visual del ensanchamiento

**EN:** `04_check_broadening.py --data <folder> --class G --resolution desi` plots the mean SHARP vs BROADENED synthetic spectrum next to the mean REAL spectrum (full range + a zoom on a line). If broadening is right, the broadened synthetic matches the real line widths. Use `--resolution 1800` for LAMOST.
**ES:** `04_check_broadening.py --data <folder> --class G --resolution desi` grafica el sintetico medio AGUDO vs ENSANCHADO junto al REAL medio (rango completo + zoom en una linea). Si el ensanchamiento es correcto, el sintetico ensanchado coincide con el ancho de las lineas reales. Usa `--resolution 1800` para LAMOST.

## Step 5: improved normalization / Paso 5: normalizacion mejorada

**EN:** The broadening step showed that matching the instrument resolution barely
moved the real accuracy (0.66 -> 0.67): the remaining sim->real gap is a
**continuum / normalization** offset, not line width. `05_normalization.py` adds a
robust `continuum_normalize_iter` (low-order polynomial + asymmetric sigma clipping),
applied **identically** to synthetic and real, and produces
`figures/normalization_progress.png` comparing the real DESI accuracy across four
stages: baseline -> +broadening -> +improved normalization -> +both. The old figures
are untouched, so the chart shows the development progress. `project_lib.py` now
accepts an optional `normalizer=` argument everywhere (default keeps the previous
percentile behaviour, so earlier results are reproducible).

**ES:** El paso de ensanchamiento mostro que igualar la resolucion casi no movio la
accuracy real (0.66 -> 0.67): la brecha sim->real que queda es un offset de
**continuo / normalizacion**, no de ancho de linea. `05_normalization.py` agrega una
`continuum_normalize_iter` robusta (polinomio de grado bajo + recorte sigma
asimetrico), aplicada **igual** a sinteticos y reales, y genera
`figures/normalization_progress.png` comparando la accuracy real de DESI en cuatro
etapas: baseline -> +ensanchamiento -> +normalizacion mejorada -> +ambas. Las figuras
viejas no se tocan, asi el grafico muestra el progreso del desarrollo. `project_lib.py`
ahora acepta un argumento opcional `normalizer=` en todas partes (por defecto mantiene
el comportamiento por percentil previo, asi los resultados anteriores son reproducibles).

## Step 8: SHAP interpretability / Paso 8: interpretabilidad SHAP

**EN:** `08_shap.py --data-npz sim_100k.npz --model rf_large_model.joblib` runs a
TreeExplainer on the trained RF and plots, per wavelength, how strongly each pixel
pushes the decision toward K vs G (`figures/shap_importance.png`), overlaid on the
chosen physical lines and compared with the RF Gini importance. It also prints the
top wavelengths and whether they fall on a known line (H-delta, Ca I, G-band CH,
H-gamma, H-beta). This is the project's core "interpretability" deliverable: it shows
the classifier decides using physically meaningful lines, not noise. No TransformerPayne
needed.
**ES:** `08_shap.py --data-npz sim_100k.npz --model rf_large_model.joblib` corre un
TreeExplainer sobre el RF y grafica, por longitud de onda, cuanto empuja cada pixel la
decision hacia K vs G (`figures/shap_importance.png`), sobre las lineas fisicas y
comparado con la importancia Gini. Ademas imprime las longitudes de onda top y si caen
sobre una linea conocida. Es el entregable central de "interpretabilidad": muestra que
el clasificador decide con lineas fisicas, no con ruido. No necesita TransformerPayne.

## Step 9: second emulator (The Payne) / Paso 9: segundo emulador (The Payne)

**EN:** The project requires TWO emulators. `payne.py` defines a Payne-style MLP
(Ting et al. 2019: labels -> spectrum) and `09_payne_compare.py` trains it on a
TransformerPayne library (TP plays the role of the ground-truth physics), then builds
one RF per emulator and compares their **real DESI accuracy**
(`figures/emulator_comparison.png`) plus a reconstruction check
(`figures/payne_reconstruction.png`). If The Payne is a good surrogate, both emulators
give a similar sim->real transfer.
`python 09_payne_compare.py --data proyecto_desi/espectros_balanceados_desi`
**ES:** El proyecto requiere DOS emuladores. `payne.py` define un MLP estilo Payne
(Ting et al. 2019: etiquetas -> espectro) y `09_payne_compare.py` lo entrena con una
libreria de TransformerPayne (TP hace de fisica verdadera), luego arma un RF por
emulador y compara su **accuracy real de DESI** (`figures/emulator_comparison.png`) mas
un chequeo de reconstruccion (`figures/payne_reconstruction.png`).

## Second real dataset: LAMOST / Segundo dataset real: LAMOST

**EN:** LAMOST LRS is a second, independent real dataset (many more spectra, LOWER
resolution R~1800). Workflow, mirroring DESI but at LAMOST resolution:
```bash
# 1) FITS -> per-class labelled CSVs (same format as Cata's DESI folders)
python ingest_lamost.py --in lamost_fits --out proyecto_desi/espectros_lamost
# 2) generate 100k synthetic broadened to LAMOST resolution (1800, not 'desi')
python 06_generate_large.py --n 50000 --out sim_100k_lamost.npz --jobs 24 --batch 16 --resolution 1800
# 3) train + evaluate on the real LAMOST spectra
python 07_train_large.py --data-npz sim_100k_lamost.npz --real proyecto_desi/espectros_lamost --norm iterative
```
The only difference vs DESI is `--resolution 1800` in generation (LAMOST has a lower,
roughly constant R). You then have TWO independent real accuracies (DESI + LAMOST).
Note: `ingest_lamost.py` keeps only the 4000-5000 A window; LAMOST FITS layouts vary
between releases, so if a file is not read it prints a message — adapt `read_lamost_fits`.
**ES:** LAMOST LRS es un segundo dataset real independiente (muchos mas espectros, MENOR
resolucion R~1800). El unico cambio vs DESI es `--resolution 1800` en la generacion.
Asi obtienes DOS accuracies reales independientes (DESI + LAMOST).
