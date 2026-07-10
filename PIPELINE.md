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

## Run order / Orden de ejecucion

```bash
conda activate astro-jax
python 01_generate_train.py                                   # -> figures/ + rf_sim2real_model.joblib
python 02_evaluate_real_desi.py  <espectros_balanceados_desi> # -> real accuracy + confusion
python 03_broadening.py --data <espectros_balanceados_desi> --resolution desi   # DESI
# LAMOST:  python 03_broadening.py --data <espectros_lamost> --resolution 1800
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
