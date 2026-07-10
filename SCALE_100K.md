# Scaling to ~100k / Escalar a ~100k

**EN:** The professor's target is a large synthetic training set (~100,000 spectra) so
the parameter space (Teff, log g, abundances) is densely sampled. The 100k are
**synthetic training** spectra — the real DESI set (Cata's labelled spectra) stays as
the evaluation set. Generation is decoupled from training: generate once to a `.npz`,
then train as often as you like.

**ES:** El objetivo del profe es un set de entrenamiento sintetico grande (~100.000
espectros) para muestrear densamente el espacio de parametros (Teff, log g,
abundancias). Los 100k son espectros de **entrenamiento sintetico** — el set real de
DESI (los espectros etiquetados de Cata) sigue siendo el set de evaluacion. La
generacion esta desacoplada del entrenamiento: generar una vez a un `.npz` y luego
entrenar las veces que quieras.

## The two new scripts / Los dos scripts nuevos

| Script | EN | ES | Needs TP? |
|---|---|---|---|
| `06_generate_large.py` | batched generation (jax.vmap+jit, GPU-aware) -> `.npz` | generacion por lotes (jax.vmap+jit, usa GPU) -> `.npz` | yes |
| `07_train_large.py` | train RF from `.npz` (fixed hyperparams) + real eval | entrena RF desde `.npz` (hiperparams fijos) + eval real | no |

Both use the **winning pipeline** from step 5: `--resolution desi` + `--norm iterative`
(R(lambda) broadening + iterative continuum normalization), which reached the best
sim->real transfer (0.66 -> 0.74). `--norm` must be the **same** in both scripts.

### Compute: GPU *or* many CPU cores / Computo: GPU *o* muchos nucleos CPU

**EN:** The emulator forward passes are the bottleneck. Two fast paths:
- **GPU (Colab):** `--jobs 1 --batch 2000` — JAX uses the GPU automatically.
- **Many-core CPU VM (e.g. 24 cores):** `--jobs 24 --batch 500` — generation is
  embarrassingly parallel, so `06` fans it out with multiprocessing (one worker per
  core, each pinned to a single thread to avoid oversubscription). The RF in `07`
  already uses all cores via `n_jobs=-1`.

**Run `01_generate_train.py` once first** so the TransformerPayne weights are cached on
disk; the parallel workers then load from cache instead of each re-downloading.

**ES:** Los forward del emulador son el cuello de botella. Dos caminos rapidos:
- **GPU (Colab):** `--jobs 1 --batch 2000` — JAX usa la GPU automaticamente.
- **VM CPU con muchos nucleos (p.ej. 24):** `--jobs 24 --batch 500` — la generacion es
  vergonzosamente paralela, asi que `06` la reparte con multiprocessing (un worker por
  nucleo, cada uno con un solo hilo para no sobre-suscribir). El RF de `07` ya usa todos
  los nucleos con `n_jobs=-1`.

**Corre `01_generate_train.py` una vez primero** para cachear los pesos de
TransformerPayne en disco; asi los workers cargan del cache en vez de descargar cada uno.

## Steps to succeed / Pasos para tener exito

1. **Smoke test first (1k).** `python 06_generate_large.py --n 500 --out sim_1k.npz --jobs 8`
   then `python 07_train_large.py --data-npz sim_1k.npz --real proyecto_desi/espectros_balanceados_desi --norm iterative`.
   Confirms the chain works before the long run. / Prueba rapida antes del run largo.
2. **Decide where to run.** GPU (Colab, `--jobs 1`) or a many-core CPU VM
   (`--jobs <cores>`) — see the compute section above. / GPU o VM CPU multinucleo.
3. **Generate 100k.** On a 24-core VM:
   `python 06_generate_large.py --n 50000 --out sim_100k.npz --jobs 24 --batch 500`
   (50k G + 50k K). It prints throughput + ETA and drops any non-finite spectra. Tune
   `--jobs` to your core count and lower `--batch` if a worker runs out of memory.
4. **Train + evaluate.** `python 07_train_large.py --data-npz sim_100k.npz
   --real proyecto_desi/espectros_balanceados_desi --norm iterative`. Saves
   `rf_large_model.joblib` and `figures/desi_real_confusion_100k.png`.
5. **Compare.** Put the new real accuracy next to the 200-spectra result (0.74) to show
   whether more synthetic data further improves the sim->real transfer.

## Memory & performance / Memoria y rendimiento

- **Storage / Almacenamiento:** 100k x 1000 float32 = ~400 MB in RAM, less on disk
  (compressed `.npz`). Default is float32; `--float64` doubles it (only if needed).
- **Generation cost / Costo de generacion:** dominated by the emulator forward passes.
  `--batch` controls the vmap batch size (bigger = faster on GPU, more memory). On CPU
  expect this to take a long time (tens of minutes to hours); on a Colab GPU it is much
  faster. Broadening + normalization are cheap per spectrum.
- **RF training / Entrenamiento RF:** GridSearch+CV (the small pipeline) is too slow at
  100k, so `07` uses **fixed** hyperparameters (`--n-estimators 200 --max-depth none
  --min-samples-leaf 2`, `criterion=entropy`, `class_weight=balanced`, `n_jobs=-1`).
  Expect a few minutes and a few GB of RAM. Increase `--min-samples-leaf` if memory is
  tight.

## Colab GPU (recommended for 100k)

```python
# 1) install
!pip -q install transformer-payne==0.10 "numpy<2" scikit-learn joblib
!pip -q install --upgrade "jax[cuda12]"    # GPU build of JAX

# 2) upload project_lib.py, 06_generate_large.py, 07_train_large.py
#    (and proyecto_desi/espectros_balanceados_desi/ if you also want the real eval here)
from google.colab import files; files.upload()

# 3) generate on the GPU
!python 06_generate_large.py --n 50000 --out sim_100k.npz --batch 2000

# 4a) train on Colab, or 4b) download sim_100k.npz and train locally with 07_train_large.py
!python 07_train_large.py --data-npz sim_100k.npz --real proyecto_desi/espectros_balanceados_desi --norm iterative
from google.colab import files; files.download("sim_100k.npz")
```

**EN:** Check `JAX devices:` in the `06` output — it should list a GPU (e.g.
`cuda:0`). If it only shows `cpu`, JAX did not find the GPU (reinstall the CUDA build /
enable the GPU runtime in Colab).
**ES:** Revisa `JAX devices:` en la salida de `06` — debe listar una GPU (p.ej.
`cuda:0`). Si solo muestra `cpu`, JAX no encontro la GPU (reinstala la build CUDA /
activa el runtime GPU en Colab).

## Where this plugs in / Donde encaja

`06_generate_large.py` -> `sim_100k.npz` -> `07_train_large.py` -> real accuracy +
`figures/desi_real_confusion_100k.png`. Steps 01-05 stay as the small, fast,
fully-interpretable pipeline (figures, broadening study, normalization progression);
06-07 are the large-scale version for the professor's 100k target.
