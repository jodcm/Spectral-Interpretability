# Spectral Interpretability — Sim-to-Real Stellar Classification (G/K)

Course AS4501 (Astroinformatica). This repository contains the **sim-to-real** branch
built with **TransformerPayne**: generate synthetic G/K spectra, train a Random Forest,
apply it to real DESI spectra, and interpret which spectral lines drive the decision.

**EN:** A Random Forest is trained on synthetic **G/K** spectra emulated with
TransformerPayne over **4000–5000 Å** (the emulator's valid range and the range of the
real DESI data). The trained model is then applied to real DESI spectra (sim-to-real
transfer), and RF feature importance is used for interpretability.

**ES:** Se entrena un Random Forest con espectros sintéticos **G/K** generados con
TransformerPayne en **4000–5000 Å** (el rango válido del emulador y el de los datos DESI
reales). El modelo se aplica luego a espectros DESI reales (transferencia sim-a-real) y se
usa la importancia de features del RF para la interpretabilidad.

## Contents / Contenido

| File | EN | ES |
|---|---|---|
| `project_lib.py` | Shared library: common grid, spectral lines, simulation, DESI cleaning, RF training, LSF broadening, label utilities | Librería compartida: grilla común, líneas, simulación, limpieza DESI, entrenamiento RF, ensanchamiento LSF, utilidades de etiquetas |
| `transformerpayne_y_rf.ipynb` | Main notebook: generate → train → apply to DESI → figures | Notebook principal: generar → entrenar → aplicar a DESI → figuras |
| `next_steps.ipynb` | Wavelength-dependent LSF broadening experiment (domain-shift mitigation) | Experimento de ensanchamiento LSF dependiente de λ (mitigación del domain shift) |
| `requirements.txt` | Dependencies (conda env `astro-jax`, Python 3.10) | Dependencias (entorno conda `astro-jax`, Python 3.10) |
| `desi_sample.npz` | Cached real DESI spectra (on the common grid) | Espectros DESI reales en caché (grilla común) |
| `desi_example_spectrum.csv` | One real DESI spectrum, to test the CSV pipeline | Un espectro DESI real, para probar la pipeline CSV |
| `figures/` | Result figures | Figuras de resultados |

## Setup

```bash
conda create -n astro-jax python=3.10 -y
conda activate astro-jax
pip install -r requirements.txt
python -m ipykernel install --user --name astro-jax --display-name "Python (astro-jax)"
```

Then open `transformerpayne_y_rf.ipynb` in VS Code, select the **Python (astro-jax)**
kernel, and **Run All**. The first run downloads the TransformerPayne weights (~7.5 MB,
needs internet once).

## Current scope / Alcance actual

- **Classes / Clases:** G and K only (F is inside TransformerPayne's range but omitted for now).
- **Range / Rango:** 4000–5000 Å (paper-valid; matches the real DESI data).
- **Sample / Muestra:** 100 spectra per class (200 total, balanced).

## Key results / Resultados clave

- **Simulated G/K test accuracy ≈ 0.958** — real but non-trivial (G and K are neighboring
  types); the earlier 100 % with F/G/K reflected an easier, well-separated task.
- **Feature importance** concentrates on real spectral lines — the Balmer series
  (H-delta, H-gamma, H-beta, stronger in G) plus a forest of metal lines (stronger in K)
  → physically interpretable, the goal of the project.
- **Sim-to-real transfer** and **wavelength-dependent LSF broadening** to reduce the
  domain shift are in `next_steps.ipynb`. A real, labelled accuracy is obtained by feeding
  labelled DESI G/K spectra into `evaluate_on_labeled` (see `project_lib.py`).

## Next steps / Próximos pasos

- Evaluate on a **balanced, labelled** real G/K set (limited by available G stars) for a
  true DESI accuracy.
- Scale up (professor's target) with equal real and synthetic counts — the real data is
  the bottleneck.

---
*Parts of this repository (code formatting, bilingual text) were prepared with AI
assistance; the scientific approach, decisions and results are the author's own and were
reviewed by the author.*
