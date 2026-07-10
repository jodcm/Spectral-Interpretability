"""
01_generate_train.py
====================
EN: Step 1 of the pipeline. Generate balanced synthetic G/K spectra with
    TransformerPayne, train the Random Forest, and save the milestone figures
    (within-class variability, confusion matrix, feature importance) plus the
    trained model and its predictions on the cached real DESI sample.
ES: Paso 1 de la pipeline. Genera espectros sinteticos G/K balanceados con
    TransformerPayne, entrena el Random Forest y guarda las figuras del milestone
    (variabilidad intra-clase, matriz de confusion, importancia de features) mas
    el modelo entrenado y sus predicciones sobre la muestra DESI real en cache.

Run / Uso:  python 01_generate_train.py
Requires / Requiere: env astro-jax (TransformerPayne). First run downloads the
weights (~7.5 MB, needs internet once).
"""
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")            # EN: no GUI needed for a script | ES: sin GUI en un script
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import jax
jax.config.update("jax_enable_x64", True)
import transformer_payne as tp
import project_lib as P

# --------------------------------------------------------------------------
CLASSES = ("G", "K")             # EN: classes to present | ES: clases a presentar
N_PER_CLASS = 100                # EN: spectra per class  | ES: espectros por clase
DESI_CACHE = "desi_sample.npz"   # EN: cached real DESI    | ES: DESI real en cache
# --------------------------------------------------------------------------

os.makedirs("figures", exist_ok=True)


def mark_lines(ax):
    """EN: mark the chosen physical lines (in range).
    ES: marca las lineas fisicas elegidas (dentro del rango)."""
    for name, (lam, _) in P.SPECTRAL_LINES.items():
        if P.WMIN <= lam <= P.WMAX:
            ax.axvline(lam, color="grey", ls="--", lw=0.8, alpha=0.7)


# EN: 1) Download emulator and build the balanced simulated dataset.
# ES: 1) Descargar el emulador y construir el dataset simulado balanceado.
print("Loading TransformerPayne weights (first run downloads them)...")
emu = tp.TransformerPayne.download()
df = P.build_balanced_dataset(emu, classes=CLASSES, n_per_class=N_PER_CLASS, sigma_noise=0.02)
print("spectra per class:", df["spectral_type"].value_counts().to_dict())

# EN: within-class variability (why the sim task is easy/hard).
# ES: variabilidad intra-clase (por que la tarea sim es facil/dificil).
fig, axes = plt.subplots(len(CLASSES), 1, sharex=True, figsize=(11, 5))
axes = np.atleast_1d(axes)
for ax, c in zip(axes, CLASSES):
    sub = np.vstack(df[df.spectral_type == c]["normalized_intensity"])
    mean, std = sub.mean(0), sub.std(0)
    ax.plot(P.WAVE_GRID, mean, "k", lw=1, label=f"mean {c}")
    ax.fill_between(P.WAVE_GRID, mean - std, mean + std, color="C0", alpha=0.25, label=r"$\pm1\sigma$")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylabel("Norm. int.")
    mark_lines(ax)
axes[-1].set_xlabel(r"Rest-frame wavelength [$\AA$]")
axes[0].set_title("Within-class spectral variability (simulated)")
plt.tight_layout()
plt.savefig("figures/within_class_variance.png", dpi=130, bbox_inches="tight")
plt.close()

# EN: 2) Train the Random Forest and save the confusion matrix.
# ES: 2) Entrenar el Random Forest y guardar la matriz de confusion.
res = P.train_rf(df, classes=CLASSES)
print("best params:", res["best_params"])
print("simulated test accuracy:", round(res["accuracy"], 3))
cm = confusion_matrix(res["y_test"], res["y_pred"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=res["classes"], yticklabels=res["classes"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"RF - confusion matrix (acc={res['accuracy']:.3f})")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()

# EN: 3) Feature importance vs the physical lines (interpretability).
# ES: 3) Importancia de features vs las lineas fisicas (interpretabilidad).
imp = res["model"].feature_importances_
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(P.WAVE_GRID, imp, color="crimson", lw=1.2)
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel("RF importance")
ax.set_title("RF feature importance vs chosen lines")
ax.grid(alpha=0.3)
mark_lines(ax)
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()

# EN: 4) Save the trained model FIRST (does not depend on the network).
# ES: 4) Guardar el modelo entrenado PRIMERO (no depende de la red).
import joblib
joblib.dump({"model": res["model"], "classes": res["classes"], "wave_grid": res["wave_grid"]},
            "rf_sim2real_model.joblib")
print("saved rf_sim2real_model.joblib")

# EN: 5) OPTIONAL: apply the RF to a small LIVE DESI sample (needs SPARCL online).
#        Skipped automatically if SPARCL is unreachable. The REAL evaluation is done
#        offline with 02_evaluate_real_desi.py on the team's labelled spectra.
# ES: 5) OPCIONAL: aplicar el RF a una pequena muestra DESI EN VIVO (necesita SPARCL online).
#        Se omite si SPARCL no responde. La evaluacion REAL se hace offline con
#        02_evaluate_real_desi.py sobre los espectros etiquetados del equipo.
try:
    use_cache = False
    if os.path.exists(DESI_CACHE):
        d = np.load(DESI_CACHE)
        if "wave_grid" in d and len(d["wave_grid"]) == len(P.WAVE_GRID) and np.allclose(d["wave_grid"], P.WAVE_GRID):
            Xdesi = d["X"]
            use_cache = True
    if not use_cache:
        print("Downloading a small real DESI sample (optional; needs SPARCL online)...")
        Xdesi, info = P.build_desi_dataset(n=80)
        np.savez_compressed(DESI_CACHE, X=Xdesi, wave_grid=P.WAVE_GRID)
    pred = np.array(res["classes"])[res["model"].predict(Xdesi)]
    u, c = np.unique(pred, return_counts=True)
    print("DESI predicted-class distribution:", dict(zip(u.tolist(), c.tolist())))
    P.export_sim2real(res, Xdesi, tag="sim2real")
except Exception as e:
    print("Skipped optional live DESI sample (SPARCL unreachable):", str(e)[:100])

print("\nStep 1 done. Figures in figures/, model in rf_sim2real_model.joblib.")
print("Next: python 02_evaluate_real_desi.py proyecto_desi/espectros_balanceados_desi")
