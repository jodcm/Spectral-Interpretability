"""
02_evaluate_real_desi.py
========================
EN: Step 2 of the pipeline. Measure the REAL accuracy of the RF (trained on
    synthetic spectra in step 1) on real, LABELLED DESI G/K spectra from Cata
    (per-class folders). This turns the prediction distribution into a true score.
ES: Paso 2 de la pipeline. Mide la ACCURACY REAL del RF (entrenado en espectros
    sinteticos en el paso 1) sobre espectros DESI G/K reales y ETIQUETADOS de Cata
    (carpetas por clase). Convierte la distribucion de predicciones en un puntaje
    real.

Run / Uso:  python 02_evaluate_real_desi.py [path_to_espectros_balanceados_desi]
Does NOT need TransformerPayne (only the saved model + the labelled CSVs).
No necesita TransformerPayne (solo el modelo guardado + los CSV etiquetados).
"""
import os
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import project_lib as P

# --------------------------------------------------------------------------
MODEL_PATH = "rf_sim2real_model.joblib"       # EN: RF from step 1 | ES: RF del paso 1
N_PER_CLASS = 150                             # EN: real per class | ES: reales por clase
DESI_DIR = sys.argv[1] if len(sys.argv) > 1 else "proyecto_desi/espectros_balanceados_desi"
# --------------------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    sys.exit(f"[ERROR] Model '{MODEL_PATH}' not found. Run 01_generate_train.py first.")
if not os.path.isdir(DESI_DIR):
    sys.exit(f"[ERROR] Labelled real spectra folder not found: '{DESI_DIR}'.\n"
             f"        Pass the path as an argument: python 02_evaluate_real_desi.py <path>")

bundle = joblib.load(MODEL_PATH)
classes = list(bundle["classes"])
print("Model classes:", classes, "| grid:",
      round(float(bundle["wave_grid"][0]), 1), "-", round(float(bundle["wave_grid"][-1]), 1), "A")

# EN: load balanced, labelled real spectra through the same pipeline as the sim data
# ES: cargar espectros reales etiquetados y balanceados por la misma pipeline que los sim
X_real, y_true, counts = P.load_labeled_desi_folder(
    DESI_DIR, classes=classes, n_per_class=N_PER_CLASS, balanced=True, seed=0)
print("Real labelled spectra per class:", counts, "-> N =", len(y_true))
if len(y_true) == 0:
    sys.exit("[ERROR] No spectrum loaded (check the CSV format / wavelength range).")

ev = P.evaluate_on_labeled({"model": bundle["model"], "classes": classes}, X_real, y_true)
print("\n=== REAL DESI accuracy: %.3f ===" % ev["accuracy"])
print(ev["report"])
print("Confusion matrix (rows = true, cols = predicted):")
print(ev["confusion_matrix"])

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(ev["confusion_matrix"], annot=True, fmt="d", cmap="Greens",
            xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True (real DESI label)")
ax.set_title("Real DESI %s (sim-trained RF) - acc=%.3f, N=%d"
             % ("/".join(classes), ev["accuracy"], len(y_true)))
plt.tight_layout()
plt.savefig("figures/desi_real_labeled_confusion.png", dpi=130, bbox_inches="tight")
print("saved figures/desi_real_labeled_confusion.png")
