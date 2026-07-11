"""
08_shap.py
==========
EN: SHAP interpretability of the Random Forest (the project's core "spectral
    interpretability" goal). Uses a TreeExplainer on the trained RF and a sample of
    the synthetic spectra to compute, per wavelength, how much each pixel pushes the
    prediction toward K vs G. Plots mean |SHAP| vs wavelength next to the chosen
    physical lines (H-delta, Ca I, G-band CH, H-gamma, H-beta), compares it with the
    RF's built-in Gini importance, and lists the top wavelengths (and whether they
    fall on a known line). This turns the classifier from a black box into a
    physically readable statement: "the RF separates G from K mainly using these
    lines".
ES: Interpretabilidad SHAP del Random Forest (el objetivo central de
    "interpretabilidad espectral" del proyecto). Usa un TreeExplainer sobre el RF
    entrenado y una muestra de los espectros sinteticos para calcular, por longitud de
    onda, cuanto empuja cada pixel la prediccion hacia K vs G. Grafica el |SHAP| medio
    vs longitud de onda junto a las lineas fisicas elegidas (H-delta, Ca I, banda CH,
    H-gamma, H-beta), lo compara con la importancia Gini del RF y lista las longitudes
    de onda mas importantes (y si caen sobre una linea conocida). Convierte al
    clasificador de caja negra en un enunciado fisico legible.

Run / Uso:
    python 08_shap.py --data-npz sim_100k.npz --model rf_large_model.joblib
    # or with the small-pipeline model: --model rf_sim2real_model.joblib

Requires / Requiere: shap, scikit-learn, matplotlib, numpy (no TransformerPayne).
"""
import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import project_lib as P


def main():
    ap = argparse.ArgumentParser(description="SHAP interpretability of the RF classifier")
    ap.add_argument("--data-npz", default="sim_100k.npz", dest="data_npz",
                    help="synthetic set from 06_generate_large.py (provides X, y)")
    ap.add_argument("--model", default="rf_large_model.joblib",
                    help="trained RF (from 07_train_large.py or 01_generate_train.py)")
    ap.add_argument("--nsample", type=int, default=800, help="spectra to explain (per run)")
    ap.add_argument("--tol", type=float, default=6.0, help="A tolerance to match a peak to a line")
    args = ap.parse_args()

    import joblib
    import shap

    os.makedirs("figures", exist_ok=True)

    # EN: 1) load model + a stratified sample of the synthetic spectra
    # ES: 1) cargar modelo + una muestra estratificada de los espectros sinteticos
    bundle = joblib.load(args.model)
    model = bundle["model"]
    classes = list(bundle["classes"])
    wave = np.asarray(bundle.get("wave_grid", P.WAVE_GRID), dtype=float)

    d = np.load(args.data_npz, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"]).astype(str)
    rng = np.random.RandomState(0)
    idx = []
    per = max(args.nsample // len(classes), 1)
    for c in classes:
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, size=min(per, len(ci)), replace=False))
    idx = np.array(idx)
    Xs = X[idx]
    print(f"model={args.model} classes={classes} | explaining {Xs.shape[0]} spectra")

    # EN: 2) SHAP values (TreeExplainer is exact + fast for RFs)
    # ES: 2) valores SHAP (TreeExplainer es exacto y rapido para RFs)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs, check_additivity=False)

    # EN: normalize SHAP output shape across shap versions -> per-class (n, feat)
    # ES: normaliza la forma de salida entre versiones de shap -> por clase (n, feat)
    def per_class_matrix(sv, k):
        if isinstance(sv, list):                 # old API: list[class] -> (n, feat)
            return np.asarray(sv[k])
        sv = np.asarray(sv)
        if sv.ndim == 3:                         # new API: (n, feat, class) or (class, n, feat)
            if sv.shape[-1] == len(classes):
                return sv[:, :, k]
            return sv[k]
        return sv                                # binary single-array fallback

    # EN: focus on the K class (index in `classes`); |SHAP| averaged over spectra
    # ES: foco en la clase K; |SHAP| promediado sobre espectros
    k_idx = classes.index("K") if "K" in classes else len(classes) - 1
    shap_k = per_class_matrix(sv, k_idx)
    mean_abs_shap = np.abs(shap_k).mean(axis=0)
    mean_abs_shap = mean_abs_shap / (mean_abs_shap.max() + 1e-12)

    gini = model.feature_importances_
    gini = gini / (gini.max() + 1e-12)

    # EN: 3) top wavelengths by SHAP + line coincidence
    # ES: 3) longitudes de onda top por SHAP + coincidencia con lineas
    order = np.argsort(mean_abs_shap)[::-1][:10]
    line_items = [(lam, name) for name, (lam, _) in P.SPECTRAL_LINES.items()
                  if P.WMIN <= lam <= P.WMAX]
    print("\nTop-10 wavelengths by |SHAP| (toward %s):" % classes[k_idx])
    for r in order:
        lam = wave[r]
        near = [nm for (ll, nm) in line_items if abs(ll - lam) <= args.tol]
        tag = ("  <- " + ", ".join(near)) if near else ""
        print(f"  {lam:8.2f} A   shap={mean_abs_shap[r]:.3f}{tag}")

    # EN: 4) plot SHAP vs Gini vs physical lines
    # ES: 4) graficar SHAP vs Gini vs lineas fisicas
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(wave, mean_abs_shap, color="#1b4a72", lw=1.3, label="mean |SHAP| (toward %s)" % classes[k_idx])
    ax.plot(wave, gini, color="crimson", lw=1.0, alpha=0.55, label="RF Gini importance")
    for lam, name in line_items:
        ax.axvline(lam, color="grey", ls="--", lw=0.8, alpha=0.7)
        ax.text(lam, 1.02, name, rotation=90, va="bottom", ha="center", fontsize=7, color="grey")
    ax.set_xlim(P.WMIN, P.WMAX)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("normalized importance")
    ax.set_title("SHAP interpretability: which wavelengths drive the G/K decision")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = "figures/shap_importance.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("\nsaved", out)


if __name__ == "__main__":
    main()
