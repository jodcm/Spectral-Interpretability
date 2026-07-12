"""
12_misclassified.py
===================
EN: The decisive diagnostic. The sim-trained RF reaches ~0.95 on SDSS but only ~0.76 on
    DESI, and the DESI errors are strongly ASYMMETRIC: many true G predicted as K, few
    the other way. Two explanations remain (giants and the Teff domain were already
    ruled out by 07 --min-logg / --teff-range):

      (A) LABEL PRECISION. G/K is a cut at Teff = 5250 K. If DESI's MWS temperatures are
          noisy or biased high, stars labelled G that actually sit just above the cut are
          really K -> the model says K and is counted wrong. Signature: the misclassified
          G stars PILE UP just above 5250 K.
      (B) SPECTRAL / INSTRUMENTAL. DESI's blue channel has poor throughput at 4000-5000 A
          (noise, flux calibration). Signature: the misclassified G stars are SPREAD over
          the whole G range, including hot G stars far from the boundary that should be
          trivially easy.

    This script separates the two: it predicts every real spectrum, keeps its catalogue
    Teff, and shows where the errors sit in temperature. Run it on DESI and on SDSS --
    the contrast between the two is the answer.
ES: El diagnostico decisivo. El RF entrenado en sinteticos llega a ~0.95 en SDSS pero
    solo ~0.76 en DESI, y los errores de DESI son muy ASIMETRICOS: muchas G verdaderas
    predichas como K, pocas al reves. Quedan dos explicaciones (gigantes y dominio de
    Teff ya fueron descartados con 07 --min-logg / --teff-range):

      (A) PRECISION DE LAS ETIQUETAS. G/K es un corte en Teff = 5250 K. Si las
          temperaturas MWS de DESI son ruidosas o estan sesgadas hacia arriba, estrellas
          etiquetadas G que en realidad estan apenas sobre el corte son K -> el modelo
          dice K y se cuenta como error. Firma: las G mal clasificadas se AMONTONAN justo
          encima de 5250 K.
      (B) ESPECTRAL / INSTRUMENTAL. El canal azul de DESI tiene poca sensibilidad en
          4000-5000 A. Firma: las G mal clasificadas estan REPARTIDAS por todo el rango,
          incluso G calientes lejos del borde que deberian ser faciles.

Run / Uso:
    python 12_misclassified.py --data-npz sim_100k.npz --real proyecto_desi/espectros_balanceados_desi
    python 12_misclassified.py --data-npz sim_50k_sdss.npz --real proyecto_desi/espectros_sdss

Requires / Requiere: numpy, pandas, scikit-learn, matplotlib (NO TransformerPayne).
"""
import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import project_lib as P


def load_real_with_meta(base_dir, classes, n_per_class, normalizer, seed=0):
    """EN: like load_labeled_desi_folder, but ALSO returns the catalogue Teff/logg
        of every spectrum, so we can ask WHERE in temperature the errors are.
    ES: como load_labeled_desi_folder, pero ADEMAS devuelve el Teff/logg de catalogo
        de cada espectro, para preguntar DONDE en temperatura estan los errores."""
    import random
    rng = random.Random(seed)
    X, meta = [], []
    for c in classes:
        files = sorted(glob.glob(os.path.join(base_dir, str(c), "*.csv")))
        rng.shuffle(files)
        kept = 0
        for f in files:
            if n_per_class is not None and kept >= n_per_class:
                break
            try:
                x = P.load_desi_csv(f, normalizer=normalizer)
                head = pd.read_csv(f, nrows=1)
            except Exception:
                continue
            if x is None or not np.all(np.isfinite(x)):
                continue
            low = {k.lower(): k for k in head.columns}
            if "teff" not in low:
                continue
            X.append(x)
            meta.append({
                "clase": c,
                "teff": float(head[low["teff"]].iloc[0]),
                "logg": float(head[low["logg"]].iloc[0]) if "logg" in low else np.nan,
            })
            kept += 1
    return np.vstack(X), pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser(description="Where in Teff do the misclassifications sit?")
    ap.add_argument("--data-npz", default="sim_100k.npz", dest="data_npz")
    ap.add_argument("--real", default="proyecto_desi/espectros_balanceados_desi")
    ap.add_argument("--norm", default="iterative", choices=["iterative", "percentile"])
    ap.add_argument("--real-n", type=int, default=500, dest="real_n",
                    help="real spectra per class (more than usual -> better statistics)")
    ap.add_argument("--n-estimators", type=int, default=200, dest="n_estimators")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    os.makedirs("figures", exist_ok=True)
    normalizer = P.continuum_normalize_iter if args.norm == "iterative" else P.continuum_normalize
    tag = os.path.basename(os.path.normpath(args.real)).replace("espectros_", "")

    # EN: 1) train the RF on the synthetic set | ES: 1) entrenar el RF con los sinteticos
    d = np.load(args.data_npz, allow_pickle=True)
    Xs = np.asarray(d["X"], dtype=np.float32)
    ys_txt = np.asarray(d["y"]).astype(str)
    classes = sorted(np.unique(ys_txt).tolist())
    ci = {c: i for i, c in enumerate(classes)}
    ys = np.array([ci[t] for t in ys_txt])
    rf = RandomForestClassifier(n_estimators=args.n_estimators, criterion="entropy",
                                class_weight="balanced", min_samples_leaf=2,
                                n_jobs=-1, random_state=23).fit(Xs, ys)
    print(f"RF trained on {Xs.shape[0]} synthetic spectra, classes={classes}")

    # EN: 2) predict every real spectrum, keeping its Teff | ES: predecir y guardar Teff
    X_real, meta = load_real_with_meta(args.real, classes, args.real_n, normalizer)
    meta["pred"] = np.asarray(classes)[rf.predict(X_real)]
    meta["correct"] = meta["pred"] == meta["clase"]
    acc = float(meta["correct"].mean())
    print(f"real [{tag}]: N={len(meta)}  accuracy={acc:.3f}")

    cut = P.TEFF_CUT_KG
    print(f"\nG/K boundary = {cut:.0f} K")
    for c in classes:
        sub = meta[meta.clase == c]
        wrong = sub[~sub.correct]
        right = sub[sub.correct]
        if len(sub) == 0:
            continue
        print(f"\n[{c}] N={len(sub)}  errors={len(wrong)} ({100*len(wrong)/len(sub):.0f}%)")
        if len(wrong):
            print(f"    misclassified Teff: median={wrong.teff.median():6.0f}  "
                  f"(5-95%: {wrong.teff.quantile(.05):.0f}-{wrong.teff.quantile(.95):.0f})")
        if len(right):
            print(f"    correct        Teff: median={right.teff.median():6.0f}  "
                  f"(5-95%: {right.teff.quantile(.05):.0f}-{right.teff.quantile(.95):.0f})")
        if len(wrong):
            near = int((np.abs(wrong.teff - cut) < 250).sum())
            print(f"    errors within +-250 K of the boundary: {near}/{len(wrong)} "
                  f"= {100*near/len(wrong):.0f}%   <-- HIGH => label-precision problem")

    # EN: 3) the decisive plot: accuracy vs distance from the G/K boundary
    # ES: 3) el grafico decisivo: accuracy vs distancia al borde G/K
    meta["dist"] = np.abs(meta["teff"] - cut)
    bins = [0, 100, 200, 300, 500, 750, 1000, 1500, 3000]
    centers, accs, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (meta.dist >= lo) & (meta.dist < hi)
        if m.sum() >= 5:
            centers.append((lo + hi) / 2)
            accs.append(float(meta.loc[m, "correct"].mean()))
            ns.append(int(m.sum()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, c in zip(axes[:2], classes):
        sub = meta[meta.clase == c]
        rng_all = (sub.teff.min(), sub.teff.max())
        ax.hist(sub[sub.correct].teff, bins=25, range=rng_all, alpha=0.7,
                color="#2e8b57", label="correct")
        ax.hist(sub[~sub.correct].teff, bins=25, range=rng_all, alpha=0.8,
                color="crimson", label="misclassified")
        ax.axvline(cut, color="black", ls="--", lw=1.4)
        ax.text(cut, ax.get_ylim()[1] * 0.96, " G/K cut", fontsize=8, va="top")
        ax.set_xlabel("catalogue Teff [K]"); ax.set_ylabel("count")
        ax.set_title(f"[{tag}] true {c}: where are the errors?")
        ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(centers, accs, "o-", color="#1b4a72", lw=1.8, ms=7)
    for x0, y0, n0 in zip(centers, accs, ns):
        ax.annotate(f"n={n0}", (x0, y0), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="grey")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_xlabel("|Teff - 5250 K|  (distance from the G/K boundary)")
    ax.set_ylabel("accuracy")
    ax.set_title("Rising = boundary/label problem\nFlat & low = spectral/instrumental")

    plt.tight_layout()
    out = f"figures/misclassified_{tag}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")
    print("\nREAD IT LIKE THIS / LEER ASI:")
    print("  right panel RISES steeply  -> errors are boundary stars -> LABEL PRECISION (A)")
    print("  right panel stays FLAT/low -> even easy stars fail      -> SPECTRAL/INSTRUMENTAL (B)")


if __name__ == "__main__":
    main()
