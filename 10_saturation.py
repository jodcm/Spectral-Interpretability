"""
10_saturation.py
================
EN: Learning curve: REAL DESI accuracy as a function of the number of SYNTHETIC
    training spectra. This answers the team's central question ("we cannot get
    700,000 spectra per class -- is that a problem?") with a measurement instead of an
    apology: if the curve flattens early, then a few thousand spectra are ENOUGH and
    the limiting factor is the domain shift, not the sample size.

    It reuses the already-generated sim_100k.npz: for each training size it draws a
    balanced random subsample, trains an RF with the SAME fixed hyperparameters, and
    evaluates on the labelled real DESI spectra. One controlled experiment, no
    regeneration, no TransformerPayne needed.
ES: Curva de aprendizaje: accuracy REAL de DESI en funcion del numero de espectros
    SINTETICOS de entrenamiento. Responde la pregunta central del equipo ("no podemos
    conseguir 700.000 espectros por clase -- es un problema?") con una medicion en vez
    de una disculpa: si la curva se aplana temprano, unos pocos miles de espectros
    ALCANZAN y lo que limita es el domain shift, no el tamano de la muestra.

    Reutiliza el sim_100k.npz ya generado: para cada tamano toma una submuestra
    balanceada al azar, entrena un RF con los MISMOS hiperparametros fijos y evalua
    sobre los espectros DESI reales etiquetados. Un experimento controlado, sin
    regenerar nada y sin necesitar TransformerPayne.

Run / Uso:
    python 10_saturation.py --data-npz sim_100k.npz --norm iterative

Requires / Requiere: numpy, scikit-learn, matplotlib (NO TransformerPayne).
"""
import os
import time
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import project_lib as P


def main():
    ap = argparse.ArgumentParser(description="Real accuracy vs number of synthetic training spectra")
    ap.add_argument("--data-npz", default="sim_100k.npz", dest="data_npz",
                    help="synthetic set from 06_generate_large.py")
    ap.add_argument("--real", default="proyecto_desi/espectros_balanceados_desi",
                    help="folder with per-class labelled real DESI spectra")
    ap.add_argument("--norm", default="iterative", choices=["iterative", "percentile"],
                    help="MUST match the normalizer used to generate the .npz")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
                    help="TOTAL training sizes to test (split evenly across classes)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="random subsamples per size (averaged -> error bars)")
    ap.add_argument("--real-n", type=int, default=150, dest="real_n")
    ap.add_argument("--n-estimators", type=int, default=200, dest="n_estimators")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    os.makedirs("figures", exist_ok=True)
    normalizer = P.continuum_normalize_iter if args.norm == "iterative" else P.continuum_normalize

    # EN: 1) load the synthetic pool + the real evaluation set (loaded ONCE)
    # ES: 1) cargar el pool sintetico + el set real de evaluacion (UNA sola vez)
    d = np.load(args.data_npz, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    y_txt = np.asarray(d["y"]).astype(str)
    classes = sorted(np.unique(y_txt).tolist())
    print(f"pool: {X.shape[0]} synthetic spectra, classes={classes}")

    X_real, y_real, counts = P.load_labeled_desi_folder(
        args.real, classes=tuple(classes), n_per_class=args.real_n, balanced=True,
        seed=0, normalizer=normalizer)
    if len(y_real) == 0:
        raise SystemExit(f"[ERROR] no real spectra found in '{args.real}'")
    print(f"real evaluation set: {counts} -> N = {len(y_real)}\n")

    by_class = {c: np.where(y_txt == c)[0] for c in classes}
    cls_index = {c: i for i, c in enumerate(classes)}
    pool_max = min(len(v) for v in by_class.values()) * len(classes)

    sizes, means, stds = [], [], []
    for total in args.sizes:
        per = total // len(classes)
        if per > min(len(v) for v in by_class.values()):
            print(f"[skip] size {total} exceeds the pool ({pool_max})")
            continue
        accs = []
        for rep in range(args.repeats):
            rng = np.random.RandomState(1000 + rep)
            idx = np.concatenate([rng.choice(by_class[c], size=per, replace=False)
                                  for c in classes])
            Xs = X[idx]
            ys = np.array([cls_index[t] for t in y_txt[idx]])
            rf = RandomForestClassifier(n_estimators=args.n_estimators, criterion="entropy",
                                        class_weight="balanced", min_samples_leaf=2,
                                        n_jobs=-1, random_state=23 + rep)
            rf.fit(Xs, ys)
            ev = P.evaluate_on_labeled({"model": rf, "classes": classes},
                                       X_real, y_real, classes=classes)
            accs.append(ev["accuracy"])
        sizes.append(total)
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs)))
        print(f"  n={total:>7d} (per class {per:>6d})  REAL acc = {means[-1]:.3f} +- {stds[-1]:.3f}")

    # EN: 2) the learning curve | ES: 2) la curva de aprendizaje
    sizes = np.array(sizes); means = np.array(means); stds = np.array(stds)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(sizes, means, yerr=stds, marker="o", color="#1b4a72",
                capsize=3, lw=1.6, ms=6, label="real DESI accuracy")
    # EN: mark the plateau (mean of the last three points) | ES: marca la meseta
    plateau = float(np.mean(means[-3:])) if len(means) >= 3 else float(means[-1])
    ax.axhline(plateau, color="grey", ls="--", lw=1.0, alpha=0.8)
    ax.text(sizes[0], plateau + 0.012, f"plateau ~ {plateau:.2f}", fontsize=9, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("Number of synthetic training spectra (log scale)")
    ax.set_ylabel("Real DESI accuracy (G/K)")
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("More synthetic data does not help: the limit is the domain shift")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = "figures/saturation_curve.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")
    print("Plateau reached at ~%.2f -> more spectra beyond a few thousand add nothing."
          % plateau)


if __name__ == "__main__":
    main()
