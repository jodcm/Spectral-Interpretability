"""
15_summary.py
=============
EN: The reference scale. A single number like "0.76 on DESI" is meaningless on its own --
    good or bad compared to WHAT? This script puts every result on one axis so the
    sim->real transfer can finally be judged:

      (1) SIM-TRAINED  -> real survey      = our actual method ("source-only")
      (2) REAL-TRAINED -> same real survey = the UPPER BOUND ("target-supervised"):
          how well can ANY model do with this data and these labels?
      (3) the GAP between (1) and (2)      = the true domain shift, by definition.
      (4) CROSS-SURVEY real->real          = do the two instruments even agree with each
          other? This is measured with NO simulation involved at all.

    Reporting (1) and (2) together is standard practice in domain-adaptation work. If the
    upper bound on DESI is also low, then DESI's data/labels are the ceiling and our 0.76
    is close to optimal. If the upper bound is high, the gap is real and belongs to the
    simulation.
ES: La escala de referencia. Un numero suelto como "0.76 en DESI" no significa nada --
    bueno o malo comparado con QUE? Este script pone todos los resultados en un mismo eje:

      (1) ENTRENADO EN SIM  -> survey real     = nuestro metodo ("source-only")
      (2) ENTRENADO EN REAL -> mismo survey    = la COTA SUPERIOR ("target-supervised")
      (3) la BRECHA entre (1) y (2)            = el domain shift, por definicion
      (4) ENTRE SURVEYS real->real             = coinciden los dos instrumentos entre si?
          Se mide SIN simulacion de por medio.

Run / Uso:
    python 15_summary.py --sim-desi sim_10k_masked.npz --sim-sdss sim_50k_sdss.npz \
        --desi proyecto_desi/espectros_balanceados_desi --sdss proyecto_desi/espectros_sdss

Requires / Requiere: numpy, scikit-learn, matplotlib (NO TransformerPayne).
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


def make_rf(n_estimators=250, seed=23):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_estimators, criterion="entropy",
                                  class_weight="balanced", min_samples_leaf=2,
                                  n_jobs=-1, random_state=seed)


def sim_to_real(npz_path, real_dir, classes, normalizer, real_n):
    """EN: train on the synthetic .npz, test on the real survey. ES: sim -> real."""
    from sklearn.metrics import accuracy_score
    if not (npz_path and os.path.exists(npz_path)) or not os.path.isdir(real_dir):
        return None
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    y_txt = np.asarray(d["y"]).astype(str)
    cls = [c for c in classes if c in set(y_txt)]
    if not cls:
        return None
    keep = np.isin(y_txt, cls)
    ci = {c: i for i, c in enumerate(cls)}
    rf = make_rf().fit(X[keep], np.array([ci[t] for t in y_txt[keep]]))
    Xr, yr, _ = P.load_labeled_desi_folder(real_dir, classes=tuple(cls), n_per_class=real_n,
                                           balanced=True, seed=0, normalizer=normalizer)
    if len(yr) == 0:
        return None
    return float(accuracy_score(np.array([ci[t] for t in yr]), rf.predict(Xr)))


def real_upper_bound(real_dir, classes, normalizer, n, test_size=0.25):
    """EN: train AND test on the same real survey -> the ceiling. ES: la cota superior."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    if not os.path.isdir(real_dir):
        return None, None
    X, y_txt, _ = P.load_labeled_desi_folder(real_dir, classes=tuple(classes), n_per_class=n,
                                             balanced=True, seed=0, normalizer=normalizer)
    if len(y_txt) < 40:
        return None, None
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[t] for t in y_txt])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=23, stratify=y)
    rf = make_rf().fit(Xtr, ytr)
    return float(accuracy_score(yte, rf.predict(Xte))), rf


def cross_survey(rf_a, real_b, classes, normalizer, n):
    """EN: model trained on real survey A, tested on real survey B. ES: entre surveys."""
    from sklearn.metrics import accuracy_score
    if rf_a is None or not os.path.isdir(real_b):
        return None
    X, y_txt, _ = P.load_labeled_desi_folder(real_b, classes=tuple(classes), n_per_class=n,
                                             balanced=True, seed=0, normalizer=normalizer)
    if len(y_txt) == 0:
        return None
    ci = {c: i for i, c in enumerate(classes)}
    return float(accuracy_score(np.array([ci[t] for t in y_txt]), rf_a.predict(X)))


def main():
    ap = argparse.ArgumentParser(description="Put every result on one reference scale")
    ap.add_argument("--sim-desi", default="sim_10k_masked.npz", dest="sim_desi",
                    help="synthetic .npz generated at DESI resolution")
    ap.add_argument("--sim-sdss", default="sim_50k_sdss.npz", dest="sim_sdss",
                    help="synthetic .npz generated at SDSS resolution (R=2000)")
    ap.add_argument("--desi", default="proyecto_desi/espectros_balanceados_desi")
    ap.add_argument("--sdss", default="proyecto_desi/espectros_sdss")
    ap.add_argument("--classes", nargs="+", default=["G", "K"])
    # EN: 'iterative' is the BEST configuration. 'masked' fixed H-beta but over-deepened
    #     the other lines and made DESI's sim->real WORSE (0.760 -> 0.637) -- kept as a
    #     documented negative result. ES: 'iterative' es la MEJOR configuracion.
    ap.add_argument("--norm", default="iterative", choices=["iterative", "masked", "percentile"])
    ap.add_argument("--real-n", type=int, default=400, dest="real_n")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    normalizer = P.get_normalizer(args.norm)
    classes = args.classes
    lbl = "/".join(classes)
    print(f"classes = {lbl}   normalizer = {args.norm}\n")

    res = {}
    res["desi_sim"] = sim_to_real(args.sim_desi, args.desi, classes, normalizer, args.real_n)
    res["desi_real"], rf_desi = real_upper_bound(args.desi, classes, normalizer, args.real_n * 2)
    res["sdss_sim"] = sim_to_real(args.sim_sdss, args.sdss, classes, normalizer, args.real_n)
    res["sdss_real"], _ = real_upper_bound(args.sdss, classes, normalizer, args.real_n * 2)
    res["cross"] = cross_survey(rf_desi, args.sdss, classes, normalizer, args.real_n)

    def s(v):
        return "  n/a " if v is None else f"{v:.3f}"

    print("=" * 62)
    print(f"{'':<34}{'DESI':>12}{'SDSS':>12}")
    print("-" * 62)
    print(f"{'(1) sim-trained  -> real':<34}{s(res['desi_sim']):>12}{s(res['sdss_sim']):>12}")
    print(f"{'(2) real-trained -> real (CEILING)':<34}{s(res['desi_real']):>12}{s(res['sdss_real']):>12}")
    print("-" * 62)
    for k, name in (("desi", "DESI"), ("sdss", "SDSS")):
        a, b = res[f"{k}_sim"], res[f"{k}_real"]
        if a is not None and b is not None:
            print(f"    domain shift on {name:<5}: {b - a:+.3f}   "
                  f"(ceiling {b:.3f} - sim {a:.3f})")
    if res["cross"] is not None:
        print(f"\n(4) cross-survey  real DESI -> real SDSS : {res['cross']:.3f}"
              "   (NO simulation involved)")
    print("=" * 62)
    print("\nHOW TO READ / COMO LEER:")
    print("  small domain shift  -> our sim-trained model is near the ceiling; the")
    print("                         limit is the DATA (noise, labels), not the emulator.")
    print("  large domain shift  -> the simulation really is missing something.")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(8.5, 5))
    surveys, sims, reals = [], [], []
    for k, name in (("desi", "DESI"), ("sdss", "SDSS")):
        if res[f"{k}_sim"] is not None or res[f"{k}_real"] is not None:
            surveys.append(name)
            sims.append(res[f"{k}_sim"] or 0.0)
            reals.append(res[f"{k}_real"] or 0.0)
    x = np.arange(len(surveys))
    w = 0.36
    b1 = ax.bar(x - w / 2, sims, w, color="#2e6da4", label="sim-trained (our method)")
    b2 = ax.bar(x + w / 2, reals, w, color="#9aa0a6", label="real-trained (upper bound)")
    for bars in (b1, b2):
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                        f"{b.get_height():.3f}", ha="center", fontsize=9)
    # EN: annotate the domain shift | ES: anotar el domain shift
    for i, (a, b) in enumerate(zip(sims, reals)):
        if a > 0 and b > 0:
            ax.annotate("", xy=(i + w / 2, b), xytext=(i - w / 2, a),
                        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.4))
            ax.text(i, (a + b) / 2 + 0.03, f"gap {b - a:+.3f}", ha="center",
                    color="crimson", fontsize=9, fontweight="bold")
    if res["cross"] is not None:
        ax.axhline(res["cross"], color="darkorange", ls="--", lw=1.3)
        ax.text(len(surveys) - 0.5, res["cross"] + 0.012,
                f"cross-survey real DESI->SDSS = {res['cross']:.3f}",
                ha="right", color="darkorange", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(surveys)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"accuracy ({lbl})")
    ax.set_title("The reference scale: how far is our sim->real model from the ceiling?\n"
                 f"(normalizer: {args.norm})", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"figures/summary_scale_{''.join(classes)}_{args.norm}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
