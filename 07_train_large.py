"""
07_train_large.py
=================
EN: Train the Random Forest on the LARGE synthetic set produced by
    06_generate_large.py (.npz), then evaluate the REAL DESI accuracy. At 100k the
    GridSearch+CV of the small pipeline is far too slow, so here the hyperparameters
    are FIXED (good defaults; override on the command line). Needs NO TransformerPayne
    (pure scikit-learn), so it runs anywhere in a few minutes once the .npz exists.
ES: Entrena el Random Forest con el set sintetico GRANDE producido por
    06_generate_large.py (.npz) y luego evalua la ACCURACY REAL de DESI. A 100k el
    GridSearch+CV de la pipeline chica es demasiado lento, asi que aca los
    hiperparametros son FIJOS (buenos por defecto; se pueden sobrescribir por linea de
    comandos). NO necesita TransformerPayne (solo scikit-learn), corre en minutos una
    vez que existe el .npz.

Run / Uso:
    python 07_train_large.py --data-npz sim_100k.npz \
        --real proyecto_desi/espectros_balanceados_desi --norm iterative

Requires / Requiere: numpy, scikit-learn, joblib, matplotlib (no TransformerPayne).
IMPORTANT: --norm must MATCH the normalizer used in 06_generate_large.py, so the real
    spectra are normalized the same way as the synthetic ones.
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
import seaborn as sns

import project_lib as P


def main():
    ap = argparse.ArgumentParser(description="Train RF on the large .npz + real DESI eval")
    ap.add_argument("--data-npz", required=True, dest="data_npz", help="output of 06_generate_large.py")
    ap.add_argument("--real", default="proyecto_desi/espectros_balanceados_desi",
                    help="folder with per-class labelled real DESI spectra")
    ap.add_argument("--norm", default="masked", choices=["masked", "iterative", "percentile"],
                    help="MUST match the normalizer used to generate the .npz. "
                         "'masked' = line-masked continuum fit (the H-beta fix)")
    ap.add_argument("--real-n", type=int, default=150, dest="real_n", help="real spectra per class")
    # EN: restrict the REAL sample to the emulator's training domain (dwarfs only!).
    #     The DESI sample contains ~17% giants (logg<3.5) that the synthetic grid never
    #     covered -> use --min-logg 3.5 (or 4.0) to test the selection-mismatch hypothesis.
    # ES: restringe la muestra REAL al dominio de entrenamiento (solo enanas!).
    ap.add_argument("--min-logg", type=float, default=None, dest="min_logg",
                    help="drop real stars below this log g (e.g. 3.5 removes giants)")
    ap.add_argument("--max-logg", type=float, default=None, dest="max_logg")
    ap.add_argument("--teff-range", type=float, nargs=2, default=None, dest="teff_range",
                    metavar=("TMIN", "TMAX"),
                    help="keep only real stars with TMIN <= Teff <= TMAX")
    # EN: The team's DESI CSVs have NO redshift column -> the spectra are in the
    #     OBSERVED frame and their lines are displaced by the star's radial velocity
    #     (sigma ~ 220 km/s = up to +-12 A). Since the RF reads the Balmer depth at
    #     FIXED pixels, a displaced line looks weaker -> G gets misread as K.
    #     --rv-correct estimates the shift per spectrum by cross-correlation against
    #     the mean synthetic spectrum and brings it back to rest.
    # ES: Los CSV DESI del equipo NO traen redshift -> espectros en el marco OBSERVADO,
    #     lineas corridas por la velocidad radial. --rv-correct estima el corrimiento
    #     por correlacion cruzada contra el sintetico medio y lo devuelve al reposo.
    ap.add_argument("--rv-correct", action="store_true", dest="rv_correct",
                    help="correct each real spectrum for its radial velocity (cross-correlation)")
    ap.add_argument("--max-shift", type=float, default=15.0, dest="max_shift",
                    help="max |shift| searched, in Angstrom")
    # EN: Wavelength selection. 13_spectra_compare.py showed that DESI's H-beta is 20%
    #     SHALLOWER than SDSS's at the same Teff, while every other line agrees. Since
    #     SHAP showed the RF leans almost entirely on H-beta, that single broken line can
    #     explain the whole G->K bias. --exclude masks a wavelength range so we can test
    #     the classifier WITHOUT it. --wave-min/--wave-max restrict the range instead.
    # ES: Seleccion de longitudes de onda. 13_spectra_compare.py mostro que la H-beta de
    #     DESI es 20% MAS PLANA que la de SDSS al mismo Teff, mientras el resto coincide.
    #     Como SHAP mostro que el RF depende casi solo de H-beta, esa unica linea rota
    #     puede explicar todo el sesgo G->K. --exclude enmascara un rango para probarlo.
    ap.add_argument("--wave-min", type=float, default=None, dest="wave_min",
                    help="use only wavelengths >= this (A)")
    ap.add_argument("--wave-max", type=float, default=None, dest="wave_max",
                    help="use only wavelengths <= this (A)")
    ap.add_argument("--exclude", type=float, nargs=2, default=None,
                    metavar=("LMIN", "LMAX"),
                    help="EXCLUDE this wavelength range, e.g. --exclude 4830 4900 (H-beta)")
    ap.add_argument("--tag", default=None,
                    help="label for the output files (default: derived from --real)")
    ap.add_argument("--n-estimators", type=int, default=200, dest="n_estimators")
    ap.add_argument("--max-depth", default="none", dest="max_depth", help="int or 'none'")
    ap.add_argument("--min-samples-leaf", type=int, default=2, dest="min_samples_leaf")
    ap.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    ap.add_argument("--out-model", default="rf_large_model.joblib", dest="out_model")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib

    os.makedirs("figures", exist_ok=True)
    normalizer = P.get_normalizer(args.norm)
    max_depth = None if str(args.max_depth).lower() == "none" else int(args.max_depth)
    # EN: tag = short name of the real survey, so runs do not overwrite each other
    # ES: tag = nombre corto del survey real, para que las corridas no se pisen
    tag = args.tag or os.path.basename(os.path.normpath(args.real)).replace("espectros_", "")
    if args.min_logg is not None:
        tag += f"_logg{args.min_logg:g}+"

    # EN: 1) load the large synthetic set | ES: 1) cargar el set sintetico grande
    d = np.load(args.data_npz, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    y_txt = np.asarray(d["y"]).astype(str)
    classes = sorted(np.unique(y_txt).tolist())
    cls_index = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_index[t] for t in y_txt])
    print(f"loaded {args.data_npz}: X={X.shape} classes={classes} "
          f"counts={ {c:int((y_txt==c).sum()) for c in classes} }")

    # EN: wavelength mask -- applied IDENTICALLY to the synthetic and the real features.
    # ES: mascara de longitud de onda -- aplicada IGUAL a los features sinteticos y reales.
    wmask = np.ones(len(P.WAVE_GRID), dtype=bool)
    if args.wave_min is not None:
        wmask &= P.WAVE_GRID >= args.wave_min
        tag += f"_wmin{args.wave_min:.0f}"
    if args.wave_max is not None:
        wmask &= P.WAVE_GRID <= args.wave_max
        tag += f"_wmax{args.wave_max:.0f}"
    if args.exclude is not None:
        wmask &= ~((P.WAVE_GRID >= args.exclude[0]) & (P.WAVE_GRID <= args.exclude[1]))
        tag += f"_excl{args.exclude[0]:.0f}-{args.exclude[1]:.0f}"
    if not wmask.all():
        print("wavelength mask: keeping %d/%d pixels (%.0f-%.0f A)"
              % (wmask.sum(), len(wmask), P.WAVE_GRID[wmask].min(), P.WAVE_GRID[wmask].max()))
        X = X[:, wmask]

    # EN: 2) train RF with FIXED hyperparameters (fast at 100k)
    # ES: 2) entrenar RF con hiperparametros FIJOS (rapido a 100k)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          random_state=23, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf, criterion="entropy",
        class_weight="balanced", n_jobs=-1, random_state=23)
    t0 = time.time()
    rf.fit(Xtr, ytr)
    print(f"RF trained in {time.time()-t0:.0f}s on {Xtr.shape[0]} spectra")
    sim_acc = accuracy_score(yte, rf.predict(Xte))
    print("simulated test accuracy:", round(sim_acc, 3))

    out_model = args.out_model if args.out_model != "rf_large_model.joblib" \
        else f"rf_large_{tag}.joblib"
    joblib.dump({"model": rf, "classes": classes, "wave_grid": P.WAVE_GRID[wmask],
                 "norm": args.norm}, out_model)
    print("saved", out_model)

    # EN: 3) REAL DESI accuracy with the MATCHING normalizer
    # ES: 3) accuracy REAL de DESI con el normalizador CONCORDANTE
    if os.path.isdir(args.real):
        # EN: RV template = mean synthetic spectrum on the FULL grid (mask comes later)
        # ES: template RV = sintetico medio en la grilla COMPLETA (la mascara va despues)
        rv_template = None
        if args.rv_correct:
            Xfull = np.asarray(d["X"], dtype=np.float32)
            rv_template = Xfull.mean(axis=0).astype(float)
            tag += "_rv"
        X_real, y_real, counts = P.load_labeled_desi_folder(
            args.real, classes=tuple(classes), n_per_class=args.real_n,
            balanced=True, seed=0, normalizer=normalizer,
            min_logg=args.min_logg, max_logg=args.max_logg, teff_range=args.teff_range,
            rv_template=rv_template, max_shift=args.max_shift)
        # EN: same wavelength mask as the training features | ES: misma mascara
        if not wmask.all() and X_real.shape[1] == len(wmask):
            X_real = X_real[:, wmask]
        if args.rv_correct and "_rv_std_kms" in counts:
            print("RV correction: measured shift median=%.2f A, std=%.2f A (RV std ~ %.0f km/s)"
                  % (counts.pop("_rv_shift_median_A"), counts.pop("_rv_shift_std_A"),
                     counts.pop("_rv_std_kms")))
        cuts = []
        if args.min_logg is not None:
            cuts.append(f"logg >= {args.min_logg:g} (giants removed)")
        if args.max_logg is not None:
            cuts.append(f"logg <= {args.max_logg:g}")
        if args.teff_range is not None:
            cuts.append(f"Teff in [{args.teff_range[0]:.0f}, {args.teff_range[1]:.0f}] K")
        if cuts:
            print("real-sample cuts:", "; ".join(cuts))
        print("real labelled spectra per class:", counts, "-> N =", len(y_real))
        if len(y_real) == 0:
            raise SystemExit("[ERROR] no real spectra left after the cuts.")
        ev = P.evaluate_on_labeled({"model": rf, "classes": classes}, X_real, y_real,
                                   classes=classes)
        print("\n=== REAL accuracy [%s] (RF trained on %d synthetic): %.3f ==="
              % (tag, X.shape[0], ev["accuracy"]))
        print(ev["report"])

        cm = ev["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True (real label)")
        ax.set_title("Real %s %s - acc=%.3f, N=%d"
                     % (tag, "/".join(classes), ev["accuracy"], len(y_real)))
        plt.tight_layout()
        out = f"figures/real_confusion_{tag}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
        print("saved", out)
    else:
        print("[skip real eval] folder not found:", args.real)


if __name__ == "__main__":
    main()
