"""
14_train_real.py
================
EN: Train the Random Forest DIRECTLY ON REAL SPECTRA (no emulator involved). This gives
    two things the sim->real study was missing:

    1) AN UPPER BOUND. Everything so far trained on synthetic spectra and tested on real
       ones. We never asked: how well can ANY model do on this real data with these
       labels? If a model trained on real DESI spectra ALSO tops out around 0.85, then
       DESI's data/labels are the ceiling -- and our sim->real transfer (0.76) is close
       to optimal, not broken. This is the reference every sim->real paper needs.

    2) F, G AND K. The synthetic side is limited to G/K because TransformerPayne is only
       validated for dwarfs at ~4000-6000 K. Training on REAL data has NO such limit:
       any class with labelled spectra can be used. Cata's DESI folder has 1000 spectra
       for every class (O B A F G K M), so F/G/K works out of the box.

    Also supports CROSS-SURVEY transfer: train on one survey, test on another
    (--test-real). Real DESI -> real SDSS tells us how much of the gap is instrumental,
    completely independently of the simulation.
ES: Entrena el Random Forest DIRECTAMENTE CON ESPECTROS REALES (sin emulador). Aporta dos
    cosas que faltaban:

    1) UNA COTA SUPERIOR. Hasta ahora entrenabamos con sinteticos y probabamos en reales.
       Nunca preguntamos: cuanto puede lograr CUALQUIER modelo con estos datos y estas
       etiquetas? Si un modelo entrenado con DESI real tambien se estanca en ~0.85,
       entonces el techo lo ponen los datos/etiquetas de DESI y nuestra transferencia
       sim->real (0.76) esta cerca del optimo, no rota.

    2) F, G Y K. El lado sintetico esta limitado a G/K porque TransformerPayne solo esta
       validado para enanas de ~4000-6000 K. Entrenar con datos REALES no tiene ese
       limite: sirve cualquier clase con espectros etiquetados. La carpeta DESI de Cata
       tiene 1000 espectros por clase (O B A F G K M), asi que F/G/K funciona directo.

    Tambien soporta transferencia ENTRE SURVEYS: entrenar en uno y probar en otro
    (--test-real). DESI real -> SDSS real mide cuanto del gap es instrumental, con total
    independencia de la simulacion.

Run / Uso:
    # upper bound on DESI, F/G/K:
    python 14_train_real.py --real proyecto_desi/espectros_balanceados_desi --classes F G K
    # upper bound on SDSS, G/K:
    python 14_train_real.py --real proyecto_desi/espectros_sdss --classes G K
    # cross-survey: train on DESI, test on SDSS
    python 14_train_real.py --real proyecto_desi/espectros_balanceados_desi \
                            --test-real proyecto_desi/espectros_sdss --classes G K

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
import seaborn as sns

import project_lib as P


def main():
    ap = argparse.ArgumentParser(description="Train the RF directly on REAL spectra")
    ap.add_argument("--real", default="proyecto_desi/espectros_balanceados_desi",
                    help="folder with per-class labelled real spectra (training set)")
    ap.add_argument("--test-real", default=None, dest="test_real",
                    help="OPTIONAL: a different survey to test on (cross-survey transfer)")
    ap.add_argument("--classes", nargs="+", default=["F", "G", "K"],
                    help="spectral classes to use, e.g. F G K  (real data has no emulator limit)")
    ap.add_argument("--n", type=int, default=800, dest="n",
                    help="real spectra per class for training")
    ap.add_argument("--test-n", type=int, default=200, dest="test_n",
                    help="real spectra per class for the cross-survey test")
    ap.add_argument("--norm", default="iterative", choices=["iterative", "masked", "percentile"])
    ap.add_argument("--min-logg", type=float, default=None, dest="min_logg")
    ap.add_argument("--test-size", type=float, default=0.25, dest="test_size")
    ap.add_argument("--n-estimators", type=int, default=300, dest="n_estimators")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    os.makedirs("figures", exist_ok=True)
    normalizer = P.get_normalizer(args.norm)
    classes = tuple(args.classes)
    tag = os.path.basename(os.path.normpath(args.real)).replace("espectros_", "")
    tag += "_" + "".join(classes)

    # EN: 1) load the REAL training set | ES: 1) cargar el set REAL de entrenamiento
    X, y_txt, counts = P.load_labeled_desi_folder(
        args.real, classes=classes, n_per_class=args.n, balanced=True, seed=0,
        normalizer=normalizer, min_logg=args.min_logg)
    if len(y_txt) == 0:
        raise SystemExit(f"[ERROR] no spectra found in '{args.real}' for classes {classes}")
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[t] for t in y_txt])
    print(f"REAL training set [{tag}]: {counts} -> N = {len(y)}")

    # EN: 2) train/test split ON THE REAL DATA -> the upper bound
    # ES: 2) split train/test SOBRE LOS REALES -> la cota superior
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          random_state=23, stratify=y)
    rf = RandomForestClassifier(n_estimators=args.n_estimators, criterion="entropy",
                                class_weight="balanced", min_samples_leaf=2,
                                n_jobs=-1, random_state=23)
    rf.fit(Xtr, ytr)
    y_pred = rf.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    print(f"\n=== UPPER BOUND: RF trained on REAL {tag}, tested on held-out REAL: {acc:.3f} ===")
    print(classification_report(yte, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(yte, y_pred)
    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 3, 1.3 * len(classes) + 2.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (real label)")
    ax.set_title(f"REAL-trained RF [{tag}] - acc={acc:.3f}, N={len(yte)}")
    plt.tight_layout()
    out = f"figures/real_trained_{tag}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", out)

    # EN: 3) OPTIONAL cross-survey transfer (real -> real, no simulation involved)
    # ES: 3) OPCIONAL transferencia entre surveys (real -> real, sin simulacion)
    if args.test_real:
        Xb, yb_txt, cb = P.load_labeled_desi_folder(
            args.test_real, classes=classes, n_per_class=args.test_n, balanced=True,
            seed=0, normalizer=normalizer, min_logg=args.min_logg)
        if len(yb_txt) == 0:
            print(f"[skip cross-survey] no spectra in '{args.test_real}'")
            return
        tb = os.path.basename(os.path.normpath(args.test_real)).replace("espectros_", "")
        yb = np.array([ci[t] for t in yb_txt])
        pb = rf.predict(Xb)
        ab = accuracy_score(yb, pb)
        print(f"\n=== CROSS-SURVEY: trained on REAL {tag} -> tested on REAL {tb}: {ab:.3f} ===")
        print(f"    ({cb} -> N = {len(yb)})")
        print(classification_report(yb, pb, target_names=classes, zero_division=0))
        print("\n  If this is ALSO low, the two instruments disagree -> the gap is")
        print("  INSTRUMENTAL, independent of the simulation. If it is high, the")
        print("  instruments agree and the sim->real gap is the emulator's fault.")

        cm = confusion_matrix(yb, pb)
        fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 3, 1.3 * len(classes) + 2.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel(f"True ({tb} label)")
        ax.set_title(f"Cross-survey: trained on {tag} -> {tb}  acc={ab:.3f}")
        plt.tight_layout()
        out = f"figures/cross_survey_{tag}_to_{tb}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
        print("saved", out)


if __name__ == "__main__":
    main()
