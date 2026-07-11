"""
09_payne_compare.py
===================
EN: Second-emulator study (project requirement: two emulators). Steps:
    1) Generate a library of G/K spectra with TransformerPayne (the winning pipeline:
       R(lambda) broadening + iterative normalization) AND keep their labels.
    2) Train a Payne-style MLP emulator (payne.py) to reproduce those spectra from the
       labels -> a fast surrogate = the SECOND emulator.
    3) Sample NEW G/K labels, generate spectra with BOTH emulators, train one RF per
       emulator, and evaluate the REAL DESI accuracy of each. If The Payne is a good
       surrogate, the two emulators give a similar sim->real transfer.
    Saves the Payne model, a reconstruction-check figure, and a bar chart comparing the
    real DESI accuracy of the two emulators.
ES: Estudio del segundo emulador (requisito del proyecto: dos emuladores). Pasos:
    1) Generar una libreria de espectros G/K con TransformerPayne (pipeline ganadora:
       ensanchamiento R(lambda) + normalizacion iterativa) y guardar sus etiquetas.
    2) Entrenar un emulador MLP estilo Payne (payne.py) para reproducir esos espectros
       desde las etiquetas -> un sustituto rapido = el SEGUNDO emulador.
    3) Sortear NUEVAS etiquetas G/K, generar espectros con AMBOS emuladores, entrenar
       un RF por emulador y evaluar la ACCURACY REAL de DESI de cada uno. Si The Payne
       es buen sustituto, ambos dan una transferencia sim->real parecida.
    Guarda el modelo Payne, una figura de chequeo de reconstruccion y un grafico de
    barras comparando la accuracy real de los dos emuladores.

Run / Uso:
    python 09_payne_compare.py --data proyecto_desi/espectros_balanceados_desi

Requires / Requiere: env astro-jax (TransformerPayne) + scikit-learn.
"""
import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import transformer_payne as tp
import project_lib as P
from payne import PayneEmulator, labels_to_matrix

CLASSES = ("G", "K")
BROADEN_R = "desi"      # EN: winning pipeline | ES: pipeline ganadora
NORMALIZER = P.continuum_normalize_iter


def df_labels(df):
    """EN: extract label dicts from a build_balanced_dataset DataFrame.
    ES: extrae dicts de etiquetas de un DataFrame de build_balanced_dataset."""
    dicts = []
    for _, row in df.iterrows():
        d = {"logteff": row["logteff"], "logg": row["logg"]}
        d.update(row["abundances"])
        dicts.append(d)
    return dicts


def rf_real_accuracy_from_arrays(X, y_txt, X_real, y_real):
    """EN: train an RF on (X, y_txt) and return its real accuracy + sim accuracy.
    ES: entrena un RF con (X, y_txt) y devuelve su accuracy real + sim."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    classes = list(CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[t] for t in y_txt])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, criterion="entropy",
                                class_weight="balanced", n_jobs=-1, random_state=23)
    rf.fit(Xtr, ytr)
    sim = accuracy_score(yte, rf.predict(Xte))
    ev = P.evaluate_on_labeled({"model": rf, "classes": classes}, X_real, y_real, classes=classes)
    return ev["accuracy"], sim


def main():
    ap = argparse.ArgumentParser(description="Compare TransformerPayne vs a Payne MLP emulator.")
    ap.add_argument("--data", default="proyecto_desi/espectros_balanceados_desi",
                    help="folder with per-class labelled real DESI spectra")
    ap.add_argument("--n-train-payne", type=int, default=4000, dest="n_train_payne",
                    help="TransformerPayne spectra per class to TRAIN the Payne MLP")
    ap.add_argument("--n-eval", type=int, default=1500, dest="n_eval",
                    help="spectra per class to build each emulator's RF training set")
    ap.add_argument("--real-n", type=int, default=150, dest="real_n")
    ap.add_argument("--sigma-noise", type=float, default=0.02, dest="sigma_noise")
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"[ERROR] real spectra folder not found: '{args.data}'")
    os.makedirs("figures", exist_ok=True)

    print("Loading TransformerPayne weights...")
    emu = tp.TransformerPayne.download()

    # EN: 1) TP library (spectra + labels) to TRAIN the Payne | ES: libreria TP para entrenar Payne
    print("1) generating TransformerPayne library to train the Payne MLP...")
    df_lib = P.build_balanced_dataset(emu, classes=CLASSES, n_per_class=args.n_train_payne,
                                      broaden_R=BROADEN_R, normalizer=NORMALIZER,
                                      sigma_noise=0.0, base_seed=100)   # noise-free targets
    lib_labels = df_labels(df_lib)
    lib_spectra = np.vstack(df_lib["normalized_intensity"].to_numpy())

    print("2) training the Payne MLP emulator...")
    payne = PayneEmulator().fit(lib_labels, lib_spectra)
    payne.save("payne_model.joblib")
    recon = payne.predict_spectra(lib_labels)
    rmse = float(np.sqrt(np.mean((recon - lib_spectra) ** 2)))
    print("   Payne reconstruction RMSE on the library: %.4f" % rmse)

    # EN: reconstruction-check figure (a TP spectrum vs its Payne reconstruction)
    # ES: figura de chequeo (un espectro TP vs su reconstruccion Payne)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(P.WAVE_GRID, lib_spectra[0], color="#1b4a72", lw=1.2, label="TransformerPayne (target)")
    ax.plot(P.WAVE_GRID, recon[0], color="crimson", lw=1.0, alpha=0.8, label="The Payne (MLP reconstruction)")
    for nm, (lam, _) in P.SPECTRAL_LINES.items():
        if P.WMIN <= lam <= P.WMAX:
            ax.axvline(lam, color="grey", ls="--", lw=0.7, alpha=0.6)
    ax.set_xlim(P.WMIN, P.WMAX); ax.set_xlabel(r"Wavelength [$\AA$]"); ax.set_ylabel("Norm. intensity")
    ax.set_title("The Payne MLP reproduces TransformerPayne (RMSE=%.4f)" % rmse)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout(); plt.savefig("figures/payne_reconstruction.png", dpi=130, bbox_inches="tight"); plt.close()

    # EN: 3) fresh labels -> spectra from BOTH emulators -> one RF each -> real accuracy
    # ES: 3) etiquetas nuevas -> espectros de AMBOS emuladores -> un RF cada uno -> accuracy real
    print("3) building an RF from each emulator and evaluating real DESI accuracy...")
    df_new = P.build_balanced_dataset(emu, classes=CLASSES, n_per_class=args.n_eval,
                                      broaden_R=BROADEN_R, normalizer=NORMALIZER,
                                      sigma_noise=args.sigma_noise, base_seed=500)
    y_new = df_new["spectral_type"].to_numpy().astype(str)
    X_tp = np.vstack(df_new["normalized_intensity"].to_numpy())

    # EN: Payne spectra for the SAME labels (+ same noise level) | ES: espectros Payne mismas etiquetas
    new_labels = df_labels(df_new)
    nprng = np.random.RandomState(500)
    X_pn = payne.predict_spectra(new_labels)
    X_pn = X_pn + nprng.normal(0.0, args.sigma_noise, size=X_pn.shape)

    # EN: real labelled spectra (iterative norm, matching the training) | ES: reales (norm iterativa)
    X_real, y_real, counts = P.load_labeled_desi_folder(
        args.data, classes=CLASSES, n_per_class=args.real_n, balanced=True, seed=0,
        normalizer=NORMALIZER)
    print("   real labelled spectra per class:", counts, "-> N =", len(y_real))

    acc_tp_real, acc_tp_sim = rf_real_accuracy_from_arrays(X_tp, y_new, X_real, y_real)
    acc_pn_real, acc_pn_sim = rf_real_accuracy_from_arrays(X_pn, y_new, X_real, y_real)
    print("\n=== Emulator comparison (real DESI accuracy) ===")
    print("  TransformerPayne : sim %.3f | REAL %.3f" % (acc_tp_sim, acc_tp_real))
    print("  The Payne (MLP)  : sim %.3f | REAL %.3f" % (acc_pn_sim, acc_pn_real))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["TransformerPayne", "The Payne\n(MLP)"], [acc_tp_real, acc_pn_real],
                  color=["#1b4a72", "#2e6da4"])
    for b, v in zip(bars, [acc_tp_real, acc_pn_real]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1); ax.set_ylabel("Real DESI accuracy (G/K)")
    ax.set_title("Two emulators: sim->real transfer compared")
    plt.tight_layout(); out = "figures/emulator_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print("\nsaved figures/payne_reconstruction.png and", out)


if __name__ == "__main__":
    main()
