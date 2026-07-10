"""
05_normalization.py
===================
EN: Step 5 (domain-shift mitigation, part 2). The broadening step (03) showed that
    matching the instrument RESOLUTION barely moved the real accuracy (0.66 -> 0.67):
    the remaining sim->real gap comes from the CONTINUUM / NORMALIZATION, not the
    line widths. This script tests an IMPROVED continuum normalization
    (`continuum_normalize_iter`: low-order polynomial + asymmetric sigma clipping),
    applied IDENTICALLY to the synthetic and the real spectra, and compares the REAL
    DESI accuracy across four development stages to show the progress:
        (1) baseline          : sharp synthetic  + percentile normalization
        (2) + R(lambda) broad. : broadened synth. + percentile normalization
        (3) + improved norm    : sharp synthetic  + iterative normalization
        (4) + both             : broadened synth. + iterative normalization
    The old figures (confusion_matrix, broadening_real_accuracy) are NOT touched;
    this adds a new figure figures/normalization_progress.png.
ES: Paso 5 (mitigacion del domain shift, parte 2). El paso de ensanchamiento (03)
    mostro que igualar la RESOLUCION casi no movio la accuracy real (0.66 -> 0.67):
    la brecha sim->real que queda viene del CONTINUO / NORMALIZACION, no del ancho de
    linea. Este script prueba una normalizacion de continuo MEJORADA
    (`continuum_normalize_iter`: polinomio de grado bajo + recorte sigma asimetrico),
    aplicada IGUAL a los sinteticos y a los reales, y compara la ACCURACY REAL de DESI
    en cuatro etapas de desarrollo para mostrar el progreso:
        (1) baseline           : sintetico agudo    + normalizacion por percentil
        (2) + ensanch. R(lambda): sintetico ensanch. + normalizacion por percentil
        (3) + norm. mejorada    : sintetico agudo    + normalizacion iterativa
        (4) + ambas             : sintetico ensanch. + normalizacion iterativa
    Las figuras viejas (confusion_matrix, broadening_real_accuracy) NO se tocan; esto
    agrega una figura nueva figures/normalization_progress.png.

Run / Uso:
    python 05_normalization.py --data proyecto_desi/espectros_balanceados_desi

Requires / Requiere: env astro-jax (TransformerPayne).
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

CLASSES = ("G", "K")


def real_accuracy(emu, data_dir, n_sim, real_n, broaden_R, normalizer):
    """EN: Train the RF on synthetic spectra (optionally broadened, with the given
        normalizer) and return its REAL accuracy on the labelled DESI spectra loaded
        with the SAME normalizer. ES: Entrena el RF con sinteticos (opcionalmente
        ensanchados, con el normalizador dado) y devuelve su accuracy REAL sobre los
        espectros DESI etiquetados cargados con el MISMO normalizador."""
    df = P.build_balanced_dataset(emu, classes=CLASSES, n_per_class=n_sim,
                                  broaden_R=broaden_R, normalizer=normalizer)
    res = P.train_rf(df, classes=CLASSES)
    X_real, y_real, counts = P.load_labeled_desi_folder(
        data_dir, classes=CLASSES, n_per_class=real_n, balanced=True, seed=0,
        normalizer=normalizer)
    ev = P.evaluate_on_labeled(res, X_real, y_real)
    return ev["accuracy"], res["accuracy"], counts


def main():
    ap = argparse.ArgumentParser(description="Real accuracy across normalization/broadening stages.")
    ap.add_argument("--data", default="proyecto_desi/espectros_balanceados_desi",
                    help="folder with per-class labelled real spectra (G/, K/, ...)")
    ap.add_argument("--n", type=int, default=100, help="simulated spectra per class")
    ap.add_argument("--real-n", type=int, default=150, dest="real_n",
                    help="real spectra per class")
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"[ERROR] Labelled real spectra folder not found: '{args.data}'")
    os.makedirs("figures", exist_ok=True)

    print("Loading TransformerPayne weights...")
    emu = tp.TransformerPayne.download()

    # EN: the four development stages | ES: las cuatro etapas de desarrollo
    stages = [
        ("baseline\n(percentile)",     None,   None),                     # sharp + percentile
        ("+ R(lambda)\nbroadening",    "desi", None),                     # broadened + percentile
        ("+ improved\nnormalization",  None,   P.continuum_normalize_iter),  # sharp + iterative
        ("+ both",                     "desi", P.continuum_normalize_iter),  # broadened + iterative
    ]

    labels, real_accs = [], []
    for label, broaden_R, normalizer in stages:
        acc_real, acc_sim, counts = real_accuracy(
            emu, args.data, args.n, args.real_n, broaden_R, normalizer)
        norm_name = "iterative" if normalizer is not None else "percentile"
        broad_name = "R(lambda)" if broaden_R is not None else "sharp"
        print(f"[{label.replace(chr(10), ' ')}] norm={norm_name:10s} synth={broad_name:9s} "
              f"| sim acc={acc_sim:.3f} | REAL acc={acc_real:.3f} | real per class={counts}")
        labels.append(label)
        real_accs.append(acc_real)

    # EN: progression bar chart (shows the development) | ES: barras de progreso
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#9aa0a6", "#6b9bd1", "#2e6da4", "#1b4a72"]
    bars = ax.bar(labels, real_accs, color=colors)
    for b, v in zip(bars, real_accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.axhline(real_accs[0], color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Real DESI accuracy (G/K)")
    ax.set_title("Sim->real progress: broadening vs improved normalization")
    plt.tight_layout()
    out = "figures/normalization_progress.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("\nsaved", out)
    print("Baseline real acc = %.3f -> best stage = %.3f (delta = %+.3f)"
          % (real_accs[0], max(real_accs), max(real_accs) - real_accs[0]))


if __name__ == "__main__":
    main()
