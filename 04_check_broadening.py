"""
04_check_broadening.py
======================
EN: Visual sanity check for the LSF broadening. Plots, for one spectral class:
    (1) the mean SHARP synthetic spectrum (high resolution, TransformerPayne),
    (2) the mean synthetic spectrum BROADENED to the target resolution, and
    (3) the mean REAL spectrum. If the broadening is correct, the broadened
    synthetic lines match the width/depth of the real lines (curve 2 ~ curve 3),
    while the sharp synthetic (curve 1) has much deeper/narrower lines. Confirms
    that the Gaussian convolution is applied to the SYNTHETIC spectra and moves
    them TOWARD the real resolution.
ES: Chequeo visual del ensanchamiento LSF. Grafica, para una clase espectral:
    (1) el espectro sintetico medio AGUDO (alta resolucion, TransformerPayne),
    (2) el sintetico medio ENSANCHADO a la resolucion objetivo, y (3) el espectro
    REAL medio. Si el ensanchamiento es correcto, las lineas sinteticas ensanchadas
    coinciden en ancho/profundidad con las reales (curva 2 ~ curva 3), mientras que
    el sintetico agudo (curva 1) tiene lineas mucho mas profundas/angostas. Confirma
    que la convolucion gaussiana se aplica a los SINTETICOS y los acerca a la real.

Run / Uso:
    python 04_check_broadening.py --data proyecto_desi/espectros_balanceados_desi --class G --resolution desi
    python 04_check_broadening.py --data proyecto_desi/espectros_lamost         --class K --resolution 1800

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


def parse_resolution(value):
    """EN: 'desi' -> R(lambda); a number -> constant R. ES: 'desi' -> R(lambda); numero -> R constante."""
    if str(value).lower() == "desi":
        return "desi", "DESI R(lambda)"
    r = float(value)
    return r, f"R={r:.0f}"


def main():
    ap = argparse.ArgumentParser(description="Visual check: sharp vs broadened synthetic vs real.")
    ap.add_argument("--data", default="proyecto_desi/espectros_balanceados_desi",
                    help="folder with per-class labelled real spectra")
    ap.add_argument("--class", dest="cls", default="G", choices=["O", "B", "A", "F", "G", "K", "M"],
                    help="spectral class to check")
    ap.add_argument("--resolution", default="desi",
                    help="'desi' (R(lambda)) or a number, e.g. 1800 for LAMOST")
    ap.add_argument("--nsim", type=int, default=40, help="synthetic spectra to average")
    ap.add_argument("--nreal", type=int, default=100, help="real spectra to average")
    ap.add_argument("--zoom", type=float, nargs=2, default=[4830.0, 4895.0],
                    help="wavelength window to zoom (default around H-beta 4862 A)")
    args = ap.parse_args()

    broaden_R, res_label = parse_resolution(args.resolution)
    os.makedirs("figures", exist_ok=True)

    print("Loading TransformerPayne weights...")
    emu = tp.TransformerPayne.download()

    # EN: same stars, sharp vs broadened (same base_seed -> matched parameters)
    # ES: mismas estrellas, agudo vs ensanchado (mismo base_seed -> parametros iguales)
    df_sharp = P.build_balanced_dataset(emu, classes=(args.cls,), n_per_class=args.nsim, base_seed=100)
    df_broad = P.build_balanced_dataset(emu, classes=(args.cls,), n_per_class=args.nsim, base_seed=100,
                                        broaden_R=broaden_R)
    mean_sharp = np.vstack(df_sharp["normalized_intensity"]).mean(0)
    mean_broad = np.vstack(df_broad["normalized_intensity"]).mean(0)

    # EN: mean real spectrum of the same class | ES: espectro real medio de la misma clase
    X_real, y_real, counts = P.load_labeled_desi_folder(
        args.data, classes=(args.cls,), n_per_class=args.nreal, balanced=False, seed=0)
    if len(y_real) == 0:
        raise SystemExit(f"[ERROR] No real spectra for class {args.cls} in '{args.data}'.")
    mean_real = np.nanmean(X_real, axis=0)
    print(f"averaged {args.nsim} synthetic and {counts.get(args.cls, 0)} real '{args.cls}' spectra")

    wg = P.WAVE_GRID

    def draw(ax):
        ax.plot(wg, mean_sharp, color="#c0c0c0", lw=1.0, label="synthetic (sharp)")
        ax.plot(wg, mean_broad, color="#2e6da4", lw=1.3, label=f"synthetic broadened ({res_label})")
        ax.plot(wg, mean_real, color="crimson", lw=1.3, alpha=0.85, label="real (mean)")
        for name, (lam, _) in P.SPECTRAL_LINES.items():
            if P.WMIN <= lam <= P.WMAX:
                ax.axvline(lam, color="grey", ls="--", lw=0.6, alpha=0.5)
        ax.set_ylabel("Norm. intensity")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    draw(ax1)
    ax1.set_title(f"Broadening check - class {args.cls} ({res_label}): "
                  f"broadened synthetic should match the real lines")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_xlim(P.WMIN, P.WMAX)

    draw(ax2)
    ax2.set_xlim(args.zoom[0], args.zoom[1])
    ax2.set_xlabel(r"Wavelength [$\AA$]")
    ax2.set_title(f"Zoom {args.zoom[0]:.0f}-{args.zoom[1]:.0f} A (line width/depth comparison)")

    plt.tight_layout()
    out = f"figures/broadening_check_{args.cls}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
