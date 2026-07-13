"""
13_spectra_compare.py
=====================
EN: Stop guessing -- LOOK. Three hypotheses for the DESI gap (giants, Teff domain,
    radial velocity) were all tested and all failed. What we know for sure:
      * SDSS reaches 0.95 and errs ONLY at the G/K boundary (= the model is fine).
      * DESI reaches 0.76 and misclassifies G stars across the WHOLE range, even the
        hottest ones, always in the same direction: G -> K.
      * Anything that WEAKENS the Balmer lines makes a G look like a K (G is defined by
        strong Balmer; K has weak Balmer anyway and is immune).

    So: compare the MEAN real spectrum of DESI vs SDSS in the SAME narrow Teff window.
    Same stars, physically speaking -- any difference is instrumental. This measures the
    Balmer line depths and the continuum shape directly, with no model in between.
ES: Basta de suponer -- MIRAR. Tres hipotesis para la brecha de DESI (gigantes, dominio
    de Teff, velocidad radial) fueron probadas y las tres fallaron. Lo que sabemos:
      * SDSS llega a 0.95 y solo se equivoca en el borde G/K (= el modelo esta bien).
      * DESI llega a 0.76 y confunde G en TODO el rango, siempre en la misma direccion.
      * Todo lo que DEBILITA las lineas de Balmer hace que una G parezca K.

    Entonces: comparar el espectro real MEDIO de DESI vs SDSS en la MISMA ventana de Teff.
    Fisicamente son las mismas estrellas -- cualquier diferencia es instrumental.

Run / Uso:
    python 13_spectra_compare.py --a proyecto_desi/espectros_balanceados_desi \
                                 --b proyecto_desi/espectros_sdss \
                                 --class G --teff 5500 5900

Requires / Requiere: numpy, pandas, matplotlib (NO TransformerPayne).
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


def mean_spectrum(base_dir, cls, teff_lo, teff_hi, normalizer, nmax=250):
    """EN: mean normalized spectrum of one class inside a Teff window.
    ES: espectro normalizado medio de una clase dentro de una ventana de Teff."""
    rows, teffs = [], []
    for f in sorted(glob.glob(os.path.join(base_dir, str(cls), "*.csv"))):
        if len(rows) >= nmax:
            break
        try:
            head = pd.read_csv(f, nrows=1)
        except Exception:
            continue
        low = {k.lower(): k for k in head.columns}
        if "teff" not in low:
            continue
        t = float(head[low["teff"]].iloc[0])
        if not (teff_lo <= t <= teff_hi):
            continue
        try:
            x = P.load_desi_csv(f, normalizer=normalizer)
        except Exception:
            continue
        if x is None or not np.all(np.isfinite(x)):
            continue
        rows.append(x)
        teffs.append(t)
    if not rows:
        return None, 0, np.nan
    return np.vstack(rows).mean(axis=0), len(rows), float(np.mean(teffs))


def line_depth(spec, lam, half=6.0):
    """EN: depth of a line = 1 - min(flux) in a small window around lam.
    ES: profundidad de linea = 1 - min(flujo) en una ventana alrededor de lam."""
    m = (P.WAVE_GRID > lam - half) & (P.WAVE_GRID < lam + half)
    return float(1.0 - np.nanmin(spec[m]))


def main():
    ap = argparse.ArgumentParser(description="Compare the mean real spectra of two surveys")
    ap.add_argument("--a", default="proyecto_desi/espectros_balanceados_desi", help="survey A")
    ap.add_argument("--b", default="proyecto_desi/espectros_sdss", help="survey B")
    ap.add_argument("--class", dest="cls", default="G", help="spectral class")
    ap.add_argument("--teff", type=float, nargs=2, default=[5500, 5900],
                    metavar=("TMIN", "TMAX"), help="Teff window (same stars in both)")
    ap.add_argument("--norm", default="masked", choices=["masked", "iterative", "percentile"])
    ap.add_argument("--nmax", type=int, default=250)
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    normalizer = P.get_normalizer(args.norm)
    na = os.path.basename(os.path.normpath(args.a)).replace("espectros_", "")
    nb = os.path.basename(os.path.normpath(args.b)).replace("espectros_", "")

    sa, ca, ta = mean_spectrum(args.a, args.cls, *args.teff, normalizer=normalizer, nmax=args.nmax)
    sb, cb, tb = mean_spectrum(args.b, args.cls, *args.teff, normalizer=normalizer, nmax=args.nmax)
    if sa is None or sb is None:
        raise SystemExit("[ERROR] not enough spectra in that Teff window for one of the surveys.")
    print(f"[{na}] N={ca}  <Teff>={ta:.0f} K")
    print(f"[{nb}] N={cb}  <Teff>={tb:.0f} K")

    # EN: measure the Balmer depths -- the physical driver of the G/K decision
    # ES: medir las profundidades de Balmer -- lo que decide G/K (segun SHAP)
    print(f"\nLine depths (1 - min flux) for class {args.cls}, Teff {args.teff[0]:.0f}-{args.teff[1]:.0f} K:")
    print(f"{'line':<12}{na:>12}{nb:>12}   ratio A/B")
    lines = {k: v[0] for k, v in P.SPECTRAL_LINES.items() if P.WMIN <= v[0] <= P.WMAX}
    for name, lam in sorted(lines.items(), key=lambda kv: kv[1]):
        da, db = line_depth(sa, lam), line_depth(sb, lam)
        print(f"{name:<12}{da:>12.3f}{db:>12.3f}   {da/max(db,1e-9):>8.2f}")
    print("\nratio < 1  => the lines are SHALLOWER in %s -> its G stars look cooler -> K" % na)

    # EN: continuum slope (flux calibration check) | ES: pendiente del continuo
    def slope(s):
        return float(np.polyfit(P.WAVE_GRID, s, 1)[0] * 1000)  # per 1000 A
    print(f"\ncontinuum slope per 1000 A:  {na}={slope(sa):+.3f}   {nb}={slope(sb):+.3f}")

    # ------------------------------------------------------------------ plot
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(P.WAVE_GRID, sa, color="#2e6da4", lw=1.2, label=f"{na} (N={ca}, <Teff>={ta:.0f} K)")
    ax.plot(P.WAVE_GRID, sb, color="crimson", lw=1.2, alpha=0.85,
            label=f"{nb} (N={cb}, <Teff>={tb:.0f} K)")
    for name, lam in lines.items():
        ax.axvline(lam, color="grey", ls="--", lw=0.7, alpha=0.6)
    ax.set_xlim(P.WMIN, P.WMAX)
    ax.set_ylabel("Norm. intensity")
    ax.set_title(f"Mean REAL spectrum, class {args.cls}, same Teff window "
                 f"({args.teff[0]:.0f}-{args.teff[1]:.0f} K) - any difference is INSTRUMENTAL")
    ax.legend(loc="lower left", fontsize=9)

    # EN: zooms on the three Balmer lines that SHAP said drive the decision
    for i, (name, lam) in enumerate([("H-delta", 4102.9), ("H-gamma", 4341.7), ("H-beta", 4862.7)]):
        axz = fig.add_subplot(gs[1, i])
        axz.plot(P.WAVE_GRID, sa, color="#2e6da4", lw=1.6, label=na)
        axz.plot(P.WAVE_GRID, sb, color="crimson", lw=1.6, alpha=0.85, label=nb)
        axz.axvline(lam, color="grey", ls="--", lw=0.8)
        axz.set_xlim(lam - 25, lam + 25)
        axz.set_xlabel(r"$\lambda$ [$\AA$]")
        axz.set_title(f"{name} {lam:.0f} A", fontsize=10)
        if i == 0:
            axz.set_ylabel("Norm. intensity")
            axz.legend(fontsize=8)

    plt.tight_layout()
    out = f"figures/spectra_compare_{na}_vs_{nb}_{args.cls}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
