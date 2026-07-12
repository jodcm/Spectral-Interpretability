"""
11_compare_real_samples.py
==========================
EN: Diagnose WHY the sim-trained RF reaches ~0.95 on SDSS but only ~0.76 on DESI.
    Both real datasets carry teff/logg columns, so we can compare their stellar
    populations directly against the SYNTHETIC training distribution.

    Two hypotheses this tests:
    (1) DWARFS vs GIANTS. The synthetic spectra are dwarfs only (project_lib
        LOGG_RANGE = 4.0-5.0). If the DESI sample contains giants (logg < 3.5), those
        stars are physically outside the training distribution -> guaranteed errors,
        and a large part of the "domain shift" would simply be a SELECTION problem.
    (2) BOUNDARY STARS. G/K is a Teff cut (~5250 K). If the DESI sample has many stars
        piled up near that boundary and SDSS does not, the DESI task is intrinsically
        harder and the accuracy difference is partly a selection artifact.

    Prints the statistics and saves a figure with the Teff and log g distributions of
    both real samples, with the synthetic training ranges shaded.
ES: Diagnostica POR QUE el RF entrenado en sinteticos llega a ~0.95 en SDSS pero solo
    ~0.76 en DESI. Ambos datasets reales traen columnas teff/logg, asi que podemos
    comparar sus poblaciones estelares contra la distribucion SINTETICA de entrenamiento.

    Dos hipotesis:
    (1) ENANAS vs GIGANTES. Los sinteticos son solo enanas (LOGG_RANGE = 4.0-5.0). Si la
        muestra DESI trae gigantes (logg < 3.5), esas estrellas estan fuera de la
        distribucion de entrenamiento -> errores garantizados, y gran parte del "domain
        shift" seria en realidad un problema de SELECCION.
    (2) ESTRELLAS EN EL BORDE. G/K es un corte en Teff (~5250 K). Si DESI tiene muchas
        estrellas cerca del borde y SDSS no, la tarea de DESI es intrinsecamente mas
        dificil y la diferencia es en parte un artefacto de seleccion.

Run / Uso:
    python 11_compare_real_samples.py --desi proyecto_desi/espectros_balanceados_desi \
                                      --sdss proyecto_desi/espectros_sdss

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

# EN: dwarf/giant boundary | ES: frontera enana/gigante
LOGG_DWARF_CUT = 3.5


def collect(base_dir, classes=("G", "K")):
    """EN: read teff/logg from the per-class CSV headers (first row of each file).
    ES: lee teff/logg de los CSV por clase (primera fila de cada archivo)."""
    rows = []
    for c in classes:
        for f in sorted(glob.glob(os.path.join(base_dir, str(c), "*.csv"))):
            try:
                head = pd.read_csv(f, nrows=1)
            except Exception:
                continue
            low = {k.lower(): k for k in head.columns}
            if "teff" not in low:
                continue
            teff = float(head[low["teff"]].iloc[0])
            logg = float(head[low["logg"]].iloc[0]) if "logg" in low else np.nan
            rows.append({"clase": c, "teff": teff, "logg": logg})
    return pd.DataFrame(rows)


def report(name, df):
    if len(df) == 0:
        print(f"[{name}] no spectra with teff column found.")
        return
    print(f"\n=== {name} (N = {len(df)}) ===")
    for c, sub in df.groupby("clase"):
        t = sub["teff"].dropna()
        g = sub["logg"].dropna()
        print(f"  [{c}] N={len(sub):4d} | Teff median={t.median():6.0f} "
              f"({t.quantile(0.05):.0f}-{t.quantile(0.95):.0f})", end="")
        if len(g):
            n_giant = int((g < LOGG_DWARF_CUT).sum())
            print(f" | logg median={g.median():.2f} | GIANTS (logg<{LOGG_DWARF_CUT}): "
                  f"{n_giant}/{len(g)} = {100*n_giant/len(g):.0f}%")
        else:
            print(" | no logg column")
    # EN: stars near the G/K boundary (hard cases) | ES: estrellas cerca del borde
    t = df["teff"].dropna()
    near = int((np.abs(t - P.TEFF_CUT_KG) < 250).sum())
    print(f"  stars within +-250 K of the G/K boundary ({P.TEFF_CUT_KG:.0f} K): "
          f"{near}/{len(t)} = {100*near/max(len(t),1):.0f}%")


def main():
    ap = argparse.ArgumentParser(description="Compare the real DESI vs SDSS stellar samples")
    ap.add_argument("--desi", default="proyecto_desi/espectros_balanceados_desi")
    ap.add_argument("--sdss", default="proyecto_desi/espectros_sdss")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    d_desi = collect(args.desi)
    d_sdss = collect(args.sdss)
    report("DESI (real acc ~0.76)", d_desi)
    report("SDSS (real acc ~0.95)", d_sdss)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Teff ---
    ax = axes[0]
    for df, name, col in [(d_desi, "DESI", "#2e6da4"), (d_sdss, "SDSS", "crimson")]:
        if len(df):
            ax.hist(df["teff"].dropna(), bins=30, alpha=0.55, label=f"{name} (real)", color=col)
    ax.axvline(P.TEFF_CUT_KG, color="black", ls="--", lw=1.2)
    ax.text(P.TEFF_CUT_KG, ax.get_ylim()[1] * 0.95, " G/K cut", fontsize=8, va="top")
    # EN: shade the synthetic training ranges | ES: sombrear los rangos sinteticos
    for c, (lo, hi) in P.TEFF_RANGES.items():
        if c in ("G", "K"):
            ax.axvspan(lo, hi, color="grey", alpha=0.12)
    ax.set_xlabel("Teff [K]"); ax.set_ylabel("count")
    ax.set_title("Teff: real samples vs synthetic ranges (shaded)")
    ax.legend(fontsize=8)

    # --- logg ---
    ax = axes[1]
    for df, name, col in [(d_desi, "DESI", "#2e6da4"), (d_sdss, "SDSS", "crimson")]:
        g = df["logg"].dropna() if len(df) else pd.Series(dtype=float)
        if len(g):
            ax.hist(g, bins=30, alpha=0.55, label=f"{name} (real)", color=col)
    ax.axvspan(P.LOGG_RANGE[0], P.LOGG_RANGE[1], color="green", alpha=0.15,
               label="synthetic training range (dwarfs)")
    ax.axvline(LOGG_DWARF_CUT, color="red", ls="--", lw=1.2)
    ax.text(LOGG_DWARF_CUT, ax.get_ylim()[1] * 0.95, " giants <-", fontsize=8, va="top",
            ha="right", color="red")
    ax.set_xlabel("log g"); ax.set_ylabel("count")
    ax.set_title("log g: are there GIANTS outside the training range?")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = "figures/real_sample_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")
    print("\nIf DESI contains many giants (logg < 3.5) and SDSS does not, a large part of")
    print("the DESI 'domain shift' is a SELECTION problem, not a physics limit.")


if __name__ == "__main__":
    main()
