"""
03_broadening.py
================
EN: Step 3 (domain-shift mitigation). Train two Random Forests on G/K spectra:
    one on the sharp simulated spectra ("before") and one on spectra broadened to
    a target instrument resolution ("after"). Then compare the REAL accuracy of
    both on the labelled real spectra, to see whether broadening closes the
    sim -> real gap. The resolution is a command-line argument, so you can switch
    between DESI ("desi", wavelength-dependent R(lambda)) and LAMOST (e.g. 1800)
    without editing the code.
ES: Paso 3 (mitigacion del domain shift). Entrena dos Random Forests con espectros
    G/K: uno con los espectros simulados agudos ("before") y otro con espectros
    ensanchados a una resolucion instrumental objetivo ("after"). Luego compara la
    ACCURACY REAL de ambos sobre los espectros reales etiquetados, para ver si el
    ensanchamiento cierra la brecha sim -> real. La resolucion es un argumento de
    linea de comandos, asi cambias entre DESI ("desi", R(lambda) dependiente de
    lambda) y LAMOST (p.ej. 1800) sin tocar el codigo.

Run / Uso:
    # DESI (wavelength-dependent R(lambda)):
    python 03_broadening.py --data proyecto_desi/espectros_balanceados_desi --resolution desi
    # LAMOST LRS (constant R ~ 1800):
    python 03_broadening.py --data proyecto_desi/espectros_lamost --resolution 1800

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


def parse_resolution(value):
    """EN: 'desi' -> wavelength-dependent R(lambda); a number -> constant R (e.g. LAMOST 1800).
    ES: 'desi' -> R(lambda) dependiente de lambda; un numero -> R constante (p.ej. LAMOST 1800)."""
    if str(value).lower() == "desi":
        return "desi", "DESI R(lambda)"
    r = float(value)
    return r, f"R={r:.0f}"


def main():
    ap = argparse.ArgumentParser(description="Compare real accuracy before/after LSF broadening.")
    ap.add_argument("--data", default="proyecto_desi/espectros_balanceados_desi",
                    help="folder with per-class labelled real spectra (G/, K/, ...)")
    ap.add_argument("--resolution", default="desi",
                    help="'desi' (R(lambda)) or a number for constant R, e.g. 1800 for LAMOST LRS")
    ap.add_argument("--n", type=int, default=100, help="simulated spectra per class")
    ap.add_argument("--real-n", type=int, default=150, dest="real_n",
                    help="real spectra per class")
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"[ERROR] Labelled real spectra folder not found: '{args.data}'")
    broaden_R, res_label = parse_resolution(args.resolution)

    os.makedirs("figures", exist_ok=True)
    print("Loading TransformerPayne weights...")
    emu = tp.TransformerPayne.download()

    # EN: "before" = sharp simulated spectra | ES: "before" = espectros simulados agudos
    df_before = P.build_balanced_dataset(emu, classes=CLASSES, n_per_class=args.n)
    res_before = P.train_rf(df_before, classes=CLASSES)

    # EN: "after" = spectra broadened to the chosen resolution
    # ES: "after" = espectros ensanchados a la resolucion elegida
    df_after = P.build_balanced_dataset(emu, classes=CLASSES, n_per_class=args.n, broaden_R=broaden_R)
    res_after = P.train_rf(df_after, classes=CLASSES)

    print("simulated test accuracy  before:", round(res_before["accuracy"], 3),
          "| after:", round(res_after["accuracy"], 3))

    # EN: load labelled real spectra o