"""
16_attention.py
===============
EN: Interpretability of the EMULATOR -- the other half of "Spectral Interpretability".
    So far 08_shap.py explained the CLASSIFIER (which wavelengths the Random Forest uses).
    This script asks what TRANSFORMERPAYNE itself has learned: which spectral lines does
    it associate with which chemical element?

    The TransformerPayne paper motivates its attention mechanism exactly this way -- the
    model should learn element<->line associations. But raw attention weights are famously
    hard to read. The functionally equivalent and DIRECTLY interpretable quantity is the
    SENSITIVITY of the emitted spectrum to each label:

        S_X(lambda) = d flux(lambda) / d [X/H]

    computed by central finite differences (perturb one abundance, re-emit the spectrum).
    S_X(lambda) answers the question exactly: "which wavelengths does element X control?"

    THE FALSIFIABLE TEST. In 4000-5000 A some elements have strong lines (Ca I 4226.7,
    the CH G-band at 4300-4315 driven by C, many Fe lines, Ti, Mn) while others have
    essentially NONE (Na, S, N, K -- their strong lines lie outside this window). If
    TransformerPayne has learned real physics, the elements WITHOUT lines here must show
    ~ZERO sensitivity, and the others must peak ON their known lines. If instead every
    element shows diffuse sensitivity everywhere, the emulator has learned correlations,
    not physics. This is a real test that can fail.

    The script also TRIES to pull the true attention maps out of the model; if the package
    does not expose them it says so and falls back to the sensitivity analysis, which is
    the scientifically meaningful result either way.
ES: Interpretabilidad del EMULADOR -- la otra mitad de "Spectral Interpretability".
    08_shap.py explico al CLASIFICADOR (que longitudes de onda usa el Random Forest). Este
    script pregunta que aprendio TRANSFORMERPAYNE: que lineas asocia a que elemento?

    El paper de TransformerPayne motiva su atencion justamente asi. Pero los pesos de
    atencion son dificiles de leer. La cantidad equivalente y DIRECTAMENTE interpretable es
    la SENSIBILIDAD del espectro emitido a cada etiqueta:

        S_X(lambda) = d flujo(lambda) / d [X/H]

    calculada por diferencias finitas centrales. Responde exactamente: "que longitudes de
    onda controla el elemento X?"

    LA PRUEBA FALSABLE. En 4000-5000 A algunos elementos tienen lineas fuertes (Ca I 4226.7,
    la banda CH 4300-4315 por el C, muchas de Fe, Ti, Mn) y otros NO tienen practicamente
    ninguna (Na, S, N, K -- sus lineas fuertes caen fuera de esta ventana). Si
    TransformerPayne aprendio fisica real, los elementos SIN lineas aca deben dar sensibilidad
    ~CERO, y los demas deben tener su maximo SOBRE sus lineas conocidas. Si en cambio todos
    los elementos muestran sensibilidad difusa por todas partes, el emulador aprendio
    correlaciones, no fisica. Es una prueba que puede fallar.

Run / Uso:
    python 16_attention.py --class G
    python 16_attention.py --class K --delta 0.3

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

# --------------------------------------------------------------------------
# EN: Reference line list for 4000-5000 A -- the GROUND TRUTH we test the emulator
#     against. "expected: none" = this element has no strong lines in this window, so a
#     physically correct emulator must show ~zero sensitivity to it here.
# ES: Lista de lineas de referencia para 4000-5000 A -- la VERDAD contra la que probamos
#     al emulador. "expected: none" = sin lineas fuertes en esta ventana.
# --------------------------------------------------------------------------
REFERENCE_LINES = {
    "Fe": [4045.8, 4063.6, 4071.7, 4383.5, 4404.8, 4415.1, 4461.7, 4482.2, 4919.0, 4957.6],
    "Ca": [4226.7, 4425.4, 4435.7, 4454.8],
    "C":  [4300.0, 4305.0, 4313.0],          # CH G-band / banda CH
    "Mg": [4571.1, 4702.9],
    "Ti": [4533.2, 4534.8, 4981.7, 4991.1],
    "Mn": [4041.4, 4823.5, 4030.8],
    "Na": [],                                 # EN/ES: Na D is at 5890 A -> OUTSIDE
    "S":  [],                                 # EN/ES: no strong optical S lines here
    "N":  [],                                 # EN/ES: no atomic N lines in cool-star optical
    "K":  [],                                 # EN/ES: K I resonance at 7665 A -> OUTSIDE
}


def main():
    ap = argparse.ArgumentParser(description="What did TransformerPayne learn? (emulator interpretability)")
    ap.add_argument("--class", dest="cls", default="G", choices=["F", "G", "K"],
                    help="spectral type of the reference star")
    ap.add_argument("--delta", type=float, default=0.25,
                    help="abundance step for the finite difference, in dex")
    ap.add_argument("--tol", type=float, default=4.0,
                    help="A tolerance when matching a sensitivity peak to a known line")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    wave = P.WAVE_GRID
    log_wave = np.log10(wave)
    mu = 1.0

    print("Loading TransformerPayne weights...")
    emu = tp.TransformerPayne.download()

    # EN: reference star in the middle of its class | ES: estrella de referencia
    t_min, t_max = P.TEFF_RANGES[args.cls]
    teff = 0.5 * (t_min + t_max)
    base = {"logteff": np.log10(teff), "logg": 4.5}
    base.update({el: 0.0 for el in P.VARIED_ELEMENTS})
    print(f"reference star: {args.cls}, Teff={teff:.0f} K, logg=4.5, [X/H]=0 for all elements")

    def emit(params):
        return np.asarray(emu(log_wave, mu, emu.to_parameters(params))[:, 0], dtype=float)

    flux0 = emit(base)

    # ----------------------------------------------------------------------
    # EN: 1) Sensitivity S_X(lambda) = d flux / d [X/H] by central differences
    # ES: 1) Sensibilidad por diferencias finitas centrales
    # ----------------------------------------------------------------------
    print(f"\ncomputing d(flux)/d[X/H] for {len(P.VARIED_ELEMENTS)} elements "
          f"(delta = {args.delta} dex, {2*len(P.VARIED_ELEMENTS)} emulator calls)...")
    S = {}
    for el in P.VARIED_ELEMENTS:
        hi = dict(base); hi[el] = base[el] + args.delta
        lo = dict(base); lo[el] = base[el] - args.delta
        S[el] = (emit(hi) - emit(lo)) / (2.0 * args.delta)

    # EN: normalize by the continuum level so the units are "relative depth per dex"
    # ES: normalizar por el continuo -> "profundidad relativa por dex"
    cont = np.nanmedian(flux0)
    Smat = np.vstack([np.abs(S[el]) / cont for el in P.VARIED_ELEMENTS])

    # ----------------------------------------------------------------------
    # EN: 2) THE TEST -- does each element peak on its own known lines?
    # ES: 2) LA PRUEBA -- cada elemento tiene su maximo sobre sus propias lineas?
    # ----------------------------------------------------------------------
    total = Smat.sum(axis=1)
    order = np.argsort(total)[::-1]
    print("\n" + "=" * 78)
    print(f"{'element':<8}{'total sens.':>13}{'peak at':>10}{'  known line?':<22}{'verdict'}")
    print("-" * 78)
    strongest = total.max() + 1e-12
    verdicts = {}
    for i in order:
        el = P.VARIED_ELEMENTS[i]
        rel = total[i] / strongest
        peak_lam = float(wave[np.argmax(Smat[i])])
        refs = REFERENCE_LINES.get(el, [])
        near = [l for l in refs if abs(l - peak_lam) <= args.tol]
        if not refs:
            # EN: element has NO lines here -> we EXPECT near-zero sensitivity
            ok = rel < 0.10
            verdict = "OK (no lines here, ~0 sens.)" if ok else "!! sensitive but has NO lines"
            note = "expected: none"
        else:
            ok = bool(near)
            verdict = "OK (peaks on its line)" if ok else "peak not on a known line"
            note = f"{near[0]:.1f} A" if near else "-"
        verdicts[el] = ok
        print(f"{el:<8}{rel:>13.3f}{peak_lam:>10.1f}   {note:<19}{verdict}")
    print("=" * 78)
    n_ok = sum(verdicts.values())
    print(f"\n{n_ok}/{len(verdicts)} elements behave as physics demands.")
    print("Elements with NO lines in 4000-5000 A (Na, S, N, K) MUST show ~zero sensitivity;")
    print("if they do, TransformerPayne learned physics, not correlations.")

    # ----------------------------------------------------------------------
    # EN: 3) try to read the REAL attention maps (may not be exposed by the package)
    # ES: 3) intentar leer los mapas de atencion REALES
    # ----------------------------------------------------------------------
    print("\n--- attention maps ---")
    try:
        found = [a for a in dir(emu) if "atten" in a.lower()]
        if found:
            print("model exposes:", found)
            print("(inspect these to plot raw attention; the sensitivity analysis above is")
            print(" the interpretable equivalent and is what we report.)")
        else:
            print("The transformer_payne package does not expose attention weights through its")
            print("public API. The sensitivity analysis d(flux)/d[X/H] above measures the same")
            print("thing (which line each element controls) directly and exactly -- and unlike")
            print("attention weights it cannot be misread.")
    except Exception as e:
        print("attention introspection failed:", str(e)[:80])

    # ----------------------------------------------------------------------
    # EN: 4) figures | ES: 4) figuras
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.0, 1.1], hspace=0.35)

    # --- the reference spectrum
    ax = fig.add_subplot(gs[0])
    ax.plot(wave, flux0 / cont, color="black", lw=0.9)
    for nm, (lam, _) in P.SPECTRAL_LINES.items():
        if P.WMIN <= lam <= P.WMAX:
            ax.axvline(lam, color="grey", ls="--", lw=0.7, alpha=0.6)
    ax.set_xlim(P.WMIN, P.WMAX); ax.set_ylabel("norm. flux")
    ax.set_title(f"TransformerPayne interpretability - reference {args.cls} star "
                 f"(Teff={teff:.0f} K): which element controls which line?")

    # --- the heatmap: elements x wavelength
    ax = fig.add_subplot(gs[1])
    rows = [P.VARIED_ELEMENTS[i] for i in order]
    M = Smat[order]
    M = M / (M.max(axis=1, keepdims=True) + 1e-12)      # normalize each row
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                   extent=[P.WMIN, P.WMAX, -0.5, len(rows) - 0.5])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{el}  ({total[P.VARIED_ELEMENTS.index(el)]/strongest:.2f})"
                        for el in rows], fontsize=9)
    # EN: mark the known lines of each element on its own row
    for j, el in enumerate(rows):
        for lam in REFERENCE_LINES.get(el, []):
            if P.WMIN <= lam <= P.WMAX:
                ax.plot(lam, j, marker="|", color="cyan", ms=11, mew=1.6)
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("element  (relative total sensitivity)")
    ax.set_title(r"$|\partial\,\mathrm{flux}\,/\,\partial\,[X/H]|$   "
                 "(cyan ticks = known lines of that element)", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01, label="sensitivity (row-normalized)")

    # --- the elements with the strongest response, as curves
    ax = fig.add_subplot(gs[2])
    for el in rows[:3]:
        i = P.VARIED_ELEMENTS.index(el)
        ax.plot(wave, Smat[i] / (Smat[i].max() + 1e-12), lw=1.1, label=el)
    for el in ("Na", "S", "N", "K"):
        if el in P.VARIED_ELEMENTS:
            i = P.VARIED_ELEMENTS.index(el)
            ax.plot(wave, Smat[i] / (strongest / len(wave) * 50 + 1e-12), lw=0.8, alpha=0.5,
                    ls=":", label=f"{el} (no lines here)")
    ax.set_xlim(P.WMIN, P.WMAX)
    ax.set_xlabel(r"Wavelength [$\AA$]"); ax.set_ylabel("rel. sensitivity")
    ax.set_title("Top-3 responding elements vs the elements that have NO lines here", fontsize=10)
    ax.legend(fontsize=8, ncol=4)

    out = f"figures/emulator_sensitivity_{args.cls}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")

    # EN: also save the matrix so it can be compared with the SHAP importance
    # ES: guardar la matriz para compararla con la importancia SHAP
    np.savez_compressed(f"emulator_sensitivity_{args.cls}.npz",
                        wave=wave, elements=np.array(P.VARIED_ELEMENTS), S=Smat)
    print(f"saved emulator_sensitivity_{args.cls}.npz "
          "(compare with SHAP: do the lines the RF uses carry real elemental information?)")


if __name__ == "__main__":
    main()
