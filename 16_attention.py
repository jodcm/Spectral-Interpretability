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
    # EN: strong, well-known lines of each element in 4000-5000 A (NIST / standard lists)
    # ES: lineas fuertes y conocidas de cada elemento en 4000-5000 A
    "Fe": [4045.81, 4063.59, 4071.74, 4132.06, 4143.87, 4202.03, 4260.47, 4271.15,
           4271.76, 4325.76, 4383.55, 4404.75, 4415.12, 4494.56, 4528.61, 4871.32,
           4890.75, 4918.99, 4920.50, 4957.60],
    "Ca": [4226.73, 4283.01, 4289.37, 4302.53, 4318.65, 4355.08, 4425.44, 4435.68,
           4454.78, 4455.89],
    "C":  [4290.0, 4300.0, 4305.0, 4310.0, 4315.0],      # CH G-band (molecular)
    "Mg": [4167.27, 4351.91, 4571.10, 4702.99],
    "Ti": [4443.80, 4468.49, 4501.27, 4533.24, 4534.78, 4548.76, 4681.91, 4981.73,
           4991.07, 4999.50],
    "Mn": [4030.75, 4033.06, 4034.48, 4041.36, 4055.54, 4451.58, 4754.04, 4783.42,
           4823.52],
    # EN: these four have NO strong lines in this window -- their strong lines lie OUTSIDE
    #     (Na D 5890 A, K I 7665 A). This is the falsifiable half of the test.
    # ES: estos cuatro NO tienen lineas fuertes aca -- las suyas caen FUERA.
    "Na": [],
    "S":  [],
    "N":  [],
    "K":  [],
}


def enrichment(S_row, wave, lines, half=2.0):
    """EN: the honest test. Is this element's sensitivity systematically HIGHER at its own
        lines than everywhere else?  enrichment = <S at its lines> / <S everywhere>.
        A single-peak match is worthless here: Fe has hundreds of lines, so which single
        pixel happens to be the global maximum is luck. The enrichment ratio is not.
    ES: la prueba honesta. Es la sensibilidad de este elemento sistematicamente MAS ALTA
        sobre sus propias lineas que en el resto?  enrichment = <S en sus lineas> / <S total>.
    -> ratio (1.0 = no preference, >1 = the element does control its own lines)
    """
    if not lines:
        return np.nan
    m = np.zeros_like(wave, dtype=bool)
    for lam in lines:
        m |= np.abs(wave - lam) <= half
    if m.sum() == 0 or (~m).sum() == 0:
        return np.nan
    return float(S_row[m].mean() / (S_row.mean() + 1e-12))


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
    strongest = total.max() + 1e-12

    # ---------------- TEST A: elements WITHOUT lines here must be silent ----------------
    print("\n" + "=" * 76)
    print("TEST A -- elements whose STRONG lines lie OUTSIDE 4000-5000 A must be SILENT")
    print("-" * 76)
    silent_ok = True
    for i in order:
        el = P.VARIED_ELEMENTS[i]
        if REFERENCE_LINES.get(el):
            continue
        rel = total[i] / strongest
        ok = rel < 0.10
        silent_ok &= ok
        print(f"  {el:<4} relative sensitivity = {rel:5.3f}   "
              f"{'OK (silent, as physics demands)' if ok else '!! LOUD but has no lines'}")
    print(f"\n  -> TEST A {'PASSED' if silent_ok else 'FAILED'}")

    # ---------------- TEST B: does each element control ITS OWN lines? ----------------
    # EN: NOT a single-peak match -- Fe has hundreds of lines, so the global maximum pixel
    #     is luck. We ask instead whether the sensitivity is systematically ENRICHED on the
    #     element's own lines relative to its own average.
    print("\n" + "=" * 76)
    print("TEST B -- is each element's sensitivity ENRICHED on its own lines?")
    print("          enrichment = <S at its own lines> / <S everywhere>   (1.0 = no preference)")
    print("-" * 76)
    enr = {}
    for i in order:
        el = P.VARIED_ELEMENTS[i]
        lines = REFERENCE_LINES.get(el, [])
        if not lines:
            continue
        e = enrichment(Smat[i], wave, lines)
        enr[el] = e
        verdict = ("OK (controls its own lines)" if e > 1.5 else
                   "weak" if e > 1.15 else "!! no preference for its own lines")
        print(f"  {el:<4} rel.sens={total[i]/strongest:5.3f}   enrichment = {e:5.2f}x   {verdict}")

    # ---------------- TEST C: the cross-matrix -- the DECISIVE one ----------------
    # EN: at the lines of element X, which element responds most strongly? If the emulator
    #     learned physics, the DIAGONAL must dominate: Ca's lines must be controlled by Ca,
    #     not by Fe. This is the test that a correlation-learner cannot pass.
    # ES: en las lineas del elemento X, que elemento responde mas fuerte? La DIAGONAL debe
    #     dominar. Es la prueba que un modelo que aprendio correlaciones no puede pasar.
    print("\n" + "=" * 76)
    print("TEST C -- CROSS-MATRIX: at the lines of element X (columns), which element")
    print("          responds (rows)?  The DIAGONAL must dominate.")
    print("-" * 76)
    with_lines = [el for el in P.VARIED_ELEMENTS if REFERENCE_LINES.get(el)]
    Cx = np.zeros((len(with_lines), len(with_lines)))
    for a, el_resp in enumerate(with_lines):          # who responds
        i = P.VARIED_ELEMENTS.index(el_resp)
        for b, el_line in enumerate(with_lines):      # at whose lines
            m = np.zeros_like(wave, dtype=bool)
            for lam in REFERENCE_LINES[el_line]:
                m |= np.abs(wave - lam) <= 2.0
            Cx[a, b] = Smat[i][m].mean() if m.any() else np.nan
    Cxn = Cx / (Cx.max(axis=0, keepdims=True) + 1e-12)   # normalize per column
    hdr = "".join(f"{el:>7}" for el in with_lines)
    print(f"{'responds':<10}{'| at the lines of ->':<0}")
    print(f"{'':<10}{hdr}")
    n_diag = 0
    for a, el in enumerate(with_lines):
        row = "".join(f"{Cxn[a, b]:>7.2f}" for b in range(len(with_lines)))
        winner = with_lines[int(np.argmax(Cxn[:, a]))] if True else ""
        print(f"{el:<10}{row}")
    for b, el in enumerate(with_lines):
        if with_lines[int(np.argmax(Cxn[:, b]))] == el:
            n_diag += 1
    print("-" * 76)
    print(f"  -> the strongest responder is the CORRECT element for "
          f"{n_diag}/{len(with_lines)} line groups")
    print("     (Fe responds everywhere because it has hundreds of blended lines --")
    print("      the meaningful question is whether Ca beats Fe ON CALCIUM's lines.)")
    print("=" * 76)

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

    # --- EN: the SAME information as the heatmap, but as curves on ONE COMMON SCALE.
    #     Every curve is divided by the SAME number (the global maximum), so the heights
    #     are directly comparable. That is the whole point: the elements WITHOUT lines in
    #     this window must lie flat on zero next to the ones that do have lines.
    #     ES: la MISMA informacion que el heatmap, pero como curvas en UNA ESCALA COMUN.
    ax = fig.add_subplot(gs[2])
    gmax = Smat.max() + 1e-12                      # EN/ES: one common divisor
    colors = {"Fe": "#1f77b4", "C": "#ff7f0e", "Mg": "#2ca02c",
              "Ca": "#9467bd", "Ti": "#8c564b", "Mn": "#e377c2"}
    for el in rows[:3]:
        i = P.VARIED_ELEMENTS.index(el)
        ax.plot(wave, Smat[i] / gmax, lw=1.0, color=colors.get(el),
                label=f"{el} (has lines here)")
    for el in ("Na", "S", "N", "K"):
        if el in P.VARIED_ELEMENTS:
            i = P.VARIED_ELEMENTS.index(el)
            ax.plot(wave, Smat[i] / gmax, lw=1.4, ls="-", alpha=0.95,
                    color="crimson" if el == "Na" else "grey",
                    label=f"{el} (NO lines here)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(P.WMIN, P.WMAX)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("sensitivity\n(common scale)")
    ax.set_title("Same scale for all: elements WITH lines here (colour) vs elements "
                 "WITHOUT lines here (red/grey, flat on zero)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper right")

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
