"""
17_attention_maps.py
====================
EN: Reproduce Figure 9 of the TransformerPayne paper -- the REAL attention maps.

    The transformer_payne package does not expose attention weights. But the model is a
    Flax module whose MHA block computes `attn_weights = softmax(...)` internally, so we
    subclass MHA with mathematically identical code that additionally `sow()`s the
    attention, and swap the class in the package namespace. The subclass keeps the name
    "MHA", so the parameter-tree paths are unchanged and the pretrained weights still load.

    WHAT THE ATTENTION ACTUALLY IS. TransformerPayne is a CROSS-attention model:
      * the QUERY is the (frequency-encoded) WAVELENGTH -- a single token
      * the KEYS/VALUES are the STELLAR PARAMETERS, embedded into `no_tokens` tokens
    So one attention map answers: *at this wavelength, which parameter-token does this head
    look at?* That is exactly the paper's claim: a head should attend to the token carrying
    Ca when it emits the Ca line.

    THE FIGURE (like paper Fig. 9):
      * top    : token <-> parameter map -- which token carries which stellar parameter
                 (computed by perturbing each parameter and watching which token moves)
      * bottom : for the most specialized heads, attention over tokens (y) at each
                 spectral line (x)
    Combining them gives the element<->line association: line L -> head attends to token T
    -> token T carries element E.

ES: Reproduce la Figura 9 del paper de TransformerPayne -- los mapas de atencion REALES.

    El paquete no expone los pesos de atencion. Pero el modelo es un modulo Flax cuyo bloque
    MHA calcula `attn_weights = softmax(...)` internamente, asi que creamos una subclase de
    MHA con matematica identica que ademas hace `sow()` de la atencion, y la intercambiamos
    en el namespace del paquete. La subclase mantiene el nombre "MHA", asi las rutas del
    arbol de parametros no cambian y los pesos preentrenados siguen cargando.

    QUE ES LA ATENCION AQUI. TransformerPayne es un modelo de atencion CRUZADA:
      * la QUERY es la LONGITUD DE ONDA (codificada en frecuencia) -- un solo token
      * las KEYS/VALUES son los PARAMETROS ESTELARES, embebidos en `no_tokens` tokens
    Un mapa de atencion responde: *en esta longitud de onda, a que token de parametros mira
    esta cabeza?* Justo lo que afirma el paper.

Run / Uso:
    python 17_attention_maps.py --class G
    python 17_attention_maps.py --class G --heads 6

Requires / Requiere: env astro-jax (TransformerPayne + flax).
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
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze

import transformer_payne as tp
import transformer_payne.transformer_payne as tpm
import project_lib as P

# --------------------------------------------------------------------------
# EN: PATCH -- an MHA that also sows its attention weights and its key/value tokens.
#     Identical maths to the original; only the two sow() calls are added.
# ES: PARCHE -- un MHA que ademas expone sus pesos de atencion y sus tokens key/value.
# --------------------------------------------------------------------------
_MHA_ORIG = tpm.MHA


class _MHAWithAttn(_MHA_ORIG):
    @nn.compact
    def __call__(self, inputs_q, inputs_kv, train=True):
        num_heads = self.dim // self.dim_head
        dtype = inputs_q.dtype
        stddev = self.dim ** -0.5
        ini = nn.initializers.truncated_normal(
            stddev=self.sigma * stddev / jnp.array(.87962566103423978, dtype))
        w_q = self.param('w_q', ini, (self.dim, num_heads, self.dim_head), dtype=dtype)
        w_k = self.param('w_k', ini, (self.dim, num_heads, self.dim_head), dtype=dtype)
        w_v = self.param('w_v', ini, (self.dim, num_heads, self.dim_head), dtype=dtype)
        w_o = self.param('w_o', ini, (num_heads, self.dim_head, self.dim), dtype=dtype)

        q = jnp.einsum('...li,ihd->...lhd', inputs_q, w_q)
        k = jnp.einsum('...li,ihd->...lhd', inputs_kv, w_k)
        v = jnp.einsum('...li,ihd->...lhd', inputs_kv, w_v)

        scaling = 1 / jnp.sqrt(self.dim_head).astype(dtype)
        scores = jnp.einsum('...qhd,...khd->...hqk',
                            q * scaling, k * scaling * self.alpha_att)
        attn = nn.softmax(scores, axis=-1)

        self.sow('intermediates', 'attn', attn)        # <<< the attention map
        self.sow('intermediates', 'kv', inputs_kv)     # <<< the parameter tokens

        ctx = jnp.einsum('...hqk,...khd->...qhd', attn, v)
        out = jnp.einsum('...lhd,hdm->...lm', ctx, w_o)
        return out


_MHAWithAttn.__name__ = "MHA"      # EN/ES: keep the parameter-tree path identical!
tpm.MHA = _MHAWithAttn


# --------------------------------------------------------------------------
# EN: PATCH 2 -- THE SUBTLE ONE. The model wraps the per-wavelength network in
#     nn.vmap(..., variable_axes={'params': None}). A lifted Flax transform only carries
#     through the collections listed in `variable_axes` -- so our sown 'intermediates'
#     were computed and then silently DISCARDED by the vmap. Adding 'intermediates': 0
#     lets them out, with the wavelength as the leading axis.
# ES: PARCHE 2 -- EL SUTIL. El modelo envuelve la red por longitud de onda en
#     nn.vmap(..., variable_axes={'params': None}). Una transformacion "lifted" de Flax
#     solo deja pasar las colecciones listadas en `variable_axes`, asi que nuestros
#     'intermediates' se calculaban y luego se DESCARTABAN. Agregar 'intermediates': 0
#     los deja salir, con la longitud de onda como eje principal.
# --------------------------------------------------------------------------
class _TPModelWithAttn(tpm.TransformerPayneModel):
    @nn.compact
    def __call__(self, inputs, train):
        log_waves, p = inputs
        TP = nn.vmap(
            tpm.TransformerPayneModelWave,
            in_axes=((None, 0),), out_axes=0,
            variable_axes={'params': None, 'intermediates': 0},   # <<< THE FIX
            split_rngs={'params': False})
        return TP(name="transformer_payne",
                  dim=self.dim, dim_ff_multiplier=self.dim_ff_multiplier,
                  no_tokens=self.no_tokens, no_layers=self.no_layers,
                  dim_head=self.dim_head, out_dim=self.out_dim, input_dim=self.input_dim,
                  min_period=self.min_period, max_period=self.max_period,
                  bias_dense=self.bias_dense, bias_attention=self.bias_attention,
                  activation_fn=self.activation_fn,
                  output_activation_fn=self.output_activation_fn,
                  init_att_q=self.init_att_q, init_att_o=self.init_att_o,
                  emb_init=self.emb_init, ff_init=self.ff_init, head_init=self.head_init,
                  sigma=self.sigma, alpha_emb=self.alpha_emb, alpha_att=self.alpha_att,
                  reference_depth=self.reference_depth,
                  reference_width=self.reference_width)((p, log_waves))


_TPModelWithAttn.__name__ = "TransformerPayneModel"
tpm.TransformerPayneModel = _TPModelWithAttn


# --------------------------------------------------------------------------
def forward_with_attention(emu, log_waves, mu, spectral_parameters):
    """EN: ONE forward pass over ALL wavelengths at once (the model is vmapped over them).
        -> (output, {layer: attn (n_waves, heads, tokens)}, kv_tokens (tokens, dim))
    ES: UN solo forward sobre TODAS las longitudes de onda (el modelo esta vmapeado)."""
    p_all = jnp.concatenate([spectral_parameters, jnp.atleast_1d(mu)], axis=0)
    p_all = (p_all - emu.min_parameters) / (emu.max_parameters - emu.min_parameters)
    out, state = emu.model.apply(
        {"params": freeze(emu.model_definition.emulator_weights)},
        (jnp.asarray(log_waves), p_all),
        train=False,
        mutable=["intermediates"],
    )
    if "intermediates" not in state:
        raise SystemExit("[ERROR] attention was not captured -- the Flax patch did not take.")

    attn, kv = {}, None
    def walk(d, path=""):
        nonlocal kv
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{path}/{k}")
            elif k == "attn":
                a = np.asarray(v[0])                  # (n_waves, heads, seq_q, tokens)
                attn[path.strip('/')] = a[:, :, 0, :] if a.ndim == 4 else a
            elif k == "kv" and kv is None:
                k0 = np.asarray(v[0])                 # (n_waves, tokens, dim)
                kv = k0[0] if k0.ndim == 3 else k0    # tokens depend only on the parameters
    walk(state["intermediates"])
    return np.asarray(out), attn, kv


def main():
    ap = argparse.ArgumentParser(description="Reproduce the TransformerPayne attention maps (paper Fig. 9)")
    ap.add_argument("--class", dest="cls", default="G", choices=["F", "G", "K"])
    ap.add_argument("--heads", type=int, default=6,
                    help="how many of the most specialized heads to plot")
    ap.add_argument("--delta", type=float, default=0.3,
                    help="perturbation (dex) for the token<->parameter map")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)

    print("Loading TransformerPayne weights...")
    emu = tp.TransformerPayne.download()
    m = emu.model
    n_heads = m.dim // m.dim_head
    print(f"architecture: {m.no_layers} layers x {n_heads} heads, "
          f"{m.no_tokens} parameter tokens, dim={m.dim}")

    # EN: the reference star | ES: la estrella de referencia
    t_min, t_max = P.TEFF_RANGES[args.cls]
    teff = 0.5 * (t_min + t_max)
    base = {"logteff": np.log10(teff), "logg": 4.5}
    base.update({el: 0.0 for el in P.VARIED_ELEMENTS})
    pvec = emu.to_parameters(base)
    mu = 1.0

    # EN: the wavelengths we probe = the physical lines | ES: las lineas fisicas
    LINES = {
        "H-delta 4103": 4102.9, "Mn 4041": 4041.4, "Ca I 4227": 4226.7,
        "CH G-band 4305": 4305.0, "H-gamma 4342": 4341.7, "Fe 4384": 4383.5,
        "Fe 4405": 4404.8, "Ti 4534": 4534.8, "Mg 4571": 4571.1,
        "H-beta 4863": 4862.7, "Fe 4919": 4919.0, "Ti 4982": 4981.7,
    }
    names = list(LINES.keys())
    waves = np.array([LINES[k] for k in names])

    # ----------------------------------------------------------------------
    # EN: 1) attention at every line | ES: 1) atencion en cada linea
    # ----------------------------------------------------------------------
    print(f"\none vmapped forward pass over {len(waves)} line wavelengths, capturing attention...")
    _, A, kv0 = forward_with_attention(emu, np.log10(waves), mu, pvec)
    layers = sorted(A.keys())
    print(f"captured attention from {len(layers)} attention blocks; "
          f"shape per block = {A[layers[0]].shape}  (lines, heads, tokens)")

    # ----------------------------------------------------------------------
    # EN: 2) token <-> parameter map: perturb each label, see which token moves
    # ES: 2) mapa token <-> parametro: perturbar cada etiqueta y ver que token se mueve
    # ----------------------------------------------------------------------
    print("building the token <-> parameter map...")
    labels = ["logteff", "logg"] + list(P.VARIED_ELEMENTS)
    n_tok = kv0.shape[0]
    probe = np.log10(np.array([4500.0]))
    T = np.zeros((n_tok, len(labels)))
    for j, lab in enumerate(labels):
        hi, lo = dict(base), dict(base)
        step = 0.02 if lab == "logteff" else args.delta
        hi[lab] = base[lab] + step
        lo[lab] = base[lab] - step
        _, _, kv_hi = forward_with_attention(emu, probe, mu, emu.to_parameters(hi))
        _, _, kv_lo = forward_with_attention(emu, probe, mu, emu.to_parameters(lo))
        d = kv_hi - kv_lo                              # (tokens, dim)
        T[:, j] = np.linalg.norm(d.reshape(n_tok, -1), axis=1)
    Tn = T / (T.max(axis=0, keepdims=True) + 1e-12)   # normalize per parameter

    # ----------------------------------------------------------------------
    # EN: 3) pick the most SPECIALIZED heads: those whose attention varies most
    #        across the lines (a head that attends to the same token everywhere is
    #        uninformative). ES: elegir las cabezas mas especializadas.
    # ----------------------------------------------------------------------
    cands = []
    for layer in layers:
        M = A[layer]                                   # (lines, heads, tokens)
        for h in range(M.shape[1]):
            spec = float(M[:, h, :].std(axis=0).mean())   # variation across lines
            cands.append((spec, layer, h))
    cands.sort(reverse=True)
    top = cands[:args.heads]
    print("\nmost specialized heads (attention varies most across lines):")
    for spec, layer, h in top:
        print(f"   {layer}  head {h}   specialization = {spec:.4f}")

    # ----------------------------------------------------------------------
    # EN: 4) QUANTITATIVE read-out -- do not make the reader squint at a heatmap.
    #        For every selected head: at each line, which token does it attend to most?
    #        If a head attends to the SAME token at all three Balmer lines, it is a
    #        "hydrogen head". That is the paper's claim, made checkable.
    # ES: 4) Lectura CUANTITATIVA -- si una cabeza mira el MISMO token en las tres lineas
    #        de Balmer, es una "cabeza de hidrogeno". La afirmacion del paper, verificable.
    # ----------------------------------------------------------------------
    BALMER = [n for n in names if n.startswith("H-")]
    print("\n" + "=" * 78)
    print("WHICH TOKEN DOES EACH HEAD LOOK AT, LINE BY LINE?")
    print("=" * 78)
    for spec, layer, h in top:
        M = A[layer][:, h, :]                          # (lines, tokens)
        tok = M.argmax(axis=1)
        lname = layer.replace("transformer_payne/", "")
        print(f"\n  [{lname}  head {h}]   specialization = {spec:.4f}")
        for i, nm in enumerate(names):
            bar = "#" * int(round(20 * M[i, tok[i]] / (M.max() + 1e-12)))
            print(f"     {nm:<16} -> token {tok[i]:2d}   {M[i, tok[i]]:.3f}  {bar}")
        bt = [tok[names.index(b)] for b in BALMER]
        if len(set(bt)) == 1:
            print(f"     >>> ALL {len(BALMER)} BALMER LINES point at the SAME token ({bt[0]}) "
                  f"-> this is a HYDROGEN / TEMPERATURE head")

    # ----------------------------------------------------------------------
    # EN: 5) the figure -- paper Fig. 9 style, with a readable layout:
    #        shared x-axis, tick labels ONLY on the bottom panel, head name as an inset.
    # ES: 5) la figura -- estilo Fig. 9, con layout legible.
    # ----------------------------------------------------------------------
    nrow = 1 + len(top)
    fig, axes = plt.subplots(nrow, 1, figsize=(12, 2.9 + 1.75 * len(top)),
                             gridspec_kw={"height_ratios": [1.35] + [1.0] * len(top),
                                          "hspace": 0.18})

    # --- top: token <-> parameter
    ax = axes[0]
    im = ax.imshow(Tn, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Stellar parameter", fontsize=9, labelpad=6)
    ax.set_ylabel("Token index", fontsize=9)
    ax.text(0.005, 1.42, f"Which token carries which stellar parameter?   "
            f"(reference {args.cls} star, Teff={teff:.0f} K)",
            transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)

    # --- the attention maps
    for r, (spec, layer, h) in enumerate(top, start=1):
        ax = axes[r]
        M = A[layer][:, h, :].T                        # (tokens, lines)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="Reds")
        ax.set_xticks(range(len(names)))
        ax.set_ylabel("Token", fontsize=9)
        lname = layer.replace("transformer_payne/", "").replace("MHA_", "layer ")
        # EN: head name INSIDE the axes -> can never collide with the panel above
        ax.text(0.008, 0.90, f"{lname}, head {h}", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top",
                bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2.0))
        if r == nrow - 1:
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Spectral line", fontsize=9)
        else:
            ax.set_xticklabels([])
        fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)

    fig.suptitle("TransformerPayne attention maps (cf. paper Fig. 9)\n"
                 "at each spectral line, which parameter-token does the head attend to?",
                 fontsize=12.5, y=0.995)
    out = f"figures/attention_maps_{args.cls}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")

    np.savez_compressed(f"attention_maps_{args.cls}.npz",
                        token_param=Tn, labels=np.array(labels),
                        lines=np.array(names), wavelengths=waves,
                        **{f"attn_{i}": A[l] for i, l in enumerate(layers)})
    print(f"saved attention_maps_{args.cls}.npz")
    print("\nHOW TO READ / COMO LEER:")
    print("  top panel  : token T carries parameter P")
    print("  lower panel: at line L, this head attends to token T")
    print("  -> chain them: line L is emitted by looking at the token that carries element E.")


if __name__ == "__main__":
    main()
