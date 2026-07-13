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
def forward_with_attention(emu, log_wave, mu, spectral_parameters):
    """EN: one forward pass, returning (output, {layer: attn}, kv_tokens).
        attn per layer has shape (num_heads, seq_q, no_tokens).
    ES: un forward, devolviendo (salida, {capa: atencion}, tokens)."""
    p_all = jnp.concatenate([spectral_parameters, jnp.atleast_1d(mu)], axis=0)
    p_all = (p_all - emu.min_parameters) / (emu.max_parameters - emu.min_parameters)
    out, state = emu.model.apply(
        {"params": freeze(emu.model_definition.emulator_weights)},
        (jnp.atleast_1d(log_wave), p_all),
        train=False,
        mutable=["intermediates"],
    )
    inter = state["intermediates"]

    attn, kv = {}, None
    def walk(d, path=""):
        nonlocal kv
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{path}/{k}")
            elif k == "attn":
                a = np.asarray(v[0])
                attn[path] = np.squeeze(a)             # (heads, seq_q, tokens)
            elif k == "kv" and kv is None:
                kv = np.asarray(v[0])
    walk(inter)
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
    print(f"\nrunning {len(waves)} forward passes (one per line) and capturing attention...")
    A = {}   # layer -> (n_lines, heads, tokens)
    kv0 = None
    for i, w in enumerate(waves):
        _, attn, kv = forward_with_attention(emu, np.log10(w), mu, pvec)
        if kv0 is None:
            kv0 = kv
        for layer, a in attn.items():
            a = np.atleast_2d(a)                      # (heads, tokens) after squeeze
            if a.ndim == 3:                           # (heads, seq_q, tokens)
                a = a[:, 0, :]
            A.setdefault(layer, np.zeros((len(waves), a.shape[0], a.shape[1])))
            A[layer][i] = a
    layers = sorted(A.keys())
    print(f"captured attention from {len(layers)} attention blocks; "
          f"shape per block = {A[layers[0]].shape} (lines, heads, tokens)")

    # ----------------------------------------------------------------------
    # EN: 2) token <-> parameter map: perturb each label, see which token moves
    # ES: 2) mapa token <-> parametro: perturbar cada etiqueta y ver que token se mueve
    # ----------------------------------------------------------------------
    print("building the token <-> parameter map...")
    labels = ["logteff", "logg"] + list(P.VARIED_ELEMENTS)
    n_tok = kv0.shape[-2] if kv0.ndim >= 2 else kv0.shape[0]
    T = np.zeros((n_tok, len(labels)))
    for j, lab in enumerate(labels):
        hi, lo = dict(base), dict(base)
        step = 0.02 if lab == "logteff" else args.delta
        hi[lab] = base[lab] + step
        lo[lab] = base[lab] - step
        _, _, kv_hi = forward_with_attention(emu, np.log10(4500.0), mu, emu.to_parameters(hi))
        _, _, kv_lo = forward_with_attention(emu, np.log10(4500.0), mu, emu.to_parameters(lo))
        d = np.squeeze(kv_hi) - np.squeeze(kv_lo)     # (tokens, dim)
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
    # EN: 4) the figure -- paper Fig. 9 style
    # ----------------------------------------------------------------------
    nrow = 1 + len(top)
    fig = plt.figure(figsize=(13, 2.6 + 1.9 * len(top)))
    gs = fig.add_gridspec(nrow, 1, hspace=0.55)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(Tn, aspect="auto", origin="lower", cmap="Reds")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Token index")
    ax.set_xlabel("Spectrum parameters")
    ax.set_title(f"Which token carries which stellar parameter?  "
                 f"(reference {args.cls} star, Teff={teff:.0f} K)", fontsize=11)
    fig.colorbar(im, ax=ax, pad=0.01)

    for r, (spec, layer, h) in enumerate(top, start=1):
        ax = fig.add_subplot(gs[r])
        M = A[layer][:, h, :].T                       # (tokens, lines)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="Reds")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_ylabel("Token index")
        lname = layer.strip("/").replace("/", " ")
        ax.set_title(f"Attention map -- {lname}, head {h}", fontsize=10)
        fig.colorbar(im, ax=ax, pad=0.01)

    fig.suptitle("TransformerPayne attention maps (cf. paper Fig. 9): "
                 "at each spectral line, which parameter-token does the head look at?",
                 fontsize=12, y=0.995)
    out = f"figures/attention_maps_{args.cls}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
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
