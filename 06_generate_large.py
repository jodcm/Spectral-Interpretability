"""
06_generate_large.py
====================
EN: Large-scale synthetic generation (the professor's ~100k target). Generates up
    to N spectra per class with TransformerPayne, using the WINNING pipeline
    (R(lambda) broadening + iterative continuum normalization) that reached the best
    sim->real transfer, and writes X (float32) + y to a compressed .npz so that
    training is DECOUPLED from generation (generate once, train many times).

    Runs on a GPU automatically if JAX sees one (Colab). On a CPU box, use `--jobs`
    to fan the (embarrassingly parallel) generation out across cores with
    multiprocessing: each worker builds its own emulator, jits once, and processes a
    shard. This is the fast path on a many-core CPU VM (e.g. 24 cores -> `--jobs 24`).
ES: Generacion sintetica a gran escala (el objetivo ~100k del profe). Genera hasta N
    espectros por clase con TransformerPayne usando la pipeline GANADORA
    (ensanchamiento R(lambda) + normalizacion de continuo iterativa) y guarda X
    (float32) + y en un .npz comprimido para DESACOPLAR el entrenamiento de la
    generacion (generar una vez, entrenar muchas).

    Usa GPU automaticamente si JAX ve una (Colab). En una maquina CPU, usa `--jobs`
    para repartir la generacion (vergonzosamente paralela) entre los nucleos con
    multiprocessing: cada worker construye su propio emulador, jitea una vez y procesa
    un shard. Es el camino rapido en una VM CPU con muchos nucleos (p.ej. 24 nucleos
    -> `--jobs 24`).

Run / Uso:
    # Quick smoke test (1k total) / prueba rapida:
    python 06_generate_large.py --n 500 --out sim_1k.npz --jobs 8
    # Full 100k on a 24-core CPU VM / completo en VM de 24 nucleos:
    python 06_generate_large.py --n 50000 --out sim_100k.npz --jobs 24 --batch 500
    # Full 100k on a GPU (Colab) / en GPU:
    python 06_generate_large.py --n 50000 --out sim_100k.npz --jobs 1 --batch 2000

Requires / Requiere: env astro-jax (TransformerPayne). Run 01 once first so the TP
    weights are cached on disk (the workers then load from cache, no re-download).
    See SCALE_100K.md.
"""
import os
import sys
import time
import argparse
import warnings
import random
import numpy as np
warnings.filterwarnings("ignore")

import project_lib as P

MU = 1.0

# --------------------------------------------------------------------------
# ES: Sorteo de parametros (mismo esquema que project_lib.generate_simulated)
# EN: Parameter sampling (same scheme as project_lib.generate_simulated)
# --------------------------------------------------------------------------
def sample_params(spectral_type, n, seed):
    """-> list of n parameter dicts (Teff, logg, abundances), reproducible."""
    rng = random.Random(seed)
    t_min, t_max = P.TEFF_RANGES[spectral_type]
    dicts = []
    for _ in range(n):
        teff = rng.uniform(t_min, t_max)
        logg = rng.uniform(*P.LOGG_RANGE)
        abundances = {el: rng.uniform(*P.ABUND_RANGE) for el in P.VARIED_ELEMENTS}
        dicts.append({"logteff": np.log10(teff), "logg": logg, **abundances})
    return dicts


def _resolve_broaden(resolution):
    if str(resolution).lower() == "none":
        return None
    if str(resolution).lower() == "desi":
        return P.desi_resolution
    return float(resolution)


def _make_gen_wave(broaden_R, oversample):
    wave_grid = P.WAVE_GRID
    if broaden_R is not None:
        return wave_grid, np.linspace(wave_grid[0], wave_grid[-1], len(wave_grid) * int(oversample))
    return wave_grid, wave_grid


# --------------------------------------------------------------------------
# ES: Estado y funciones de worker (multiprocessing, deben ser importables)
# EN: Worker state and functions (multiprocessing, must be top-level/picklable)
# --------------------------------------------------------------------------
_W = {}


def _init_worker(resolution, norm, oversample, float64, sigma_noise, batch):
    # ES: 1 hilo por worker para no sobre-suscribir los nucleos | EN: 1 thread/worker
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    import jax
    if float64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import transformer_payne as tp

    broaden_R = _resolve_broaden(resolution)
    wave_grid, gen_wave = _make_gen_wave(broaden_R, oversample)
    log_gen_wave = np.asarray(np.log10(gen_wave))
    emu = tp.TransformerPayne.download()

    def one(pvec):
        return emu(log_gen_wave, MU, pvec)[:, 0]
    batch_spectra = jax.jit(jax.vmap(one))

    _W.update(emu=emu, jnp=jnp, batch_spectra=batch_spectra,
              broaden_R=broaden_R, wave_grid=wave_grid, gen_wave=gen_wave,
              normalizer=(P.continuum_normalize_iter if norm == "iterative"
                          else P.continuum_normalize),
              sigma_noise=sigma_noise, batch=batch,
              do_broaden=(broaden_R is not None))


def _gen_shard(task):
    """EN: generate one shard -> (global_start, cls, X_chunk float32).
    ES: genera un shard -> (global_start, cls, X_chunk float32)."""
    global_start, cls, dicts, noise_seed = task
    emu = _W["emu"]; jnp = _W["jnp"]; batch_spectra = _W["batch_spectra"]
    gen_wave = _W["gen_wave"]; wave_grid = _W["wave_grid"]
    normalizer = _W["normalizer"]; do_broaden = _W["do_broaden"]
    broaden_R = _W["broaden_R"]; sigma = _W["sigma_noise"]; batch = _W["batch"]
    npr = np.random.RandomState(noise_seed)
    out = np.zeros((len(dicts), len(wave_grid)), dtype=np.float32)
    row = 0
    for b0 in range(0, len(dicts), batch):
        pmat = jnp.stack([emu.to_parameters(d) for d in dicts[b0:b0 + batch]])
        inten = np.asarray(batch_spectra(pmat))
        for k in range(inten.shape[0]):
            flux = inten[k].astype(float)
            if do_broaden:
                flux = P.broaden_to_resolution(gen_wave, flux, broaden_R)
                flux = np.interp(wave_grid, gen_wave, flux)
            norm = normalizer(flux) + npr.normal(0.0, sigma, size=len(wave_grid))
            out[row] = norm
            row += 1
    return global_start, cls, out


def _shard_indices(n, jobs):
    """EN: split range(n) into ~jobs contiguous chunks. ES: parte range(n) en ~jobs trozos."""
    step = int(np.ceil(n / max(jobs, 1)))
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def main():
    ap = argparse.ArgumentParser(description="Large-scale synthetic G/K generation -> .npz")
    ap.add_argument("--n", type=int, default=50000, help="spectra PER class (100k total = 50000)")
    ap.add_argument("--classes", nargs="+", default=["G", "K"], help="spectral classes")
    ap.add_argument("--out", default="sim_100k.npz", help="output .npz path")
    ap.add_argument("--jobs", type=int, default=1, help="CPU worker processes (e.g. 24). 1 = single process / GPU")
    ap.add_argument("--batch", type=int, default=32,
                    help="emulator batch size per vmap call. LOWER this if you hit "
                         "'Out of memory' (each spectrum on the fine grid is memory-heavy; "
                         "peak RAM ~ jobs * batch). Try 16-32 on CPU, larger only on GPU.")
    ap.add_argument("--resolution", default="desi", help="'desi' R(lambda), a number, or 'none'")
    ap.add_argument("--norm", default="masked",
                    choices=["masked", "iterative", "percentile"],
                    help="continuum normalizer (must match training). 'masked' = line-masked "
                         "continuum fit, the H-beta fix (recommended).")
    ap.add_argument("--sigma-noise", type=float, default=0.02, dest="sigma_noise")
    ap.add_argument("--oversample", type=int, default=6, help="fine-grid factor before broadening")
    ap.add_argument("--float64", action="store_true", help="use x64 (slower; default float32)")
    ap.add_argument("--base-seed", type=int, default=100, dest="base_seed")
    args = ap.parse_args()

    wave_grid = P.WAVE_GRID
    n_total = args.n * len(args.classes)
    X = np.zeros((n_total, len(wave_grid)), dtype=(np.float64 if args.float64 else np.float32))
    y = np.empty(n_total, dtype="<U1")

    # EN: sample all parameter dicts up front (cheap, deterministic) and build shards
    # ES: sortear todos los dicts de parametros al inicio (barato, determinista) y armar shards
    tasks = []
    class_offsets = {}
    off = 0
    for ci, c in enumerate(args.classes):
        class_offsets[c] = off
        dicts = sample_params(c, args.n, seed=args.base_seed + ci)
        shards = _shard_indices(args.n, args.jobs if args.jobs > 1 else 1)
        for si, (a, b) in enumerate(shards):
            noise_seed = args.base_seed + ci * 1000 + si
            tasks.append((off + a, c, dicts[a:b], noise_seed))
        off += args.n

    t0 = time.time()
    init_args = (args.resolution, args.norm, args.oversample, args.float64,
                 args.sigma_noise, args.batch)

    if args.jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        print(f"CPU multiprocessing: {args.jobs} workers, {len(tasks)} shards, {n_total} spectra total")
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx,
                                 initializer=_init_worker, initargs=init_args) as ex:
            futs = [ex.submit(_gen_shard, t) for t in tasks]
            for fut in as_completed(futs):
                gstart, cls, chunk = fut.result()
                X[gstart:gstart + len(chunk)] = chunk
                y[gstart:gstart + len(chunk)] = cls
                done += len(chunk)
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{n_total}  | {rate:.0f} spec/s  | ETA {(n_total-done)/max(rate,1e-9):.0f}s", flush=True)
    else:
        # EN: single process (GPU or 1 core). ES: un solo proceso (GPU o 1 nucleo).
        print("JAX devices:", __import__("jax").devices())
        _init_worker(*init_args)
        done = 0
        for t in tasks:
            gstart, cls, chunk = _gen_shard(t)
            X[gstart:gstart + len(chunk)] = chunk
            y[gstart:gstart + len(chunk)] = cls
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{n_total}  | {rate:.0f} spec/s  | ETA {(n_total-done)/max(rate,1e-9):.0f}s", flush=True)

    good = np.isfinite(X).all(axis=1)
    if not good.all():
        print(f"dropping {np.sum(~good)} non-finite spectra")
        X, y = X[good], y[good]

    meta = dict(resolution=str(args.resolution), norm=args.norm,
                classes=",".join(args.classes))
    np.savez_compressed(args.out, X=X, y=y, wave_grid=wave_grid,
                        **{f"meta_{k}": v for k, v in meta.items()})
    dt = time.time() - t0
    print(f"\nsaved {args.out}  shape={X.shape}  dtype={X.dtype}  in {dt:.0f}s "
          f"({X.shape[0]/max(dt,1e-9):.0f} spec/s)")
    print("Next: python 07_train_large.py --data-npz %s --real proyecto_desi/espectros_balanceados_desi --norm %s"
          % (args.out, args.norm))


if __name__ == "__main__":
    main()
