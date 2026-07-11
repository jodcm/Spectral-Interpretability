"""
00_fetch_sdss.py
================
EN: Fetch a SECOND real dataset yourself: SDSS DR16. Same idea as 00_fetch_desi.py,
    but a DIFFERENT instrument -> an independent check of the sim->real transfer.
    Why SDSS instead of LAMOST: SPARCL (the client we already use for DESI) also
    serves SDSS DR16 and BOSS DR16, and the NOIRLab Data Lab hosts the SDSS stellar
    parameters (sppparams: teffadop/loggadop/fehadop). So we get labelled real spectra
    with the SAME code path -- no FITS parsing, no registration, no waiting for anyone.

    Output = per-class CSVs in exactly the format the rest of the pipeline expects, so
    02/03/07 work unchanged. SDSS resolving power in 4000-5000 A is R ~ 2000, so train
    the synthetic side with --resolution 2000 (not 'desi').
ES: Consigue tu mismo un SEGUNDO dataset real: SDSS DR16. Misma idea que
    00_fetch_desi.py pero con OTRO instrumento -> chequeo independiente de la
    transferencia sim->real. Por que SDSS y no LAMOST: SPARCL (el cliente que ya usamos
    para DESI) tambien sirve SDSS DR16 y BOSS DR16, y el NOIRLab Data Lab tiene los
    parametros estelares de SDSS (sppparams: teffadop/loggadop/fehadop). Asi obtenemos
    espectros reales etiquetados con el MISMO codigo -- sin parsear FITS, sin registro,
    sin depender de nadie.

    Salida = CSV por clase en el formato exacto que espera el resto de la pipeline.
    El poder resolutivo de SDSS en 4000-5000 A es R ~ 2000, asi que entrena el lado
    sintetico con --resolution 2000 (no 'desi').

Run / Uso:
    python 00_fetch_sdss.py --classes G K --n 300
    # then / luego:
    python 06_generate_large.py --n 50000 --out sim_100k_sdss.npz --jobs 24 --batch 16 --resolution 2000
    python 07_train_large.py --data-npz sim_100k_sdss.npz --real proyecto_desi/espectros_sdss --norm iterative

Requires / Requiere:  pip install astro-datalab sparclclient
"""
import os
import time
import argparse
import numpy as np
import pandas as pd

import project_lib as P

# EN: SDSS resolving power in our window (approximately constant ~2000).
# ES: poder resolutivo de SDSS en nuestra ventana (aprox. constante ~2000).
SDSS_R = 2000.0

# EN: same Teff ranges as the DESI fetcher (real-data selection)
# ES: mismos rangos de Teff que el fetcher de DESI (seleccion de datos reales)
TEFF_RANGES_REAL = {
    "O": (30000, 100000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}


def build_query(tmin, tmax, limit):
    """EN: SDSS DR16 spectra of STARS with SSPP stellar parameters in a Teff range.
        specobj = clean spectra (no sky/QA/duplicates); sppparams = SEGUE Stellar
        Parameter Pipeline (teffadop = adopted effective temperature).
    ES: Espectros SDSS DR16 de ESTRELLAS con parametros SSPP en un rango de Teff.
        specobj = espectros limpios; sppparams = pipeline de parametros estelares."""
    return f"""
    SELECT
        s.specobjid,
        p.teffadop AS teff,
        p.loggadop AS logg,
        p.fehadop  AS feh
    FROM sdss_dr16.specobj  AS s
    JOIN sdss_dr16.sppparams AS p ON s.specobjid = p.specobjid
    WHERE s.class = 'STAR'
      AND s.zwarning = 0
      AND p.teffadop >= {tmin}
      AND p.teffadop <  {tmax}
      AND p.loggadop > 3.5
    LIMIT {limit}
    """


def main():
    ap = argparse.ArgumentParser(description="Search + download real SDSS DR16 spectra per class.")
    ap.add_argument("--classes", nargs="+", default=["G", "K"], help="spectral classes, e.g. G K")
    ap.add_argument("--n", type=int, default=300, help="spectra per class")
    ap.add_argument("--out", default="proyecto_desi/espectros_sdss",
                    help="output folder (one subfolder per class)")
    ap.add_argument("--release", default="SDSS-DR16",
                    help="SPARCL data release: SDSS-DR16 or BOSS-DR16")
    args = ap.parse_args()

    from dl import queryClient as qc
    from sparcl.client import SparclClient

    client = SparclClient(connect_timeout=30, read_timeout=1800)

    for c in args.classes:
        if c not in TEFF_RANGES_REAL:
            print(f"[skip] unknown class '{c}'")
            continue
        tmin, tmax = TEFF_RANGES_REAL[c]
        print(f"\n[{c}] querying SDSS DR16 (SSPP Teff {tmin}-{tmax} K)...")
        try:
            df = qc.query(sql=build_query(tmin, tmax, args.n * 3), fmt="pandas")
        except Exception as e:
            print(f"[{c}] query failed: {str(e)[:150]}")
            print("    -> check the schema at https://datalab.noirlab.edu/data-explorer "
                  "(table names sdss_dr16.specobj / sdss_dr16.sppparams may differ per release)")
            continue
        df = df.drop_duplicates(subset="specobjid").head(args.n)
        print(f"[{c}] {len(df)} stars found. Downloading spectra via SPARCL ({args.release})...")

        outdir = os.path.join(args.out, c)
        os.makedirs(outdir, exist_ok=True)
        saved = 0
        for _, row in df.iterrows():
            sid = int(row["specobjid"])
            fpath = os.path.join(outdir, f"sdss_{c}_{sid}_4000_5000A.csv")
            if os.path.exists(fpath):
                saved += 1
                continue
            try:
                recs = client.retrieve_by_specid(
                    [sid], include=["wavelength", "flux", "ivar"],
                    dataset_list=[args.release]).records
                if not recs:
                    continue
                s = recs[0]
                sdf = pd.DataFrame({
                    "wavelength": np.asarray(s.wavelength, dtype=float),
                    "flux": np.asarray(s.flux, dtype=float),
                    "ivar": np.asarray(s.ivar, dtype=float),
                    "specobjid": sid, "clase_aprox": c,
                    "teff": row["teff"], "logg": row["logg"], "feh": row["feh"],
                })
                # EN: cut to the common pipeline range | ES: recortar al rango comun
                sdf = sdf[(sdf["wavelength"] >= P.WMIN) & (sdf["wavelength"] <= P.WMAX)]
                if len(sdf) == 0:
                    continue
                sdf.to_csv(fpath, index=False)
                saved += 1
                time.sleep(0.05)
            except Exception as e:
                print(f"  [{sid}] error: {str(e)[:80]}")
        print(f"[{c}] saved {saved} spectra -> {outdir}")

    print(f"\nDone. SDSS has R ~ {SDSS_R:.0f} in 4000-5000 A, so generate the synthetic side with:")
    print(f"  python 06_generate_large.py --n 50000 --out sim_100k_sdss.npz --jobs 24 --batch 16 --resolution {SDSS_R:.0f}")
    print(f"  python 07_train_large.py --data-npz sim_100k_sdss.npz --real {args.out} --norm iterative")


if __name__ == "__main__":
    main()
