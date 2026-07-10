"""
00_fetch_desi.py
================
EN: Step 0 (optional). SEARCH real stars in DESI (x Gaia x MWS) by spectral class
    and DOWNLOAD their spectra, saving them as per-class CSVs ready for
    02_evaluate_real_desi.py. Combines the NOIRLab Data Lab query (Cata's logic)
    with the SPARCL spectrum download in a single script. Classes and the number
    per class are command-line parameters, so you can run "search -> download ->
    train" entirely yourself.
ES: Paso 0 (opcional). BUSCA estrellas reales en DESI (x Gaia x MWS) por clase
    espectral y DESCARGA sus espectros, guardandolos como CSV por clase listos para
    02_evaluate_real_desi.py. Combina la query del NOIRLab Data Lab (logica de Cata)
    con la descarga SPARCL en un solo script. Las clases y la cantidad por clase son
    parametros, asi puedes correr "buscar -> descargar -> entrenar" tu mismo.

Run / Uso:
    python 00_fetch_desi.py --classes G K --n 300
    python 00_fetch_desi.py --classes F G K --n 500 --out proyecto_desi/espectros_balanceados_desi

Requires / Requiere:  pip install astro-datalab sparclclient
    (Data Lab may need a free NOIRLab account for big queries /
     el Data Lab puede requerir una cuenta gratis de NOIRLab para queries grandes.)
"""
import os
import argparse
import time
import numpy as np
import pandas as pd

import project_lib as P     # EN: for the common wavelength range | ES: para el rango comun

# EN: approximate Teff ranges (K) per spectral class (real-data selection).
# ES: rangos aproximados de Teff (K) por clase espectral (seleccion de datos reales).
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
    """EN: DESI DR1 x Gaia DR3 x MWS query for STAR spectra in a Teff range, with
        reliable-parallax cuts. Uses m.param[4]=Teff directly in SQL (no parsing bug).
    ES: Query DESI DR1 x Gaia DR3 x MWS para espectros STAR en un rango de Teff, con
        cortes de paralaje confiable. Usa m.param[4]=Teff directo en SQL (sin bug de parseo)."""
    return f"""
    SELECT
        z.targetid,
        m.param[1] AS feh,
        m.param[4] AS teff,
        m.param[5] AS logg,
        g.parallax,
        g.parallax_error
    FROM desi_dr1.x1p5__zpix__gaia_dr3__gaia_source AS x
    JOIN desi_dr1.zpix        AS z ON x.id1 = z.id
    JOIN gaia_dr3.gaia_source AS g ON x.id2 = g.source_id
    JOIN desi_dr1.mws         AS m ON z.targetid = m.targetid
    WHERE z.spectype = 'STAR'
      AND z.zwarn = 0
      AND z.zcat_primary = true
      AND g.parallax IS NOT NULL
      AND g.parallax_error IS NOT NULL
      AND g.parallax > 0
      AND g.parallax / g.parallax_error > 3.0
      AND m.param[4] >= {tmin}
      AND m.param[4] <  {tmax}
    ORDER BY g.parallax / g.parallax_error DESC
    LIMIT {limit}
    """


def main():
    ap = argparse.ArgumentParser(description="Search + download real DESI spectra per class.")
    ap.add_argument("--classes", nargs="+", default=["G", "K"],
                    help="spectral classes to fetch, e.g. G K  or  F G K")
    ap.add_argument("--n", type=int, default=300, help="spectra per class")
    ap.add_argument("--out", default="proyecto_desi/espectros_balanceados_desi",
                    help="output folder (one subfolder per class)")
    args = ap.parse_args()

    # EN: imported here so the message is clear if the packages are missing.
    # ES: importado aqui para que el mensaje sea claro si faltan los paquetes.
    from dl import queryClient as qc
    from sparcl.client import SparclClient

    client = SparclClient(connect_timeout=30, read_timeout=1800)

    for c in args.classes:
        if c not in TEFF_RANGES_REAL:
            print(f"[skip] unknown class '{c}'")
            continue
        tmin, tmax = TEFF_RANGES_REAL[c]
        print(f"\n[{c}] querying DESI x Gaia x MWS (Teff {tmin}-{tmax} K)...")
        # EN: ask for 3x to have spares after de-duplication and download errors.
        # ES: pedir 3x para tener repuestos tras deduplicar y errores de descarga.
        df = qc.query(sql=build_query(tmin, tmax, args.n * 3), fmt="pandas")
        df = df.drop_duplicates(subset="targetid").head(args.n)
        print(f"[{c}] {len(df)} stars found. Downloading spectra via SPARCL...")

        outdir = os.path.join(args.out, c)
        os.makedirs(outdir, exist_ok=True)
        saved = 0
        for _, row in df.iterrows():
            tid = int(row["targetid"])
            fpath = os.path.join(outdir, f"desi_{c}_{tid}_4000_5000A.csv")
            if os.path.exists(fpath):
                saved += 1
                continue
            try:
                recs = client.retrieve_by_specid(
                    [tid], include=["wavelength", "flux", "ivar", "model"]).records
                if not recs:
                    continue
                s = recs[0]
                sdf = pd.DataFrame({
                    "wavelength": s.wavelength, "flux": s.flux,
                    "ivar": s.ivar, "model": s.model,
                    "targetid": tid, "clase_aprox": c,
                    "teff": row["teff"], "logg": row["logg"], "feh": row["feh"],
                })
                # EN: cut to the common range used by the pipeline (project_lib)
                # ES: recortar al rango comun usado por la pipeline (project_lib)
                sdf = sdf[(sdf["wavelength"] >= P.WMIN) & (sdf["wavelength"] <= P.WMAX)]
                if len(sdf) == 0:
                    continue
                sdf.to_csv(fpath, index=False)
                saved += 1
                time.sleep(0.05)
            except Exception as e:
                print(f"  [{tid}] error: {str(e)[:80]}")
        print(f"[{c}] saved {saved} spectra -> {outdir}")

    print(f"\nDone. Now:  python 02_evaluate_real_desi.py {args.out}")


if __name__ == "__main__":
    main()
