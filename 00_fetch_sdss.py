"""
00_fetch_sdss.py
================
EN: Fetch a SECOND real dataset yourself: SDSS DR16 -- via SDSS SkyServer directly
    (astroquery.sdss), NOT via NOIRLab/SPARCL. Reason: the NOIRLab SPARCL host
    (astrosparcl.datalab.noirlab.edu) is currently unreachable (connection timeouts),
    so anything depending on it fails. SDSS SkyServer is a completely independent
    infrastructure and works on its own.

    Labels come from the SEGUE Stellar Parameter Pipeline (sppParams.teffadop), joined
    to the clean spectra table (SpecObj). Spectra are downloaded with SDSS.get_spectra
    and written as per-class CSVs in exactly the format the rest of the pipeline
    expects -- so 02/03/07/10 all work unchanged.

    SDSS is a DIFFERENT instrument than DESI -> an independent check of the sim->real
    transfer. Its resolving power in 4000-5000 A is R ~ 2000, so generate the synthetic
    side with --resolution 2000 (not 'desi').
ES: Consigue tu mismo un SEGUNDO dataset real: SDSS DR16 -- directamente por SDSS
    SkyServer (astroquery.sdss), NO por NOIRLab/SPARCL. Motivo: el host SPARCL de
    NOIRLab no responde (timeouts), asi que todo lo que dependa de el falla. SkyServer
    es una infraestructura independiente y funciona por su cuenta.

    Las etiquetas vienen del SEGUE Stellar Parameter Pipeline (sppParams.teffadop),
    unido a la tabla de espectros limpios (SpecObj). Los espectros se descargan con
    SDSS.get_spectra y se guardan como CSV por clase en el formato exacto que espera el
    resto de la pipeline.

    SDSS es un instrumento DISTINTO a DESI -> chequeo independiente de la transferencia
    sim->real. Su resolucion en 4000-5000 A es R ~ 2000, asi que genera el lado
    sintetico con --resolution 2000 (no 'desi').

Run / Uso:
    python 00_fetch_sdss.py --classes G K --n 300
    # then / luego:
    python 06_generate_large.py --n 25000 --out sim_50k_sdss.npz --jobs 24 --batch 16 --resolution 2000
    python 07_train_large.py --data-npz sim_50k_sdss.npz --real proyecto_desi/espectros_sdss --norm iterative

Requires / Requiere:  pip install astroquery
"""
import os
import time
import argparse
import numpy as np
import pandas as pd

import project_lib as P

# EN: SDSS resolving power in our window (approximately constant ~2000).
SDSS_R = 2000.0

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
    """EN: SkyServer SQL (T-SQL dialect: TOP n, not LIMIT). Clean stellar spectra
        (SpecObj) joined with SEGUE stellar parameters (sppParams). loggadop > 3.5
        keeps dwarfs, matching the synthetic LOGG_RANGE.
        IMPORTANT: astroquery's SDSS.get_spectra(matches=...) builds the download URL
        from the columns run2d, plate, mjd and fiberID -- they MUST be present and
        named exactly like that (fiberID with capital ID), otherwise you get a
        KeyError: 'run2d'.
    ES: SQL de SkyServer (dialecto T-SQL: TOP n, no LIMIT). Espectros estelares limpios
        (SpecObj) unidos con parametros estelares SEGUE (sppParams).
        IMPORTANTE: SDSS.get_spectra(matches=...) de astroquery arma la URL de descarga
        con las columnas run2d, plate, mjd y fiberID -- deben estar y llamarse
        exactamente asi (fiberID con ID mayuscula), si no da KeyError: 'run2d'."""
    return f"""
    SELECT TOP {limit}
        s.specobjid,
        s.plate   AS plate,
        s.mjd     AS mjd,
        s.fiberid AS fiberID,
        s.run2d   AS run2d,
        p.teffadop AS teff, p.loggadop AS logg, p.fehadop AS feh
    FROM SpecObj AS s
    JOIN sppParams AS p ON s.specobjid = p.specobjid
    WHERE s.class = 'STAR'
      AND s.zWarning = 0
      AND p.teffadop >= {tmin}
      AND p.teffadop <  {tmax}
      AND p.loggadop > 3.5
    """


def spectrum_from_hdulist(hdu):
    """EN: SDSS spec FITS -> (wavelength[A], flux, ivar, redshift).
        HDU1: loglam/flux/ivar ; HDU2: Z (redshift).
    ES: FITS de espectro SDSS -> (longitud de onda[A], flujo, ivar, redshift)."""
    d = hdu[1].data
    wave = 10.0 ** np.asarray(d["loglam"], dtype=float)
    flux = np.asarray(d["flux"], dtype=float)
    ivar = np.asarray(d["ivar"], dtype=float)
    try:
        z = float(np.asarray(hdu[2].data["Z"]).ravel()[0])
    except Exception:
        z = 0.0
    return wave, flux, ivar, z


def main():
    ap = argparse.ArgumentParser(description="Search + download real SDSS DR16 spectra per class.")
    ap.add_argument("--classes", nargs="+", default=["G", "K"], help="spectral classes, e.g. G K")
    ap.add_argument("--n", type=int, default=300, help="spectra per class")
    ap.add_argument("--out", default="proyecto_desi/espectros_sdss",
                    help="output folder (one subfolder per class)")
    ap.add_argument("--dr", type=int, default=16, help="SDSS data release")
    ap.add_argument("--batch", type=int, default=20, help="spectra per download request")
    args = ap.parse_args()

    from astroquery.sdss import SDSS

    for c in args.classes:
        if c not in TEFF_RANGES_REAL:
            print(f"[skip] unknown class '{c}'")
            continue
        tmin, tmax = TEFF_RANGES_REAL[c]
        print(f"\n[{c}] querying SDSS DR{args.dr} SkyServer (SSPP Teff {tmin}-{tmax} K)...")
        try:
            tab = SDSS.query_sql(build_query(tmin, tmax, args.n), data_release=args.dr)
        except Exception as e:
            print(f"[{c}] query failed: {str(e)[:150]}")
            continue
        if tab is None or len(tab) == 0:
            print(f"[{c}] no stars found in this Teff range.")
            continue
        print(f"[{c}] {len(tab)} stars found. Downloading spectra from SkyServer...")

        outdir = os.path.join(args.out, c)
        os.makedirs(outdir, exist_ok=True)
        saved = 0

        # EN: download in small batches (one request per batch) | ES: descargar por lotes
        for b0 in range(0, len(tab), args.batch):
            sub = tab[b0:b0 + args.batch]
            try:
                sps = SDSS.get_spectra(matches=sub, data_release=args.dr)
            except Exception as e:
                print(f"  [batch {b0}] download error: {str(e)[:80]}")
                continue
            if sps is None:
                continue
            for row, hdu in zip(sub, sps):
                if hdu is None:
                    continue
                sid = int(row["specobjid"])
                fpath = os.path.join(outdir, f"sdss_{c}_{sid}_4000_5000A.csv")
                if os.path.exists(fpath):
                    saved += 1
                    continue
                try:
                    wave, flux, ivar, z = spectrum_from_hdulist(hdu)
                    sdf = pd.DataFrame({
                        "wavelength": wave, "flux": flux, "ivar": ivar,
                        "redshift": z, "specobjid": sid, "clase_aprox": c,
                        "teff": float(row["teff"]), "logg": float(row["logg"]),
                        "feh": float(row["feh"]),
                    })
                    # EN: cut to the common pipeline range | ES: recortar al rango comun
                    sdf = sdf[(sdf["wavelength"] >= P.WMIN) & (sdf["wavelength"] <= P.WMAX)]
                    if len(sdf) == 0:
                        continue
                    sdf.to_csv(fpath, index=False)
                    saved += 1
                except Exception as e:
                    print(f"  [{sid}] parse error: {str(e)[:70]}")
            time.sleep(0.2)
        print(f"[{c}] saved {saved} spectra -> {outdir}")

    print(f"\nDone. SDSS has R ~ {SDSS_R:.0f} in 4000-5000 A, so:")
    print(f"  python 06_generate_large.py --n 25000 --out sim_50k_sdss.npz --jobs 24 --batch 16 --resolution {SDSS_R:.0f}")
    print(f"  python 07_train_large.py --data-npz sim_50k_sdss.npz --real {args.out} --norm iterative")


if __name__ == "__main__":
    main()
