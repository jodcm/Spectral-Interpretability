"""
ingest_lamost.py
================
EN: Convert LAMOST LRS spectra (FITS files that Cata downloads) into the SAME
    per-class CSV format used by the DESI pipeline, so the training/evaluation
    scripts work unchanged. Input: a folder with one subfolder per class
    (<in>/G/*.fits, <in>/K/*.fits, ...). Output: <out>/<class>/*.csv with columns
    wavelength, flux, clase_aprox (+ teff if present in the header).
ES: Convierte espectros LAMOST LRS (archivos FITS que descarga Cata) al MISMO
    formato CSV por clase que usa la pipeline DESI, para que los scripts de
    entrenamiento/evaluacion funcionen sin cambios. Entrada: una carpeta con una
    subcarpeta por clase (<in>/G/*.fits, <in>/K/*.fits, ...). Salida:
    <out>/<class>/*.csv con columnas wavelength, flux, clase_aprox (+ teff si esta).

Run / Uso:
    python ingest_lamost.py --in lamost_fits --out proyecto_desi/espectros_lamost

Then / Luego:
    train with LAMOST resolution:  broaden_R = 1800  (P.LAMOST_LRS_R)
    entrenar con la resolucion LAMOST:  broaden_R = 1800

NOTE / NOTA:
    LAMOST FITS layouts differ slightly between data releases. This reader tries the
    common LRS conventions; if a file is not read, print its HDU structure and adapt.
    Los formatos FITS de LAMOST varian entre releases. Este lector prueba las
    convenciones LRS comunes; si un archivo no se lee, imprime su estructura y adapta.
Requires / Requiere:  astropy
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

import project_lib as P


def read_lamost_fits(path):
    """EN: return (wavelength[A], flux, teff) from a LAMOST LRS FITS, trying the
        common layouts. ES: devuelve (wavelength[A], flux, teff) de un FITS LAMOST LRS."""
    from astropy.io import fits
    wave = flux = None
    teff = np.nan
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        teff = float(hdr.get("TEFF", hdr.get("Z_TEFF", np.nan)) or np.nan)
        data = hdul[0].data
        # EN: LAMOST LRS combined spectrum: HDU0 = 2D array, row0=flux, row2=wavelength
        # ES: espectro combinado LAMOST LRS: HDU0 = array 2D, fila0=flux, fila2=wavelength
        if data is not None and getattr(data, "ndim", 0) == 2 and data.shape[0] >= 3:
            flux = np.asarray(data[0], dtype=float)
            wave = np.asarray(data[2], dtype=float)
        # EN: fallback 1: 1D flux + WCS in the header (linear or log10)
        # ES: alternativa 1: flux 1D + WCS en el header (lineal o log10)
        elif data is not None and getattr(data, "ndim", 0) == 1:
            flux = np.asarray(data, dtype=float)
            n = len(flux)
            crval1 = hdr.get("CRVAL1")
            cdelt1 = hdr.get("CD1_1", hdr.get("CDELT1"))
            if crval1 is not None and cdelt1 is not None:
                w = float(crval1) + float(cdelt1) * np.arange(n)
                is_log = str(hdr.get("CTYPE1", "")).upper().startswith("LOG") or np.nanmax(w) < 10
                wave = 10.0 ** w if is_log else w
        # EN: fallback 2: a binary table with WAVELENGTH / FLUX columns
        # ES: alternativa 2: una tabla binaria con columnas WAVELENGTH / FLUX
        if wave is None:
            for h in hdul[1:]:
                names = [c.upper() for c in getattr(getattr(h, "columns", None), "names", []) or []]
                if "WAVELENGTH" in names and "FLUX" in names:
                    wave = np.asarray(h.data["WAVELENGTH"], dtype=float).ravel()
                    flux = np.asarray(h.data["FLUX"], dtype=float).ravel()
                    break
    return wave, flux, teff


def main():
    ap = argparse.ArgumentParser(description="Ingest LAMOST LRS FITS into per-class CSVs.")
    ap.add_argument("--in", dest="indir", required=True,
                    help="input folder with per-class subfolders of FITS (<in>/G, <in>/K, ...)")
    ap.add_argument("--out", default="proyecto_desi/espectros_lamost",
                    help="output folder (one subfolder per class)")
    args = ap.parse_args()

    classes = [d for d in sorted(os.listdir(args.indir))
               if os.path.isdir(os.path.join(args.indir, d))]
    if not classes:
        raise SystemExit(f"[ERROR] No per-class subfolders in '{args.indir}' (expected G/, K/, ...)")

    for c in classes:
        files = []
        for pat in ("*.fits", "*.fits.gz", "*.fit", "*.fit.gz"):
            files += glob.glob(os.path.join(args.indir, c, pat))
        outdir = os.path.join(args.out, c)
        os.makedirs(outdir, exist_ok=True)
        saved = 0
        for f in sorted(files):
            try:
                wave, flux, teff = read_lamost_fits(f)
            except Exception as e:
                print(f"  [{os.path.basename(f)}] read error: {str(e)[:70]}")
                continue
            if wave is None or flux is None or len(wave) == 0:
                print(f"  [{os.path.basename(f)}] could not extract wavelength/flux -> check FITS layout")
                continue
            sdf = pd.DataFrame({"wavelength": np.asarray(wave, float),
                                "flux": np.asarray(flux, float),
                                "clase_aprox": c, "teff": teff})
            sdf = sdf[(sdf["wavelength"] >= P.WMIN) & (sdf["wavelength"] <= P.WMAX)]
            if len(sdf) == 0:
                continue
            base = os.path.splitext(os.path.basename(f))[0]
            sdf.to_csv(os.path.join(outdir, f"lamost_{c}_{base}.csv"), index=False)
            saved += 1
        print(f"[{c}] {len(files)} FITS -> {saved} CSV in {outdir}")

    print(f"\nDone. Train with LAMOST resolution, e.g. in 03_broadening.py set broaden_R=1800,")
    print(f"then:  python 02_evaluate_real_desi.py {args.out}")


if __name__ == "__main__":
    main()
