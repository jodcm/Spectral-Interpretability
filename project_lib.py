"""
project_lib.py
==============
ES: Funciones compartidas para el proyecto "Spectral Interpretability" (AS4501).
EN: Shared functions for the "Spectral Interpretability" project (AS4501).

ES: Reune en un solo lugar las decisiones clave del milestone:
EN: Gathers the key milestone decisions in a single place:
  * ES: Grilla de longitudes de onda COMUN para datos simulados y reales (B4).
    EN: COMMON wavelength grid for simulated and real data (B4).
  * ES: Lista de lineas espectrales fisicamente discriminantes (B3).
    EN: List of physically discriminating spectral lines (B3).
  * ES: Generacion de espectros simulados balanceados con ruido unificado (A2/B2).
    EN: Balanced simulated-spectra generation with unified noise (A2/B2).
  * ES: Normalizacion de continuo identica para sim y real (transferencia del RF).
    EN: Identical continuum normalization for sim and real (RF transfer).
  * ES: Descarga, limpieza y remuestreo de espectros reales DESI a la grilla comun.
    EN: Download, cleaning and resampling of real DESI spectra onto the common grid.
  * ES: Entrenamiento del Random Forest y utilidades de interpretacion.
    EN: Random Forest training and interpretation utilities.

ES: Las longitudes de onda estan en Angstrom y en VACIO (como TransformerPayne y DESI).
EN: Wavelengths are in Angstrom and in VACUUM (like TransformerPayne and DESI).
"""

import numpy as np
import pandas as pd
# NOTE: shared library for the Spectral Interpretability project (see README).

# --------------------------------------------------------------------------
# B4 - ES: Grilla espectral comun (sim + real) | EN: Common spectral grid
# --------------------------------------------------------------------------
# ES: Rango 4000-5000 A: es el rango VALIDO del paper de TransformerPayne y el
#     rango en que el equipo (Cata) descargo los espectros DESI reales, para que
#     sim y real coincidan. Muestreo ~1 A/pixel.
# EN: Range 4000-5000 A: the VALID range of the TransformerPayne paper and the
#     range in which the team (Cata) downloaded the real DESI spectra, so sim and
#     real match. Sampling ~1 A/pixel.
WMIN, WMAX = 4000.0, 5000.0
NWV = 1000
WAVE_GRID = np.linspace(WMIN, WMAX, NWV)

# --------------------------------------------------------------------------
# B3 - ES: Lineas para separar F/G/K (vacio, A) | EN: Lines to separate F/G/K
# --------------------------------------------------------------------------
# ES/EN: name -> (central_wavelength, physical note ES | EN)
# ES: Solo lineas dentro de 4000-5000 A (Ca II H&K, Mg b, Na D y H-alpha quedan
#     fuera de este rango). Sirven para separar G de K: Balmer mas fuerte en G,
#     lineas metalicas / banda CH mas fuertes en K.
# EN: Only lines within 4000-5000 A (Ca II H&K, Mg b, Na D and H-alpha fall
#     outside). They separate G from K: Balmer stronger in G, metal lines / CH
#     band stronger in K.
SPECTRAL_LINES = {
    "H-delta":   (4102.9, "Balmer, mas fuerte en G | Balmer, stronger in G"),
    "Ca I":      (4226.7, "metalica, fuerte en K | metal line, strong in K"),
    "G-band CH": (4305.0, "banda CH ~4300, fuerte en K | CH band ~4300, strong in K"),
    "H-gamma":   (4341.7, "Balmer, mas fuerte en G | Balmer, stronger in G"),
    "H-beta":    (4862.7, "Balmer, mas fuerte en G | Balmer, stronger in G"),
}

# --------------------------------------------------------------------------
# B2 - ES: Rangos de temperatura por tipo (K) | EN: Temperature ranges per type
# --------------------------------------------------------------------------
# ES: Los pesos por defecto valen para enanas G/K (~4000-6000 K); F (6100-7000 K)
#     queda parcialmente fuera de validez.
# EN: Default weights are valid for G/K dwarfs (~4000-6000 K); F (6100-7000 K)
#     is partially outside the valid range.
TEFF_RANGES = {
    "F": (6100.0, 7000.0),
    "G": (5300.0, 6000.0),
    "K": (4000.0, 5200.0),
}

LOGG_RANGE = (4.0, 5.0)          # ES: enanas | EN: dwarfs
ABUND_RANGE = (-1.0, 1.0)        # [X/H] dex
# ES: Elementos cuya abundancia se varia | EN: Elements whose abundance is varied
VARIED_ELEMENTS = ["Fe", "Na", "Mg", "C", "Ca", "K", "S", "N", "Mn", "Ti"]


# --------------------------------------------------------------------------
# ES: Normalizacion de continuo (igual para sim y real)
# EN: Continuum normalization (identical for sim and real)
# --------------------------------------------------------------------------
def continuum_normalize(flux, window=151, percentile=88):
    """ES: Normaliza dividiendo por una estimacion del continuo (envolvente
        superior con filtro de percentil alto), robusta a lineas y ruido. Se
        aplica IGUAL a sim y real para que el RF transfiera de sim a DESI.
    EN: Normalize by dividing by a continuum estimate (upper envelope via a
        high-percentile filter), robust to lines and noise. Applied IDENTICALLY
        to sim and real so the RF transfers from sim to DESI.
    """
    from scipy.ndimage import percentile_filter
    flux = np.asarray(flux, dtype=float)
    continuum = percentile_filter(flux, percentile, size=window, mode="nearest")
    continuum[continuum == 0] = np.nan
    return flux / continuum


def continuum_normalize_iter(flux, deg=4, niter=5, low_sigma=1.5, high_sigma=3.0):
    """ES: Normalizacion de continuo MEJORADA (siguiente paso del domain shift).
        Ajusta un polinomio de grado bajo al continuo con recorte sigma ASIMETRICO:
        en cada iteracion descarta los puntos muy por DEBAJO del ajuste (lineas de
        absorcion) y los outliers muy por encima (rayos cosmicos / emision), y
        reajusta. Asi el continuo sigue la envolvente superior de forma robusta y
        con la MISMA forma para sim y real, reduciendo el offset de continuo que
        el filtro de percentil dejaba entre ambos (la causa del gap sim->real que
        el ensanchamiento no cerraba). Reemplazo directo de `continuum_normalize`.
    EN: IMPROVED continuum normalization (next step against the domain shift). Fits
        a low-order polynomial continuum with ASYMMETRIC sigma clipping: each
        iteration rejects points far BELOW the fit (absorption lines) and outliers
        far above (cosmics / emission), then refits. The continuum thus tracks the
        upper envelope robustly and with the SAME shape for sim and real, reducing
        the continuum offset that the percentile filter left between them (the cause
        of the sim->real gap that broadening did not close). Drop-in replacement for
        `continuum_normalize` (same call signature `f(flux)`).
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    # ES: x escalado a [-1, 1] para estabilidad numerica del polyfit
    # EN: x scaled to [-1, 1] for numerical stability of polyfit
    x = np.linspace(-1.0, 1.0, n)
    mask = np.isfinite(flux)
    if mask.sum() <= deg + 1:
        return flux / np.nanmedian(flux[mask]) if mask.any() else flux
    for _ in range(niter):
        coef = np.polyfit(x[mask], flux[mask], deg)
        cont = np.polyval(coef, x)
        resid = flux - cont
        std = np.std(resid[mask])
        if std == 0:
            break
        # ES: recorte asimetrico (absorcion abajo, cosmicos arriba)
        # EN: asymmetric clip (absorption below, cosmics above)
        new_mask = np.isfinite(flux) & (resid > -low_sigma * std) & (resid < high_sigma * std)
        if new_mask.sum() <= deg + 1 or new_mask.sum() == mask.sum():
            mask = new_mask if new_mask.sum() > deg + 1 else mask
            break
        mask = new_mask
    coef = np.polyfit(x[mask], flux[mask], deg)
    cont = np.polyval(coef, x)
    cont[cont == 0] = np.nan
    return flux / cont


# --------------------------------------------------------------------------
# NEXT STEP 1 - ES: Ensanchamiento instrumental (LSF) | EN: Instrumental (LSF) broadening
# --------------------------------------------------------------------------
# ES: FWHM -> sigma para una gaussiana | EN: FWHM -> sigma for a Gaussian
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# ES: Resolucion representativa de DESI. R = lambda/FWHM ~ 2500 en el azul.
#     A 5000 A eso son ~2.0 A FWHM; usamos un FWHM fijo como aproximacion simple.
# EN: Representative DESI resolution. R = lambda/FWHM ~ 2500 in the blue.
#     At 5000 A that is ~2.0 A FWHM; we use a fixed FWHM as a simple approximation.
DESI_LSF_FWHM_A = 2.0


def gaussian_lsf_broaden(wave, flux, fwhm_A=DESI_LSF_FWHM_A):
    """ES: Convoluciona un espectro con una LSF gaussiana de ancho `fwhm_A` (en A)
        para imitar la resolucion instrumental (p.ej. DESI). Asume muestreo
        aprox. uniforme en `wave`. Conserva la anchura equivalente (solo reparte
        la linea): las lineas se vuelven mas anchas y menos profundas.
    EN: Convolve a spectrum with a Gaussian LSF of width `fwhm_A` (in A) to mimic
        the instrumental resolution (e.g. DESI). Assumes ~uniform sampling in
        `wave`. Conserves equivalent width (only spreads the line): lines become
        broader and shallower.
    """
    from scipy.ndimage import gaussian_filter1d
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    dlam = float(np.median(np.diff(wave)))
    sigma_pix = (fwhm_A * FWHM_TO_SIGMA) / dlam
    if sigma_pix <= 0:
        return flux
    return gaussian_filter1d(flux, sigma_pix, mode="nearest")


# --------------------------------------------------------------------------
# NEXT STEP 1b - ES: Resolucion DESI dependiente de lambda | EN: wavelength-dependent DESI R
# --------------------------------------------------------------------------
# ES: R = lambda/FWHM de DESI NO es constante: crece del azul al rojo. Tabla
#     aproximada (brazos azul+rojo, ~3800-6700 A) segun el instrumento DESI.
# EN: DESI's R = lambda/FWHM is NOT constant: it grows from blue to red. Approximate
#     table (blue+red arms, ~3800-6700 A) after the DESI instrument.
DESI_R_TABLE_WAVE = np.array([3800., 4500., 5000., 5500., 5930., 6200., 6700.])
DESI_R_TABLE_R = np.array([2000., 2400., 2700., 3000., 3200., 3400., 3800.])


def desi_resolution(wave):
    """ES: Poder resolutivo R(lambda) de DESI, interpolado de la tabla. Devuelve R
        para cada longitud de onda (crece del azul al rojo).
    EN: DESI resolving power R(lambda), interpolated from the table. Returns R for
        each wavelength (grows from blue to red).
    """
    wave = np.asarray(wave, dtype=float)
    return np.interp(wave, DESI_R_TABLE_WAVE, DESI_R_TABLE_R)


def broaden_to_resolution(wave, flux, R):
    """ES: Ensancha un espectro a un poder resolutivo R = lambda/FWHM que puede
        DEPENDER de lambda (DESI R varia). Metodo exacto por cambio de coordenada:
        en la variable u con du = R(lambda) d(ln lambda), una gaussiana de sigma
        CONSTANTE equivale a FWHM = lambda/R(lambda) en lambda. Asi una sola
        convolucion cubre R variable. Conserva la anchura equivalente.
    EN: Broaden a spectrum to a resolving power R = lambda/FWHM that may DEPEND on
        lambda (DESI R varies). Exact via a coordinate change: in the variable u
        with du = R(lambda) d(ln lambda), a CONSTANT-sigma Gaussian equals
        FWHM = lambda/R(lambda) in lambda. One convolution thus covers a variable
        R. Conserves equivalent width.

    ES/EN: `R` puede ser escalar, callable R(wave), o array alineado con `wave`.
    """
    from scipy.ndimage import gaussian_filter1d
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if callable(R):
        Rv = np.asarray(R(wave), dtype=float)
    elif np.ndim(R) == 0:
        Rv = np.full_like(wave, float(R))
    else:
        Rv = np.asarray(R, dtype=float)
    lnw = np.log(wave)
    # ES: u = integral acumulada de R d(ln lambda) | EN: u = cumulative integral
    u = np.concatenate([[0.0], np.cumsum(0.5 * (Rv[1:] + Rv[:-1]) * np.diff(lnw))])
    n = len(wave)
    u_uni = np.linspace(u[0], u[-1], n)
    flux_u = np.interp(u_uni, u, flux)
    du = (u_uni[-1] - u_uni[0]) / (n - 1)
    sigma_pix = FWHM_TO_SIGMA / du
    smoothed = gaussian_filter1d(flux_u, sigma_pix, mode="nearest")
    return np.interp(u, u_uni, smoothed)


# --------------------------------------------------------------------------
# A2 / B2 - ES: Generacion de espectros simulados | EN: Simulated-spectra generation
# --------------------------------------------------------------------------
def generate_simulated(emulator, spectral_type, n=80, sigma_noise=0.02,
                       wave_grid=WAVE_GRID, mu=1.0, random_seed=0,
                       broaden_fwhm_A=None, broaden_R=None, oversample=6,
                       normalizer=None):
    """ES: Genera `n` espectros simulados de un tipo espectral en la grilla comun.
        Misma grilla para todas las clases (B4); ruido UNIFICADO en el espacio
        ya normalizado (mismo nivel para F/G/K, evita fuga por el ruido); mismo
        normalizador de continuo que los datos reales.
    EN: Generate `n` simulated spectra of one spectral type on the common grid.
        Same grid for all classes (B4); UNIFIED noise in the already-normalized
        space (same level for F/G/K, avoids noise leakage); same continuum
        normalizer as the real data.

    NEXT STEP 1 - ES: Si `broaden_fwhm_A` no es None, el espectro se genera en una
        grilla FINA (oversample x mas densa), se convoluciona con una LSF gaussiana
        de ese FWHM (imita la resolucion DESI) y se remuestrea a `wave_grid`. Asi
        las lineas simuladas dejan de ser mas profundas/agudas que las reales
        (mitiga el domain shift sim->real que empujaba todo a F).
    EN: If `broaden_fwhm_A` is not None, the spectrum is generated on a FINE grid
        (oversample x denser), convolved with a Gaussian LSF of that FWHM (mimics
        the DESI resolution) and resampled to `wave_grid`. This stops the simulated
        lines from being deeper/sharper than the real ones (mitigates the sim->real
        domain shift that pushed everything to F).

    ES: Devuelve un DataFrame (spectral_type, normalized_intensity, logteff, logg,
        abundances). | EN: Returns a DataFrame with those columns.
    """
    import random
    rng = random.Random(random_seed)
    nprng = np.random.RandomState(random_seed)
    t_min, t_max = TEFF_RANGES[spectral_type]
    # ES: normalizador de continuo (por defecto el de percentil, para no cambiar
    #     resultados previos); se aplica IGUAL a sim y real | EN: continuum
    #     normalizer (default = percentile, to preserve previous results); applied
    #     IDENTICALLY to sim and real.
    norm_fn = normalizer or continuum_normalize

    # NEXT STEP 1b - ES: R(lambda) dependiente de lambda (acepta "desi")
    #                EN: wavelength-dependent R(lambda) (accepts "desi")
    if isinstance(broaden_R, str) and broaden_R.lower() == "desi":
        broaden_R = desi_resolution
    do_broaden = (broaden_fwhm_A is not None) or (broaden_R is not None)
    # ES: grilla de generacion (fina si hay ensanchamiento) | EN: generation grid
    if do_broaden:
        gen_wave = np.linspace(wave_grid[0], wave_grid[-1],
                               len(wave_grid) * int(oversample))
    else:
        gen_wave = wave_grid
    log_gen_wave = np.log10(gen_wave)

    rows = []
    for _ in range(n):
        teff = rng.uniform(t_min, t_max)
        logg = rng.uniform(*LOGG_RANGE)
        abundances = {el: rng.uniform(*ABUND_RANGE) for el in VARIED_ELEMENTS}
        params = {"logteff": np.log10(teff), "logg": logg, **abundances}
        spectrum = emulator(log_gen_wave, mu, emulator.to_parameters(params))
        intensity = np.asarray(spectrum[:, 0], dtype=float)
        # NEXT STEP 1 - ES: ensanchar en flujo ANTES de normalizar y remuestrear
        #               EN: broaden in flux BEFORE normalizing and resampling
        if broaden_R is not None:
            # ES: R(lambda) dependiente de lambda | EN: wavelength-dependent R(lambda)
            intensity = broaden_to_resolution(gen_wave, intensity, broaden_R)
            intensity = np.interp(wave_grid, gen_wave, intensity)
        elif broaden_fwhm_A is not None:
            # ES: FWHM fijo en A | EN: fixed FWHM in A
            intensity = gaussian_lsf_broaden(gen_wave, intensity, broaden_fwhm_A)
            intensity = np.interp(wave_grid, gen_wave, intensity)
        # ES: normalizar y luego anadir ruido | EN: normalize then add noise
        norm = norm_fn(intensity)
        norm = norm + nprng.normal(0.0, sigma_noise, size=norm.shape)
        rows.append({
            "spectral_type": spectral_type,
            "normalized_intensity": norm,
            "logteff": np.log10(teff),
            "logg": logg,
            "abundances": abundances,
        })
    return pd.DataFrame(rows)


def build_balanced_dataset(emulator, classes=("G", "K"), n_per_class=80,
                           sigma_noise=0.02, base_seed=100,
                           broaden_fwhm_A=None, broaden_R=None, oversample=6,
                           normalizer=None):
    """ES: Dataset balanceado (mismo n por clase) en la grilla comun.
    EN: Balanced dataset (same n per class) on the common grid.

    NEXT STEP 1 - ES/EN: pasa `broaden_fwhm_A` (FWHM fijo) o `broaden_R`
        (poder resolutivo dependiente de lambda, p.ej. "desi") para ensanchar.
    """
    dfs = []
    for i, c in enumerate(classes):
        dfs.append(generate_simulated(emulator, c, n=n_per_class,
                                      sigma_noise=sigma_noise,
                                      random_seed=base_seed + i,
                                      broaden_fwhm_A=broaden_fwhm_A,
                                      broaden_R=broaden_R,
                                      oversample=oversample,
                                      normalizer=normalizer))
    return pd.concat(dfs, ignore_index=True)


# --------------------------------------------------------------------------
# A2 / B2 / B5 - ES: Descarga y limpieza DESI (SPARCL) | EN: DESI download & cleaning
# --------------------------------------------------------------------------
def fetch_desi_stars(n=60, data_release="DESI-DR1", connect_timeout=30,
                     read_timeout=600, batch=20):
    """ES: Descarga `n` espectros estelares primarios de DESI (SPARCL).
    EN: Download `n` primary stellar spectra from DESI (SPARCL).

    ES/EN: -> list of dicts (wavelength, flux, ivar, mask, redshift).
    """
    from sparcl.client import SparclClient
    client = SparclClient(connect_timeout=connect_timeout, read_timeout=read_timeout)
    cons = {"data_release": [data_release], "spectype": ["STAR"], "specprimary": [True]}
    found = client.find(outfields=["sparcl_id", "specid", "redshift"],
                        constraints=cons, limit=n)
    ids = [r["sparcl_id"] for r in found.records]
    records = []
    for i in range(0, len(ids), batch):
        got = client.retrieve(uuid_list=ids[i:i + batch],
                              include=["sparcl_id", "wavelength", "flux",
                                       "ivar", "mask", "redshift", "spectype"])
        for rec in got.records:
            if rec.get("wavelength") is None:
                continue
            records.append({
                "wavelength": np.asarray(rec["wavelength"], dtype=float),
                "flux": np.asarray(rec["flux"], dtype=float),
                "ivar": np.asarray(rec["ivar"], dtype=float),
                "mask": np.asarray(rec["mask"], dtype=float),
                "redshift": float(rec.get("redshift", 0.0) or 0.0),
            })
    return records


def clean_desi_spectrum(rec, wave_grid=WAVE_GRID, normalizer=None):
    """ES: Limpia y remuestrea un espectro DESI a la grilla comun en reposo:
        1) a reposo lambda/(1+z); 2) enmascarar pixeles malos (mask!=0 o ivar<=0);
        3) interpolar NaN + normalizar continuo; 4) remuestrear a WAVE_GRID.
    EN: Clean and resample a DESI spectrum onto the common rest-frame grid:
        1) rest frame lambda/(1+z); 2) mask bad pixels (mask!=0 or ivar<=0);
        3) interpolate NaN + continuum-normalize; 4) resample to WAVE_GRID.

    ES/EN: -> normalized_intensity on WAVE_GRID, or None if range not covered.
    """
    wave = rec["wavelength"] / (1.0 + rec["redshift"])
    flux = rec["flux"].copy()
    bad = (rec["mask"] != 0) | (rec["ivar"] <= 0) | ~np.isfinite(flux)
    flux[bad] = np.nan
    # ES: cobertura suficiente | EN: sufficient coverage
    if np.nanmin(wave) > WMIN + 5 or np.nanmax(wave) < WMAX - 5:
        return None
    good = np.isfinite(flux)
    if good.sum() < 0.5 * len(flux):
        return None
    flux_interp = np.interp(wave, wave[good], flux[good])
    norm_fn = normalizer or continuum_normalize
    norm = norm_fn(flux_interp)
    return np.interp(wave_grid, wave, norm)


def build_desi_dataset(n=60, wave_grid=WAVE_GRID, **kwargs):
    """ES: Descarga, limpia y remuestrea un lote DESI. Devuelve (X, info).
    EN: Download, clean and resample a DESI batch. Returns (X, info)."""
    records = fetch_desi_stars(n=n, **kwargs)
    rows, redshifts = [], []
    for rec in records:
        norm = clean_desi_spectrum(rec, wave_grid=wave_grid)
        if norm is not None and np.all(np.isfinite(norm)):
            rows.append(norm)
            redshifts.append(rec["redshift"])
    X = np.vstack(rows) if rows else np.empty((0, len(wave_grid)))
    return X, {"redshift": np.array(redshifts), "n_raw": len(records), "n_clean": len(rows)}


def load_desi_csv(path, wave_col=None, flux_col=None, redshift=None,
                  ivar_col=None, mask_col=None, wave_grid=WAVE_GRID,
                  normalizer=None):
    """ES: Carga UN espectro real desde un CSV y lo lleva a la grilla comun.
        Para el formato del equipo (Cata): columnas wavelength, flux y
        opcionalmente ivar/mask/redshift. Detecta nombres y acepta `loglam`.
    EN: Load ONE real spectrum from a CSV and bring it onto the common grid.
        For the team format (Cata): columns wavelength, flux and optionally
        ivar/mask/redshift. Auto-detects names and accepts `loglam`.

    ES/EN: -> normalized_intensity on wave_grid (ready for model.predict([x])),
           or None if it does not cover the common range.
    """
    tab = pd.read_csv(path)
    low = {c.lower(): c for c in tab.columns}

    def pick(candidates, given):
        if given is not None:
            return given
        for k in candidates:
            if k in low:
                return low[k]
        return None

    wave_name = pick(["wavelength", "wave", "lambda", "loglam", "wl", "lam"], wave_col)
    flux_name = pick(["flux", "intensity", "f", "model", "spec"], flux_col)
    if wave_name is None or flux_name is None:
        raise ValueError(f"No wave/flux columns found in {list(tab.columns)}")

    wave = tab[wave_name].to_numpy(dtype=float)
    # ES: loglam (SDSS/DESI) -> lineal | EN: loglam (SDSS/DESI) -> linear
    if wave_name.lower() == "loglam" or np.nanmax(wave) < 10:
        wave = 10.0 ** wave
    flux = tab[flux_name].to_numpy(dtype=float)

    z_name = pick(["redshift", "z"], None)
    if redshift is None:
        redshift = float(tab[z_name].iloc[0]) if z_name is not None else 0.0

    ivar_name = pick(["ivar", "inverse_variance"], ivar_col)
    mask_name = pick(["mask", "and_mask", "bitmask"], mask_col)
    rec = {
        "wavelength": wave,
        "flux": flux,
        "ivar": tab[ivar_name].to_numpy(float) if ivar_name is not None else np.ones_like(flux),
        "mask": tab[mask_name].to_numpy(float) if mask_name is not None else np.zeros_like(flux),
        "redshift": float(redshift),
    }
    return clean_desi_spectrum(rec, wave_grid=wave_grid, normalizer=normalizer)


def export_sim2real(result, X_real, out_dir=".", tag="sim2real"):
    """ES: Exporta el resultado Sim->Real para la mezcla del equipo: modelo RF
        (joblib) + CSV con la prediccion por espectro real.
    EN: Export the Sim->Real result for the team merge: RF model (joblib) +
        CSV with the per-spectrum prediction.
    """
    import os
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"rf_{tag}_model.joblib")
    joblib.dump({"model": result["model"], "classes": result["classes"],
                 "wave_grid": result["wave_grid"]}, model_path)
    pred = result["model"].predict(X_real)
    labels = np.array(result["classes"])[pred]
    proba = result["model"].predict_proba(X_real)
    table = pd.DataFrame(proba, columns=[f"p_{c}" for c in result["classes"]])
    table.insert(0, "pred", labels)
    csv_path = os.path.join(out_dir, f"{tag}_desi_predictions.csv")
    table.to_csv(csv_path, index=False)
    return model_path, csv_path


# --------------------------------------------------------------------------
# A3 / B6 - ES: Random Forest e interpretacion | EN: Random Forest and interpretation
# --------------------------------------------------------------------------
def df_to_xy(df, classes, feature="normalized_intensity", wave_grid=WAVE_GRID):
    """ES: Convierte un DataFrame de espectros en (X, y) con columnas = longitud de onda.
    EN: Turn a spectra DataFrame into (X, y) with columns = wavelength."""
    from sklearn.preprocessing import LabelEncoder
    X = pd.DataFrame(np.vstack(df[feature].to_numpy()), columns=np.round(wave_grid, 2))
    encoder = LabelEncoder()
    encoder.classes_ = np.array(classes)
    y = encoder.transform(df["spectral_type"])
    return X, y, encoder


def train_rf(df, classes=("G", "K"), feature="normalized_intensity",
             wave_grid=WAVE_GRID, test_size=0.3, random_state=23,
             grid=None):
    """ES: Entrena un Random Forest con GridSearch + CV estratificada.
        Devuelve un dict con modelo, splits y predicciones de test.
    EN: Train a Random Forest with GridSearch + stratified CV.
        Returns a dict with model, splits and test predictions.
    """
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X, y, encoder = df_to_xy(df, classes, feature=feature, wave_grid=wave_grid)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    if grid is None:
        grid = {"n_estimators": [40, 80], "max_depth": [5, 10, None],
                "criterion": ["gini", "entropy"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    search = GridSearchCV(rf, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    return {
        "model": best, "encoder": encoder, "classes": list(classes),
        "X_train": X_train, "X_test": X_test, "y_train": y_train,
        "y_test": y_test, "y_pred": y_pred,
        "best_params": search.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=classes),
        "wave_grid": wave_grid,
    }


# --------------------------------------------------------------------------
# NEXT STEP 2 - ES: Etiquetas reales F/G/K y evaluacion | EN: Real F/G/K labels & eval
# --------------------------------------------------------------------------
# ES: Cortes de temperatura (K) coherentes con TEFF_RANGES (en los huecos entre
#     rangos): K < 5250 <= G < 6050 <= F. | EN: Temperature cuts consistent with
#     TEFF_RANGES (in the gaps between ranges): K < 5250 <= G < 6050 <= F.
TEFF_CUT_KG = 5250.0
TEFF_CUT_GF = 6050.0


def teff_to_class(teff):
    """ES: Mapea temperatura efectiva (K) a tipo espectral F/G/K, para construir
        etiquetas verdaderas a partir del Teff del VAC de estrellas de DESI (MWS)
        o del catalogo del equipo. Acepta escalar o array.
    EN: Map effective temperature (K) to spectral type F/G/K, to build ground-truth
        labels from the DESI stellar (MWS) Teff VAC or the team catalog. Accepts a
        scalar or an array.
    """
    teff = np.asarray(teff, dtype=float)
    out = np.where(teff < TEFF_CUT_KG, "K",
                   np.where(teff < TEFF_CUT_GF, "G", "F"))
    return out.item() if out.ndim == 0 else out


def evaluate_on_labeled(result, X_real, y_true, classes=None):
    """ES: Evalua el RF (entrenado en simulados) sobre datos reales CON etiquetas
        verdaderas -> accuracy real + matriz de confusion + reporte. `y_true` son
        etiquetas de texto ('F'/'G'/'K'). Usar cuando lleguen los datos etiquetados
        de Cata (K=1242, G=82) o el cross-match con el VAC de Teff.
    EN: Evaluate the RF (trained on simulated) on real data WITH ground-truth
        labels -> real accuracy + confusion matrix + report. `y_true` are text
        labels ('F'/'G'/'K'). Use once Cata's labeled data (K=1242, G=82) or the
        Teff-VAC cross-match is available.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 classification_report)
    model = result["model"] if isinstance(result, dict) else result
    classes = list(classes or (result["classes"] if isinstance(result, dict)
                               else ["F", "G", "K"]))
    y_true = np.asarray(y_true).astype(str)
    pred_idx = model.predict(X_real)
    y_pred = np.asarray(classes)[pred_idx]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes),
        "report": classification_report(y_true, y_pred, labels=classes,
                                        target_names=classes, zero_division=0),
        "classes": classes,
        "y_pred": y_pred,
    }


# --------------------------------------------------------------------------
# NEXT STEP 3 - ES: Cargar espectros DESI reales ETIQUETADOS (carpetas por clase)
#              EN: Load LABELLED real DESI spectra (per-class folders)
# --------------------------------------------------------------------------
def load_labeled_desi_folder(base_dir, classes=("G", "K"), n_per_class=None,
                             balanced=True, wave_grid=WAVE_GRID, seed=0,
                             normalizer=None, min_logg=None, max_logg=None,
                             teff_range=None):
    """ES: Carga espectros DESI reales ETIQUETADOS desde carpetas por clase
        (base_dir/<clase>/*.csv, formato del equipo con columnas wavelength/flux
        y teff/logg/feh). Cada espectro pasa por `load_desi_csv` (misma
        normalizacion y grilla que los simulados). Junto con `evaluate_on_labeled`
        da la ACCURACY REAL sobre DESI.
    EN: Load LABELLED real DESI spectra from per-class folders (base_dir/<class>/
        *.csv, the team format with wavelength/flux and teff/logg/feh). Each
        spectrum goes through `load_desi_csv` (same normalization and grid as the
        simulated data). Together with `evaluate_on_labeled` it gives the REAL
        accuracy on DESI.

    ES/EN: -> (X_real, y_true, per_class_counts).

    NEXT STEP 4 - EN: `min_logg` / `max_logg` / `teff_range` restrict the REAL sample to
        the region the emulator was actually trained on. This matters: the synthetic
        grid is DWARFS ONLY (LOGG_RANGE = 4.0-5.0), but the real DESI sample contains
        ~17% GIANTS (logg < 3.5) and K stars below the synthetic 4000 K floor. Those
        stars are out-of-distribution -> guaranteed errors. Filtering them out tests
        whether the "domain shift" is really a physics limit or just a SELECTION
        mismatch. (SDSS, which naturally contains no giants, already reaches ~0.95.)
    ES: `min_logg` / `max_logg` / `teff_range` restringen la muestra REAL a la region en
        la que el emulador fue realmente entrenado. Importa: la grilla sintetica es SOLO
        ENANAS (LOGG_RANGE = 4.0-5.0), pero la muestra DESI real trae ~17% de GIGANTES
        (logg < 3.5) y estrellas K por debajo del piso sintetico de 4000 K. Esas
        estrellas estan fuera de distribucion -> errores garantizados. Filtrarlas prueba
        si el "domain shift" es un limite fisico o solo un desajuste de SELECCION.
    """
    import os
    import glob
    import random
    rng = random.Random(seed)

    def passes_filter(path):
        """EN: read teff/logg from the CSV header row and apply the cuts.
        ES: lee teff/logg de la primera fila del CSV y aplica los cortes."""
        if min_logg is None and max_logg is None and teff_range is None:
            return True
        try:
            head = pd.read_csv(path, nrows=1)
        except Exception:
            return False
        low = {k.lower(): k for k in head.columns}
        if min_logg is not None or max_logg is not None:
            if "logg" not in low:
                return False
            g = float(head[low["logg"]].iloc[0])
            if not np.isfinite(g):
                return False
            if min_logg is not None and g < min_logg:
                return False
            if max_logg is not None and g > max_logg:
                return False
        if teff_range is not None:
            if "teff" not in low:
                return False
            t = float(head[low["teff"]].iloc[0])
            if not np.isfinite(t) or t < teff_range[0] or t > teff_range[1]:
                return False
        return True

    by_class = {}
    for c in classes:
        files = sorted(glob.glob(os.path.join(base_dir, str(c), "*.csv")))
        rng.shuffle(files)
        rows = []
        for f in files:
            if n_per_class is not None and len(rows) >= n_per_class:
                break
            if not passes_filter(f):
                continue
            try:
                x = load_desi_csv(f, wave_grid=wave_grid, normalizer=normalizer)
            except Exception:
                x = None
            if x is not None and np.all(np.isfinite(x)):
                rows.append(x)
        by_class[c] = rows
    # ES: balancear a la clase mas pequena | EN: balance to the smallest class
    if balanced and by_class:
        m = min(len(v) for v in by_class.values())
        by_class = {c: v[:m] for c, v in by_class.items()}
    X_rows, y = [], []
    for c in classes:
        for x in by_class.get(c, []):
            X_rows.append(x)
            y.append(c)
    X = np.vstack(X_rows) if X_rows else np.empty((0, len(wave_grid)))
    return X, np.array(y), {c: len(by_class.get(c, [])) for c in classes}
