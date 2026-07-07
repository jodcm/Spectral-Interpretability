from dl import queryClient as qc
import pandas as pd

N_POR_CLASE = 1000
LIMITE_QUERY = 2000

# Rangos aproximados por temperatura efectiva
# Ojo: esto es clasificación aproximada por Teff, no clasificación espectral oficial visual.
CLASES = {
    "O": (30000, 100000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}

def construir_query(clase, tmin, tmax):
    return f"""
    SELECT
        z.targetid,
        z.survey,
        z.program,
        z.spectype,
        z.zwarn,
        z.zcat_primary,

        g.source_id,
        g.ra AS gaia_ra,
        g.dec AS gaia_dec,
        g.parallax,
        g.parallax_error,
        g.pmra,
        g.pmdec,
        g.phot_g_mean_mag,
        g.phot_bp_mean_mag,
        g.phot_rp_mean_mag,

        m.param[1] AS feh,
        m.param[2] AS alphafe,
        m.param[3] AS log10micro,
        m.param[4] AS teff,
        m.param[5] AS logg

    FROM desi_dr1.x1p5__zpix__gaia_dr3__gaia_source AS x
    JOIN desi_dr1.zpix AS z
        ON x.id1 = z.id
    JOIN gaia_dr3.gaia_source AS g
        ON x.id2 = g.source_id
    JOIN desi_dr1.mws AS m
        ON z.targetid = m.targetid

    WHERE z.spectype = 'STAR'
      AND z.zwarn = 0
      AND z.zcat_primary = true

      -- Filtros Gaia: fuente con paralaje/movimiento propio confiable
      AND g.parallax IS NOT NULL
      AND g.parallax_error IS NOT NULL
      AND g.parallax > 0
      AND g.parallax_error > 0
      AND g.parallax/g.parallax_error > 3.0
      AND g.pmra IS NOT NULL
      AND g.pmdec IS NOT NULL

      -- Rango de temperatura de la clase
      AND m.param[4] >= {tmin}
      AND m.param[4] < {tmax}

    ORDER BY g.parallax/g.parallax_error DESC
    LIMIT {LIMITE_QUERY}
    """

todas = []

for clase, (tmin, tmax) in CLASES.items():
    print(f"\nBuscando clase {clase}: Teff entre {tmin} y {tmax} K")

    query = construir_query(clase, tmin, tmax)
    df = qc.query(sql=query, fmt="pandas")

    df = df.drop_duplicates(subset="targetid")
    df["clase_aprox"] = clase
    df["parallax_snr"] = df["parallax"] / df["parallax_error"]

    df = df.sort_values("parallax_snr", ascending=False)

    if len(df) >= N_POR_CLASE:
        df = df.head(N_POR_CLASE)
        print(f"OK: seleccionadas {len(df)} estrellas {clase}")
    else:
        print(f"ADVERTENCIA: solo se encontraron {len(df)} estrellas {clase}")

    todas.append(df)

muestra = pd.concat(todas, ignore_index=True)

# Guardar muestra completa
muestra.to_csv("muestra_balanceada_desi_1000_por_clase.csv", index=False)

# Guardar una tabla resumen
resumen = muestra["clase_aprox"].value_counts().sort_index()
resumen.to_csv("resumen_muestra_balanceada.csv")

print("\nResumen final:")
print(resumen)

print("\nGuardado:")
print("muestra_balanceada_desi_1000_por_clase.csv")
print("resumen_muestra_balanceada.csv")
