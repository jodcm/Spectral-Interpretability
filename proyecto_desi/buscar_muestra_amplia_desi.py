from dl import queryClient as qc
import pandas as pd
import numpy as np
from io import StringIO

N_OBJETOS = 10000

def float_converter(x):
    try:
        return np.array(x[1:-1].split(), dtype=float)
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

query = f"""
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

    m.param

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

  AND g.parallax IS NOT NULL
  AND g.parallax_error IS NOT NULL
  AND g.parallax > 0.5
  AND g.parallax_error > 0
  AND g.parallax/g.parallax_error > 3.0

  AND g.pmra IS NOT NULL
  AND g.pmdec IS NOT NULL

LIMIT {N_OBJETOS}
"""

print(f"Consultando muestra amplia de {N_OBJETOS} estrellas DESI + Gaia + MWS...")
res = qc.query(sql=query)

stars = pd.read_csv(StringIO(res), converters={"param": float_converter})
stars = stars.drop_duplicates(subset="targetid")

# DESI MWS:
# param = [Fe/H], [a/Fe], log10micro, Teff, logg
stars["feh"] = stars["param"].apply(lambda x: x[0])
stars["alphafe"] = stars["param"].apply(lambda x: x[1])
stars["log10micro"] = stars["param"].apply(lambda x: x[2])
stars["teff"] = stars["param"].apply(lambda x: x[3])
stars["logg"] = stars["param"].apply(lambda x: x[4])

def tipo_espectral(teff):
    if pd.isna(teff):
        return "sin_teff"
    elif teff >= 30000:
        return "O"
    elif 10000 <= teff < 30000:
        return "B"
    elif 7500 <= teff < 10000:
        return "A"
    elif 6000 <= teff < 7500:
        return "F"
    elif 5200 <= teff < 6000:
        return "G"
    elif 3700 <= teff < 5200:
        return "K"
    elif 2400 <= teff < 3700:
        return "M"
    else:
        return "fuera_rango"

stars["tipo_espectral_aprox"] = stars["teff"].apply(tipo_espectral)
stars["parallax_snr"] = stars["parallax"] / stars["parallax_error"]

stars = stars.sort_values("parallax_snr", ascending=False)

print("\nPrimeras estrellas:")
print(stars[[
    "targetid", "teff", "logg", "feh",
    "parallax", "parallax_error", "parallax_snr",
    "tipo_espectral_aprox"
]].head(20))

print("\nConteo por tipo espectral aproximado:")
print(stars["tipo_espectral_aprox"].value_counts())

stars.to_csv("muestra_amplia_desi_gaia.csv", index=False)
print("\nGuardado: muestra_amplia_desi_gaia.csv")
