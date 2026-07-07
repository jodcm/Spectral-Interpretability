from dl import queryClient as qc
import pandas as pd
import numpy as np
from io import StringIO

def float_converter(x):
    try:
        return np.array(x[1:-1].split(','), dtype=float)
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

query = """
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

  AND lower(z.survey) = 'main'
  AND lower(z.program) = 'bright'

  AND g.parallax IS NOT NULL
  AND g.parallax_error IS NOT NULL
  AND g.parallax > 1.0
  AND g.parallax_error > 0
  AND g.parallax/g.parallax_error > 5.0

  AND g.pmra IS NOT NULL
  AND g.pmdec IS NOT NULL

LIMIT 5000
"""

print("Consultando muestra DESI + Gaia + MWS...")
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

# Dominio compatible con el paper de TransformerPayne
clean = stars[
    (stars["teff"] >= 4000) & (stars["teff"] <= 6000) &
    (stars["logg"] >= 4.0) & (stars["logg"] <= 5.0) &
    (stars["feh"] >= -2.0) & (stars["feh"] <= 1.0)
].copy()

def tipo_fgk(teff):
    if 4000 <= teff < 5200:
        return "K"
    elif 5200 <= teff <= 6000:
        return "G"
    else:
        return "fuera_rango"

clean["tipo_fgk"] = clean["teff"].apply(tipo_fgk)
clean["parallax_snr"] = clean["parallax"] / clean["parallax_error"]

clean = clean.sort_values("parallax_snr", ascending=False)

print("\nMuestra limpia:")
print(clean[[
    "targetid", "teff", "logg", "feh",
    "parallax", "parallax_error", "parallax_snr", "tipo_fgk"
]].head(20))

print("\nConteo por tipo:")
print(clean["tipo_fgk"].value_counts())

clean.to_csv("muestra_pura_desi_mws_gaia.csv", index=False)
print("\nGuardado: muestra_pura_desi_mws_gaia.csv")
