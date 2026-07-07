from sparcl.client import SparclClient
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

CATALOGO = "muestra_balanceada_desi_1000_por_clase.csv"

# Cantidad máxima por clase.
# Si quieres probar primero, cambia a 5 o 10.
# Para tu muestra final, déjalo en 1000.
MAX_POR_CLASE = 1000

# Rango compatible con el paper de TransformerPayne
WMIN = 4000
WMAX = 5000

client = SparclClient(connect_timeout=30, read_timeout=5400)

df = pd.read_csv(CATALOGO)

# Asegurar que existe la columna de clase
if "clase_aprox" not in df.columns:
    raise ValueError("El catálogo no tiene columna 'clase_aprox'.")

# Seleccionar máximo 1000 por clase
df_sel = (
    df.groupby("clase_aprox", group_keys=False)
      .head(MAX_POR_CLASE)
      .reset_index(drop=True)
)

print("Espectros a descargar por clase:")
print(df_sel["clase_aprox"].value_counts().sort_index())

os.makedirs("espectros_balanceados_desi", exist_ok=True)
os.makedirs("figuras_espectros_desi", exist_ok=True)

errores = []

for i, row in df_sel.iterrows():
    targetid = int(row["targetid"])
    clase = row["clase_aprox"]

    carpeta_clase = f"espectros_balanceados_desi/{clase}"
    os.makedirs(carpeta_clase, exist_ok=True)

    archivo_csv = f"{carpeta_clase}/desi_{clase}_{targetid}_4000_5000A.csv"

    # Si ya existe, lo saltamos
    if os.path.exists(archivo_csv):
        print(f"[{i+1}/{len(df_sel)}] Ya existe {targetid}, saltando...")
        continue

    print(f"[{i+1}/{len(df_sel)}] Descargando {clase} TARGETID {targetid}...")

    try:
        spectra = client.retrieve_by_specid(
            [targetid],
            include=["wavelength", "flux", "ivar", "model"]
        )

        if len(spectra.records) == 0:
            print(f"  No se encontró espectro para {targetid}")
            errores.append((targetid, clase, "sin records"))
            continue

        spec = spectra.records[0]

        wave = spec.wavelength
        flux = spec.flux
        ivar = spec.ivar
        model = spec.model

        spec_df = pd.DataFrame({
            "wavelength": wave,
            "flux": flux,
            "ivar": ivar,
            "model": model,
            "targetid": targetid,
            "clase_aprox": clase,
            "teff": row.get("teff", None),
            "logg": row.get("logg", None),
            "feh": row.get("feh", None)
        })

        # Recorte al rango usado por TransformerPayne
        spec_df = spec_df[
            (spec_df["wavelength"] >= WMIN) &
            (spec_df["wavelength"] <= WMAX)
        ].copy()

        if len(spec_df) == 0:
            print(f"  Espectro vacío en 4000-5000 Å para {targetid}")
            errores.append((targetid, clase, "sin datos en rango"))
            continue

        spec_df.to_csv(archivo_csv, index=False)

        # Guardar solo algunas figuras para no llenar la carpeta
        n_figs_clase = len([
            f for f in os.listdir("figuras_espectros_desi")
            if f.startswith(f"desi_{clase}_")
        ])

        if n_figs_clase < 5:
            archivo_png = f"figuras_espectros_desi/desi_{clase}_{targetid}_4000_5000A.png"

            plt.figure(figsize=(12, 4))
            plt.plot(spec_df["wavelength"], spec_df["flux"], label="Flujo DESI")
            plt.plot(spec_df["wavelength"], spec_df["model"], label="Modelo DESI", alpha=0.8)
            plt.xlabel("Longitud de onda [Å]")
            plt.ylabel("Flujo")
            plt.title(f"DESI {clase} - TARGETID {targetid}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(archivo_png, dpi=200)
            plt.close()

        time.sleep(0.1)

    except Exception as e:
        print(f"  Error con {targetid}: {e}")
        errores.append((targetid, clase, str(e)))

# Guardar errores
if errores:
    err_df = pd.DataFrame(errores, columns=["targetid", "clase_aprox", "error"])
    err_df.to_csv("errores_descarga_espectros.csv", index=False)
    print("\nHubo errores. Guardado: errores_descarga_espectros.csv")

print("\nDescarga terminada.")

# Resumen final de archivos descargados
conteo = {}
for clase in sorted(df_sel["clase_aprox"].unique()):
    carpeta = f"espectros_balanceados_desi/{clase}"
    if os.path.exists(carpeta):
        conteo[clase] = len([f for f in os.listdir(carpeta) if f.endswith(".csv")])
    else:
        conteo[clase] = 0

print("\nEspectros descargados por clase:")
for clase, n in conteo.items():
    print(f"{clase}: {n}")
