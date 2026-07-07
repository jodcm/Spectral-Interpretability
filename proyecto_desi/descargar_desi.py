from sparcl.client import SparclClient
import matplotlib.pyplot as plt
import pandas as pd

client = SparclClient()

# Ejemplo oficial de un TARGETID DESI
targetid = 39628427912810449

spectra = client.retrieve_by_specid(
    [targetid],
    include=["wavelength", "flux", "ivar", "model", "wave_sigma"]
)

spec = spectra.records[0]

wave = spec.wavelength
flux = spec.flux
ivar = spec.ivar
model = spec.model

# Guardar datos como CSV
df = pd.DataFrame({
    "wavelength": wave,
    "flux": flux,
    "ivar": ivar,
    "model": model
})

df.to_csv(f"desi_spectrum_{targetid}.csv", index=False)

# Graficar y guardar imagen
plt.figure(figsize=(12, 4))
plt.plot(wave, flux, label="Flujo DESI")
plt.plot(wave, model, label="Modelo", alpha=0.8)
plt.xlabel("Longitud de onda [Angstrom]")
plt.ylabel("Flujo")
plt.title(f"Espectro DESI - TARGETID {targetid}")
plt.legend()
plt.tight_layout()
plt.savefig(f"desi_spectrum_{targetid}.png", dpi=200)

print("Descarga lista")
print(f"Archivo de datos: desi_spectrum_{targetid}.csv")
print(f"Imagen: desi_spectrum_{targetid}.png")
