#!/usr/bin/env bash
# ============================================================================
# setup_env.sh — Spectral Interpretability environment on a Linux VM
# EN: Creates the `astro-jax` conda env, installs deps, caches the TransformerPayne
#     weights, and runs a tiny generation+train smoke test to confirm the VM works.
# ES: Crea el env conda `astro-jax`, instala dependencias, cachea los pesos de
#     TransformerPayne y corre una prueba pequena de generacion+entrenamiento para
#     confirmar que la VM funciona.
#
# Usage / Uso:   bash setup_env.sh
# ============================================================================
set -e

ENV_NAME="astro-jax"
PY_VER="3.10"

# ---------------------------------------------------------------------------
# 0) EN: Make sure conda is available. If not, install Miniconda first (see notes
#       at the bottom). ES: Asegura que conda este disponible; si no, instala
#       Miniconda primero (ver notas al final).
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda first (see notes at the bottom of this script)."
  exit 1
fi
# EN: allow 'conda activate' inside this script | ES: permite 'conda activate' aqui
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1) EN: Create the env (Python 3.10) | ES: Crea el env (Python 3.10)
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[info] env '${ENV_NAME}' already exists — reusing it."
else
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# 2) EN: Install dependencies (CPU JAX) | ES: Instala dependencias (JAX CPU)
# ---------------------------------------------------------------------------
python -m pip install --upgrade pip
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 3) EN: Cache the TransformerPayne weights ONCE (so the parallel workers load
#       them from disk instead of each re-downloading).
#    ES: Cachea los pesos de TransformerPayne UNA vez (para que los workers
#       paralelos los carguen del disco en vez de descargar cada uno).
# ---------------------------------------------------------------------------
python - <<'PY'
import transformer_payne as tp
tp.TransformerPayne.download()
print("TransformerPayne weights cached OK")
PY

# ---------------------------------------------------------------------------
# 4) EN: Smoke test — small parallel generation + train. If this prints a real
#       accuracy, the VM is ready for the full 100k run.
#    ES: Prueba — generacion paralela chica + entrenamiento. Si imprime una
#       accuracy real, la VM esta lista para el run completo de 100k.
# ---------------------------------------------------------------------------
CORES="$(nproc)"
echo "[info] detected ${CORES} CPU cores"
python 06_generate_large.py --n 500 --out sim_1k.npz --jobs "$(( CORES > 8 ? 8 : CORES ))"
if [ -d "proyecto_desi/espectros_balanceados_desi" ]; then
  python 07_train_large.py --data-npz sim_1k.npz --real proyecto_desi/espectros_balanceados_desi --norm iterative
else
  echo "[warn] proyecto_desi/espectros_balanceados_desi not found — skipping real eval in the smoke test."
  python 07_train_large.py --data-npz sim_1k.npz --real /nonexistent --norm iterative
fi

echo ""
echo "=== setup done ==="
echo "Full 100k run:  python 06_generate_large.py --n 50000 --out sim_100k.npz --jobs ${CORES} --batch 500"
echo "Then:           python 07_train_large.py --data-npz sim_100k.npz --real proyecto_desi/espectros_balanceados_desi --norm iterative"

# ============================================================================
# NOTES / NOTAS
# ---------------------------------------------------------------------------
# EN: If conda is missing, install Miniconda (Linux x86_64):
# ES: Si falta conda, instala Miniconda (Linux x86_64):
#
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
#   conda init bash    # then open a new shell / luego abre una shell nueva
#
# EN: No conda? A plain venv works too:
# ES: Sin conda? Un venv normal tambien sirve:
#   python3.10 -m venv astro-jax && source astro-jax/bin/activate
#   pip install -r requirements.txt
# ============================================================================
