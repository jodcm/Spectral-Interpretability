@echo off
REM ============================================================================
REM setup_env.bat - Spectral Interpretability environment on a WINDOWS VM
REM EN: Creates the `astro-jax` conda env, installs deps, caches the
REM     TransformerPayne weights, and runs a tiny generation+train smoke test.
REM ES: Crea el env conda `astro-jax`, instala dependencias, cachea los pesos de
REM     TransformerPayne y corre una prueba pequena de generacion+entrenamiento.
REM
REM Usage / Uso (in the "Anaconda Prompt", inside the repo folder):
REM     cd C:\path\to\Spectral-Interpretability
REM     setup_env.bat
REM ============================================================================
setlocal enabledelayedexpansion

set ENV_NAME=astro-jax
set PY_VER=3.10

REM ---------------------------------------------------------------------------
REM 0) EN: conda must be on PATH -> run this from the "Anaconda Prompt".
REM    ES: conda debe estar en el PATH -> ejecuta esto desde el "Anaconda Prompt".
REM ---------------------------------------------------------------------------
where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] conda not found. Open the "Anaconda Prompt" ^(Start menu^) and run this from there.
  exit /b 1
)

REM ---------------------------------------------------------------------------
REM 1) EN: Create the env if it does not exist | ES: Crea el env si no existe
REM ---------------------------------------------------------------------------
conda env list | findstr /b /c:"%ENV_NAME% " >nul 2>nul
if errorlevel 1 (
  echo [info] creating env %ENV_NAME% ^(python %PY_VER%^)...
  call conda create -n %ENV_NAME% python=%PY_VER% -y
) else (
  echo [info] env %ENV_NAME% already exists - reusing it.
)

call conda activate %ENV_NAME%
if errorlevel 1 (
  echo [ERROR] could not activate %ENV_NAME%. Run "conda init cmd.exe" once, reopen the prompt, and retry.
  exit /b 1
)

REM ---------------------------------------------------------------------------
REM 2) EN: Install dependencies (CPU JAX) | ES: Instala dependencias (JAX CPU)
REM ---------------------------------------------------------------------------
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] pip install failed. Check the messages above.
  exit /b 1
)

REM ---------------------------------------------------------------------------
REM 3) EN: Cache the TransformerPayne weights ONCE (so the parallel workers load
REM       them from disk instead of each re-downloading).
REM    ES: Cachea los pesos de TransformerPayne UNA vez.
REM ---------------------------------------------------------------------------
python -c "import transformer_payne as tp; tp.TransformerPayne.download(); print('TransformerPayne weights cached OK')"
if errorlevel 1 (
  echo [ERROR] could not download TransformerPayne weights. Check internet access.
  exit /b 1
)

REM ---------------------------------------------------------------------------
REM 4) EN: Smoke test - small parallel generation + train.
REM    ES: Prueba - generacion paralela chica + entrenamiento.
REM ---------------------------------------------------------------------------
echo [info] detected %NUMBER_OF_PROCESSORS% CPU cores
python 06_generate_large.py --n 500 --out sim_1k.npz --jobs 8
if errorlevel 1 (
  echo [ERROR] generation smoke test failed.
  exit /b 1
)

if exist "proyecto_desi\espectros_balanceados_desi" (
  python 07_train_large.py --data-npz sim_1k.npz --real proyecto_desi\espectros_balanceados_desi --norm iterative
) else (
  echo [warn] proyecto_desi\espectros_balanceados_desi not found - skipping real eval in the smoke test.
  python 07_train_large.py --data-npz sim_1k.npz --real nonexistent --norm iterative
)

echo.
echo === setup done ===
echo Full 100k run:  python 06_generate_large.py --n 50000 --out sim_100k.npz --jobs %NUMBER_OF_PROCESSORS% --batch 500
echo Then:           python 07_train_large.py --data-npz sim_100k.npz --real proyecto_desi\espectros_balanceados_desi --norm iterative

endlocal
