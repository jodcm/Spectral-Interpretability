@echo off
REM ============================================================================
REM run_all.bat - runs the FULL analysis on the Windows VM, step by step.
REM EN: Executes every step of the study in the right order, skipping the
REM     expensive generation steps if their .npz already exists. Prints a clear
REM     header before each step and a summary of failures at the end.
REM ES: Ejecuta todos los pasos del estudio en orden, saltando la generacion
REM     costosa si el .npz ya existe. Imprime un encabezado por paso y un
REM     resumen de fallos al final.
REM
REM Usage / Uso (Anaconda Prompt, in the repo folder):
REM     conda activate astro-jax
REM     run_all.bat
REM
REM To save everything to a log file / para guardar todo en un log:
REM     run_all.bat > run_all_log.txt 2>&1
REM ============================================================================
setlocal enabledelayedexpansion

set DESI=proyecto_desi\espectros_balanceados_desi
set SDSS=proyecto_desi\espectros_sdss
set SIM_DESI=sim_10k_masked.npz
set SIM_SDSS=sim_10k_sdss_masked.npz
set NPC=5000
set JOBS=%NUMBER_OF_PROCESSORS%
set BATCH=16
set FAILED=

echo.
echo ############################################################
echo #  Spectral Interpretability - full run
echo #  started %DATE% %TIME%
echo #  cores detected: %JOBS%
echo ############################################################

REM ---------------------------------------------------------------- checks
python -c "import transformer_payne, jax, sklearn, shap" 2>nul
if errorlevel 1 (
  echo [ERROR] environment not ready. Run:  conda activate astro-jax
  echo         ^(and once:  pip install -r requirements.txt^)
  exit /b 1
)
if not exist "%DESI%" (
  echo [ERROR] real DESI folder not found: %DESI%
  exit /b 1
)
if not exist "%SDSS%" (
  echo [WARN] real SDSS folder not found: %SDSS%
  echo        Fetch it first:  python 00_fetch_sdss.py --classes G K --n 300
  echo        Steps involving SDSS will be skipped.
)

REM ================================================================ STEP 1
echo.
echo ============================================================
echo  STEP 1/9  Verify the H-beta fix (masked normalization)
echo            -^> the H-beta ratio DESI/SDSS must move 0.80 -^> 1.00
echo ============================================================
if exist "%SDSS%" (
  python 13_spectra_compare.py --a %DESI% --b %SDSS% --class G --teff 5500 5900 --norm masked
  if errorlevel 1 set FAILED=!FAILED! [1-hbeta-check]
) else (
  echo [skip] needs SDSS
)

REM ================================================================ STEP 2
echo.
echo ============================================================
echo  STEP 2/9  Generate synthetic spectra at DESI resolution
echo            (10k is plenty - the saturation curve plateaus at ~2000)
echo ============================================================
if exist "%SIM_DESI%" (
  echo [skip] %SIM_DESI% already exists. Delete it to regenerate.
) else (
  python 06_generate_large.py --n %NPC% --out %SIM_DESI% --jobs %JOBS% --batch %BATCH% --resolution desi --norm masked
  if errorlevel 1 set FAILED=!FAILED! [2-generate-desi]
)

REM ================================================================ STEP 3
echo.
echo ============================================================
echo  STEP 3/9  Generate synthetic spectra at SDSS resolution (R=2000)
echo ============================================================
if exist "%SIM_SDSS%" (
  echo [skip] %SIM_SDSS% already exists. Delete it to regenerate.
) else (
  python 06_generate_large.py --n %NPC% --out %SIM_SDSS% --jobs %JOBS% --batch %BATCH% --resolution 2000 --norm masked
  if errorlevel 1 set FAILED=!FAILED! [3-generate-sdss]
)

REM ================================================================ STEP 4
echo.
echo ============================================================
echo  STEP 4/9  SIM -^> REAL : train on synthetic, test on real DESI
echo ============================================================
python 07_train_large.py --data-npz %SIM_DESI% --real %DESI% --norm masked
if errorlevel 1 set FAILED=!FAILED! [4-sim2real-desi]

REM ================================================================ STEP 5
echo.
echo ============================================================
echo  STEP 5/9  SIM -^> REAL : train on synthetic, test on real SDSS
echo ============================================================
if exist "%SDSS%" (
  python 07_train_large.py --data-npz %SIM_SDSS% --real %SDSS% --norm masked
  if errorlevel 1 set FAILED=!FAILED! [5-sim2real-sdss]
) else (
  echo [skip] needs SDSS
)

REM ================================================================ STEP 6
echo.
echo ============================================================
echo  STEP 6/9  UPPER BOUND : train on REAL DESI, test on real DESI
echo            (G/K, then the F/G/K extra experiment)
echo ============================================================
python 14_train_real.py --real %DESI% --classes G K --norm masked
if errorlevel 1 set FAILED=!FAILED! [6a-real-GK]
python 14_train_real.py --real %DESI% --classes F G K --norm masked
if errorlevel 1 set FAILED=!FAILED! [6b-real-FGK]

REM ================================================================ STEP 7
echo.
echo ============================================================
echo  STEP 7/9  CROSS-SURVEY : train on real DESI -^> test on real SDSS
echo            (no simulation involved - do the instruments agree?)
echo ============================================================
if exist "%SDSS%" (
  python 14_train_real.py --real %DESI% --test-real %SDSS% --classes G K --norm masked
  if errorlevel 1 set FAILED=!FAILED! [7-cross-survey]
) else (
  echo [skip] needs SDSS
)

REM ================================================================ STEP 8
echo.
echo ============================================================
echo  STEP 8/9  THE REFERENCE SCALE : sim-trained vs the ceiling
echo            -^> figures/summary_scale_GK.png  (the key figure)
echo ============================================================
python 15_summary.py --sim-desi %SIM_DESI% --sim-sdss %SIM_SDSS% --desi %DESI% --sdss %SDSS% --classes G K --norm masked
if errorlevel 1 set FAILED=!FAILED! [8-summary]

REM ================================================================ STEP 9
echo.
echo ============================================================
echo  STEP 9/9  SHAP interpretability on the new model
echo ============================================================
if exist rf_large_balanceados_desi.joblib (
  python 08_shap.py --data-npz %SIM_DESI% --model rf_large_balanceados_desi.joblib
  if errorlevel 1 set FAILED=!FAILED! [9-shap]
) else (
  echo [skip] model not found - step 4 must succeed first.
)

REM ---------------------------------------------------------------- summary
echo.
echo ############################################################
echo #  DONE  %DATE% %TIME%
if "%FAILED%"=="" (
  echo #  All steps completed successfully.
) else (
  echo #  FAILED STEPS: %FAILED%
)
echo #
echo #  Key figures in figures\ :
echo #    summary_scale_GK.png            ^<- the reference scale (main result)
echo #    spectra_compare_*_G.png         ^<- the H-beta fix
echo #    real_confusion_*.png            ^<- sim-^>real confusion matrices
echo #    real_trained_*.png              ^<- upper bound
echo #    cross_survey_*.png              ^<- DESI -^> SDSS
echo #    shap_importance.png             ^<- interpretability
echo ############################################################

endlocal
