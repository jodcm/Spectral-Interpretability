@echo off
REM ============================================================================
REM run_all.bat - runs the FULL analysis on the Windows VM, step by step.
REM
REM EN: MAIN PATH uses the 'iterative' normalizer -- the BEST configuration
REM     (DESI 0.760, SDSS 0.953). The 'masked' normalizer (the H-beta fix) is run
REM     afterwards as a documented COMPARISON: it does repair H-beta (depth ratio
REM     DESI/SDSS 0.80 -> 0.94) but over-deepens every other line and makes DESI's
REM     sim->real WORSE (0.760 -> 0.637). Both summaries are produced so the two
REM     can be shown side by side.
REM ES: El CAMINO PRINCIPAL usa el normalizador 'iterative' -- la MEJOR configuracion.
REM     El 'masked' (el fix de H-beta) se corre despues como COMPARACION documentada:
REM     arregla H-beta pero profundiza de mas el resto y EMPEORA DESI.
REM
REM Usage / Uso (Anaconda Prompt, in the repo folder):
REM     conda activate astro-jax
REM     run_all.bat
REM     run_all.bat > run_all_log.txt 2>&1     ^<- to save a log
REM ============================================================================
setlocal enabledelayedexpansion

set DESI=proyecto_desi\espectros_balanceados_desi
set SDSS=proyecto_desi\espectros_sdss
REM --- main path: iterative (best) ---
set SIM_DESI=sim_10k_iter.npz
set SIM_SDSS=sim_10k_sdss_iter.npz
REM --- comparison: masked (H-beta fix) ---
set SIM_DESI_M=sim_10k_masked.npz
set SIM_SDSS_M=sim_10k_sdss_masked.npz

set NPC=5000
set JOBS=%NUMBER_OF_PROCESSORS%
set BATCH=16
set FAILED=

echo.
echo ############################################################
echo #  Spectral Interpretability - full run
echo #  started %DATE% %TIME%   ^| cores: %JOBS%
echo ############################################################

python -c "import transformer_payne, jax, sklearn, shap" 2>nul
if errorlevel 1 (
  echo [ERROR] environment not ready. Run:  conda activate astro-jax
  exit /b 1
)
if not exist "%DESI%" ( echo [ERROR] missing %DESI% & exit /b 1 )
if not exist "%SDSS%" (
  echo [WARN] missing %SDSS% - fetch with: python 00_fetch_sdss.py --classes G K --n 300
  echo        SDSS steps will be skipped.
)

REM ================================================================ STEP 1
echo.
echo ============================================================
echo  STEP 1/10  Generate synthetic - DESI resolution, ITERATIVE norm
echo ============================================================
if exist "%SIM_DESI%" ( echo [skip] %SIM_DESI% exists. ) else (
  python 06_generate_large.py --n %NPC% --out %SIM_DESI% --jobs %JOBS% --batch %BATCH% --resolution desi --norm iterative
  if errorlevel 1 set FAILED=!FAILED! [1]
)

REM ================================================================ STEP 2
echo.
echo ============================================================
echo  STEP 2/10  Generate synthetic - SDSS resolution (R=2000), ITERATIVE
echo ============================================================
if exist "%SIM_SDSS%" ( echo [skip] %SIM_SDSS% exists. ) else (
  python 06_generate_large.py --n %NPC% --out %SIM_SDSS% --jobs %JOBS% --batch %BATCH% --resolution 2000 --norm iterative
  if errorlevel 1 set FAILED=!FAILED! [2]
)

REM ================================================================ STEP 3
echo.
echo ============================================================
echo  STEP 3/10  SIM -^> REAL  : synthetic -^> real DESI   (iterative)
echo ============================================================
python 07_train_large.py --data-npz %SIM_DESI% --real %DESI% --norm iterative
if errorlevel 1 set FAILED=!FAILED! [3]

REM ================================================================ STEP 4
echo.
echo ============================================================
echo  STEP 4/10  SIM -^> REAL  : synthetic -^> real SDSS   (iterative)
echo ============================================================
if exist "%SDSS%" (
  python 07_train_large.py --data-npz %SIM_SDSS% --real %SDSS% --norm iterative
  if errorlevel 1 set FAILED=!FAILED! [4]
) else ( echo [skip] needs SDSS )

REM ================================================================ STEP 5
echo.
echo ============================================================
echo  STEP 5/10  UPPER BOUND : real-trained on DESI  (G/K, then F/G/K)
echo             -^> how well can ANY model do with this data?
echo ============================================================
python 14_train_real.py --real %DESI% --classes G K --norm iterative
if errorlevel 1 set FAILED=!FAILED! [5a]
python 14_train_real.py --real %DESI% --classes F G K --norm iterative
if errorlevel 1 set FAILED=!FAILED! [5b]

REM ================================================================ STEP 6
echo.
echo ============================================================
echo  STEP 6/10  UPPER BOUND : real-trained on SDSS  (G/K)
echo ============================================================
if exist "%SDSS%" (
  python 14_train_real.py --real %SDSS% --classes G K --norm iterative
  if errorlevel 1 set FAILED=!FAILED! [6]
) else ( echo [skip] needs SDSS )

REM ================================================================ STEP 7
echo.
echo ============================================================
echo  STEP 7/10  CROSS-SURVEY : real DESI -^> real SDSS  (no simulation)
echo ============================================================
if exist "%SDSS%" (
  python 14_train_real.py --real %DESI% --test-real %SDSS% --classes G K --norm iterative
  if errorlevel 1 set FAILED=!FAILED! [7]
) else ( echo [skip] needs SDSS )

REM ================================================================ STEP 8
echo.
echo ============================================================
echo  STEP 8/10  *** THE REFERENCE SCALE *** (iterative = best config)
echo             -^> figures\summary_scale_GK_iterative.png   MAIN FIGURE
echo ============================================================
python 15_summary.py --sim-desi %SIM_DESI% --sim-sdss %SIM_SDSS% --desi %DESI% --sdss %SDSS% --classes G K --norm iterative
if errorlevel 1 set FAILED=!FAILED! [8]

REM ================================================================ STEP 9
echo.
echo ============================================================
echo  STEP 9/10  COMPARISON: the 'masked' H-beta experiment
echo             -^> repairs H-beta but hurts DESI overall (negative result)
echo ============================================================
if exist "%SDSS%" (
  python 13_spectra_compare.py --a %DESI% --b %SDSS% --class G --teff 5500 5900 --norm iterative
  python 13_spectra_compare.py --a %DESI% --b %SDSS% --class G --teff 5500 5900 --norm masked
)
if exist "%SIM_DESI_M%" ( echo [skip] %SIM_DESI_M% exists. ) else (
  python 06_generate_large.py --n %NPC% --out %SIM_DESI_M% --jobs %JOBS% --batch %BATCH% --resolution desi --norm masked
)
if exist "%SIM_SDSS_M%" ( echo [skip] %SIM_SDSS_M% exists. ) else (
  python 06_generate_large.py --n %NPC% --out %SIM_SDSS_M% --jobs %JOBS% --batch %BATCH% --resolution 2000 --norm masked
)
python 15_summary.py --sim-desi %SIM_DESI_M% --sim-sdss %SIM_SDSS_M% --desi %DESI% --sdss %SDSS% --classes G K --norm masked
if errorlevel 1 set FAILED=!FAILED! [9]

REM ================================================================ STEP 10
echo.
echo ============================================================
echo  STEP 10/10  SHAP interpretability on the best model
echo ============================================================
if exist rf_large_balanceados_desi.joblib (
  python 08_shap.py --data-npz %SIM_DESI% --model rf_large_balanceados_desi.joblib
  if errorlevel 1 set FAILED=!FAILED! [10]
) else ( echo [skip] step 3 must succeed first. )

REM ---------------------------------------------------------------- summary
echo.
echo ############################################################
echo #  DONE  %DATE% %TIME%
if "%FAILED%"=="" ( echo #  All steps completed successfully. ) else ( echo #  FAILED: %FAILED% )
echo #
echo #  MAIN FIGURE : figures\summary_scale_GK_iterative.png
echo #  comparison  : figures\summary_scale_GK_masked.png
echo #  H-beta fix  : figures\spectra_compare_*_G.png
echo #  upper bound : figures\real_trained_*.png
echo #  cross-survey: figures\cross_survey_*.png
echo #  SHAP        : figures\shap_importance.png
echo ############################################################

endlocal
