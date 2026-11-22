@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  train_case.cmd  --  centralized traffic-light policy trainer for ONE grid case
REM
REM  Builds/trains a CENTRALIZED (single-agent, global-view) model that controls
REM  every intersection's traffic light, then evaluates it against a fixed-time
REM  and a random baseline.  By default it runs for the 3x4 scenario
REM  (flow_3_4.json / roadnet_3_4.json).
REM
REM  This is the FIXED-ARG-PARSER version.  The original used
REM    if "!ARG!"=="-episodes" ( ... & set "EP=%~2" & shift & shift & goto parse )
REM  which silently dropped every value flag: %~2 is expanded at PARSE time of
REM  the parenthesized block, and `goto` inside an &-chain in parens is
REM  unreliable in CMD, so "-episodes 300 -seed 7" was reported as "unknown
REM  argument" and the defaults were used instead.  The parser below captures
REM  the value into !VAL! once per loop and dispatches each value flag to a
REM  shared :takeval + :shift2 routine, which is bulletproof.
REM
REM  Usage:
REM    train_case.cmd                            default: 3x4, 150 ep, 180s, ctrl 10s
REM    train_case.cmd -episodes 300              longer training            (NOW WORKS)
REM    train_case.cmd -duration 300 -control-interval 8
REM    train_case.cmd -rows 3 -cols 4            explicit (default) -> flow_3_4.json
REM    train_case.cmd -seed 42 -noopen           reproducible, no auto-open
REM    train_case.cmd -use-existing              skip gen; reuse flow_R_C.json in -outdir
REM    train_case.cmd -max-vehicles 1            single-car scenario
REM    train_case.cmd -entropy 0.02 -lr 0.008    tune the new optimizer
REM    train_case.cmd -h                         full help
REM
REM  Requirements (same folder):
REM    centralized_train.py   (the model / trainer)
REM    grid_sim_all.py        v3.0.1+  (only needed to generate flow_R_C.json)
REM    numpy                  (pip install numpy)
REM    matplotlib             (optional; for the train_curve.png plot)
REM ============================================================================

if /I "%~1"=="-h"      goto help
if /I "%~1"=="--help"  goto help
if /I "%~1"=="-help"   goto help
if /I "%~1"=="help"    goto help
if /I "%~1"=="/?"      goto help

REM --- Defaults ---------------------------------------------------------------
set "ROWS=3"
set "COLS=4"
set "X=150"
set "VEHICLES=4000"
set "GENDUR=300"
set "EPISODES=150"
set "DURATION=180"
set "CTRL=10"
set "DT=0.5"
set "SEED=0"
set "LR=0.005"
set "GAMMA=0.95"
set "EVALS=5"
set "MAXVEH=0"
set "ENTROPY=0.01"
set "GRADCLIP=5.0"
set "BETADECAY=0.97"
set "LRDECAY=0.99"
set "EVALEVERY=10"
set "PATIENCE=0"
set "OUTDIR=runs\c3_4"
set "NOOPEN=0"
set "VERBOSE=0"
set "USEEXISTING=0"

REM --- Parse arguments (robust: capture !VAL! once, dispatch via :takeval) ----
:parse
if "%~1"=="" goto parse_done
set "ARG=%~1"
set "VAL=%~2"

REM --- boolean flags (consume one token) ---
if /I "!ARG!"=="-noopen"        set "NOOPEN=1"
if /I "!ARG!"=="-noopen"        goto shift1
if /I "!ARG!"=="-verbose"       set "VERBOSE=1"
if /I "!ARG!"=="-verbose"       goto shift1
if /I "!ARG!"=="-use-existing"  set "USEEXISTING=1"
if /I "!ARG!"=="-use-existing"  goto shift1

REM --- value flags (consume two tokens: flag + value) ---
REM     Each: call :takeval VARNAME "!VAL!"  -> sets VAR or exits 2 on missing.
REM     Then jump to :shift2 which double-shifts and returns to :parse.
if /I "!ARG!"=="-rows"               call :takeval ROWS        "!VAL!"        || exit /b 2
if /I "!ARG!"=="-rows"               goto shift2
if /I "!ARG!"=="-cols"               call :takeval COLS        "!VAL!"        || exit /b 2
if /I "!ARG!"=="-cols"               goto shift2
if /I "!ARG!"=="-x"                  call :takeval X           "!VAL!"        || exit /b 2
if /I "!ARG!"=="-x"                  goto shift2
if /I "!ARG!"=="-vehicles"           call :takeval VEHICLES    "!VAL!"        || exit /b 2
if /I "!ARG!"=="-vehicles"           goto shift2
if /I "!ARG!"=="-gendur"             call :takeval GENDUR      "!VAL!"        || exit /b 2
if /I "!ARG!"=="-gendur"             goto shift2
if /I "!ARG!"=="-episodes"           call :takeval EPISODES    "!VAL!"        || exit /b 2
if /I "!ARG!"=="-episodes"           goto shift2
if /I "!ARG!"=="-duration"           call :takeval DURATION    "!VAL!"        || exit /b 2
if /I "!ARG!"=="-duration"           goto shift2
if /I "!ARG!"=="-control-interval"   call :takeval CTRL        "!VAL!"        || exit /b 2
if /I "!ARG!"=="-control-interval"   goto shift2
if /I "!ARG!"=="-dt"                 call :takeval DT          "!VAL!"        || exit /b 2
if /I "!ARG!"=="-dt"                 goto shift2
if /I "!ARG!"=="-seed"               call :takeval SEED        "!VAL!"        || exit /b 2
if /I "!ARG!"=="-seed"               goto shift2
if /I "!ARG!"=="-lr"                 call :takeval LR          "!VAL!"        || exit /b 2
if /I "!ARG!"=="-lr"                 goto shift2
if /I "!ARG!"=="-gamma"              call :takeval GAMMA       "!VAL!"        || exit /b 2
if /I "!ARG!"=="-gamma"              goto shift2
if /I "!ARG!"=="-eval-episodes"      call :takeval EVALS       "!VAL!"        || exit /b 2
if /I "!ARG!"=="-eval-episodes"      goto shift2
if /I "!ARG!"=="-max-vehicles"       call :takeval MAXVEH      "!VAL!"        || exit /b 2
if /I "!ARG!"=="-max-vehicles"       goto shift2
if /I "!ARG!"=="-entropy"            call :takeval ENTROPY     "!VAL!"        || exit /b 2
if /I "!ARG!"=="-entropy"            goto shift2
if /I "!ARG!"=="-grad-clip"          call :takeval GRADCLIP    "!VAL!"        || exit /b 2
if /I "!ARG!"=="-grad-clip"          goto shift2
if /I "!ARG!"=="-beta-decay"         call :takeval BETADECAY   "!VAL!"        || exit /b 2
if /I "!ARG!"=="-beta-decay"         goto shift2
if /I "!ARG!"=="-lr-decay"           call :takeval LRDECAY     "!VAL!"        || exit /b 2
if /I "!ARG!"=="-lr-decay"           goto shift2
if /I "!ARG!"=="-eval-every"         call :takeval EVALEVERY   "!VAL!"        || exit /b 2
if /I "!ARG!"=="-eval-every"         goto shift2
if /I "!ARG!"=="-patience"           call :takeval PATIENCE    "!VAL!"        || exit /b 2
if /I "!ARG!"=="-patience"           goto shift2
if /I "!ARG!"=="-outdir"             call :takeval OUTDIR      "!VAL!"        || exit /b 2
if /I "!ARG!"=="-outdir"             goto shift2

REM --- unknown: warn + consume one token ---
echo WARN: unknown argument '!ARG!' ^(ignored^). Use -h for help.
goto shift1

:shift1
shift
goto parse

:shift2
shift
shift
goto parse

:takeval
REM %1 = variable name, %2 = value (may be empty)
if "%~2"=="" (
    echo ERROR: !ARG! needs a value.
    exit /b 2
)
set "%~1=%~2"
exit /b 0
goto :eof

:parse_done

set "OUTDIR=%OUTDIR:/=\%"

REM --- Validate rows / cols against grid_sim_all.py GRID_CHOICES -------------
set "ROWS_OK=0"
for %%G in (2 3 4 5 6 7 8 10 12) do if /I "!ROWS!"=="%%G" set "ROWS_OK=1"
if "!ROWS_OK!"=="0" (
    echo ERROR: -rows !ROWS! not allowed by grid_sim_all.py.
    echo        valid: 2, 3, 4, 5, 6, 7, 8, 10, 12
    exit /b 2
)
set "COLS_OK=0"
for %%G in (2 3 4 5 6 7 8 10 12) do if /I "!COLS!"=="%%G" set "COLS_OK=1"
if "!COLS_OK!"=="0" (
    echo ERROR: -cols !COLS! not allowed by grid_sim_all.py.
    echo        valid: 2, 3, 4, 5, 6, 7, 8, 10, 12
    exit /b 2
)

REM --- Validate -gendur against gen END_CHOICES {300,600,1800,3600} ----------
set "GD_OK=0"
for %%D in (300 600 1800 3600) do if /I "!GENDUR!"=="%%D" set "GD_OK=1"
if "!GD_OK!"=="0" (
    echo ERROR: -gendur !GENDUR! not allowed by grid_sim_all.py gen --end.
    echo        valid: 300, 600, 1800, 3600
    exit /b 2
)

set "ROADNET=%OUTDIR%\roadnet_%ROWS%_%COLS%.json"
set "FLOW=%OUTDIR%\flow_%ROWS%_%COLS%.json"

REM --- Banner ----------------------------------------------------------------
echo.
echo ============================================================
echo   CENTRALIZED POLICY TRAINING  --  %ROWS%x%COLS% grid
echo   Target flow   : %FLOW%
echo   Episodes      : %EPISODES%   sim %DURATION%s   ctrl %CTRL%s   dt %DT%s
echo   Seed / lr / g : %SEED% / %LR% / %GAMMA%    eval rollouts: %EVALS%
echo   Optimizer     : Adam  entropy=%ENTROPY%  grad_clip=%GRADCLIP%
echo                  beta_decay=%BETADECAY%  lr_decay=%LRDECAY%  eval_every=%EVALEVERY%
if not "!MAXVEH!"=="0" echo   Vehicles      : cap=!MAXVEH!  ^(0 = full flow^)
if "!MAXVEH!"=="0"    echo   Vehicles      : full flow
echo   Out dir       : %OUTDIR%
echo ============================================================
echo.

REM --- Detect python ---------------------------------------------------------
set "PYTHON="
for %%P in (python py python3) do (
    if not defined PYTHON (
        where %%P >nul 2>&1
        if not errorlevel 1 set "PYTHON=%%P"
    )
)
if not defined PYTHON (
    echo ERROR: no Python interpreter found on PATH.
    exit /b 1
)
echo Using Python: %PYTHON%

REM --- Check required files --------------------------------------------------
set "MISSING="
if not exist "centralized_train.py" set "MISSING=!MISSING! centralized_train.py"
if not "%MISSING%"=="" (
    echo ERROR: missing file^(s^):%MISSING%
    echo        cd to the folder containing centralized_train.py and rerun.
    exit /b 1
)

REM --- Check numpy (required by centralized_train.py) -----------------------
%PYTHON% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo ERROR: numpy is not installed.
    echo        Install with:  %PYTHON% -m pip install numpy
    exit /b 1
)

REM --- matplotlib (optional, for the plot) ----------------------------------
%PYTHON% -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo INFO: matplotlib not installed -- train_curve.png will be skipped.
    echo       Install with:  %PYTHON% -m pip install matplotlib
    echo.
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM ============================================================================
REM Stage 1/2 -- ensure inputs (roadnet_R_C.json + flow_R_C.json)
REM ============================================================================
if "%USEEXISTING%"=="1" (
    if exist "%ROADNET%" if exist "%FLOW%" (
        echo ^>^>^> stage 1/2  skipped ^(-use-existing: reusing %FLOW%^)
        goto stage_train
    )
    echo ERROR: -use-existing but %ROADNET% / %FLOW% not found under %OUTDIR%.
    exit /b 1
)
if exist "%ROADNET%" if exist "%FLOW%" (
    echo ^>^>^> stage 1/2  inputs already present ^(reusing^)
    echo       %ROADNET%
    echo       %FLOW%
    goto stage_train
)

if not exist "grid_sim_all.py" (
    echo ERROR: grid_sim_all.py not found -- needed to generate flow_%ROWS%_%COLS%.json.
    echo        Either place it here, or supply existing inputs with -use-existing.
    exit /b 1
)
echo ^>^>^> stage 1/2  generate  ^(grid %ROWS%x%COLS%, %VEHICLES% veh, demand --end %GENDUR%s^)
set "CMD=%PYTHON% grid_sim_all.py gen %X% --rows %ROWS% --cols %COLS% --total-vehicles %VEHICLES% --end %GENDUR% --out-dir "%OUTDIR%" --summary --quiet"
if "%VERBOSE%"=="1" echo + %CMD%
%CMD%
if errorlevel 1 (
    echo ERROR: generation of %FLOW% failed.
    exit /b 1
)
if not exist "%FLOW%" (
    echo ERROR: generation ran but %FLOW% was not written.
    exit /b 1
)

REM ============================================================================
REM Stage 2/2 -- train the centralized policy
REM ============================================================================
:stage_train
echo.
echo ^>^>^> stage 2/2  train centralized policy on %FLOW%
set "CMD=%PYTHON% centralized_train.py --roadnet "%ROADNET%" --flow "%FLOW%" --episodes %EPISODES% --duration %DURATION% --control-interval %CTRL% --dt %DT% --seed %SEED% --lr %LR% --gamma %GAMMA% --eval-episodes %EVALS% --max-vehicles %MAXVEH% --entropy %ENTROPY% --grad-clip %GRADCLIP% --beta-decay %BETADECAY% --lr-decay %LRDECAY% --eval-every %EVALEVERY% --patience %PATIENCE% --out-dir "%OUTDIR%""
if "%VERBOSE%"=="1" (
    echo + %CMD%
) else (
    set "CMD=%CMD% --quiet"
)
%CMD%
if errorlevel 1 (
    echo ERROR: centralized training failed.
    exit /b 1
)

REM --- Open results ----------------------------------------------------------
if "%NOOPEN%"=="0" (
    if exist "%OUTDIR%\train_curve.png" ( echo opening %OUTDIR%\train_curve.png & start "" "%OUTDIR%\train_curve.png" )
    if exist "%OUTDIR%\eval_summary.txt" ( echo opening %OUTDIR%\eval_summary.txt & start "" "%OUTDIR%\eval_summary.txt" )
)

REM --- Summary ---------------------------------------------------------------
echo.
echo ============================================================
echo   DONE
echo ============================================================
echo   Outputs in %OUTDIR% :
if exist "%OUTDIR%\policy.json"        echo     policy.json         trained centralized weights W + value head v
if exist "%OUTDIR%\train_log.json"     echo     train_log.json      per-episode curve + eval + best snapshot
if exist "%OUTDIR%\train_curve.png"    echo     train_curve.png     learning curve + eval bars
if exist "%OUTDIR%\eval_summary.txt"   echo     eval_summary.txt    trained vs fixed-time vs random
echo.
echo   Retrain      :  train_case.cmd -episodes 300 -seed 7
echo   Other grid   :  train_case.cmd -rows 4 -cols 4 -outdir runs\c4_4
echo   Single car   :  train_case.cmd -max-vehicles 1 -episodes 300
echo   Full help    :  train_case.cmd -h
echo.
goto :eof

REM ============================================================================
REM :help
REM ============================================================================
:help
echo.
echo train_case.cmd -- train a CENTRALIZED traffic-light policy for one grid case
echo.
echo USAGE
echo   train_case.cmd [options]
echo.
echo WHAT IT DOES
echo   1. Ensures roadnet_R_C.json + flow_R_C.json exist ^(generates via
echo      grid_sim_all.py if missing; default R=3 C=4 -> flow_3_4.json^).
echo   2. Trains ONE centralized policy ^(shared weights across all intersections^)
echo      that observes the global queue/phase state and sets every intersection's
echo      phase each control cycle, using corrected REINFORCE + a linear value
echo      head (advantage = G - V(s^), entropy bonus, Adam, grad clip, best-policy
echo      checkpointing.
echo   3. Evaluates it vs fixed-time and random baselines; writes policy.json,
echo      train_log.json, train_curve.png, eval_summary.txt.
echo.
echo OPTIONS -- grid / generation
echo   -rows N            grid rows  ^(2,3,4,5,6,7,8,10,12^)        ^(default 3^)
echo   -cols N            grid cols  ^(2,3,4,5,6,7,8,10,12^)        ^(default 4^)
echo   -x N               logistic demand parameter                ^(default 150^)
echo   -vehicles N        target vehicle count for gen             ^(default 4000^)
echo   -gendur N          gen demand window s ^(300,600,1800,3600^) ^(default 300^)
echo   -use-existing      skip gen; reuse flow_R_C.json in -outdir
echo.
echo OPTIONS -- training
echo   -episodes N        training episodes                        ^(default 150^)
echo   -duration N        sim seconds per episode                  ^(default 180^)
echo   -control-interval N seconds between decisions              ^(default 10^)
echo   -dt N              sim timestep, seconds                    ^(default 0.5^)
echo   -seed N            RNG seed                                 ^(default 0^)
echo   -lr N              Adam learning rate                       ^(default 0.005^)
echo   -gamma N           discount factor                          ^(default 0.95^)
echo   -max-vehicles N    cap total vehicles spawned               ^(default 0=full; 1=single car^)
echo.
echo OPTIONS -- optimizer knobs ^(new^)
echo   -entropy F         entropy bonus coefficient                ^(default 0.01^)
echo   -grad-clip F       global gradient-norm clip  ^(0=off^)      ^(default 5.0^)
echo   -beta-decay F      per-episode decay of entropy coef        ^(default 0.97^)
echo   -lr-decay F        per-episode decay of learning rate       ^(default 0.99^)
echo   -eval-every N      greedy eval + checkpoint every N eps     ^(default 10^)
echo   -patience N        early-stop after N stagnant evals       ^(default 0=off^)
echo.
echo OPTIONS -- output / misc
echo   -eval-episodes N   evaluation rollouts per policy           ^(default 5^)
echo   -outdir PATH       output / input directory                 ^(default runs\c3_4^)
echo   -noopen            do not auto-open the plot / summary
echo   -verbose           echo each python command
echo   -h, --help         show this help
echo.
echo EXAMPLES
echo   train_case.cmd
echo   train_case.cmd -episodes 300 -seed 7
echo   train_case.cmd -episodes 300 -duration 300 -control-interval 8
echo   train_case.cmd -rows 4 -cols 4 -vehicles 8000 -outdir runs\c4_4
echo   train_case.cmd -max-vehicles 1 -episodes 300 -duration 120 -control-interval 5
echo   train_case.cmd -use-existing -outdir runs\c3_4 -noopen
echo.
goto :eof
