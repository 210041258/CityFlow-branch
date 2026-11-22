@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  run_case.cmd  --  one-command grid simulation pipeline (batch, no PowerShell)
REM
REM  Runs exactly ONE case per invocation, end to end:
REM      generate  ->  simulate  ->  [plot]  ->  [animate]  ->  render
REM
REM  Multi-feature: presets, custom params, turn routes, rng seed, render modes,
REM  optional plot / sim_animate stages, case-reuse, clean rebuild, tag folders,
REM  capture-stride, verbose, dry-run, open-last, list, full help.
REM
REM  Usage:
REM    run_case.cmd                              default demo: 5000 veh, 4x4, 300s
REM    run_case.cmd quick                        1000 veh, 4x4, 60s
REM    run_case.cmd dense                        10000 veh, 4x4, 600s
REM    run_case.cmd megacrowd                    1M veh, 12x12 turn, 3600s
REM    run_case.cmd custom -vehicles 2500 -rows 3 -cols 3 -end 120
REM    run_case.cmd case                         reuse existing inputs, same outputs
REM    run_case.cmd case -mode gif -noopen       rerender a saved case as GIF only
REM    run_case.cmd -animate -plot               also produce flipbook + plot PNG
REM    run_case.cmd -tag smoke                   write to runs\live\smoke\
REM    run_case.cmd -clean                       wipe outdir before running
REM    run_case.cmd -mode gif                    GIF + PNG, no window
REM    run_case.cmd -mode show                   GIF + PNG + interactive window
REM    run_case.cmd -mode none                   skip render stage
REM    run_case.cmd -noopen                      don't auto-open output files
REM    run_case.cmd -dry-run                     generate only; skip sim + render
REM    run_case.cmd -verbose                     echo each python command
REM    run_case.cmd openlast                     re-open the last GIF/PNG in runs\
REM    run_case.cmd list                         list saved cases under runs\
REM    run_case.cmd -h                           show full help
REM
REM  Requirements (in the same folder):
REM    grid_sim_all.py        v3.0.1+   (required)
REM    live_matplotlib.py     v1.2.0+   (required for render stage)
REM    sim_animate.py         v1.0.1+   (optional; only with -animate)
REM    matplotlib                        (optional; auto-disables render if absent)
REM ============================================================================

REM --- Help / list fast paths -------------------------------------------------
if /I "%~1"=="-h"      goto help
if /I "%~1"=="--help"  goto help
if /I "%~1"=="-help"   goto help
if /I "%~1"=="help"    goto help
if /I "%~1"=="/?"      goto help
if /I "%~1"=="list"    goto list_mode
if /I "%~1"=="-list"   goto list_mode

REM --- Defaults ---------------------------------------------------------------
set "PRESET=demo"
set "X=150"
set "VEHICLES=0"
set "ROWS=0"
set "COLS=0"
set "START=0"
set "END=0"
set "DT=0.5"
set "SEED=0"
set "FPS=25"
set "MAXFRAMES=200"
set "CAPTUREEVERY=0"
set "MODE=both"
set "OUTDIR=runs\live"
set "TAG="
set "NOOPEN=0"
set "OPENLAST=0"
set "VERBOSE=0"
set "DRYRUN=0"
set "FORCETURN=0"
set "TLPLAN=0"
set "CLEAN=0"
set "DOPLOT=0"
set "DOANIM=0"
set "ASCIIFRAMES=50"
set "GIFFRAMES=60"
set "SKIPGEN=0"

REM --- Parse arguments --------------------------------------------------------
:parse
if "%~1"=="" goto parse_done
set "ARG=%~1"

REM Presets (no value)
if /I "!ARG!"=="quick"     ( set "PRESET=quick"     & shift & goto parse )
if /I "!ARG!"=="demo"      ( set "PRESET=demo"      & shift & goto parse )
if /I "!ARG!"=="dense"     ( set "PRESET=dense"     & shift & goto parse )
if /I "!ARG!"=="megacrowd" ( set "PRESET=megacrowd" & shift & goto parse )
if /I "!ARG!"=="custom"    ( set "PRESET=custom"    & shift & goto parse )
if /I "!ARG!"=="case"      ( set "PRESET=case"      & shift & goto parse )

REM Mode / behaviour switches (no value)
if /I "!ARG!"=="openlast"   ( set "OPENLAST=1"  & shift & goto parse )
if /I "!ARG!"=="-openlast"  ( set "OPENLAST=1"  & shift & goto parse )
if /I "!ARG!"=="noopen"     ( set "NOOPEN=1"    & shift & goto parse )
if /I "!ARG!"=="-noopen"    ( set "NOOPEN=1"    & shift & goto parse )
if /I "!ARG!"=="verbose"    ( set "VERBOSE=1"   & shift & goto parse )
if /I "!ARG!"=="-verbose"   ( set "VERBOSE=1"   & shift & goto parse )
if /I "!ARG!"=="dry-run"    ( set "DRYRUN=1"    & shift & goto parse )
if /I "!ARG!"=="-dry-run"   ( set "DRYRUN=1"    & shift & goto parse )
if /I "!ARG!"=="-turn"      ( set "FORCETURN=1" & shift & goto parse )
if /I "!ARG!"=="--turn"     ( set "FORCETURN=1" & shift & goto parse )
if /I "!ARG!"=="-tlplan"    ( set "TLPLAN=1"    & shift & goto parse )
if /I "!ARG!"=="--tlPlan"   ( set "TLPLAN=1"    & shift & goto parse )
if /I "!ARG!"=="-clean"     ( set "CLEAN=1"     & shift & goto parse )
if /I "!ARG!"=="-plot"      ( set "DOPLOT=1"    & shift & goto parse )
if /I "!ARG!"=="-animate"   ( set "DOANIM=1"    & shift & goto parse )

REM Value flags (a missing value is a hard error, not a silent default)
if /I "!ARG!"=="-preset"       ( if "%~2"=="" goto missing_val & set "PRESET=%~2"        & shift & shift & goto parse )
if /I "!ARG!"=="-x"            ( if "%~2"=="" goto missing_val & set "X=%~2"             & shift & shift & goto parse )
if /I "!ARG!"=="-vehicles"     ( if "%~2"=="" goto missing_val & set "VEHICLES=%~2"      & shift & shift & goto parse )
if /I "!ARG!"=="-rows"         ( if "%~2"=="" goto missing_val & set "ROWS=%~2"          & shift & shift & goto parse )
if /I "!ARG!"=="-cols"         ( if "%~2"=="" goto missing_val & set "COLS=%~2"          & shift & shift & goto parse )
if /I "!ARG!"=="-start"        ( if "%~2"=="" goto missing_val & set "START=%~2"         & shift & shift & goto parse )
if /I "!ARG!"=="-end"          ( if "%~2"=="" goto missing_val & set "END=%~2"           & shift & shift & goto parse )
if /I "!ARG!"=="-dt"           ( if "%~2"=="" goto missing_val & set "DT=%~2"            & shift & shift & goto parse )
if /I "!ARG!"=="-seed"         ( if "%~2"=="" goto missing_val & set "SEED=%~2"          & shift & shift & goto parse )
if /I "!ARG!"=="-fps"          ( if "%~2"=="" goto missing_val & set "FPS=%~2"           & shift & shift & goto parse )
if /I "!ARG!"=="-frames"       ( if "%~2"=="" goto missing_val & set "MAXFRAMES=%~2"     & shift & shift & goto parse )
if /I "!ARG!"=="-max-frames"   ( if "%~2"=="" goto missing_val & set "MAXFRAMES=%~2"     & shift & shift & goto parse )
if /I "!ARG!"=="-capture-every"( if "%~2"=="" goto missing_val & set "CAPTUREEVERY=%~2"  & shift & shift & goto parse )
if /I "!ARG!"=="-ascii-frames" ( if "%~2"=="" goto missing_val & set "ASCIIFRAMES=%~2"   & shift & shift & goto parse )
if /I "!ARG!"=="-gif-frames"   ( if "%~2"=="" goto missing_val & set "GIFFRAMES=%~2"     & shift & shift & goto parse )
if /I "!ARG!"=="-mode"         ( if "%~2"=="" goto missing_val & set "MODE=%~2"          & shift & shift & goto parse )
if /I "!ARG!"=="-outdir"       ( if "%~2"=="" goto missing_val & set "OUTDIR=%~2"        & shift & shift & goto parse )
if /I "!ARG!"=="-tag"          ( if "%~2"=="" goto missing_val & set "TAG=%~2"           & shift & shift & goto parse )

echo WARN: unknown argument '!ARG!' ^(ignored^). Use -h for help.
shift
goto parse

:missing_val
echo ERROR: !ARG! needs a value.
exit /b 2
:parse_done

REM Normalize path separators (runs/live -> runs\live)
set "OUTDIR=%OUTDIR:/=\%"

REM --- Tag -> per-case subfolder under the chosen outdir ---------------------
if not "%TAG%"=="" (
    if /I "!OUTDIR!"=="runs\live" (
        set "OUTDIR=runs\live\%TAG:/=\%"
    ) else (
        echo WARN: -tag ^"%TAG%^" ignored because -outdir was also given.
        set "TAG="
    )
)

REM --- Fast path: open last results ------------------------------------------
if "%OPENLAST%"=="1" goto openlast_mode

REM --- Validate enum values --------------------------------------------------
set "PRESET_OK=0"
for %%P in (quick demo dense megacrowd custom case) do (
    if /I "!PRESET!"=="%%P" set "PRESET_OK=1"
)
if "!PRESET_OK!"=="0" (
    echo ERROR: invalid preset '!PRESET!'
    echo        valid: quick, demo, dense, megacrowd, custom, case
    exit /b 2
)

set "MODE_OK=0"
for %%M in (none gif show both) do (
    if /I "!MODE!"=="%%M" set "MODE_OK=1"
)
if "!MODE_OK!"=="0" (
    echo ERROR: invalid mode '!MODE!'
    echo        valid: none, gif, show, both
    exit /b 2
)

REM --- Apply preset defaults (only for values the user did not override) -----
if "%VEHICLES%"=="0" (
    if /I "!PRESET!"=="quick"     set "VEHICLES=1000"
    if /I "!PRESET!"=="demo"      set "VEHICLES=5000"
    if /I "!PRESET!"=="dense"     set "VEHICLES=10000"
    if /I "!PRESET!"=="megacrowd" set "VEHICLES=1000000"
    if /I "!PRESET!"=="custom"    set "VEHICLES=5000"
    if /I "!PRESET!"=="case"      set "VEHICLES=5000"
)
if "%ROWS%"=="0" (
    set "ROWS=4"
    if /I "!PRESET!"=="megacrowd" set "ROWS=12"
)
if "%COLS%"=="0" (
    set "COLS=4"
    if /I "!PRESET!"=="megacrowd" set "COLS=12"
)
if "%END%"=="0" (
    if /I "!PRESET!"=="quick"     set "END=60"
    if /I "!PRESET!"=="demo"      set "END=300"
    if /I "!PRESET!"=="dense"     set "END=600"
    if /I "!PRESET!"=="megacrowd" set "END=3600"
    if /I "!PRESET!"=="custom"    set "END=120"
    if /I "!PRESET!"=="case"      set "END=300"
)

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

REM --- Validate start against gen START_CHOICES {0,60,300} -------------------
set "START_OK=0"
for %%S in (0 60 300) do if /I "!START!"=="%%S" set "START_OK=1"
if "!START_OK!"=="0" (
    echo ERROR: -start !START! not allowed by grid_sim_all.py gen.
    echo        valid: 0, 60, 300
    exit /b 2
)

REM --- Validate end > 0 ------------------------------------------------------
if !END! LEQ 0 (
    echo ERROR: -end must be a positive number of seconds.
    exit /b 2
)

REM --- Snap the demand-window --end up to the nearest allowed gen value ------
REM     gen --end choices = {-1,300,600,1800,3600}; -end here is the SIM
REM     duration (free), so we feed gen the smallest legal window >= END and
REM     pass the literal END to sim --duration / render --duration.
set "GEN_END=3600"
if !END! LEQ 1800 set "GEN_END=1800"
if !END! LEQ 600  set "GEN_END=600"
if !END! LEQ 300  set "GEN_END=300"
if !GEN_END! LEQ !START! (
    echo ERROR: demand window !GEN_END!s must be ^> start !START!s.
    echo        raise -end or lower -start.
    exit /b 2
)

REM --- Turn flag + filename suffix -------------------------------------------
if /I "%PRESET%"=="megacrowd" set "FORCETURN=1"

set "TURNFLAG="
set "TURNSUFFIX="
if "%FORCETURN%"=="1" (
    set "TURNFLAG=--turn"
    set "TURNSUFFIX=_turn"
)
set "TLPLANFLAG="
if "%TLPLAN%"=="1" set "TLPLANFLAG=--tlPlan"

REM --- Consistent file paths (respecting _turn) ------------------------------
set "ROADNET=%OUTDIR%\roadnet_%ROWS%_%COLS%%TURNSUFFIX%.json"
set "FLOW=%OUTDIR%\flow_%ROWS%_%COLS%%TURNSUFFIX%.json"
set "SIMJSON=%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%.json"
set "PNGOUT=%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%_final.png"
set "GIFOUT=%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%_matplotlib.gif"
set "PLOTPNG=%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%_plot.png"
REM  sim_animate hardcodes prefix sim_{rows}_{cols} (no _turn), so its outputs
REM  are always unsuffixed -- point these at the real names it writes.
set "ANIMGIF=%OUTDIR%\sim_%ROWS%_%COLS%_anim.gif"
set "TRACEPNG=%OUTDIR%\sim_%ROWS%_%COLS%_traces.png"
set "FRAMESTXT=%OUTDIR%\sim_%ROWS%_%COLS%_frames.txt"

REM --- Banner ----------------------------------------------------------------
set "TURNLABEL=off"
if "%FORCETURN%"=="1" set "TURNLABEL=on"
set "PLOTLABEL=no"
if "%DOPLOT%"=="1" set "PLOTLABEL=yes"
set "ANIMLABEL=no"
if "%DOANIM%"=="1" set "ANIMLABEL=yes"
set "CAPLABEL=auto"
if !CAPTUREEVERY! GTR 0 set "CAPLABEL=!CAPTUREEVERY!"

echo.
echo ============================================================
echo   GRID SIMULATION  --  %PRESET% preset   ^(one case^)
echo   Vehicles : %VEHICLES%
echo   Grid     : %ROWS% x %COLS%   ^(turn: %TURNLABEL%, tlPlan: %TLPLAN%^)
echo   Window   : %START%s -^> %END%s   ^(dt=%DT%s, seed=%SEED%^)
echo   Demand   : --end %GEN_END%s   ^(snapped to gen choices^)
echo   Mode     : %MODE%   fps=%FPS%  frames^<=%MAXFRAMES%  capture=%CAPLABEL%
echo   Extras   : plot=%PLOTLABEL%  animate=%ANIMLABEL%  clean=%CLEAN%
echo   Out dir  : %OUTDIR%
echo ============================================================
echo.

REM --- Detect python (python -> py -> python3) ------------------------------
set "PYTHON="
for %%P in (python py python3) do (
    if not defined PYTHON (
        where %%P >nul 2>&1
        if not errorlevel 1 set "PYTHON=%%P"
    )
)
if not defined PYTHON (
    echo ERROR: no Python interpreter found on PATH.
    echo        Install from https://python.org and tick "Add Python to PATH".
    exit /b 1
)
echo Using Python: %PYTHON%

REM --- Check required files --------------------------------------------------
set "MISSING="
for %%F in (grid_sim_all.py live_matplotlib.py) do (
    if not exist "%%F" set "MISSING=!MISSING! %%F"
)
if not "%MISSING%"=="" (
    echo ERROR: missing file^(s^):%MISSING%
    echo        cd to the folder containing the .py files and rerun.
    exit /b 1
)
if "%DOANIM%"=="1" (
    if not exist "sim_animate.py" (
        echo ERROR: -animate requires sim_animate.py in this folder.
        exit /b 1
    )
)

REM --- Version sanity check (3.0.1 fixed route pruning) ----------------------
%PYTHON% -c "import re,sys; s=open('grid_sim_all.py',encoding='utf-8',errors='ignore').read(); m=re.search(r'__version__\s*=\s*\"(\d+)\.(\d+)',s); v=(int(m.group(1)),int(m.group(2))) if m else (0,0); sys.exit(0 if v>=(3,0) else 1)" >nul 2>&1
if errorlevel 1 (
    echo WARN: grid_sim_all.py may be older than v3.0 -- expect
    echo       'pruned N invalid route^(s^)' warnings and fewer routes.
    echo.
)

REM --- matplotlib availability ----------------------------------------------
set "HASMPL=0"
%PYTHON% -c "import matplotlib" >nul 2>&1
if not errorlevel 1 set "HASMPL=1"
if "!HASMPL!"=="0" (
    echo INFO: matplotlib is not installed.
    echo       Rendering ^(GIF/PNG/window^) will be skipped.
    echo       Install with:  %PYTHON% -m pip install matplotlib
    if /I not "!MODE!"=="none" (
        set "MODE=none"
        echo       --^> mode switched to 'none'
    )
    echo.
)

REM --- Clean ----------------------------------------------------------------
if "%CLEAN%"=="1" (
    if exist "%OUTDIR%" (
        echo Cleaning %OUTDIR% ...
        del /q /f "%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%*.json" 2>nul
        del /q /f "%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%*.png"  2>nul
        del /q /f "%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%*.gif"  2>nul
        del /q /f "%OUTDIR%\sim_%ROWS%_%COLS%%TURNSUFFIX%*.txt"  2>nul
        del /q /f "%OUTDIR%\roadnet_%ROWS%_%COLS%%TURNSUFFIX%.json" 2>nul
        del /q /f "%OUTDIR%\flow_%ROWS%_%COLS%%TURNSUFFIX%.json"    2>nul
        echo.
    )
)

REM --- Ensure output directory ----------------------------------------------
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM --- case preset: reuse existing inputs, same output paths ----------------
if /I "%PRESET%"=="case" (
    if exist "%ROADNET%" if exist "%FLOW%" (
        set "SKIPGEN=1"
        echo [case] reusing inputs : %ROADNET%
        echo                         %FLOW%
        echo        output paths   : %SIMJSON%
        echo                         %GIFOUT%
        echo                         %PNGOUT%
        echo.
    ) else (
        echo [case] no existing input files under %OUTDIR% - will generate first.
        echo.
    )
)

REM ============================================================================
REM Stage 1/5 -- generate
REM ============================================================================
if "%SKIPGEN%"=="1" (
    echo ^>^>^> stage 1/5  skipped ^(case: reusing existing inputs^)
    goto stage_sim
)
echo ^>^>^> stage 1/5  generate  ^(demand --end %GEN_END%s^)
set "CMD=%PYTHON% grid_sim_all.py gen %X% --rows %ROWS% --cols %COLS% %TURNFLAG% %TLPLANFLAG% --total-vehicles %VEHICLES% --start %START% --end %GEN_END% --out-dir "%OUTDIR%" --summary --quiet"
if "%VERBOSE%"=="1" echo + %CMD%
%CMD%
if errorlevel 1 (
    echo ERROR: generation failed.
    exit /b 1
)

if "%DRYRUN%"=="1" (
    echo.
    echo [dry-run] stopped after generate.
    goto summary_only
)

REM ============================================================================
REM Stage 2/5 -- simulate
REM ============================================================================
:stage_sim
echo.
echo ^>^>^> stage 2/5  simulate  ^(duration %END%s, dt %DT%s^)
set "CMD=%PYTHON% grid_sim_all.py sim --roadnet "%ROADNET%" --flow "%FLOW%" --out "%SIMJSON%" --duration %END% --dt %DT% --seed %SEED%"
if "%VERBOSE%"=="1" echo + %CMD%
%CMD%
if errorlevel 1 (
    echo ERROR: simulation failed.
    exit /b 1
)

REM --- capture authoritative metrics NOW, before any render stage re-runs the
REM     simulator and (for non-turn cases) overwrites sim_R_C.json in place.
set "METRICSFILE=%TEMP%\run_case_metrics_%RANDOM%.txt"
%PYTHON% -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; open(sys.argv[2],'w').write('    finished  : {:>10,}   peak queue : {:>10,}   mean speed : {:>6.1f} km/h\n'.format(s.get('finished_vehicles',0), s.get('peak_queue',0), s.get('mean_avg_speed_kmh',0.0)))" "%SIMJSON%" "%METRICSFILE%" 2>nul

REM ============================================================================
REM Stage 3/5 -- plot (optional)
REM ============================================================================
if "%DOPLOT%"=="0" goto plot_done
echo.
echo ^>^>^> stage 3/5  plot  ^(sim_R_C_plot.png^)
REM plot writes {prefix}_plot.png next to --sim ; prefix must carry grid+turn
set "PLOTPREFIX=sim_%ROWS%_%COLS%%TURNSUFFIX%"
set "CMD=%PYTHON% grid_sim_all.py plot --sim "%SIMJSON%" --prefix %PLOTPREFIX% --quiet"
if "%VERBOSE%"=="1" echo + %CMD%
%CMD%
if errorlevel 1 (
    echo WARN: plot stage exited with code %ERRORLEVEL% ^(sim/JSON still valid^).
)
:plot_done

REM ============================================================================
REM Stage 4/5 -- animate (optional, sim_animate.py)
REM ============================================================================
if "%DOANIM%"=="0" goto anim_done
echo.
echo ^>^>^> stage 4/5  animate  ^(frames.txt + anim.gif + traces.png^)
set "CMD=%PYTHON% sim_animate.py %X% --rows %ROWS% --cols %COLS% %TURNFLAG% %TLPLANFLAG% --total-vehicles %VEHICLES% --start %START% --end %GEN_END% --duration %END% --dt %DT% --out-dir "%OUTDIR%" --ascii-frames %ASCIIFRAMES% --gif-frames %GIFFRAMES% --fps %FPS% --quiet"
if "%VERBOSE%"=="1" echo + %CMD%
%CMD%
if errorlevel 1 (
    echo WARN: animate stage exited with code %ERRORLEVEL% ^(sim/JSON still valid^).
)
:anim_done

REM ============================================================================
REM Stage 5/5 -- render (live_matplotlib.py)
REM ============================================================================
if /I "%MODE%"=="none" (
    echo.
    echo ^>^>^> stage 5/5  skipped ^(mode=none^)
    goto render_done
)
echo.
echo ^>^>^> stage 5/5  render ^(mode=%MODE%^)

set "SHOWFLAG="
if /I "!MODE!"=="both" set "SHOWFLAG=--show"
if /I "!MODE!"=="show" set "SHOWFLAG=--show"

set "CAPFLAG="
if !CAPTUREEVERY! GTR 0 set "CAPFLAG=--capture-every %CAPTUREEVERY%"

REM  live_matplotlib hardcodes sim_{rows}_{cols} for its sim-json/gif; pass
REM  --gif and --png so the GIF and final PNG land on the case's _turn names.
set "CMD=%PYTHON% live_matplotlib.py %X% --rows %ROWS% --cols %COLS% %TURNFLAG% %TLPLANFLAG% --total-vehicles %VEHICLES% --start %START% --end %GEN_END% --duration %END% --dt %DT% --out-dir "%OUTDIR%" --fps %FPS% --max-frames %MAXFRAMES% %CAPFLAG% --gif "%GIFOUT%" --png "%PNGOUT%" %SHOWFLAG%"
if "%VERBOSE%"=="1" echo + %CMD%
%CMD%
if errorlevel 1 (
    echo WARN: render stage exited with code %ERRORLEVEL% ^(sim/JSON still valid^).
)
:render_done

REM --- Auto-open results -----------------------------------------------------
if "%NOOPEN%"=="0" (
    if exist "%GIFOUT%"   ( echo opening %GIFOUT%   & start "" "%GIFOUT%" )
    if exist "%PNGOUT%"   ( echo opening %PNGOUT%   & start "" "%PNGOUT%" )
    if exist "%PLOTPNG%"  ( echo opening %PLOTPNG%  & start "" "%PLOTPNG%" )
    if exist "%ANIMGIF%"  ( echo opening %ANIMGIF%  & start "" "%ANIMGIF%" )
    if exist "%TRACEPNG%" ( echo opening %TRACEPNG% & start "" "%TRACEPNG%" )
)

REM ============================================================================
REM Summary
REM ============================================================================
:summary_only
echo.
echo ============================================================
echo   DONE
echo ============================================================
echo   Files in %OUTDIR% :
if exist "%METRICSFILE%" (
    type "%METRICSFILE%"
) else (
    echo    ^(no sim summary -- run without -dry-run^)
)
dir /a-d /b "%OUTDIR%" 2>nul
echo.
echo   Re-open later :  run_case.cmd openlast
echo   List saved    :  run_case.cmd list
echo   Try a preset  :  run_case.cmd quick ^| dense ^| megacrowd ^| case
echo   Full help     :  run_case.cmd -h
echo.
goto :eof

REM ============================================================================
REM :help
REM ============================================================================
:help
echo.
echo run_case.cmd -- one-command grid simulation pipeline ^(one case, multi features^)
echo.
echo USAGE
echo   run_case.cmd [preset] [options]
echo.
echo PRESETS  ^(pick one; each runs a single case^)
echo   quick       1000 vehicles, 4x4 grid, 60s
echo   demo        5000 vehicles, 4x4 grid, 300s        ^(default^)
echo   dense       10000 vehicles, 4x4 grid, 600s
echo   megacrowd   1,000,000 vehicles, 12x12 turn, 3600s
echo   custom      user-specified; sane defaults applied
echo   case        reuse existing roadnet+flow in -outdir; same output names
echo.
echo OPTIONS
echo   -preset NAME     quick^|demo^|dense^|megacrowd^|custom^|case
echo   -x N             logistic demand parameter             ^(default 150^)
echo   -vehicles N      target vehicle count
echo   -rows N          grid rows  ^(2,3,4,5,6,7,8,10,12^)
echo   -cols N          grid cols  ^(2,3,4,5,6,7,8,10,12^)
echo   -start N         demand start, seconds  ^(0,60,300^)    ^(default 0^)
echo   -end N           SIM duration, seconds                  ^(default per preset^)
echo                    demand window is auto-snapped to {300,600,1800,3600}
echo   -dt N            sim timestep, seconds                 ^(default 0.5^)
echo   -seed N          RNG seed                              ^(default 0^)
echo   -fps N           GIF / playback fps                    ^(default 25^)
echo   -frames N        max rendered frames                   ^(default 200^)
echo   -capture-every N frame stride ^(0 = auto for long runs^) ^(default 0^)
echo   -ascii-frames N  sim_animate flipbook length           ^(default 50^)
echo   -gif-frames N    sim_animate GIF length                ^(default 60^)
echo   -mode MODE       none^|gif^|show^|both                  ^(default both^)
echo                    none = skip render entirely
echo                    gif  = GIF + PNG, no window
echo                    show = GIF + PNG + window  ^(same as both^)
echo                    both = GIF + PNG + window
echo   -outdir PATH     output directory                      ^(default runs\live^)
echo   -tag NAME        per-case subfolder: runs\live\NAME
echo   -turn            enable turn routes ^(auto for megacrowd^)
echo   -tlplan          enable traffic-light plan in gen/animate/render
echo   -plot            also run grid_sim_all.py plot  ^(extra PNG^)
echo   -animate         also run sim_animate.py  ^(frames.txt + anim.gif + traces.png^)
echo   -clean           wipe this case's old outputs in -outdir first
echo   -noopen          do not auto-open results
echo   -verbose         echo each python command before running
echo   -dry-run         generate only; skip sim + render
echo   openlast         re-open the most recent GIF/PNG under runs\
echo   list             list saved cases under runs\
echo   -h, --help       show this help
echo.
echo EXAMPLES
echo   run_case.cmd quick
echo   run_case.cmd case
echo   run_case.cmd case -mode gif -noopen
echo   run_case.cmd custom -vehicles 2500 -rows 3 -cols 3 -end 120
echo   run_case.cmd -mode gif -frames 300 -fps 15 -tag batch1
echo   run_case.cmd megacrowd -verbose -capture-every 50
echo   run_case.cmd dense -plot -animate -clean
echo   run_case.cmd openlast
echo   run_case.cmd list
echo.
goto :eof

REM ============================================================================
REM :openlast_mode -- re-open the most recent GIF/PNG files under runs\
REM ============================================================================
:openlast_mode
echo Searching for recent .gif and .png files under runs\ ...
if not exist "runs\" (
    echo ERROR: no 'runs\' directory found.
    exit /b 1
)
set "FOUND=0"
for /F "delims=" %%F in ('dir /b /s /o-d /a-d "runs\*.gif" 2^>nul') do (
    echo opening %%F
    start "" "%%F"
    set /a FOUND+=1
    if !FOUND! geq 3 goto openlast_done
)
for /F "delims=" %%F in ('dir /b /s /o-d /a-d "runs\*.png" 2^>nul') do (
    echo opening %%F
    start "" "%%F"
    set /a FOUND+=1
    if !FOUND! geq 3 goto openlast_done
)
:openlast_done
if "%FOUND%"=="0" (
    echo No .gif or .png files found under runs\
    exit /b 1
)
goto :eof

REM ============================================================================
REM :list_mode -- list saved cases under runs\
REM ============================================================================
:list_mode
if not exist "runs\" (
    echo No 'runs\' directory found yet.
    echo Run a preset first, e.g.  run_case.cmd quick
    exit /b 1
)
echo Saved cases under runs\  ^(folders with roadnet_*.json^):
echo.
set "NCASE=0"
for /F "delims=" %%F in ('dir /b /s /a-d "runs\roadnet_*.json" 2^>nul') do (
    set /a NCASE+=1
    echo   !NCASE!  %%~dpF
    echo        %%~nxF
)
if "!NCASE!"=="0" echo   ^(none yet -- no roadnet_*.json under runs\^)
echo.
echo Re-run a case:   run_case.cmd case -outdir ^<folder^>
goto :eof
