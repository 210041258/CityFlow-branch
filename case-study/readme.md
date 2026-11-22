C:\Users\albre\Downloads\New folder>w.cmd -dry-run

============================================================
  GRID SIMULATION  --  demo preset
  Vehicles : 5000
  Grid     : 4 x 4   (turn: off)
  Window   : 0s -> 300s   (dt=0.5s, seed=0)
  Mode     : both   fps=25  frames<=200
  Out dir  : runs\live
============================================================

Using Python: python
  File "<string>", line 1
    import re,sys; s=open('grid_sim_all.py',encoding='utf-8',errors='ignore').read(); m=re.search(r'__version__\s*=\s*"(\d+)\.(\d+)',s); v=(int(m.group(1)),int(m.group(2))) if m else (0,0); sys.exit(0 if v,0) else 1)

                                                                                               ^
SyntaxError: unmatched ')'
WARN: grid_sim_all.py may be older than v3.0 ΓÇö expect
      'pruned N invalid route(s)' warnings and fewer routes.

>>> stage 1/3  generate

>>> stage 2/3  simulate
[sim] roadnet : runs\live\roadnet_4_4.json  (16 intersections, 80 roads)
[sim] flow    : runs\live\flow_4_4.json  (16 route definitions)
[sim] window  : 0 -> 300s  (dt = 0.5s)
[sim] wrote runs\live\sim_4_4.json  finished=496  peak_queue=4288  mean_speed_kmh=8.079

>>> stage 3/3  render (mode=both)
[viz] backend: qtagg
Target: 5,000 vehicles over 300s across 16 routes -> interval=0.960000s
Logistic y=201.480 -> grid 4x4, interval=0.9600s, lanes(L/S/R)=1/2/1, speed=14.93
Roadnet:   runs\live\roadnet_4_4.json
Flow:      runs\live\flow_4_4.json  (16 routes)
Summary:   runs\live\summary_4_4.json
[viz] simulating 300s at dt=0.5s (600 steps), capturing every 3 → ~200 frames
[viz] sim -> runs\live\sim_4_4.json (finished=496, peak_queue=4288)
[viz] gif -> runs\live\sim_4_4_matplotlib.gif
[viz] png -> runs\live\sim_4_4_final.png
[viz] opening interactive window — close it to exit ...
opening runs\live\sim_4_4_matplotlib.gif
opening runs\live\sim_4_4_final.png

============================================================
  DONE
============================================================
  Files in runs\live :
    finished  :        496   peak queue :      4,288   mean speed :    8.1 km/h
flow_4_4.json
roadnet_4_4.json
sim_4_4.json
sim_4_4_final.png
sim_4_4_matplotlib.gif
summary_4_4.json

  Re-open later :  run_case.cmd openlast
  Try a preset  :  run_case.cmd quick | dense | megacrowd | case
  Full help     :  run_case.cmd -h


C:\Users\albre\Downloads\New folder>w.cmd megacrowd
ERROR: invalid preset 'megacrowd '
       valid: quick, demo, dense, megacrowd, custom, case

C:\Users\albre\Downloads\New folder>w.cmd demo
ERROR: invalid preset 'demo      '
       valid: quick, demo, dense, megacrowd, custom, case

C:\Users\albre\Downloads\New folder>w.cmd -verbose

============================================================
  GRID SIMULATION  --  demo preset
  Vehicles : 5000
  Grid     : 4 x 4   (turn: off)
  Window   : 0s -> 300s   (dt=0.5s, seed=0)
  Mode     : both   fps=25  frames<=200
  Out dir  : runs\live
============================================================

Using Python: python
  File "<string>", line 1
    import re,sys; s=open('grid_sim_all.py',encoding='utf-8',errors='ignore').read(); m=re.search(r'__version__\s*=\s*"(\d+)\.(\d+)',s); v=(int(m.group(1)),int(m.group(2))) if m else (0,0); sys.exit(0 if v,0) else 1)

                                                                                               ^
SyntaxError: unmatched ')'
WARN: grid_sim_all.py may be older than v3.0 ΓÇö expect
      'pruned N invalid route(s)' warnings and fewer routes.

>>> stage 1/3  generate

>>> stage 2/3  simulate
[sim] roadnet : runs\live\roadnet_4_4.json  (16 intersections, 80 roads)
[sim] flow    : runs\live\flow_4_4.json  (16 route definitions)
[sim] window  : 0 -> 300s  (dt = 0.5s)
[sim] wrote runs\live\sim_4_4.json  finished=496  peak_queue=4288  mean_speed_kmh=8.079

>>> stage 3/3  render (mode=both)
[viz] backend: qtagg
Target: 5,000 vehicles over 300s across 16 routes -> interval=0.960000s
Logistic y=201.480 -> grid 4x4, interval=0.9600s, lanes(L/S/R)=1/2/1, speed=14.93
Roadnet:   runs\live\roadnet_4_4.json
Flow:      runs\live\flow_4_4.json  (16 routes)
Summary:   runs\live\summary_4_4.json
[viz] simulating 300s at dt=0.5s (600 steps), capturing every 3 → ~200 frames
[viz] sim -> runs\live\sim_4_4.json (finished=496, peak_queue=4288)
[viz] gif -> runs\live\sim_4_4_matplotlib.gif
[viz] png -> runs\live\sim_4_4_final.png
[viz] opening interactive window — close it to exit ...
opening runs\live\sim_4_4_matplotlib.gif
opening runs\live\sim_4_4_final.png

============================================================
  DONE
============================================================
  Files in runs\live :
    finished  :        496   peak queue :      4,288   mean speed :    8.1 km/h
flow_4_4.json
roadnet_4_4.json
sim_4_4.json
sim_4_4_final.png
sim_4_4_matplotlib.gif
summary_4_4.json

  Re-open later :  run_case.cmd openlast
  Try a preset  :  run_case.cmd quick | dense | megacrowd | case
  Full help     :  run_case.cmd -h


C:\Users\albre\Downloads\New folder>run_case.cmd custom -vehicles 2500 -rows 3 -cols 3 -end 120
WARN: unknown argument '-vehicles' (ignored). Use -h for help.
WARN: unknown argument '2500' (ignored). Use -h for help.
WARN: unknown argument '-rows' (ignored). Use -h for help.
WARN: unknown argument '3' (ignored). Use -h for help.
WARN: unknown argument '-cols' (ignored). Use -h for help.
WARN: unknown argument '3' (ignored). Use -h for help.
WARN: unknown argument '-end' (ignored). Use -h for help.
WARN: unknown argument '120' (ignored). Use -h for help.
ERROR: invalid preset 'custom    '
       valid: quick, demo, dense, megacrowd, custom

C:\Users\albre\Downloads\New folder>run_case.cmd custom -vehicles 2500 -rows 3 -cols 3 -end 120
WARN: unknown argument '-vehicles' (ignored). Use -h for help.
WARN: unknown argument '2500' (ignored). Use -h for help.
WARN: unknown argument '-rows' (ignored). Use -h for help.
WARN: unknown argument '3' (ignored). Use -h for help.
WARN: unknown argument '-cols' (ignored). Use -h for help.
WARN: unknown argument '3' (ignored). Use -h for help.
WARN: unknown argument '-end' (ignored). Use -h for help.
WARN: unknown argument '120' (ignored). Use -h for help.
ERROR: invalid preset 'custom    '
       valid: quick, demo, dense, megacrowd, custom

C:\Users\albre\Downloads\New folder>run_case.cmd custom -vehicles 2500 -rows 3 -cols 3 -end 120
WARN: unknown argument '-vehicles' (ignored). Use -h for help.
WARN: unknown argument '2500' (ignored). Use -h for help.
WARN: unknown argument '-rows' (ignored). Use -h for help.
WARN: unknown argument '3' (ignored). Use -h for help.
WARN: unknown argument '-cols' (ignored). Use -h for help.
WARN: unknown argument '3' (ignored). Use -h for help.
WARN: unknown argument '-end' (ignored). Use -h for help.
WARN: unknown argument '120' (ignored). Use -h for help.
ERROR: invalid preset 'custom    '
       valid: quick, demo, dense, megacrowd, custom

C:\Users\albre\Downloads\New folder>w.cmd demo
ERROR: invalid preset 'demo      '
       valid: quick, demo, dense, megacrowd, custom, case

C:\Users\albre\Downloads\New folder>w.cmd -dry-run

============================================================
  GRID SIMULATION  --  demo preset
  Vehicles : 5000
  Grid     : 4 x 4   (turn: off)
  Window   : 0s -> 300s   (dt=0.5s, seed=0)
  Mode     : both   fps=25  frames<=200
  Out dir  : runs\live
============================================================

Using Python: python
  File "<string>", line 1
    import re,sys; s=open('grid_sim_all.py',encoding='utf-8',errors='ignore').read(); m=re.search(r'__version__\s*=\s*"(\d+)\.(\d+)',s); v=(int(m.group(1)),int(m.group(2))) if m else (0,0); sys.exit(0 if v,0) else 1)

                                                                                               ^
SyntaxError: unmatched ')'
WARN: grid_sim_all.py may be older than v3.0 ΓÇö expect
      'pruned N invalid route(s)' warnings and fewer routes.

>>> stage 1/3  generate

>>> stage 2/3  simulate
[sim] roadnet : runs\live\roadnet_4_4.json  (16 intersections, 80 roads)
[sim] flow    : runs\live\flow_4_4.json  (16 route definitions)
[sim] window  : 0 -> 300s  (dt = 0.5s)
[sim] wrote runs\live\sim_4_4.json  finished=496  peak_queue=4288  mean_speed_kmh=8.079

>>> stage 3/3  render (mode=both)
[viz] backend: qtagg
Target: 5,000 vehicles over 300s across 16 routes -> interval=0.960000s
Logistic y=201.480 -> grid 4x4, interval=0.9600s, lanes(L/S/R)=1/2/1, speed=14.93
Roadnet:   runs\live\roadnet_4_4.json
Flow:      runs\live\flow_4_4.json  (16 routes)
Summary:   runs\live\summary_4_4.json
[viz] simulating 300s at dt=0.5s (600 steps), capturing every 3 → ~200 frames
[viz] sim -> runs\live\sim_4_4.json (finished=496, peak_queue=4288)
[viz] gif -> runs\live\sim_4_4_matplotlib.gif
[viz] png -> runs\live\sim_4_4_final.png
[viz] opening interactive window — close it to exit ...
opening runs\live\sim_4_4_matplotlib.gif
opening runs\live\sim_4_4_final.png

============================================================
  DONE
============================================================
  Files in runs\live :
    finished  :        496   peak queue :      4,288   mean speed :    8.1 km/h
flow_4_4.json
roadnet_4_4.json
sim_4_4.json
sim_4_4_final.png
sim_4_4_matplotlib.gif
summary_4_4.json

  Re-open later :  run_case.cmd openlast
  Try a preset  :  run_case.cmd quick | dense | megacrowd | case
  Full help     :  run_case.cmd -h


C:\Users\albre\Downloads\New folder>run_case.cmd

============================================================
  GRID SIMULATION  --  demo preset
  Vehicles : 5000
  Grid     : 4 x 4   (turn: off)
  Window   : 0s -> 300s   (dt=0.5s, seed=0)
  Mode     : both   fps=25  frames<=200
  Out dir  : runs\live
============================================================

Using Python: python
  File "<string>", line 1
    import re,sys; s=open('grid_sim_all.py',encoding='utf-8',errors='ignore').read(); m=re.search(r'__version__\s*=\s*"(\d+)\.(\d+)',s); v=(int(m.group(1)),int(m.group(2))) if m else (0,0); sys.exit(0 if v,0) else 1)

                                                                                               ^
SyntaxError: unmatched ')'
WARN: grid_sim_all.py may be older than v3.0 ΓÇö expect
      'pruned N invalid route(s)' warnings and fewer routes.

>>> stage 1/3  generate

>>> stage 2/3  simulate
[sim] roadnet : runs\live\roadnet_4_4.json  (16 intersections, 80 roads)
[sim] flow    : runs\live\flow_4_4.json  (16 route definitions)
[sim] window  : 0 -> 300s  (dt = 0.5s)
[sim] wrote runs\live\sim_4_4.json  finished=496  peak_queue=4288  mean_speed_kmh=8.079

>>> stage 3/3  render (mode=both)
opening runs\live\sim_4_4_anim.gif
opening runs\live\sim_4_4_plot.png
opening runs\live\sim_4_4_traces.png

============================================================
  DONE
============================================================
  Files in runs\live :
    finished  :        496   peak queue :      4,288   mean speed :    8.1 km/h
flow_4_4.json
roadnet_4_4.json
sim_4_4.json
sim_4_4_anim.gif
sim_4_4_final.png
sim_4_4_frames.txt
sim_4_4_matplotlib.gif
sim_4_4_plot.png
sim_4_4_traces.png
summary_4_4.json

  Re-open later :  run_case.cmd openlast
  Try a preset  :  run_case.cmd quick | dense | megacrowd
  Full help     :  run_case.cmd -h
