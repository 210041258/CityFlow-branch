# 🚦 Traffic Intersection Visualizer

A professional, high‑performance visualization tool for traffic intersection simulations. Reads standard `roadnet.json` and `replay.txt` formats, generates publication‑ready static frames, and animates complete traffic flow with lane‑accurate traffic lights.


---

## ✨ Features

### 🎯 Core Visualization

* Accurate road geometry from `roadnet.json`
* Solid outer lane edges and dashed lane dividers
* Lane‑accurate traffic lights (R/G/Y per lane)
* Detailed vehicles with headlights, taillights, and shadows
* Stable vehicle colors across all frames

### 🚀 Performance

* Blitted animation (static infrastructure drawn once)
* 10–50× faster than full redraw approaches
* Smooth playback with 1000+ frames

### 📊 Reporting & Export

* Console summary tables (intersections, edges, lights)
* High‑resolution static PNG (1800×1800)
* Vector PDF export
* Optional MP4 export (requires ffmpeg)

