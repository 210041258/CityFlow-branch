# 📂 CityFlow Case Studies

This folder contains **experimental scripts and algorithms** designed to evaluate active learning strategies and traffic control methods in CityFlow.  
It serves as a sandbox for testing optimization approaches, scenario variations, and algorithmic improvements.

---

## 📑 Contents

- **`adaptive_multi_execution.py`** – Framework for running multiple adaptive executions to test robustness across different seeds and settings.
- **`analytical_surrogate.py`** – Implements surrogate modeling to approximate traffic outcomes and reduce computational cost.
- **`bayesian_active_learning.py`** – Global Expected Improvement (EI) strategy for pure exploration without frontier bias.
- **`trust_region_active_learning.py`** – Fixed TuRBO trust-region optimization, ideal for tight budgets and clear local optima.
- **`trust_region_decay.py`** – Decayed TuRBO variant that maintains exploration in later rounds, balancing exploitation and exploration.
- **`scenario_advanced.py`** – Defines advanced traffic scenarios with complex dynamics.
- **`scenario_all.py`** – Unified runner that aggregates multiple scenario definitions for batch testing.
- **`scenario_extensions.py`** – Adds extended scenario definitions for specialized experiments.
- **`scenario_matrix.py`** – Generates matrix-based scenario configurations for systematic sweeps.
- **`scenario_ramps.py`** – Focused scenario modeling ramp traffic conditions.
- **`traffic_control_algorithm.py`** – Core traffic control logic and adaptive decision-making algorithms.
- **`readme.md`** – Documentation for this folder.

---

## 🔬 Algorithm Comparison

| Variant | When to Use | Mean MAE at b=120 |
|---------|-------------|-------------------|
| **Global EI** (`bayesian_active_learning.py`) | Pure exploration, no frontier bias | 3.106 |
| **Fixed TuRBO** (`trust_region_active_learning.py`) | Tight budget, clear local optimum | 2.241 |
| **Decayed TuRBO** (`trust_region_decay.py`) | Same budget, but want the last round to still explore | **1.798** |

---
💡 *Tip:* Choose the variant based on your budget constraints and whether exploration or exploitation is more critical for your experiment.
---
## 📈 Visual Comparison

```mermaid
bar
    title Mean MAE at b=120
    x-axis Variants
    y-axis MAE
    "Global EI" : 3.106
    "Fixed TuRBO" : 2.241
    "Decayed TuRBO" : 1.798


