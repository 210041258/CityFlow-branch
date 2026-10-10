# 📂 CityFlow Case Studies

This folder contains **experimental scripts and algorithms** designed to evaluate active learning strategies and traffic control methods in CityFlow.  
It serves as a sandbox for testing optimization approaches, scenario variations, and algorithmic improvements.

---

## 📑 Contents & Thesis Mapping

- **`traffic_control_algorithm.py`**  
  ↳ 📖 Ch. 3 (Methodology), Ch. 4 (System Implementation)  
  Core traffic control logic and adaptive decision-making algorithms.

- **`scenario_matrix.py`, `scenario_all.py`, `scenario_extensions.py`**  
  ↳ 📖 Ch. 4 §4.4 (Experimental Design), Table 6 (Demand Scenarios), Table 9 (Severity Scenarios)  
  Tools for generating systematic scenario sweeps and aggregating definitions.

- **`scenario_advanced.py`**  
  ↳ 📖 Ch. 6 §6.5 (Limitations — simplified communication model), Table 17 (Future Research Roadmap — Short-term)  
  Defines advanced traffic scenarios with complex dynamics.

- **`scenario_ramps.py`**  
  ↳ 📖 Ch. 5 §5.3 (Logistic Degradation Model), page 51 (Findings F2)  
  Focused scenario modeling ramp traffic conditions.

- **`adaptive_multi_execution.py`**  
  ↳ 📖 Ch. 4 §4.4 (Distributed agent execution)  
  Framework for running multiple adaptive executions to test robustness across seeds/settings.

- **`analytical_surrogate.py`, `bayesian_active_learning.py`, `trust_region_active_learning.py`, `trust_region_decay.py`**  
  ↳ 🚀 Novel extensions **not included in the thesis** — exploratory methods beyond scope.  
  Implements surrogate modeling and active learning strategies (Global EI, Fixed TuRBO, Decayed TuRBO).

- **`readme.md`**  
  Documentation for this folder.

---

## 🔬 Algorithm Comparison (Novel Extensions)

| Variant | When to Use | Mean MAE at b=120 |
|---------|-------------|-------------------|
| **Global EI** (`bayesian_active_learning.py`) | Pure exploration, no frontier bias | 3.106 |
| **Fixed TuRBO** (`trust_region_active_learning.py`) | Tight budget, clear local optimum | 2.241 |
| **Decayed TuRBO** (`trust_region_decay.py`) | Same budget, but want the last round to still explore | **1.798** |

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


