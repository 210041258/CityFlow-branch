# Frontend Replay and Algorithm Checks

## Purpose

This folder replays CityFlow-style traffic data against road-network geometry so vehicle movement and traffic-signal states can be inspected visually. `replay_simu.py` reads `roadnet.json` and `replay.txt`, builds a network view, parses frames, writes a static PNG/PDF, and can animate the frames. `current/currnet.py` provides streaming vehicle-count analysis, model fitting, and traffic-dynamics summaries.

## Algorithm-check workflow

Use replay visualization and statistical analysis as complementary checks—not as proof that a traffic-control algorithm is correct or effective.

1. **Check the inputs.** Confirm the road-network and replay paths, JSON structure, frame delimiter, and expected vehicle/light fields. The replay parser skips blank or malformed entries; record how many frames and records are retained so skipped data are visible.
2. **Check the network mapping.** Confirm that replay road IDs, intersection IDs, lanes, and traffic-light entries correspond to the network. The current visualization code assumes the central intersection ID `intersection_1_1` and a compatible roadnet structure; inspect these assumptions before using another map.
3. **Check frame progression.** Render selected frames and inspect vehicle positions, route continuity, signal colors, and frame order. A visually plausible animation is a diagnostic, not a traffic-model validation.
4. **Check aggregate behavior.** Use `currnet.py` to derive vehicle counts per frame and, when appropriate, growth, congestion peaks, and trend summaries. Its streaming count parser ignores blank lines and lines without `;`; compare parsed frame counts with the source file.
5. **Compare candidate models.** The analysis supports polynomial fits and, when SciPy is installed, exponential and logistic fits. Compare residual behavior and AIC/BIC as well as R²; a lower AIC is a relative model-selection signal for the compared candidates, not evidence of causality or out-of-sample accuracy.
6. **Validate independently.** Use a held-out time interval or additional scenarios, document seeds and settings, compare with suitable fixed-time or established baselines, and report raw observations with metrics. The replay visualizer itself does not run a controlled policy comparison.

## References and how they relate

The references below cover CityFlow and other simulators, traffic-signal reinforcement learning, multi-agent coordination, traffic-flow prediction, and surveys. They provide research context for the modeling and evaluation questions; they do not independently verify this repository's implementation. To verify an algorithm block, match the code's actual state, action, reward, data assumptions, and evaluation protocol to the relevant paper, then reproduce its stated baseline and metrics where feasible. Cite a source only for claims it actually supports.

## References

Distributed Multi-Agent Reinforcement Learning for Traffic Signal Control
https://doi.org/10.1109/ICASSP.2019.8682160

Traffic Signal Control with Deep Q-Learning and Graph-Based State Representation
https://doi.org/10.1109/ITSC.2021.9564476

Urban Traffic Flow Prediction with Deep Learning: A Survey
https://doi.org/10.1016/j.trc.2021.103290

MAXPRESSURE Control for Signalized Intersections
https://doi.org/10.3141/2259-03

Modeling and Simulation of Urban Traffic Networks with Realistic Flow Patterns
https://doi.org/10.1016/j.simpat.2018.04.002

Reinforcement Learning Methods for Adaptive Traffic Signal Control: A Review
https://doi.org/10.1109/OJITS.2022.3157413

A Deep Learning Framework for Real-Time Urban Traffic Signal Control
https://doi.org/10.1109/ITSC.2022.9914867
