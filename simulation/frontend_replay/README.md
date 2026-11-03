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

SocialLight: Distributed Cooperation Learning towardsNetwork-Wide Traffic Signal Control
https://dl.acm.org/doi/epdf/10.5555/3545946.3598809

CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario
https://dl.acm.org/doi/pdf/10.1145/3308558.3314139

IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control
https://doi.org/10.1145/3219819.3220096

Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control
https://doi.org/10.1109/TITS.2019.2913345

Traffic Signal Timing via Deep Reinforcement Learning
https://doi.org/10.1109/JAS.2016.7471730

Multiagent Reinforcement Learning for Integrated Network of Adaptive Traffic Signal Controllers (MARLIN-ATSC): Methodology and Large-Scale Application on Downtown Toronto
https://doi.org/10.1109/TITS.2013.2255288

PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network
https://doi.org/10.1145/3292500.3330949

CoLight: Learning Network-level Cooperation for Traffic Signal Control
https://doi.org/10.1145/3357384.3357902

Learning Phase Competition for Traffic Signal Control
https://proceedings.mlr.press/v97/zheng19d.html

Coordinated Deep Reinforcement Learners for Traffic Light Control
https://arxiv.org/abs/1609.01120

Using a Deep Reinforcement Learning Agent for Traffic Signal Control
https://arxiv.org/abs/1611.01142

An Experimental Review of Reinforcement Learning Algorithms for Adaptive Traffic Signal Control
https://doi.org/10.1007/978-3-319-25808-9_3

Traffic Signal Control Based on Reinforcement Learning with Graph Convolutional Neural Nets
https://doi.org/10.1109/ACCESS.2020.2995771

Multiagent Reinforcement Learning for Urban Traffic Control Using Coordination Graphs
https://doi.org/10.1007/978-3-540-87479-9_62

Simulation of Urban MObility (SUMO)
https://doi.org/10.1109/MITS.2018.2876558

Microscopic Traffic Simulation using SUMO
https://doi.org/10.1109/ITSC.2012.6338689

Multi-Agent Reinforcement Learning: A Critical Survey
https://doi.org/10.1109/TSMCC.2008.2009436

A Review of Urban Traffic Management: From Classical Methods to Reinforcement Learning and Swarm Intelligence
https://doi.org/10.1109/TITS.2022.3143077

Deep Reinforcement Learning Approaches for Urban Traffic Optimization: A Review
https://doi.org/10.3390/s22093250

Urban Traffic Control: An Overview
https://doi.org/10.1109/ACCESS.2021.3118633

Deep Learning for Short-Term Traffic Flow Prediction
https://doi.org/10.1016/j.trc.2017.02.024

Deep Reinforcement Learning for Traffic Signal Control: A Survey
https://doi.org/10.1109/ACCESS.2020.2993358

Hierarchical Reinforcement Learning for Traffic Signal Control
https://doi.org/10.1109/ITSC.2019.8917468

Adaptive Traffic Signal Control using Reinforcement Learning with Function Approximation
https://doi.org/10.1109/ITSC.2016.7795656

Large-Scale Multi-Agent Traffic Signal Control with DQN and Graph Neural Networks
https://doi.org/10.1109/ITSC.2020.9294497

A Survey of Reinforcement Learning for Traffic Signal Control
https://doi.org/10.1109/TITS.2020.2999930

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
