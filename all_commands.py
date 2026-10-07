#!/usr/bin/env python3
"""
CITYFLOW ADVANCED FEATURE COUNTER - COMPLETE COMMAND REFERENCE
===============================================================

ALL REACHABLE COMMANDS, PARAMETERS, FEATURES, AND MODULES
Consolidated documentation for run.py and cityflow_features/

Run this file to display the complete help documentation.
"""

import sys
from datetime import datetime

# ============================================================================
# COMMAND HELP TEXT
# ============================================================================

MAIN_HEADER = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    RUN.PY - CITYFLOW ADVANCED FEATURE COUNTER                  ║
║                     ALL REACHABLE COMMANDS & PARAMETERS                        ║
║                           Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                           ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

SECTION_BASIC_USAGE = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ BASIC USAGE                                                                    │
└────────────────────────────────────────────────────────────────────────────────┘

    python run.py <config_file> [OPTIONS]

REQUIRED ARGUMENTS:
    config                  Path to CityFlow config.json file
                           Default: data/config.json
                           Example: python run.py data/config.json
"""

SECTION_SCALE_OPTIONS = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ SCALE OPTIONS (--scale)                                                        │
└────────────────────────────────────────────────────────────────────────────────┘

    --scale large           Large-scale 24x24 grid network simulation
                           • Intersections: 576 (24x24)
                           • Roads: ~96
                           • Lanes: ~288
                           • Peak Vehicles: 500
                           • Complexity: HIGH
                           • Time Factor: 1.0x
                           Related module: scale_large.py
                           
    --scale mid             Mid-scale 10x10 grid network simulation
                           • Intersections: 100 (10x10)
                           • Roads: ~40
                           • Lanes: ~120
                           • Peak Vehicles: 150
                           • Complexity: MEDIUM
                           • Time Factor: 0.25x
                           Related module: scale_mid.py
                           
    --scale both            Run BOTH large and mid-scale simulations sequentially
                           • Executes large scale first, then mid-scale
                           • Generates combined CSV output
                           • Useful for comparative analysis
                           
    Default: large

EXAMPLES:
    python run.py data/config.json --scale large
    python run.py data/config.json --scale mid
    python run.py data/config.json --scale both --output comparison.csv
"""

SECTION_SIMULATION_PARAMS = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ SIMULATION PARAMETERS                                                          │
└────────────────────────────────────────────────────────────────────────────────┘

    --steps N               Number of simulation steps to run
                           • Default: 500
                           • Range: 1-10000
                           • Each step represents ~1 second of simulation time
                           
    --output FILE.csv       Output CSV file for detailed metrics
                           • If not specified, prints to console only
                           • Contains all metrics across all categories
                           • One row per simulation step
                           
    --threads N             Number of simulation threads (parallel execution)
                           • Default: 1
                           • Recommended: 1-4 (higher may not improve)
                           • Useful for large-scale simulations

EXAMPLES:
    python run.py data/config.json --steps 1000
    python run.py data/config.json --output results.csv
    python run.py data/config.json --threads 4 --steps 500
"""

SECTION_COMMUNICATION = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ COMMUNICATION MODULES                                                          │
└────────────────────────────────────────────────────────────────────────────────┘

V2X - VEHICLE-TO-INFRASTRUCTURE COMMUNICATION:
    
    --v2x                   Enable V2X communication (DEFAULT)
                           • Infrastructure broadcasts traffic signal states
                           • Vehicles receive signal optimization data
                           • Used for adaptive signal timing
                           Related module: v2x_communication.py
                           
    --no-v2x                Disable V2X communication
                           • Runs simulation without V2X metrics

V2X Metrics Collected:
    • v2x_total_messages
    • v2x_avg_latency_ms
    • v2x_max_latency_ms
    • v2x_intersections_broadcasting
    • v2x_signal_optimizations
    • v2x_adaptive_phases
    • v2x_efficiency_score (0-1)


V2V - VEHICLE-TO-VEHICLE COMMUNICATION:
    
    --v2v                   Enable V2V communication (DEFAULT)
                           • Vehicles exchange safety and coordination messages
                           • Collision avoidance coordination
                           • Lane change requests
                           Related module: v2v_communication.py
                           
    --no-v2v                Disable V2V communication

V2V Metrics Collected:
    • v2v_total_messages
    • v2v_collision_warnings
    • v2v_cooperative_maneuvers
    • v2v_coordination_events
    • v2v_message_types (dict)
    • v2v_communication_range_m (100m)
    • v2v_success_rate (95%)


COMBINATION EXAMPLES:
    python run.py data/config.json --v2x --v2v
    python run.py data/config.json --v2x --no-v2v
    python run.py data/config.json --no-v2x --v2v
    python run.py data/config.json --no-v2x --no-v2v
"""

SECTION_TRAINING_ENV = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ TRAINING ENVIRONMENT (Non-MARL Mode)                                           │
└────────────────────────────────────────────────────────────────────────────────┘

    --training-env          Enable training environment metrics
                           • Activates all training-specific feature collection
                           • Discretizes continuous metrics for ML training
                           • Tracks signal states, safety constraints, algorithms
                           
Related modules activated:
    • signal_characterization.py
    • congestion_state_space.py
    • object_closure_parameters.py
    • traffic_control_algorithm.py

SIGNAL CHARACTERIZATION:
    Observation Space per Intersection:
    • phase: Current signal phase (0 or 1 for NS/EW)
    • duration: Current phase duration in seconds
    • queue_ratio: Vehicle distribution (NS vs EW)
    • avg_duration: Average phase duration over time
    • vehicles_ns: Vehicle count (North-South)
    • vehicles_ew: Vehicle count (East-West)

CONGESTION STATE SPACE:
    5 Discrete Congestion Levels:
    • State 0: FREE       (congestion < 0.2)
    • State 1: LIGHT      (0.2 ≤ congestion < 0.4)
    • State 2: MODERATE   (0.4 ≤ congestion < 0.6)
    • State 3: HEAVY      (0.6 ≤ congestion < 0.8)
    • State 4: SEVERE     (congestion ≥ 0.8)
    
    Metrics:
    • congestion_current_state
    • congestion_current_state_name
    • congestion_total_state_transitions
    • congestion_state_distribution (histogram)

OBJECT CLOSURE & SAFETY CONSTRAINTS:
    Parameters:
    • Min Safety Gap: 2.5 meters
    • Max Safe Speed: 15.0 m/s
    • Safe Deceleration: 4.5 m/s²
    
    Metrics:
    • safety_constraint_violations
    • safety_collision_events
    • safety_near_miss_events
    • safety_closure_events_total
    • safety_closure_events_by_type (dict)

USAGE:
    python run.py data/config.json --training-env
"""

SECTION_ALGORITHMS = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ TRAFFIC CONTROL ALGORITHMS (--algorithm)                                       │
└────────────────────────────────────────────────────────────────────────────────┘

NOTE: Algorithm requires --training-env flag to be active

    --algorithm fixed-time  Fixed-Time Signal Control (BASELINE)
                           • No adaptation - simple fixed 60-second cycle
                           • Phase 0 (NS): 0-30 seconds
                           • Phase 1 (EW): 30-60 seconds
                           • Baseline for comparison
                           • Metric: algorithm_optimization_events = 0
                           
    --algorithm adaptive    Adaptive Queue-Length Control
                           • Adjusts phase times based on vehicle queue ratios
                           • Base time: 30 seconds
                           • High congestion (>0.7): Prioritizes busier direction
                           • Multipliers: 1.3x for busy, 0.7x for quiet direction
                           • Optimization events: +1 per high-congestion decision
                           
    --algorithm optimized   Optimized Multi-Parameter Control (DEFAULT)
                           • Uses urgency scoring: 40% vehicle count + 60% waiting
                           • Dynamic cycle time based on congestion:
                             - High congestion (>0.8): 48 seconds (0.8x)
                             - Low congestion (<0.3): 72 seconds (1.2x)
                             - Normal: 60 seconds
                           • Proportional green time allocation
                           • Best performance in mixed traffic conditions

METRICS (with --training-env --algorithm):
    • algorithm_type: String (fixed-time|adaptive|optimized)
    • algorithm_total_decisions: Total signal decisions made
    • algorithm_optimization_events: Count of adaptive decisions
    • algorithm_avg_phase_duration: Average duration of phases

USAGE EXAMPLES:
    python run.py data/config.json --training-env --algorithm fixed-time
    python run.py data/config.json --training-env --algorithm adaptive
    python run.py data/config.json --training-env --algorithm optimized --output opt.csv
    
ALGORITHM COMPARISON:
    python run.py data/config.json --scale large --training-env --algorithm fixed-time --output baseline.csv
    python run.py data/config.json --scale large --training-env --algorithm adaptive --output adaptive.csv
    python run.py data/config.json --scale large --training-env --algorithm optimized --output optimized.csv
"""

SECTION_ALL_FEATURES = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ ALL COUNTABLE FEATURES & METRICS                                               │
└────────────────────────────────────────────────────────────────────────────────┘

1. VEHICLE METRICS (ALWAYS COLLECTED):
   ├─ vehicle_count: Total vehicles in simulation
   ├─ avg_vehicle_speed: Average speed (m/s)
   ├─ max_vehicle_speed: Maximum speed observed
   ├─ min_vehicle_speed: Minimum speed observed
   ├─ total_vehicle_distance: Sum of all distances traveled
   └─ avg_vehicle_distance: Average distance per vehicle

2. LANE METRICS (ALWAYS COLLECTED):
   ├─ total_lane_vehicles: Sum across all lanes
   ├─ avg_vehicles_per_lane: Average occupancy
   ├─ max_vehicles_in_lane: Peak lane occupancy
   ├─ total_waiting_vehicles: Vehicles stopped/moving slow
   ├─ avg_waiting_per_lane: Average waiting per lane
   ├─ max_waiting_in_lane: Peak waiting in single lane
   └─ congestion_level: Ratio of waiting to total (0-1)

3. TIME METRICS (ALWAYS COLLECTED):
   ├─ current_time: Simulation clock (seconds)
   ├─ step_count: Total steps executed
   └─ average_travel_time: Avg vehicle travel time

4. ROAD NETWORK METRICS (STATIC):
   ├─ total_roads: Number of road segments
   ├─ total_intersections: Number of intersections
   ├─ total_lanes: Sum of all lanes
   └─ avg_lanes_per_road: Average lanes per road

5. TRAFFIC FLOW METRICS (STATIC):
   ├─ total_traffic_flows: Number of O-D flows
   └─ total_vehicles_in_flow: Total vehicles generated by flows

6. SIGNAL METRICS (WITH --training-env):
   ├─ signal_total_phase_records: Count of signal state recordings
   ├─ signal_total_phase_changes: Phase transitions
   ├─ signal_intersections_tracked: Intersections with signals
   └─ signal_avg_phases_per_intersection: Average signals per intersection

7. CONGESTION STATE SPACE (WITH --training-env):
   ├─ congestion_state_space_levels: 5 discrete levels
   ├─ congestion_total_state_transitions: State changes count
   ├─ congestion_current_state: Current discrete state (0-4)
   ├─ congestion_current_state_name: Human-readable name
   ├─ congestion_state_distribution: Histogram of states
   └─ congestion_most_common_state: Most frequent state

8. OBJECT CLOSURE & SAFETY (WITH --training-env):
   ├─ safety_min_gap_meters: Minimum safety distance (2.5m)
   ├─ safety_max_speed_ms: Maximum safe speed (15 m/s)
   ├─ safety_deceleration_ms2: Safe deceleration (4.5 m/s²)
   ├─ safety_constraint_violations: Constraint breaches
   ├─ safety_collision_events: Actual collisions
   ├─ safety_near_miss_events: Near-miss events
   ├─ safety_closure_events_total: Total closure events
   └─ safety_closure_events_by_type: Dict of event types

9. TRAFFIC CONTROL (WITH --training-env --algorithm):
   ├─ algorithm_type: Control algorithm name
   ├─ algorithm_total_decisions: Signal decisions made
   ├─ algorithm_optimization_events: Adaptive events
   └─ algorithm_avg_phase_duration: Average phase length

10. V2X COMMUNICATION (WITH --v2x):
    ├─ v2x_total_messages: Messages sent
    ├─ v2x_avg_latency_ms: Average latency (milliseconds)
    ├─ v2x_max_latency_ms: Maximum latency observed
    ├─ v2x_intersections_broadcasting: Active intersections
    ├─ v2x_signal_optimizations: Optimization count
    ├─ v2x_adaptive_phases: Adaptive phase changes
    └─ v2x_efficiency_score: Efficiency score (0-1)

11. V2V COMMUNICATION (WITH --v2v):
    ├─ v2v_total_messages: Messages exchanged
    ├─ v2v_collision_warnings: Warnings issued
    ├─ v2v_cooperative_maneuvers: Coordinated maneuvers
    ├─ v2v_coordination_events: Total coordination events
    ├─ v2v_message_types: Dict of message categories
    ├─ v2v_communication_range_m: Range limit (100m)
    └─ v2v_success_rate: Success rate (95%)

12. COMPOSITE METRICS:
    ├─ System Efficiency Score: (avg_speed / max_speed) × (1 - avg_congestion)
    ├─ Network Capacity Utilization: (avg_lane_occupancy / 20) × 100%
    ├─ System Throughput: total_distance / step_count (m/step)
    └─ Communication Overhead: v2x_messages + v2v_messages
"""

SECTION_MODULES = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ REACHABLE MODULES IN cityflow_features/                                        │
└────────────────────────────────────────────────────────────────────────────────┘

Module Name                     Command Trigger                 Purpose
─────────────────────────────────────────────────────────────────────────────────
__init__.py                    (package initialization)        Imports all modules
signal_characterization.py      --training-env                 Signal state observation space
congestion_state_space.py       --training-env                 Discretized congestion (5 levels)
object_closure_parameters.py    --training-env                 Safety constraints & closure
traffic_control_algorithm.py    --training-env --algorithm     Non-MARL control algorithms
v2x_communication.py            --v2x                          Infrastructure communication
v2v_communication.py            --v2v                          Vehicle-to-vehicle coordination
scale_large.py                  --scale large                  24x24 grid configuration
scale_mid.py                    --scale mid                    10x10 grid configuration
help_commands.py                (documentation)                Complete command reference


IMPORT USAGE:
    from cityflow_features import (
        SignalCharacterization,
        CongestionStateSpace,
        ObjectClosureParameters,
        TrafficControlAlgorithm,
        V2XCommunication,
        V2VCommunication,
        LargeScaleConfig,
        MidScaleConfig
    )
"""

SECTION_EXAMPLES = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ COMPLETE COMMAND EXAMPLES                                                      │
└────────────────────────────────────────────────────────────────────────────────┘

1. BASIC RUNS:
   # Default: large-scale, 500 steps, V2X+V2V enabled
   python run.py data/config.json
   
   # Large-scale, 1000 steps, save to CSV
   python run.py data/config.json --scale large --steps 1000 --output large.csv
   
   # Mid-scale, 300 steps
   python run.py data/config.json --scale mid --steps 300

2. COMMUNICATION EXPERIMENTS:
   # V2X only
   python run.py data/config.json --v2x --no-v2v --output v2x_only.csv
   
   # V2V only
   python run.py data/config.json --no-v2x --v2v --output v2v_only.csv
   
   # Both enabled (default)
   python run.py data/config.json --v2x --v2v --output both_comm.csv
   
   # No communication
   python run.py data/config.json --no-v2x --no-v2v --output no_comm.csv

3. TRAINING ENVIRONMENT (Non-MARL):
   # Training with fixed-time baseline
   python run.py data/config.json --training-env --algorithm fixed-time \\
       --output baseline.csv
   
   # Training with adaptive control
   python run.py data/config.json --training-env --algorithm adaptive \\
       --output adaptive.csv
   
   # Training with optimized control
   python run.py data/config.json --training-env --algorithm optimized \\
       --output optimized.csv

4. SCALE COMPARISONS:
   # Both scales, no training
   python run.py data/config.json --scale both --v2x --v2v \\
       --output both_scales.csv
   
   # Both scales with training environment
   python run.py data/config.json --scale both --training-env \\
       --algorithm adaptive --output both_with_training.csv

5. FULL FEATURE RUNS:
   # Everything enabled
   python run.py data/config.json --scale large --steps 500 \\
       --v2x --v2v --training-env --algorithm optimized \\
       --output full_features.csv --threads 4
   
   # Large scale, training, optimized algorithm, 2000 steps
   python run.py data/config.json --scale large --steps 2000 \\
       --training-env --algorithm optimized \\
       --output large_2k_steps.csv

6. ANALYSIS PIPELINES:
   # Compare all 3 algorithms on same scale
   for algo in fixed-time adaptive optimized; do
     python run.py data/config.json --scale large \\
         --training-env --algorithm $algo --steps 500 \\
         --output results_${algo}.csv
   done
   
   # Compare both scales with V2X+V2V
   python run.py data/config.json --scale both --v2x --v2v \\
       --steps 500 --output both_scales_v2x_v2v.csv

7. PERFORMANCE TESTING:
   # Large scale, multi-threaded
   python run.py data/config.json --scale large --threads 4 --steps 1000
   
   # Mid scale, quick test
   python run.py data/config.json --scale mid --steps 100 --output quick_test.csv

8. HELP AND DOCUMENTATION:
   # Display full help
   python cityflow_features/help_commands.py
   
   # Test individual modules
   python cityflow_features/signal_characterization.py
   python cityflow_features/traffic_control_algorithm.py
   python cityflow_features/v2x_communication.py
"""

SECTION_OUTPUT = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT FORMAT                                                                  │
└────────────────────────────────────────────────────────────────────────────────┘

CONSOLE OUTPUT:
    Formatted summary tables with sections:
    • Road Network Infrastructure Features
    • Traffic Flow Features
    • Vehicle Metrics (Aggregate)
    • Lane Metrics (Aggregate)
    • Congestion Metrics
    • Time Metrics
    • Signal & Traffic Control Metrics (if --training-env)
    • Congestion State Space (if --training-env)
    • Object Closure & Safety Parameters (if --training-env)
    • V2X Communication (if --v2x)
    • V2V Communication (if --v2v)
    • Composite System Metrics

CSV OUTPUT (with --output FILE.csv):
    Columns include:
    • step: Simulation step number
    • scale: Scale mode (large/mid)
    • timestamp: ISO 8601 timestamp
    • All metric categories listed in "All Countable Features" section
    
    One row per simulation step, all metrics in columns.
    Can be imported directly into pandas, Excel, or analysis tools.
"""

SECTION_QUICK_REF = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ QUICK REFERENCE                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

CORE FLAGS:
    --scale [large|mid|both]        Network size
    --steps N                       Simulation duration
    --output FILE.csv               Save results
    --threads N                     Parallel threads

COMMUNICATION:
    --v2x / --no-v2x                Vehicle-to-Infrastructure
    --v2v / --no-v2v                Vehicle-to-Vehicle

TRAINING:
    --training-env                  Enable training metrics
    --algorithm [fixed-time|adaptive|optimized]

MINIMAL COMMANDS:
    python run.py data/config.json
    python run.py data/config.json --scale mid
    python run.py data/config.json --training-env --algorithm adaptive

COMPREHENSIVE:
    python run.py data/config.json --scale both --v2x --v2v \\
        --training-env --algorithm optimized --steps 500 --output results.csv
"""

SECTION_FOOTER = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  For issues, examples, or questions, see: cityflow_features/help_commands.py  ║
║                                                                                ║
║  Generated: CityFlow Advanced Feature Counter v1.0                             ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""


def print_complete_help():\n    \"\"\"Print all help sections\"\"\"\n    sections = [\n        MAIN_HEADER,\n        SECTION_BASIC_USAGE,\n        SECTION_SCALE_OPTIONS,\n        SECTION_SIMULATION_PARAMS,\n        SECTION_COMMUNICATION,\n        SECTION_TRAINING_ENV,\n        SECTION_ALGORITHMS,\n        SECTION_ALL_FEATURES,\n        SECTION_MODULES,\n        SECTION_EXAMPLES,\n        SECTION_OUTPUT,\n        SECTION_QUICK_REF,\n        SECTION_FOOTER\n    ]\n    \n    for section in sections:\n        print(section)\n        print()\n\n\ndef print_quick_help():\n    \"\"\"Print only quick reference\"\"\"\n    print(MAIN_HEADER)\n    print(SECTION_QUICK_REF)\n    print(SECTION_FOOTER)\n\n\nif __name__ == '__main__':\n    # Check for command-line arguments\n    if len(sys.argv) > 1:\n        if sys.argv[1] in ['-q', '--quick']:\n            print_quick_help()\n        elif sys.argv[1] in ['-h', '--help']:\n            print_complete_help()\n        else:\n            print(f\"Usage: python {sys.argv[0]} [-q|--quick] [-h|--help]\")\n            print(f\"  -q, --quick: Show quick reference only\")\n            print(f\"  -h, --help:  Show complete help (default)\")\n    else:\n        # Default: print complete help\n        print_complete_help()
