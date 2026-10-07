#!/usr/bin/env python3
"""
Command Help and Usage Guide for run.py
Shows all reachable commands and their parameters
"""

COMMAND_HELP = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    RUN.PY - CITYFLOW ADVANCED FEATURE COUNTER                  ║
║                        ALL REACHABLE COMMANDS & ARGUMENTS                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
    python run.py <config_file> [OPTIONS]

REQUIRED ARGUMENTS:
    config                  Path to CityFlow config.json file
                           Default: data/config.json

═══════════════════════════════════════════════════════════════════════════════════

SCALE OPTIONS (--scale):
    
    --scale large           Large-scale 24x24 grid network simulation
                           Intersections: 576 (24x24)
                           Complexity: HIGH
                           Related module: scale_large.py
                           
    --scale mid             Mid-scale 10x10 grid network simulation
                           Intersections: 100 (10x10)
                           Complexity: MEDIUM
                           Related module: scale_mid.py
                           
    --scale both            Run BOTH large and mid-scale simulations sequentially
                           Generates combined CSV output
                           
    Default: large

═══════════════════════════════════════════════════════════════════════════════════

SIMULATION PARAMETERS:
    
    --steps N               Number of simulation steps to run
                           Default: 500
                           Range: 1-10000
                           
    --output FILE.csv       Output CSV file for detailed metrics
                           If not specified, prints to console only
                           
    --threads N             Number of simulation threads (parallel execution)
                           Default: 1
                           Recommended: 1-4 (higher may not improve performance)

═══════════════════════════════════════════════════════════════════════════════════

COMMUNICATION MODULES:

V2X (Vehicle-to-Infrastructure):
    
    --v2x                   Enable V2X communication
                           Tracks: infrastructure broadcasts, signal optimization
                           Related module: v2x_communication.py
                           
    --no-v2x                Disable V2X communication
                           Default: ENABLED
                           
Metrics collected:
    - v2x_total_messages
    - v2x_avg_latency_ms
    - v2x_intersections_broadcasting
    - v2x_signal_optimizations
    - v2x_efficiency_score

V2V (Vehicle-to-Vehicle):
    
    --v2v                   Enable V2V communication
                           Tracks: collision warnings, cooperative maneuvers
                           Related module: v2v_communication.py
                           
    --no-v2v                Disable V2V communication
                           Default: ENABLED

Metrics collected:
    - v2v_total_messages
    - v2v_collision_warnings
    - v2v_cooperative_maneuvers
    - v2v_communication_range_m
    - v2v_success_rate

═══════════════════════════════════════════════════════════════════════════════════

TRAINING ENVIRONMENT (Non-MARL Mode):
    
    --training-env          Enable training environment metrics
                           Activates: signal characterization, congestion state space,
                                      object closure parameters, traffic control
                           Related modules:
                               - signal_characterization.py
                               - congestion_state_space.py
                               - object_closure_parameters.py
                               - traffic_control_algorithm.py
                           
Metrics collected:
    - Signal state space (phase, duration, queue ratios)
    - Congestion state discretization (5 levels: FREE→SEVERE)
    - Safety constraints (collisions, near-misses, violations)
    - Algorithm performance (decisions, optimizations)

═══════════════════════════════════════════════════════════════════════════════════

TRAFFIC CONTROL ALGORITHMS (--algorithm):
    
    --algorithm fixed-time  Fixed-time signal control (baseline)
                           60-second cycle, no adaptation
                           Related module: traffic_control_algorithm.py
                           
    --algorithm adaptive    Queue-length responsive control
                           Adjusts timing based on vehicle density
                           Optimization events on high congestion
                           
    --algorithm optimized   Multi-parameter optimized control (DEFAULT)
                           Uses urgency scoring and dynamic cycle times
                           Balances waiting vehicles and queue lengths
                           
Note: Algorithm requires --training-env flag to be active

═══════════════════════════════════════════════════════════════════════════════════

FEATURES MEASURED:

1. VEHICLE METRICS:
   - Count (current, max, min, average)
   - Speed (average, max, min)
   - Distance (total, average)
   - Travel time (average)

2. LANE METRICS:
   - Vehicle count per lane
   - Waiting vehicles
   - Congestion level (0-1)

3. TIME METRICS:
   - Current simulation time
   - Step count
   - Average travel time

4. SIGNAL METRICS (with --training-env):
   - Phase records
   - Phase change count
   - Intersections tracked
   - Average phases per intersection

5. CONGESTION STATE SPACE (with --training-env):
   - Discrete state levels (5: FREE, LIGHT, MODERATE, HEAVY, SEVERE)
   - State transitions
   - Current state name
   - State distribution

6. OBJECT CLOSURE & SAFETY (with --training-env):
   - Minimum safety gap: 2.5m
   - Collision events
   - Near-miss events
   - Safety constraint violations
   - Closure events by type

7. TRAFFIC CONTROL (with --training-env --algorithm):
   - Algorithm type
   - Total decisions
   - Optimization events
   - Average phase duration

8. V2X COMMUNICATION (with --v2x):
   - Total messages sent
   - Average latency
   - Intersections broadcasting
   - Signal optimizations
   - Efficiency score

9. V2V COMMUNICATION (with --v2v):
   - Total messages
   - Collision warnings
   - Cooperative maneuvers
   - Message types distribution
   - Communication success rate

10. COMPOSITE METRICS:
    - System efficiency score
    - Network capacity utilization
    - System throughput
    - Total communication overhead

═══════════════════════════════════════════════════════════════════════════════════

EXAMPLE COMMANDS:

# 1. Basic large-scale simulation
python run.py data/config.json --scale large --steps 500

# 2. Large-scale with V2X/V2V communication
python run.py data/config.json --scale large --v2x --v2v --output results.csv

# 3. Training environment with adaptive control
python run.py data/config.json --training-env --algorithm adaptive --output train.csv

# 4. Optimized algorithm comparison
python run.py data/config.json --algorithm optimized --training-env --output optimized.csv

# 5. Both scales with all features
python run.py data/config.json --scale both --v2x --v2v --training-env \
    --algorithm optimized --steps 500 --output full_analysis.csv

# 6. Mid-scale with fixed-time baseline
python run.py data/config.json --scale mid --algorithm fixed-time \
    --training-env --steps 300 --output baseline.csv

# 7. Large-scale without communication (core metrics only)
python run.py data/config.json --scale large --no-v2x --no-v2v --steps 500

# 8. Multi-threaded simulation
python run.py data/config.json --scale large --threads 4 --steps 500

═══════════════════════════════════════════════════════════════════════════════════

REACHABLE .PY MODULES IN cityflow_features/:

Module Name                     Triggered By                    Purpose
─────────────────────────────────────────────────────────────────────────────────
run.py                         (main script)                   Orchestrates all features
signal_characterization.py      --training-env                 Signal state observations
congestion_state_space.py       --training-env                 Congestion discretization (5 levels)
object_closure_parameters.py    --training-env                 Safety constraints & closures
traffic_control_algorithm.py    --training-env --algorithm     Non-MARL control strategies
v2x_communication.py            --v2x                          Infrastructure communication
v2v_communication.py            --v2v                          Vehicle-to-vehicle coordination
scale_large.py                  --scale large                  24x24 grid network config
scale_mid.py                    --scale mid                    10x10 grid network config

═══════════════════════════════════════════════════════════════════════════════════

OUTPUT FORMAT:

CSV Output includes all metrics in columns:
    - Timestamps and step numbers
    - Vehicle metrics (count, speed, distance, travel time)
    - Lane metrics (occupancy, waiting, congestion)
    - Signal metrics (phases, changes, optimizations)
    - Communication metrics (V2X, V2V message counts/latencies)
    - Training environment metrics (state, transitions, safety)
    - Composite system metrics (efficiency, throughput, utilization)

Console Output includes formatted summary tables with:
    - Feature category sections
    - Min/Max/Average statistics
    - State distribution histograms
    - Algorithm performance comparisons

═══════════════════════════════════════════════════════════════════════════════════
"""

def print_help():
    """Print full help text"""
    print(COMMAND_HELP)

def print_quick_reference():
    """Print quick reference"""
    quick = """
QUICK REFERENCE:
    python run.py <config>                              # Default: large scale, 500 steps
    python run.py <config> --scale mid                  # Mid-scale only
    python run.py <config> --scale both                 # Both scales
    python run.py <config> --training-env               # Enable training metrics
    python run.py <config> --algorithm adaptive         # Adaptive control
    python run.py <config> --v2x --v2v                  # Enable communications
    python run.py <config> --output results.csv         # Save to CSV
    python run.py <config> --steps 1000                 # 1000 steps
"""
    print(quick)

if __name__ == '__main__':
    print_help()
