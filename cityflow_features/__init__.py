#!/usr/bin/env python3
"""
CityFlow Feature Counter - All Reachable Modules
Organized feature modules for run.py command-line argument handling

Modules:
  - signal_characterization: --training-env (signal state observation space)
  - congestion_state_space: --training-env (discretized congestion levels)
  - object_closure_parameters: --training-env (safety constraints)
  - traffic_control_algorithm: --training-env --algorithm [fixed-time|adaptive|optimized]
  - v2x_communication: --v2x (vehicle-to-infrastructure)
  - v2v_communication: --v2v (vehicle-to-vehicle)
  - scale_large: --scale large (24x24 grid)
  - scale_mid: --scale mid (10x10 grid)
  - help_commands: Documentation of all reachable commands
"""

from .signal_characterization import SignalCharacterization
from .congestion_state_space import CongestionStateSpace
from .object_closure_parameters import ObjectClosureParameters
from .traffic_control_algorithm import TrafficControlAlgorithm
from .v2x_communication import V2XCommunication
from .v2v_communication import V2VCommunication
from .scale_large import LargeScaleConfig
from .scale_mid import MidScaleConfig

__all__ = [
    'SignalCharacterization',
    'CongestionStateSpace',
    'ObjectClosureParameters',
    'TrafficControlAlgorithm',
    'V2XCommunication',
    'V2VCommunication',
    'LargeScaleConfig',
    'MidScaleConfig'
]
