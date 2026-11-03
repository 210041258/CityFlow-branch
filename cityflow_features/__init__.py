#!/usr/bin/env python3
"""Public feature exports for the CityFlow feature package.

The normalization utilities are general-purpose operational indicators that
can be used alongside the traffic features. They do not grant permissions or
authorize access to people, software, hardware, or CityFlow resources.
"""

from .signal_characterization import SignalCharacterization
from .congestion_state_space import CongestionStateSpace
from .object_closure_parameters import ObjectClosureParameters
from .traffic_control_algorithm import TrafficControlAlgorithm
from .v2x_communication import V2XCommunication
from .v2v_communication import V2VCommunication
from .scale_large import LargeScaleConfig
from .scale_mid import MidScaleConfig
from .factor_normalization import (
    FactorSpec,
    FactorNormalizer,
    PEOPLE_MANAGEMENT_FACTORS,
    SOFTWARE_FACTORS,
    HARDWARE_FACTORS,
    default_normalizers,
    normalize_operational_record,
)

__version__ = "1.1.0"

__all__ = [
    "SignalCharacterization",
    "CongestionStateSpace",
    "ObjectClosureParameters",
    "TrafficControlAlgorithm",
    "V2XCommunication",
    "V2VCommunication",
    "LargeScaleConfig",
    "MidScaleConfig",
    "FactorSpec",
    "FactorNormalizer",
    "PEOPLE_MANAGEMENT_FACTORS",
    "SOFTWARE_FACTORS",
    "HARDWARE_FACTORS",
    "default_normalizers",
    "normalize_operational_record",
]
