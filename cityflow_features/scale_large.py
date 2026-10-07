#!/usr/bin/env python3
"""
Feature Counter Module - Scale Simulation (LARGE)
Used by: run.py --scale large (24x24 grid network)
"""

import json
import os

class LargeScaleConfig:
    """Configuration and metrics for large-scale (24x24) simulation"""
    
    def __init__(self, config_path='data/config.json'):
        self.config_path = config_path
        self.scale_name = 'large'
        self.grid_size = (24, 24)  # 24x24 intersections
        self.expected_roads = 96  # 24*4 (4 directions per intersection)
        self.expected_lanes = 288  # 96 roads * 3 lanes average
        self.expected_vehicles_peak = 500  # Expected peak vehicle count
        
    def get_scale_info(self):
        """Get large-scale simulation info"""
        return {
            'scale_type': self.scale_name,
            'grid_dimensions': f"{self.grid_size[0]}x{self.grid_size[1]}",
            'total_intersections': self.grid_size[0] * self.grid_size[1],
            'expected_roads': self.expected_roads,
            'expected_lanes': self.expected_lanes,
            'expected_peak_vehicles': self.expected_vehicles_peak,
            'network_complexity': 'HIGH',
            'computation_time_factor': 1.0
        }
    
    def load_network_metrics(self):
        """Load metrics from config file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            return {}


if __name__ == '__main__':
    large_scale = LargeScaleConfig()
    print("LARGE-SCALE (24x24) Configuration:")
    for key, value in large_scale.get_scale_info().items():
        print(f"  {key}: {value}")
