#!/usr/bin/env python3
"""
Feature Counter Module - Scale Simulation (MID)
Used by: run.py --scale mid (10x10 grid network)
"""

import json
import os

class MidScaleConfig:
    """Configuration and metrics for mid-scale (10x10) simulation"""
    
    def __init__(self, config_path='data/config.json'):
        self.config_path = config_path
        self.scale_name = 'mid'
        self.grid_size = (10, 10)  # 10x10 intersections
        self.expected_roads = 40  # 10*4
        self.expected_lanes = 120  # 40 roads * 3 lanes average
        self.expected_vehicles_peak = 150  # Expected peak vehicle count
        
    def get_scale_info(self):
        """Get mid-scale simulation info"""
        return {
            'scale_type': self.scale_name,
            'grid_dimensions': f"{self.grid_size[0]}x{self.grid_size[1]}",
            'total_intersections': self.grid_size[0] * self.grid_size[1],
            'expected_roads': self.expected_roads,
            'expected_lanes': self.expected_lanes,
            'expected_peak_vehicles': self.expected_vehicles_peak,
            'network_complexity': 'MEDIUM',
            'computation_time_factor': 0.25
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
    mid_scale = MidScaleConfig()
    print("MID-SCALE (10x10) Configuration:")
    for key, value in mid_scale.get_scale_info().items():
        print(f"  {key}: {value}")
