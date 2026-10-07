#!/usr/bin/env python3
"""
Feature Counter Module - V2X Communication
Used by: run.py --v2x (Vehicle-to-Infrastructure)
"""

from collections import defaultdict

class V2XCommunication:
    """V2X (Vehicle-to-Infrastructure) Communication Module"""
    
    def __init__(self):
        self.message_count = 0
        self.message_latency = []
        self.intersection_broadcasts = defaultdict(int)
        self.v2x_efficiency = 0.0
        self.signal_optimization_events = 0
        self.adaptive_phase_changes = 0
        
    def broadcast_signal_state(self, intersection_id, phase_state, num_vehicles_aware):
        """Simulate V2X broadcast of traffic signal state"""
        self.message_count += 1
        self.intersection_broadcasts[intersection_id] += 1
        latency = 0.05 + (num_vehicles_aware * 0.001)  # ms + per-vehicle overhead
        self.message_latency.append(latency)
        
    def optimize_signal_timing(self, intersection_id, vehicle_density, congestion_level):
        """V2X-based adaptive signal timing optimization"""
        if congestion_level > 0.7:
            self.adaptive_phase_changes += 1
            self.signal_optimization_events += 1
            return True
        return False
    
    def get_metrics(self):
        """Get V2X communication metrics"""
        return {
            'v2x_total_messages': self.message_count,
            'v2x_avg_latency_ms': sum(self.message_latency) / len(self.message_latency) if self.message_latency else 0,
            'v2x_max_latency_ms': max(self.message_latency) if self.message_latency else 0,
            'v2x_intersections_broadcasting': len(self.intersection_broadcasts),
            'v2x_signal_optimizations': self.signal_optimization_events,
            'v2x_adaptive_phases': self.adaptive_phase_changes,
            'v2x_efficiency_score': self.calculate_efficiency()
        }
    
    def calculate_efficiency(self):
        """Calculate V2X communication efficiency (0-1)"""
        if not self.message_latency:
            return 0.0
        avg_latency = sum(self.message_latency) / len(self.message_latency)
        efficiency = max(0, 1.0 - (avg_latency / 50.0))
        return efficiency


if __name__ == '__main__':
    # Example usage
    v2x = V2XCommunication()
    
    # Simulate broadcasts
    for i in range(5):
        v2x.broadcast_signal_state(f'int_{i}', 'green', 20 + i*5)
        v2x.optimize_signal_timing(f'int_{i}', 30, 0.75)
    
    print("V2X Metrics:")
    for key, value in v2x.get_metrics().items():
        print(f"  {key}: {value}")
