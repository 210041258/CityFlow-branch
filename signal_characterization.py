#!/usr/bin/env python3
"""
Feature Counter Module - Signal Characterization
Used by: run.py --training-env (for signal metrics)
"""

from collections import defaultdict

class SignalCharacterization:
    """Signal state observation space for training environment"""
    
    def __init__(self, num_intersections=10):
        self.num_intersections = num_intersections
        self.signal_history = defaultdict(list)
        self.phase_duration_history = defaultdict(list)
        self.phase_changes = 0
        
    def record_signal_state(self, intersection_id, phase_index, duration, vehicle_count_ns, vehicle_count_ew):
        """Record signal state observation"""
        signal_state = {
            'phase': phase_index,
            'duration': duration,
            'vehicles_ns': vehicle_count_ns,
            'vehicles_ew': vehicle_count_ew,
            'queue_ratio': vehicle_count_ns / max(vehicle_count_ns + vehicle_count_ew, 1)
        }
        self.signal_history[intersection_id].append(signal_state)
        self.phase_duration_history[intersection_id].append(duration)
    
    def get_observation_space(self, intersection_id):
        """Get current observation space for intersection (training input)"""
        if not self.signal_history[intersection_id]:
            return {
                'phase': 0,
                'duration': 0,
                'queue_ratio': 0,
                'avg_duration': 0
            }
        
        latest = self.signal_history[intersection_id][-1]
        avg_duration = sum(self.phase_duration_history[intersection_id]) / len(self.phase_duration_history[intersection_id])
        
        return {
            'phase': latest['phase'],
            'duration': latest['duration'],
            'queue_ratio': latest['queue_ratio'],
            'avg_duration': avg_duration,
            'vehicles_ns': latest['vehicles_ns'],
            'vehicles_ew': latest['vehicles_ew']
        }
    
    def get_metrics(self):
        """Get signal characterization metrics"""
        total_phases = sum(len(history) for history in self.signal_history.values())
        total_phase_changes = self.phase_changes
        
        return {
            'signal_total_phase_records': total_phases,
            'signal_total_phase_changes': total_phase_changes,
            'signal_intersections_tracked': len(self.signal_history),
            'signal_avg_phases_per_intersection': total_phases / max(len(self.signal_history), 1)
        }


if __name__ == '__main__':
    # Example usage
    sig_char = SignalCharacterization(10)
    sig_char.record_signal_state('int_0', 0, 30, 50, 30)
    print(sig_char.get_metrics())
    print(sig_char.get_observation_space('int_0'))
