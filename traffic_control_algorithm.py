#!/usr/bin/env python3
"""
Feature Counter Module - Traffic Control Algorithms (Non-MARL)
Used by: run.py --training-env --algorithm [fixed-time|adaptive|optimized]
"""

from collections import defaultdict

class TrafficControlAlgorithm:
    """Non-MARL traffic control algorithms for training"""
    
    def __init__(self, algorithm_type='adaptive'):
        self.algorithm_type = algorithm_type  # 'fixed-time', 'adaptive', 'optimized'
        self.phase_plans = defaultdict(lambda: {'red_time': 30, 'green_time': 30})
        self.optimization_events = 0
        self.phase_decisions = []
        self.performance_history = []
        
    def fixed_time_control(self, intersection_id, current_time):
        """Fixed-time signal control (no adaptation)"""
        cycle_time = 60  # 60 seconds
        phase_index = int((current_time % cycle_time) / 30)
        return phase_index, 30  # phase, duration
    
    def adaptive_control(self, intersection_id, vehicle_count_ns, vehicle_count_ew, current_congestion):
        """Adaptive signal control based on queue lengths"""
        total_vehicles = vehicle_count_ns + vehicle_count_ew
        
        # Adjust phase times based on queue ratio
        ns_ratio = vehicle_count_ns / max(total_vehicles, 1)
        ew_ratio = vehicle_count_ew / max(total_vehicles, 1)
        
        # Base times
        base_time = 30
        
        # Adapt based on congestion
        if current_congestion > 0.7:
            # High congestion: prioritize direction with more vehicles
            if ns_ratio > 0.6:
                ns_time = int(base_time * 1.3)
                ew_time = int(base_time * 0.7)
            else:
                ns_time = int(base_time * 0.7)
                ew_time = int(base_time * 1.3)
        else:
            # Normal: balanced timing
            ns_time = base_time
            ew_time = base_time
        
        # Determine current phase (simplified)
        phase_index = 0 if ns_ratio > 0.5 else 1
        duration = ns_time if phase_index == 0 else ew_time
        
        self.optimization_events += 1
        return phase_index, duration
    
    def optimized_control(self, intersection_id, vehicle_count_ns, vehicle_count_ew, 
                         waiting_ns, waiting_ew, current_congestion):
        """Optimized signal control with multiple parameters"""
        total_vehicles = vehicle_count_ns + vehicle_count_ew
        total_waiting = waiting_ns + waiting_ew
        
        # Calculate urgency scores
        ns_urgency = (vehicle_count_ns * 0.4) + (waiting_ns * 0.6)
        ew_urgency = (vehicle_count_ew * 0.4) + (waiting_ew * 0.6)
        
        # Normalize
        max_urgency = max(ns_urgency, ew_urgency, 1)
        ns_priority = ns_urgency / max_urgency
        ew_priority = ew_urgency / max_urgency
        
        # Optimize cycle time based on congestion
        base_cycle = 60
        if current_congestion > 0.8:
            cycle_time = int(base_cycle * 0.8)  # Shorter cycles in high congestion
        elif current_congestion < 0.3:
            cycle_time = int(base_cycle * 1.2)  # Longer cycles in free flow
        else:
            cycle_time = base_cycle
        
        # Allocate green time proportionally
        ns_green = int((cycle_time - 10) * ns_priority)  # 10s for yellow phases
        ew_green = (cycle_time - 10) - ns_green
        
        # Determine phase
        phase_index = 0 if ns_priority > 0.5 else 1
        duration = ns_green if phase_index == 0 else ew_green
        
        self.optimization_events += 1
        return phase_index, duration
    
    def control_signal(self, intersection_id, vehicle_count_ns, vehicle_count_ew, 
                      waiting_ns, waiting_ew, current_time, current_congestion):
        """Main control method that dispatches to appropriate algorithm"""
        if self.algorithm_type == 'fixed-time':
            phase, duration = self.fixed_time_control(intersection_id, current_time)
        elif self.algorithm_type == 'adaptive':
            phase, duration = self.adaptive_control(intersection_id, vehicle_count_ns, vehicle_count_ew, current_congestion)
        elif self.algorithm_type == 'optimized':
            phase, duration = self.optimized_control(
                intersection_id, vehicle_count_ns, vehicle_count_ew, 
                waiting_ns, waiting_ew, current_congestion
            )
        else:
            phase, duration = 0, 30
        
        decision = {
            'intersection': intersection_id,
            'phase': phase,
            'duration': duration,
            'algorithm': self.algorithm_type
        }
        self.phase_decisions.append(decision)
        return phase, duration
    
    def get_metrics(self):
        """Get algorithm performance metrics"""
        return {
            'algorithm_type': self.algorithm_type,
            'algorithm_total_decisions': len(self.phase_decisions),
            'algorithm_optimization_events': self.optimization_events,
            'algorithm_avg_phase_duration': sum(d['duration'] for d in self.phase_decisions) / max(len(self.phase_decisions), 1) if self.phase_decisions else 0
        }


if __name__ == '__main__':
    # Example usage - test all algorithms
    for algo in ['fixed-time', 'adaptive', 'optimized']:
        print(f"\n=== {algo.upper()} ALGORITHM ===")
        controller = TrafficControlAlgorithm(algo)
        
        # Simulate multiple decisions
        for t in range(5):
            phase, duration = controller.control_signal(
                'intersection_0',
                vehicle_count_ns=50,
                vehicle_count_ew=30,
                waiting_ns=10,
                waiting_ew=5,
                current_time=t*30,
                current_congestion=0.6
            )
            print(f"Time {t*30}s: Phase {phase}, Duration {duration}s")
        
        print(f"Metrics: {controller.get_metrics()}")
