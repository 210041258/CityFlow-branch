#!/usr/bin/env python3
"""
Feature Counter Module - Object Closure Parameters (Safety Constraints)
Used by: run.py --training-env (for safety metrics)
"""

from collections import defaultdict

class ObjectClosureParameters:
    """Safety constraints and object closure parameters for training"""
    
    def __init__(self):
        self.min_safety_gap = 2.5  # meters
        self.collision_events = 0
        self.near_miss_events = 0
        self.safety_constraint_violations = 0
        self.closure_events = defaultdict(int)  # object type -> count
        self.maximum_safe_speed = 15.0  # m/s
        self.safe_deceleration = 4.5  # m/s^2
        
    def check_safety_constraint(self, vehicle_gap, relative_speed, max_speed):
        """Check if safety constraints are satisfied"""
        # Calculate required braking distance
        required_distance = (relative_speed ** 2) / (2 * self.safe_deceleration)
        
        if vehicle_gap < self.min_safety_gap:
            self.safety_constraint_violations += 1
            return False
        
        if vehicle_gap < required_distance:
            self.near_miss_events += 1
            return False
        
        return True
    
    def detect_collision(self, vehicle_gap):
        """Detect collision events"""
        if vehicle_gap <= 0:
            self.collision_events += 1
            self.closure_events['collision'] += 1
            return True
        return False
    
    def record_object_closure(self, closure_type, severity):
        """Record object closure events (intersections, crossings, etc.)"""
        self.closure_events[f"{closure_type}_{severity}"] += 1
    
    def get_metrics(self):
        """Get object closure and safety parameters"""
        return {
            'safety_min_gap_meters': self.min_safety_gap,
            'safety_max_speed_ms': self.maximum_safe_speed,
            'safety_deceleration_ms2': self.safe_deceleration,
            'safety_constraint_violations': self.safety_constraint_violations,
            'safety_collision_events': self.collision_events,
            'safety_near_miss_events': self.near_miss_events,
            'safety_closure_events_total': sum(self.closure_events.values()),
            'safety_closure_events_by_type': dict(self.closure_events)
        }


if __name__ == '__main__':
    # Example usage
    closure = ObjectClosureParameters()
    closure.check_safety_constraint(5.0, 2.0, 15.0)
    closure.detect_collision(0.5)
    closure.record_object_closure('intersection', 'high')
    print(closure.get_metrics())
