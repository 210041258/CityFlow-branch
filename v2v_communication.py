#!/usr/bin/env python3
"""
Feature Counter Module - V2V Communication
Used by: run.py --v2v (Vehicle-to-Vehicle)
"""

from collections import defaultdict

class V2VCommunication:
    """V2V (Vehicle-to-Vehicle) Communication Module"""
    
    def __init__(self):
        self.message_count = 0
        self.collision_warnings = 0
        self.cooperative_maneuvers = 0
        self.vehicle_coordination_events = defaultdict(int)
        self.message_range = 100  # meters
        self.communication_success_rate = 0.95
        
    def send_cooperative_message(self, source_id, target_ids, message_type='safety'):
        """Send V2V messages between vehicles"""
        successful = len([t for t in target_ids if self.communication_success_rate > 0.5])
        self.message_count += len(target_ids)
        self.vehicle_coordination_events[message_type] += 1
        return successful
    
    def detect_collision_risk(self, vehicle_id, nearby_vehicles, gap, relative_speed):
        """V2V collision avoidance detection"""
        if gap < 5.0 and relative_speed > 1.0:
            self.collision_warnings += 1
            self.send_cooperative_message(vehicle_id, nearby_vehicles, 'collision_warning')
            return True
        return False
    
    def coordinate_lane_change(self, source_id, target_lane, adjacent_vehicles):
        """Coordinate lane changes between vehicles"""
        if len(adjacent_vehicles) > 0:
            self.send_cooperative_message(source_id, adjacent_vehicles, 'lane_change_request')
            self.cooperative_maneuvers += 1
            return True
        return False
    
    def get_metrics(self):
        """Get V2V communication metrics"""
        total_coordination = sum(self.vehicle_coordination_events.values())
        return {
            'v2v_total_messages': self.message_count,
            'v2v_collision_warnings': self.collision_warnings,
            'v2v_cooperative_maneuvers': self.cooperative_maneuvers,
            'v2v_coordination_events': total_coordination,
            'v2v_message_types': dict(self.vehicle_coordination_events),
            'v2v_communication_range_m': self.message_range,
            'v2v_success_rate': self.communication_success_rate
        }


if __name__ == '__main__':
    # Example usage
    v2v = V2VCommunication()
    
    # Simulate collision warnings
    for i in range(3):
        v2v.detect_collision_risk(f'vehicle_{i}', ['vehicle_0', 'vehicle_1'], 4.0, 2.5)
    
    # Simulate lane changes
    for i in range(2):
        v2v.coordinate_lane_change(f'vehicle_{i}', 'left_lane', ['adjacent_1', 'adjacent_2'])
    
    print("V2V Metrics:")
    for key, value in v2v.get_metrics().items():
        print(f"  {key}: {value}")
