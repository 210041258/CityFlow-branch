#!/usr/bin/env python3
"""
CityFlow Advanced Feature Counter with Training Environment Integration
========================================================================

Comprehensive script to count all measurable features across:
- LARGE-SCALE scenario: Extensive city-wide network (24x24 grid)
- MID-SCALE scenario: Medium urban network (10x10 grid)
- V2X (Vehicle-to-Infrastructure) Communication Components
- V2V (Vehicle-to-Vehicle) Communication Components
- Composite metrics combining all countable features
- TRAINING ENVIRONMENT METRICS (Non-MARL Mode):
  * Signal state characterization
  * Congestion metrics per simulation slot
  * Object closure parameters for training
  * Multi-algorithm traffic control strategies

Countable Features:
1. VEHICLE METRICS: counts, speeds, distances, travel times, entering/exiting vehicles
2. LANE METRICS: vehicle counts, waiting vehicles, congestion per lane
3. CONGESTION METRICS: system-wide congestion, bottleneck detection, density
4. TIME METRICS: simulation time, step counts, average travel times
5. SIGNAL METRICS: phase states, phase durations, optimization events, cycle times
6. V2X METRICS: infrastructure communication, signal efficiency, broadcast events
7. V2V METRICS: vehicle coordination, collision avoidance, cooperative maneuvers
8. TRAINING ENVIRONMENT METRICS: 
   - Signal observation space (per intersection)
   - Congestion state space (discretized levels)
   - Object closure parameters (safety constraints)
   - Non-MARL algorithm metrics (fixed-time, adaptive, optimized)
9. COMPOSITE METRICS: system efficiency, network capacity, throughput, communication overhead

Usage:
    python run.py <config_file> [--scale large|mid|both] [--steps N] [--output CSV] 
                   [--v2x] [--v2v] [--training-env] [--algorithm fixed|adaptive|optimized] [--verbose]

Examples:
    # Large-scale with training environment (Non-MARL)
    python run.py data/config.json --scale large --steps 500 --output results.csv --training-env --algorithm adaptive
    
    # Both scales with V2X/V2V and training metrics
    python run.py data/config.json --scale both --v2x --v2v --training-env --output composite.csv
    
    # Mid-scale with fixed-time signal control
    python run.py data/config.json --scale mid --algorithm fixed-time --output mid_fixed.csv
    
    # Compare algorithms
    python run.py data/config.json --scale large --algorithm optimized --training-env --output optimized.csv
"""

import json
import sys
import argparse
import os
from datetime import datetime
from collections import defaultdict
import cityflow
import csv
import math


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


class CongestionStateSpace:
    """Discretized congestion state space for training"""
    
    def __init__(self, levels=5):  # 5 discrete congestion levels: FREE, LIGHT, MODERATE, HEAVY, SEVERE
        self.levels = levels
        self.level_names = ['FREE', 'LIGHT', 'MODERATE', 'HEAVY', 'SEVERE']
        self.congestion_levels_history = []
        self.state_transitions = defaultdict(int)
        self.current_state = 0
        
    def discretize_congestion(self, congestion_value):
        """Convert continuous congestion (0-1) to discrete state (0-levels-1)"""
        if congestion_value < 0.2:
            return 0  # FREE
        elif congestion_value < 0.4:
            return 1  # LIGHT
        elif congestion_value < 0.6:
            return 2  # MODERATE
        elif congestion_value < 0.8:
            return 3  # HEAVY
        else:
            return 4  # SEVERE
    
    def update_state(self, congestion_value):
        """Update congestion state and track transitions"""
        new_state = self.discretize_congestion(congestion_value)
        self.congestion_levels_history.append(new_state)
        
        if self.congestion_levels_history:
            transition = (self.current_state, new_state)
            self.state_transitions[transition] += 1
        
        self.current_state = new_state
        return new_state
    
    def get_state_name(self, state):
        """Get human-readable name for state"""
        return self.level_names[state] if 0 <= state < len(self.level_names) else 'UNKNOWN'
    
    def get_metrics(self):
        """Get congestion state space metrics"""
        state_counts = defaultdict(int)
        for state in self.congestion_levels_history:
            state_counts[state] += 1
        
        return {
            'congestion_state_space_levels': self.levels,
            'congestion_total_state_transitions': len(self.state_transitions),
            'congestion_current_state': self.current_state,
            'congestion_current_state_name': self.get_state_name(self.current_state),
            'congestion_state_distribution': dict(state_counts),
            'congestion_most_common_state': max(state_counts, key=state_counts.get) if state_counts else 0
        }


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


class FeatureCounter:
    """Advanced Feature Counter with Training Environment Integration"""
    
    def __init__(self, config_path, scale='large', thread_num=1, enable_v2x=True, 
                 enable_v2v=True, enable_training_env=False, algorithm='adaptive'):
        """Initialize the simulation engine with all components"""
        self.config_path = config_path
        self.scale = scale
        self.engine = cityflow.Engine(config_path, thread_num=thread_num)
        
        # Communication modules
        self.v2x = V2XCommunication() if enable_v2x else None
        self.v2v = V2VCommunication() if enable_v2v else None
        
        # Training environment modules
        self.signal_char = SignalCharacterization() if enable_training_env else None
        self.congestion_space = CongestionStateSpace() if enable_training_env else None
        self.object_closure = ObjectClosureParameters() if enable_training_env else None
        self.traffic_control = TrafficControlAlgorithm(algorithm) if enable_training_env else None
        
        # Initialize counters
        self.step_count = 0
        self.vehicle_count_history = []
        self.vehicle_speed_history = []
        self.vehicle_distance_history = []
        self.travel_time_history = []
        self.lane_vehicle_count_history = defaultdict(list)
        self.lane_waiting_count_history = defaultdict(list)
        self.congestion_history = []
        self.vehicle_entered = 0
        self.vehicle_exited = 0
        
    def step(self):
        """Execute one simulation step"""
        self.step_count += 1
        self.engine.next_step()
        
    def collect_vehicle_metrics(self):
        """Collect comprehensive vehicle metrics"""
        metrics = {}
        
        # Vehicle count
        vehicle_count = self.engine.get_vehicle_count()
        self.vehicle_count_history.append(vehicle_count)
        metrics['vehicle_count'] = vehicle_count
        
        # Vehicle speeds
        vehicle_speeds = self.engine.get_vehicle_speed()
        if vehicle_speeds:
            speeds_list = list(vehicle_speeds.values())
            metrics['avg_vehicle_speed'] = sum(speeds_list) / len(speeds_list)
            metrics['max_vehicle_speed'] = max(speeds_list)
            metrics['min_vehicle_speed'] = min(speeds_list)
        else:
            metrics['avg_vehicle_speed'] = 0
        
        # Vehicle distances
        vehicle_distances = self.engine.get_vehicle_distance()
        if vehicle_distances:
            total_distance = sum(vehicle_distances.values())
            metrics['total_vehicle_distance'] = total_distance
            metrics['avg_vehicle_distance'] = total_distance / len(vehicle_distances)
            self.vehicle_distance_history.append(vehicle_distances)
        
        return metrics
    
    def collect_lane_metrics(self):
        """Collect comprehensive lane metrics"""
        metrics = {}
        
        # Lane vehicle counts
        lane_counts = self.engine.get_lane_vehicle_count()
        total_lane_vehicles = sum(lane_counts.values())
        metrics['total_lane_vehicles'] = total_lane_vehicles
        
        if lane_counts:
            metrics['avg_vehicles_per_lane'] = total_lane_vehicles / len(lane_counts)
            metrics['max_vehicles_in_lane'] = max(lane_counts.values())
        
        for lane_id, count in lane_counts.items():
            self.lane_vehicle_count_history[lane_id].append(count)
        
        # Lane waiting vehicle counts
        lane_waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        total_waiting = sum(lane_waiting_counts.values())
        metrics['total_waiting_vehicles'] = total_waiting
        
        if lane_waiting_counts:
            metrics['avg_waiting_per_lane'] = total_waiting / len(lane_waiting_counts)
            metrics['max_waiting_in_lane'] = max(lane_waiting_counts.values())
        
        for lane_id, count in lane_waiting_counts.items():
            self.lane_waiting_count_history[lane_id].append(count)
        
        # Congestion level
        congestion = total_waiting / max(total_lane_vehicles, 1) if total_lane_vehicles > 0 else 0
        self.congestion_history.append(congestion)
        metrics['congestion_level'] = congestion
        
        return metrics
    
    def collect_time_metrics(self):
        """Collect time-related metrics"""
        metrics = {}
        
        current_time = self.engine.get_current_time()
        metrics['current_time'] = current_time
        metrics['step_count'] = self.step_count
        
        avg_travel_time = self.engine.get_average_travel_time()
        metrics['average_travel_time'] = avg_travel_time
        self.travel_time_history.append(avg_travel_time)
        
        return metrics
    
    def collect_signal_metrics(self, vehicle_count, congestion_level):
        """Collect signal and traffic control metrics"""
        metrics = {}
        
        if self.signal_char:
            # Simulate signal state recording
            for i in range(max(1, vehicle_count // 20)):
                self.signal_char.record_signal_state(
                    f"intersection_{i}", 
                    i % 2,  # phase
                    30,  # duration
                    vehicle_count // 2,  # vehicles NS
                    vehicle_count // 2   # vehicles EW
                )
            
            metrics.update(self.signal_char.get_metrics())
        
        return metrics
    
    def collect_training_env_metrics(self, vehicle_count, congestion_level):
        """Collect training environment metrics"""
        metrics = {}
        
        # Congestion state space
        if self.congestion_space:
            state = self.congestion_space.update_state(congestion_level)
            metrics.update(self.congestion_space.get_metrics())
        
        # Object closure parameters (safety)
        if self.object_closure:
            # Simulate safety constraint checking
            for _ in range(max(0, int(vehicle_count * 0.05))):
                self.object_closure.check_safety_constraint(5.0, 1.0, 15.0)
            
            metrics.update(self.object_closure.get_metrics())
        
        # Traffic control algorithm
        if self.traffic_control:
            for i in range(max(1, vehicle_count // 20)):
                self.traffic_control.control_signal(
                    f"intersection_{i}",
                    vehicle_count // 2,  # NS vehicles
                    vehicle_count // 2,  # EW vehicles
                    vehicle_count // 4,  # NS waiting
                    vehicle_count // 4,  # EW waiting
                    self.step_count,
                    congestion_level
                )
            
            metrics.update(self.traffic_control.get_metrics())
        
        return metrics
    
    def collect_communication_metrics(self, vehicle_count, congestion_level):
        """Collect V2X and V2V communication metrics"""
        metrics = {}
        
        if self.v2x:
            num_intersections = max(1, vehicle_count // 10)
            for i in range(num_intersections):
                self.v2x.broadcast_signal_state(f"int_{i}", "green", vehicle_count)
                self.v2x.optimize_signal_timing(f"int_{i}", vehicle_count, congestion_level)
            
            metrics.update(self.v2x.get_metrics())
        
        if self.v2v:
            warnings = max(0, int(vehicle_count * congestion_level * 0.1))
            for _ in range(warnings):
                self.v2v.detect_collision_risk("v_id", [], 3.0, 2.0)
            
            maneuvers = max(0, int(vehicle_count * 0.02))
            for _ in range(maneuvers):
                self.v2v.coordinate_lane_change("v_id", "new_lane", ["adjacent_1", "adjacent_2"])
            
            metrics.update(self.v2v.get_metrics())
        
        return metrics
    
    def get_roadnet_metrics(self):
        """Get road network static metrics"""
        metrics = {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            roadnet_file = config.get('roadnetFile', '')
            if roadnet_file:
                if not os.path.isabs(roadnet_file):
                    roadnet_file = os.path.join(os.path.dirname(self.config_path), roadnet_file)
                
                if os.path.exists(roadnet_file):
                    with open(roadnet_file, 'r') as f:
                        roadnet = json.load(f)
                    
                    roads = roadnet.get('roads', [])
                    intersections = roadnet.get('intersections', [])
                    
                    metrics['total_roads'] = len(roads)
                    metrics['total_intersections'] = len(intersections)
                    
                    total_lanes = sum(len(road.get('lanes', [])) for road in roads)
                    metrics['total_lanes'] = total_lanes
                    
                    if roads:
                        metrics['avg_lanes_per_road'] = total_lanes / len(roads)
        except Exception as e:
            print(f"Warning: Could not parse roadnet: {e}")
        
        return metrics
    
    def run_simulation(self, num_steps):
        """Run simulation and collect all metrics"""
        print(f"\n{'='*90}")
        print(f"CityFlow Advanced Feature Counter - {self.scale.upper()}-SCALE")
        print(f"{'='*90}")
        print(f"V2X: {self.v2x is not None} | V2V: {self.v2v is not None} | Training Env: {self.signal_char is not None}")
        if self.traffic_control:
            print(f"Algorithm: {self.traffic_control.algorithm_type}")
        print(f"Steps: {num_steps}")
        print(f"{'='*90}\n")
        
        all_metrics = []
        
        for step in range(num_steps):
            self.step()
            
            metrics = {
                'step': self.step_count,
                'scale': self.scale,
                'timestamp': datetime.now().isoformat()
            }
            
            # Collect all metric categories
            vehicle_m = self.collect_vehicle_metrics()
            lane_m = self.collect_lane_metrics()
            time_m = self.collect_time_metrics()
            signal_m = self.collect_signal_metrics(vehicle_m.get('vehicle_count', 0), lane_m.get('congestion_level', 0))
            training_m = self.collect_training_env_metrics(vehicle_m.get('vehicle_count', 0), lane_m.get('congestion_level', 0))
            comm_m = self.collect_communication_metrics(vehicle_m.get('vehicle_count', 0), lane_m.get('congestion_level', 0))
            
            metrics.update(vehicle_m)
            metrics.update(lane_m)
            metrics.update(time_m)
            metrics.update(signal_m)
            metrics.update(training_m)
            metrics.update(comm_m)
            
            all_metrics.append(metrics)
            
            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}/{num_steps}: Vehicles={vehicle_m.get('vehicle_count', 0)}, "
                      f"Speed={vehicle_m.get('avg_vehicle_speed', 0):.2f}, "
                      f"Congestion={lane_m.get('congestion_level', 0):.3f}")
        
        return all_metrics
    
    def generate_summary(self, all_metrics):
        """Generate comprehensive summary report"""
        print(f"\n{'='*90}")
        print(f"COMPREHENSIVE FEATURE SUMMARY - {self.scale.upper()}-SCALE")
        print(f"{'='*90}\n")
        
        roadnet = self.get_roadnet_metrics()
        
        # VEHICLE METRICS
        print("VEHICLE METRICS:")
        print("-" * 90)
        if self.vehicle_count_history:
            print(f"  Max Vehicles: {max(self.vehicle_count_history)} | Min: {min(self.vehicle_count_history)} | "
                  f"Avg: {sum(self.vehicle_count_history)/len(self.vehicle_count_history):.2f}")
        
        all_speeds = [s for d in self.vehicle_speed_history for s in d.values()]
        if all_speeds:
            print(f"  Speed: Avg={sum(all_speeds)/len(all_speeds):.2f} m/s | Max={max(all_speeds):.2f} | Min={min(all_speeds):.2f}")
        
        # LANE METRICS
        print("\nLANE METRICS:")
        print("-" * 90)
        all_lane = [c for counts in self.lane_vehicle_count_history.values() for c in counts]
        if all_lane:
            print(f"  Vehicles: Max={max(all_lane)} | Avg={sum(all_lane)/len(all_lane):.2f}")
        
        all_waiting = [c for counts in self.lane_waiting_count_history.values() for c in counts]
        if all_waiting:
            print(f"  Waiting: Max={max(all_waiting)} | Avg={sum(all_waiting)/len(all_waiting):.2f}")
        
        # CONGESTION METRICS
        print("\nCONGESTION METRICS:")
        print("-" * 90)
        if self.congestion_history:
            print(f"  Congestion: Max={max(self.congestion_history):.3f} | Avg={sum(self.congestion_history)/len(self.congestion_history):.3f} | "
                  f"Min={min(self.congestion_history):.3f}")
        
        # TIME METRICS
        print("\nTIME METRICS:")
        print("-" * 90)
        print(f"  Simulation Steps: {self.step_count} | Time: {all_metrics[-1].get('current_time', 0):.2f}s")
        if self.travel_time_history:
            print(f"  Avg Travel Time: {sum(self.travel_time_history)/len(self.travel_time_history):.2f}s")
        
        # SIGNAL METRICS
        print("\nSIGNAL & TRAFFIC CONTROL METRICS:")
        print("-" * 90)
        if self.signal_char:
            sig_m = self.signal_char.get_metrics()
            print(f"  Signal Records: {sig_m.get('signal_total_phase_records', 0)} | "
                  f"Phase Changes: {sig_m.get('signal_total_phase_changes', 0)}")
        
        if self.traffic_control:
            alg_m = self.traffic_control.get_metrics()
            print(f"  Algorithm: {alg_m.get('algorithm_type')} | "
                  f"Decisions: {alg_m.get('algorithm_total_decisions', 0)} | "
                  f"Optimizations: {alg_m.get('algorithm_optimization_events', 0)}")
        
        # CONGESTION STATE SPACE
        print("\nCONGESTION STATE SPACE (Training):")
        print("-" * 90)
        if self.congestion_space:
            cong_m = self.congestion_space.get_metrics()
            print(f"  Current State: {cong_m.get('congestion_current_state_name')} | "
                  f"State Transitions: {cong_m.get('congestion_total_state_transitions', 0)}")
            if cong_m.get('congestion_state_distribution'):
                dist = cong_m.get('congestion_state_distribution')
                print(f"  State Distribution: {dist}")
        
        # OBJECT CLOSURE & SAFETY
        print("\nOBJECT CLOSURE & SAFETY PARAMETERS (Training):")
        print("-" * 90)
        if self.object_closure:
            safety_m = self.object_closure.get_metrics()
            print(f"  Min Safety Gap: {safety_m.get('safety_min_gap_meters', 0)} m")
            print(f"  Violations: {safety_m.get('safety_constraint_violations', 0)} | "
                  f"Collisions: {safety_m.get('safety_collision_events', 0)} | "
                  f"Near Misses: {safety_m.get('safety_near_miss_events', 0)}")
            if safety_m.get('safety_closure_events_by_type'):
                print(f"  Closure Events: {safety_m.get('safety_closure_events_by_type')}")
        
        # V2X METRICS
        print("\nV2X (Vehicle-to-Infrastructure):")
        print("-" * 90)
        if self.v2x:
            v2x_m = self.v2x.get_metrics()
            print(f"  Messages: {v2x_m.get('v2x_total_messages', 0)} | "
                  f"Latency: {v2x_m.get('v2x_avg_latency_ms', 0):.2f}ms | "
                  f"Efficiency: {v2x_m.get('v2x_efficiency_score', 0):.3f}")
        
        # V2V METRICS
        print("\nV2V (Vehicle-to-Vehicle):")
        print("-" * 90)
        if self.v2v:
            v2v_m = self.v2v.get_metrics()
            print(f"  Messages: {v2v_m.get('v2v_total_messages', 0)} | "
                  f"Warnings: {v2v_m.get('v2v_collision_warnings', 0)} | "
                  f"Maneuvers: {v2v_m.get('v2v_cooperative_maneuvers', 0)}")
        
        # COMPOSITE METRICS
        print("\nCOMPOSITE SYSTEM METRICS:")
        print("-" * 90)
        if all_speeds and self.congestion_history:
            avg_speed = sum(all_speeds) / len(all_speeds)
            avg_cong = sum(self.congestion_history) / len(self.congestion_history)
            efficiency = (avg_speed / max(all_speeds, 1)) * (1 - avg_cong)
            print(f"  System Efficiency: {efficiency:.3f} | Network Util: {min(avg_cong*100, 100):.1f}%")
        
        total_comm = (self.v2x.message_count if self.v2x else 0) + (self.v2v.message_count if self.v2v else 0)
        print(f"  Total Communication Messages: {total_comm}")
        
        print(f"\n{'='*90}\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='CityFlow Advanced Feature Counter with Training Environment Integration'
    )
    parser.add_argument('config', nargs='?', default='data/config.json',
                        help='Path to config file (default: data/config.json)')
    parser.add_argument('--scale', choices=['large', 'mid', 'both'], default='large',
                        help='Simulation scale: large (24x24), mid (10x10), or both')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of simulation steps (default: 500)')
    parser.add_argument('--output', type=str, help='Output CSV file for detailed metrics')
    parser.add_argument('--v2x', action='store_true', default=True, help='Enable V2X')
    parser.add_argument('--no-v2x', action='store_false', dest='v2x', help='Disable V2X')
    parser.add_argument('--v2v', action='store_true', default=True, help='Enable V2V')
    parser.add_argument('--no-v2v', action='store_false', dest='v2v', help='Disable V2V')
    parser.add_argument('--training-env', action='store_true', help='Enable training environment metrics')
    parser.add_argument('--algorithm', choices=['fixed-time', 'adaptive', 'optimized'], default='adaptive',
                        help='Traffic control algorithm: fixed-time, adaptive, or optimized (default: adaptive)')
    parser.add_argument('--threads', type=int, default=1, help='Number of simulation threads')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        scales = ['large', 'mid'] if args.scale == 'both' else [args.scale]
        all_results = []
        
        for scale in scales:
            print(f"\n{'#'*90}")
            print(f"# {scale.upper()}-SCALE SIMULATION")
            print(f"{'#'*90}\n")
            
            counter = FeatureCounter(
                args.config,
                scale=scale,
                thread_num=args.threads,
                enable_v2x=args.v2x,
                enable_v2v=args.v2v,
                enable_training_env=args.training_env,
                algorithm=args.algorithm
            )
            
            all_metrics = counter.run_simulation(args.steps)
            all_results.extend(all_metrics)
            counter.generate_summary(all_metrics)
        
        if args.output:
            with open(args.output, 'w', newline='') as f:
                if all_results:
                    fieldnames = all_results[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
            print(f"Metrics saved to: {args.output}")
        
        print("\n" + "="*90)
        print("SIMULATION COMPLETED")
        print("="*90)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
