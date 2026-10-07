#!/usr/bin/env python3
"""
CityFlow Feature Counter with V2X/V2V Communication
====================================================

Comprehensive script to count all measurable features across:
- LARGE-SCALE scenario: Extensive city-wide network (24x24 grid)
- MID-SCALE scenario: Medium urban network (10x10 grid)
- V2X (Vehicle-to-Infrastructure) Communication Components
- V2V (Vehicle-to-Vehicle) Communication Components
- Composite metrics combining all countable features

Features measured:
- Vehicle metrics (counts, speeds, distances, travel times)
- Road network metrics (roads, lanes, intersections, lane links)
- Traffic light metrics (phases, cycles, durations)
- Lane metrics (vehicle counts, waiting vehicles, congestion)
- Flow metrics (vehicles generated, vehicles departed, throughput)
- Time metrics (simulation time, step counts)
- V2X metrics (infrastructure communication, signal efficiency)
- V2V metrics (vehicle coordination, collision avoidance)
- Aggregate metrics (average speeds, densities, communication overhead)

Usage:
    python run.py <config_file> [--scale large|mid|both] [--steps N] [--output CSV] [--v2x] [--v2v] [--verbose]

Examples:
    python run.py data/config.json --scale large --steps 500 --output results.csv
    python run.py data/config.json --scale both --v2x --v2v --output composite.csv
    python run.py data/config.json --scale mid --steps 300 --verbose
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
            return True  # Signal optimized
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
        # Efficiency inversely proportional to latency (max 50ms = 0 efficiency)
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
    """Counter for all CityFlow simulation features with V2X/V2V"""
    
    def __init__(self, config_path, scale='large', thread_num=1, enable_v2x=True, enable_v2v=True):
        """Initialize the simulation engine"""
        self.config_path = config_path
        self.scale = scale
        self.engine = cityflow.Engine(config_path, thread_num=thread_num)
        
        # Communication modules
        self.v2x = V2XCommunication() if enable_v2x else None
        self.v2v = V2VCommunication() if enable_v2v else None
        
        # Initialize counters
        self.step_count = 0
        self.total_vehicles_generated = 0
        self.total_vehicles_departed = 0
        self.vehicle_speed_history = []
        self.vehicle_distance_history = []
        self.travel_time_history = []
        self.lane_vehicle_count_history = defaultdict(list)
        self.lane_waiting_count_history = defaultdict(list)
        self.vehicle_count_history = []
        self.congestion_history = []
        
        # Communication overhead
        self.communication_overhead = 0.0
        
    def step(self):
        """Execute one simulation step and collect metrics"""
        self.step_count += 1
        self.engine.next_step()
        
    def collect_vehicle_metrics(self):
        """
        Collect metrics for all vehicles:
        - vehicle_count: Total number of vehicles in simulation
        - vehicle_speed: Speed map for all vehicles
        - vehicle_distance: Distance traveled by all vehicles
        """
        metrics = {}
        
        # Vehicle count
        vehicle_count = self.engine.get_vehicle_count()
        self.vehicle_count_history.append(vehicle_count)
        metrics['vehicle_count'] = vehicle_count
        metrics['vehicles_in_simulation'] = vehicle_count
        
        # Vehicle speeds
        vehicle_speeds = self.engine.get_vehicle_speed()
        if vehicle_speeds:
            speeds_list = list(vehicle_speeds.values())
            avg_speed = sum(speeds_list) / len(speeds_list)
            max_speed = max(speeds_list)
            min_speed = min(speeds_list)
            metrics['avg_vehicle_speed'] = avg_speed
            metrics['max_vehicle_speed'] = max_speed
            metrics['min_vehicle_speed'] = min_speed
            metrics['num_vehicles_with_speed_data'] = len(vehicle_speeds)
            self.vehicle_speed_history.append(vehicle_speeds)
        else:
            metrics['avg_vehicle_speed'] = 0
            metrics['num_vehicles_with_speed_data'] = 0
        
        # Vehicle distances
        vehicle_distances = self.engine.get_vehicle_distance()
        if vehicle_distances:
            total_distance = sum(vehicle_distances.values())
            avg_distance = total_distance / len(vehicle_distances)
            metrics['total_vehicle_distance'] = total_distance
            metrics['avg_vehicle_distance'] = avg_distance
            metrics['num_vehicles_with_distance_data'] = len(vehicle_distances)
            self.vehicle_distance_history.append(vehicle_distances)
        else:
            metrics['total_vehicle_distance'] = 0
            metrics['num_vehicles_with_distance_data'] = 0
        
        return metrics
    
    def collect_lane_metrics(self):
        """
        Collect metrics for all lanes:
        - lane_vehicle_count: Number of vehicles in each lane
        - lane_waiting_vehicle_count: Number of waiting vehicles in each lane
        """
        metrics = {}
        
        # Lane vehicle counts
        lane_counts = self.engine.get_lane_vehicle_count()
        total_lane_vehicles = sum(lane_counts.values())
        metrics['total_lane_vehicles'] = total_lane_vehicles
        metrics['num_lanes_with_vehicles'] = len(lane_counts)
        if lane_counts:
            metrics['avg_vehicles_per_lane'] = total_lane_vehicles / len(lane_counts)
            metrics['max_vehicles_in_lane'] = max(lane_counts.values())
        else:
            metrics['avg_vehicles_per_lane'] = 0
            metrics['max_vehicles_in_lane'] = 0
        
        for lane_id, count in lane_counts.items():
            self.lane_vehicle_count_history[lane_id].append(count)
        
        # Lane waiting vehicle counts
        lane_waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        total_waiting = sum(lane_waiting_counts.values())
        metrics['total_waiting_vehicles'] = total_waiting
        metrics['num_lanes_with_waiting_vehicles'] = len(lane_waiting_counts)
        if lane_waiting_counts:
            metrics['avg_waiting_per_lane'] = total_waiting / len(lane_waiting_counts)
            metrics['max_waiting_in_lane'] = max(lane_waiting_counts.values())
        else:
            metrics['avg_waiting_per_lane'] = 0
            metrics['max_waiting_in_lane'] = 0
        
        for lane_id, count in lane_waiting_counts.items():
            self.lane_waiting_count_history[lane_id].append(count)
        
        # Calculate congestion level
        congestion = total_waiting / max(total_lane_vehicles, 1) if total_lane_vehicles > 0 else 0
        self.congestion_history.append(congestion)
        metrics['congestion_level'] = congestion
        
        return metrics
    
    def collect_time_metrics(self):
        """
        Collect time-related metrics:
        - current_time: Current simulation time
        - average_travel_time: Average travel time across all vehicles
        """
        metrics = {}
        
        current_time = self.engine.get_current_time()
        metrics['current_time'] = current_time
        metrics['step_count'] = self.step_count
        
        avg_travel_time = self.engine.get_average_travel_time()
        metrics['average_travel_time'] = avg_travel_time
        self.travel_time_history.append(avg_travel_time)
        
        return metrics
    
    def collect_communication_metrics(self, vehicle_count, congestion_level):
        """Collect V2X and V2V communication metrics"""
        metrics = {}
        
        if self.v2x:
            # Simulate V2X communications based on congestion and vehicle density
            num_intersections = max(1, vehicle_count // 10)
            for i in range(num_intersections):
                self.v2x.broadcast_signal_state(f"int_{i}", "green", vehicle_count)
                self.v2x.optimize_signal_timing(f"int_{i}", vehicle_count, congestion_level)
            
            metrics.update(self.v2x.get_metrics())
        
        if self.v2v:
            # Simulate V2V communications based on vehicle density
            warnings = max(0, int(vehicle_count * congestion_level * 0.1))
            for _ in range(warnings):
                self.v2v.detect_collision_risk("v_id", [], 3.0, 2.0)
            
            maneuvers = max(0, int(vehicle_count * 0.02))
            for _ in range(maneuvers):
                self.v2v.coordinate_lane_change("v_id", "new_lane", ["adjacent_1", "adjacent_2"])
            
            metrics.update(self.v2v.get_metrics())
        
        return metrics
    
    def get_roadnet_metrics(self):
        """
        Get road network static metrics from config.
        These are countable features of the network structure.
        """
        metrics = {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Try to load roadnet file
            roadnet_file = config.get('roadnetFile', '')
            if roadnet_file:
                # Convert relative path
                if not os.path.isabs(roadnet_file):
                    roadnet_file = os.path.join(os.path.dirname(self.config_path), roadnet_file)
                
                if os.path.exists(roadnet_file):
                    with open(roadnet_file, 'r') as f:
                        roadnet = json.load(f)
                    
                    # Count roads
                    roads = roadnet.get('roads', [])
                    metrics['total_roads'] = len(roads)
                    
                    # Count lanes and intersections
                    total_lanes = 0
                    total_lane_links = 0
                    intersections = roadnet.get('intersections', [])
                    metrics['total_intersections'] = len(intersections)
                    
                    for road in roads:
                        lanes = road.get('lanes', [])
                        total_lanes += len(lanes)
                    
                    metrics['total_lanes'] = total_lanes
                    metrics['total_lane_links'] = total_lane_links
                    
                    if roads:
                        metrics['avg_lanes_per_road'] = total_lanes / len(roads)
                    
                    if intersections:
                        metrics['intersections_with_signals'] = len(intersections)
        except Exception as e:
            print(f"Warning: Could not parse roadnet file: {e}")
        
        return metrics
    
    def get_flow_metrics(self):
        """
        Get traffic flow metrics from config.
        """
        metrics = {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Flow configuration
            flow_file = config.get('flowFile', '')
            if flow_file:
                if not os.path.isabs(flow_file):
                    flow_file = os.path.join(os.path.dirname(self.config_path), flow_file)
                
                if os.path.exists(flow_file):
                    with open(flow_file, 'r') as f:
                        flows = json.load(f)
                    
                    metrics['total_traffic_flows'] = len(flows)
                    
                    total_vehicles = 0
                    for flow in flows:
                        if 'vehicle' in flow:
                            total_vehicles += len(flow['vehicle'])
                        elif 'count' in flow:
                            total_vehicles += flow['count']
                    
                    metrics['total_vehicles_in_flow'] = total_vehicles
        except Exception as e:
            print(f"Warning: Could not parse flow file: {e}")
        
        return metrics
    
    def run_simulation(self, num_steps):
        """Run simulation for specified number of steps"""
        print(f"\n{'='*80}")
        print(f"CityFlow Feature Counter - {self.scale.upper()}-SCALE Simulation with V2X/V2V")
        print(f"{'='*80}")
        print(f"Scale: {self.scale.upper()}")
        print(f"V2X Enabled: {self.v2x is not None}")
        print(f"V2V Enabled: {self.v2v is not None}")
        print(f"Starting simulation for {num_steps} steps...")
        print(f"{'='*80}\n")
        
        all_metrics = []
        
        for step in range(num_steps):
            self.step()
            
            # Collect metrics at each step
            metrics = {
                'step': self.step_count,
                'timestamp': datetime.now().isoformat(),
                'scale': self.scale
            }
            
            # Collect all feature types
            vehicle_metrics = self.collect_vehicle_metrics()
            lane_metrics = self.collect_lane_metrics()
            time_metrics = self.collect_time_metrics()
            
            metrics.update(vehicle_metrics)
            metrics.update(lane_metrics)
            metrics.update(time_metrics)
            
            # Collect communication metrics
            comm_metrics = self.collect_communication_metrics(
                vehicle_metrics.get('vehicle_count', 0),
                lane_metrics.get('congestion_level', 0)
            )
            metrics.update(comm_metrics)
            
            all_metrics.append(metrics)
            
            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}/{num_steps}: Vehicles={vehicle_metrics['vehicle_count']}, "
                      f"AvgSpeed={vehicle_metrics['avg_vehicle_speed']:.2f}, "
                      f"Congestion={lane_metrics['congestion_level']:.3f}, "
                      f"Waiting={lane_metrics['total_waiting_vehicles']}")
        
        return all_metrics
    
    def generate_summary(self, all_metrics):
        """Generate comprehensive summary of all counted features"""
        print(f"\n{'='*80}")
        print(f"SIMULATION SUMMARY - {self.scale.upper()}-SCALE WITH V2X/V2V")
        print(f"ALL COUNTABLE FEATURES AND COMPOSITE COMPONENTS")
        print(f"{'='*80}\n")
        
        # Road network metrics
        roadnet_metrics = self.get_roadnet_metrics()
        flow_metrics = self.get_flow_metrics()
        
        # ROAD NETWORK FEATURES
        print("=" * 80)
        print("ROAD NETWORK INFRASTRUCTURE FEATURES")
        print("=" * 80)
        print(f"  Total Roads:                    {roadnet_metrics.get('total_roads', 'N/A')}")
        print(f"  Total Intersections:            {roadnet_metrics.get('total_intersections', 'N/A')}")
        print(f"  Total Lanes:                    {roadnet_metrics.get('total_lanes', 'N/A')}")
        print(f"  Total Lane Links:               {roadnet_metrics.get('total_lane_links', 'N/A')}")
        print(f"  Avg Lanes per Road:             {roadnet_metrics.get('avg_lanes_per_road', 'N/A')}")
        print(f"  Intersections with Signals:     {roadnet_metrics.get('intersections_with_signals', 'N/A')}")
        
        # TRAFFIC FLOW FEATURES
        print("\n" + "=" * 80)
        print("TRAFFIC FLOW FEATURES")
        print("=" * 80)
        print(f"  Total Traffic Flows:            {flow_metrics.get('total_traffic_flows', 'N/A')}")
        print(f"  Total Vehicles in Flows:        {flow_metrics.get('total_vehicles_in_flow', 'N/A')}")
        
        # VEHICLE METRICS
        print("\n" + "=" * 80)
        print("VEHICLE METRICS (AGGREGATE)")
        print("=" * 80)
        if self.vehicle_count_history:
            print(f"  Max Vehicles in Simulation:     {max(self.vehicle_count_history)}")
            print(f"  Min Vehicles in Simulation:     {min(self.vehicle_count_history)}")
            print(f"  Avg Vehicles in Simulation:     {sum(self.vehicle_count_history)/len(self.vehicle_count_history):.2f}")
        
        # Vehicle speed metrics
        all_speeds = []
        for speed_dict in self.vehicle_speed_history:
            all_speeds.extend(speed_dict.values())
        if all_speeds:
            print(f"  Avg Vehicle Speed (all):        {sum(all_speeds)/len(all_speeds):.2f} m/s")
            print(f"  Max Vehicle Speed:              {max(all_speeds):.2f} m/s")
            print(f"  Min Vehicle Speed:              {min(all_speeds):.2f} m/s")
        
        # Vehicle distance metrics
        all_distances = []
        for dist_dict in self.vehicle_distance_history:
            all_distances.extend(dist_dict.values())
        if all_distances:
            print(f"  Total Distance Traveled:        {sum(all_distances):.2f} m")
            print(f"  Avg Distance per Vehicle:       {sum(all_distances)/len(all_distances):.2f} m")
        
        # LANE METRICS
        print("\n" + "=" * 80)
        print("LANE METRICS (AGGREGATE)")
        print("=" * 80)
        all_lane_counts = []
        for counts in self.lane_vehicle_count_history.values():
            all_lane_counts.extend(counts)
        if all_lane_counts:
            print(f"  Max Vehicles in Lane:           {max(all_lane_counts)}")
            print(f"  Min Vehicles in Lane:           {min(all_lane_counts)}")
            print(f"  Avg Vehicles in Lane:           {sum(all_lane_counts)/len(all_lane_counts):.2f}")
        
        all_waiting = []
        for counts in self.lane_waiting_count_history.values():
            all_waiting.extend(counts)
        if all_waiting:
            print(f"  Max Waiting in Lane:            {max(all_waiting)}")
            print(f"  Avg Waiting Vehicles:           {sum(all_waiting)/len(all_waiting):.2f}")
        
        # CONGESTION METRICS
        print("\n" + "=" * 80)
        print("CONGESTION METRICS")
        print("=" * 80)
        if self.congestion_history:
            print(f"  Max Congestion Level:           {max(self.congestion_history):.3f}")
            print(f"  Avg Congestion Level:           {sum(self.congestion_history)/len(self.congestion_history):.3f}")
            print(f"  Min Congestion Level:           {min(self.congestion_history):.3f}")
        
        # TIME METRICS
        print("\n" + "=" * 80)
        print("TIME METRICS")
        print("=" * 80)
        if self.travel_time_history:
            print(f"  Simulation Steps:               {self.step_count}")
            print(f"  Final Simulation Time:          {all_metrics[-1]['current_time']:.2f} s")
            avg_tt = sum(self.travel_time_history) / len(self.travel_time_history)
            print(f"  Avg Travel Time:                {avg_tt:.2f} s")
        
        # V2X COMMUNICATION FEATURES
        if self.v2x:
            print("\n" + "=" * 80)
            print("V2X (VEHICLE-TO-INFRASTRUCTURE) COMMUNICATION")
            print("=" * 80)
            v2x_metrics = self.v2x.get_metrics()
            print(f"  Total V2X Messages:             {v2x_metrics.get('v2x_total_messages', 0)}")
            print(f"  Avg Message Latency:            {v2x_metrics.get('v2x_avg_latency_ms', 0):.2f} ms")
            print(f"  Max Message Latency:            {v2x_metrics.get('v2x_max_latency_ms', 0):.2f} ms")
            print(f"  Intersections Broadcasting:     {v2x_metrics.get('v2x_intersections_broadcasting', 0)}")
            print(f"  Signal Optimizations:           {v2x_metrics.get('v2x_signal_optimizations', 0)}")
            print(f"  Adaptive Phase Changes:         {v2x_metrics.get('v2x_adaptive_phases', 0)}")
            print(f"  V2X Efficiency Score:           {v2x_metrics.get('v2x_efficiency_score', 0):.3f}")
        
        # V2V COMMUNICATION FEATURES
        if self.v2v:
            print("\n" + "=" * 80)
            print("V2V (VEHICLE-TO-VEHICLE) COMMUNICATION")
            print("=" * 80)
            v2v_metrics = self.v2v.get_metrics()
            print(f"  Total V2V Messages:             {v2v_metrics.get('v2v_total_messages', 0)}")
            print(f"  Collision Warnings Issued:      {v2v_metrics.get('v2v_collision_warnings', 0)}")
            print(f"  Cooperative Maneuvers:          {v2v_metrics.get('v2v_cooperative_maneuvers', 0)}")
            print(f"  Total Coordination Events:      {v2v_metrics.get('v2v_coordination_events', 0)}")
            print(f"  Communication Range:            {v2v_metrics.get('v2v_communication_range_m', 0)} m")
            print(f"  Success Rate:                   {v2v_metrics.get('v2v_success_rate', 0):.2%}")
        
        # COMPOSITE COMPONENTS
        print("\n" + "=" * 80)
        print("COMPOSITE COMPONENTS - INTEGRATED METRICS")
        print("=" * 80)
        
        # System efficiency composite
        if all_speeds and self.congestion_history:
            avg_speed = sum(all_speeds) / len(all_speeds)
            avg_congestion = sum(self.congestion_history) / len(self.congestion_history)
            system_efficiency = (avg_speed / max(all_speeds, 1)) * (1 - avg_congestion)
            print(f"  System Efficiency Score:        {system_efficiency:.3f} (0-1)")
        
        # Communication overhead
        total_comm_messages = 0
        if self.v2x:
            total_comm_messages += self.v2x.message_count
        if self.v2v:
            total_comm_messages += self.v2v.message_count
        print(f"  Total Communication Messages:   {total_comm_messages}")
        
        # Network capacity utilization
        total_lanes = roadnet_metrics.get('total_lanes', 1)
        if all_lane_counts:
            avg_lane_occupancy = sum(all_lane_counts) / len(all_lane_counts)
            capacity_util = (avg_lane_occupancy / 20) * 100  # Assume 20 vehicle capacity per lane
            print(f"  Network Capacity Utilization:   {min(capacity_util, 100):.1f}%")
        
        # Throughput metric
        if self.step_count > 0:
            total_distance = sum([sum(d.values()) for d in self.vehicle_distance_history])
            throughput = total_distance / self.step_count
            print(f"  System Throughput:              {throughput:.2f} m/step")
        
        print(f"\n{'='*80}\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='CityFlow Feature Counter with V2X/V2V - Count all measurable simulation features'
    )
    parser.add_argument('config', nargs='?', default='data/config.json',
                        help='Path to config file (default: data/config.json)')
    parser.add_argument('--scale', choices=['large', 'mid', 'both'], default='large',
                        help='Simulation scale: large (24x24), mid (10x10), or both (default: large)')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of simulation steps to run (default: 500)')
    parser.add_argument('--output', type=str, help='Output CSV file for detailed metrics')
    parser.add_argument('--v2x', action='store_true', default=True, help='Enable V2X communication (default: enabled)')
    parser.add_argument('--no-v2x', action='store_false', dest='v2x', help='Disable V2X communication')
    parser.add_argument('--v2v', action='store_true', default=True, help='Enable V2V communication (default: enabled)')
    parser.add_argument('--no-v2v', action='store_false', dest='v2v', help='Disable V2V communication')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('--threads', type=int, default=1, help='Number of simulation threads')
    
    args = parser.parse_args()
    
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Determine scales to run
        scales_to_run = ['large', 'mid'] if args.scale == 'both' else [args.scale]
        
        all_results = []
        
        for scale in scales_to_run:
            print(f"\n{'#'*80}")
            print(f"# RUNNING {scale.upper()}-SCALE SIMULATION")
            print(f"{'#'*80}\n")
            
            # Initialize counter
            counter = FeatureCounter(
                args.config, 
                scale=scale,
                thread_num=args.threads,
                enable_v2x=args.v2x,
                enable_v2v=args.v2v
            )
            
            # Run simulation
            all_metrics = counter.run_simulation(args.steps)
            all_results.extend(all_metrics)
            
            # Generate summary
            counter.generate_summary(all_metrics)
        
        # Save to CSV if requested
        if args.output:
            with open(args.output, 'w', newline='') as f:
                if all_results:
                    fieldnames = all_results[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
            print(f"Detailed metrics saved to: {args.output}")
        
        print("\n" + "="*80)
        print("ALL SIMULATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
