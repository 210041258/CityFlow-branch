"""
CityFlow API - SQLAlchemy ORM Models
Maps Python objects to PostgreSQL database tables
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, 
    ForeignKey, Index, func, JSON, BigInteger, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Simulation(Base):
    """Simulation metadata and lifecycle"""
    __tablename__ = 'simulations'
    
    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    config_path = Column(String(512), nullable=False)
    roadnet_path = Column(String(512), nullable=True)
    flow_path = Column(String(512), nullable=True)
    algorithm = Column(String(50), nullable=False, default='adaptive', index=True)
    status = Column(String(50), nullable=False, default='initialized', index=True)
    step_count = Column(Integer, default=0)
    total_time = Column(Float, default=0.0)
    max_steps = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    metrics = relationship("SimulationMetric", back_populates="simulation", cascade="all, delete-orphan")
    algorithm_metrics = relationship("AlgorithmMetric", back_populates="simulation", cascade="all, delete-orphan")
    congestion_analysis = relationship("CongestionAnalysis", back_populates="simulation", cascade="all, delete-orphan")
    vehicles = relationship("Vehicle", back_populates="simulation", cascade="all, delete-orphan")
    intersections = relationship("Intersection", back_populates="simulation", cascade="all, delete-orphan")
    signal_states = relationship("SignalState", back_populates="simulation", cascade="all, delete-orphan")
    v2x_communications = relationship("V2XCommunication", back_populates="simulation", cascade="all, delete-orphan")
    v2v_communications = relationship("V2VCommunication", back_populates="simulation", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="simulation", cascade="all, delete-orphan")
    exports = relationship("SimulationExport", back_populates="simulation", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
        Index('idx_algorithm', 'algorithm'),
    )


class SimulationMetric(Base):
    """Aggregated metrics per simulation step"""
    __tablename__ = 'simulation_metrics'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    simulated_time = Column(Float, default=0.0)
    total_vehicles = Column(Integer, default=0)
    avg_speed = Column(Float, default=0.0)
    max_speed = Column(Float, default=0.0)
    min_speed = Column(Float, default=0.0)
    vehicles_waiting = Column(Integer, default=0)
    total_travel_distance = Column(Float, default=0.0)
    total_wait_time = Column(Float, default=0.0)
    congestion_level = Column(Float, default=0.0, index=True)
    congestion_state = Column(String(20), nullable=True)
    avg_queue_length = Column(Float, default=0.0)
    throughput = Column(Integer, default=0)
    emission_co2 = Column(Float, default=0.0)
    emission_nox = Column(Float, default=0.0)
    
    simulation = relationship("Simulation", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_simulation_step', 'simulation_id', 'step'),
        Index('idx_metrics_timestamp', 'timestamp'),
    )


class AlgorithmMetric(Base):
    """Traffic control algorithm performance metrics"""
    __tablename__ = 'algorithm_metrics'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    algorithm_type = Column(String(50), nullable=False)
    total_decisions = Column(Integer, default=0)
    optimization_events = Column(Integer, default=0)
    avg_phase_duration = Column(Float, default=0.0)
    phase_changes = Column(Integer, default=0)
    decision_latency = Column(Float, default=0.0)
    effectiveness_score = Column(Float, default=0.0)
    
    simulation = relationship("Simulation", back_populates="algorithm_metrics")
    
    __table_args__ = (
        Index('idx_simulation_algorithm', 'simulation_id', 'algorithm_type'),
        Index('idx_algo_step', 'step'),
    )


class CongestionAnalysis(Base):
    """Congestion state tracking and transitions"""
    __tablename__ = 'congestion_analysis'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    congestion_value = Column(Float, default=0.0)
    congestion_state = Column(Integer, default=0)  # 0=FREE, 1=LIGHT, 2=MODERATE, 3=HEAVY, 4=SEVERE
    state_transitions = Column(Integer, default=0)
    most_common_state = Column(Integer, nullable=True)
    state_duration_seconds = Column(Integer, default=0)
    affected_intersections = Column(Integer, default=0)
    
    simulation = relationship("Simulation", back_populates="congestion_analysis")
    
    __table_args__ = (
        Index('idx_simulation_congestion', 'simulation_id', 'congestion_state'),
        Index('idx_cong_step', 'step'),
    )


class Vehicle(Base):
    """Vehicle information and routing"""
    __tablename__ = 'vehicles'
    
    id = Column(String(36), primary_key=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False, index=True)
    vehicle_type = Column(String(50), default='car', index=True)
    route_id = Column(String(100), nullable=True)
    source_junction = Column(String(100), nullable=True)
    destination_junction = Column(String(100), nullable=True)
    
    simulation = relationship("Simulation", back_populates="vehicles")
    trajectories = relationship("VehicleTrajectory", back_populates="vehicle", cascade="all, delete-orphan")


class VehicleTrajectory(Base):
    """High-frequency vehicle position and state updates"""
    __tablename__ = 'vehicle_trajectories'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    vehicle_id = Column(String(36), ForeignKey('vehicles.id', ondelete='CASCADE'), nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    x = Column(Float, default=0.0)
    y = Column(Float, default=0.0)
    speed = Column(Float, default=0.0)
    acceleration = Column(Float, default=0.0)
    direction = Column(Float, default=0.0)
    road_id = Column(String(100), nullable=True, index=True)
    lane_id = Column(String(100), nullable=True)
    is_waiting = Column(Boolean, default=False)
    wait_time = Column(Float, default=0.0)
    
    vehicle = relationship("Vehicle", back_populates="trajectories")
    
    __table_args__ = (
        Index('idx_simulation_step_traj', 'simulation_id', 'step'),
        Index('idx_vehicle_step', 'vehicle_id', 'step'),
    )


class Intersection(Base):
    """Traffic intersection configuration"""
    __tablename__ = 'intersections'
    
    id = Column(String(100), primary_key=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=True, index=True)
    junction_name = Column(String(255), nullable=True)
    junction_type = Column(String(50), default='traffic_light')
    x = Column(Float, default=0.0)
    y = Column(Float, default=0.0)
    num_phases = Column(Integer, default=2)
    cycle_time = Column(Integer, default=60)
    
    simulation = relationship("Simulation", back_populates="intersections")
    signal_states = relationship("SignalState", back_populates="intersection", cascade="all, delete-orphan")


class SignalState(Base):
    """Traffic light signal state per step"""
    __tablename__ = 'signal_states'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    intersection_id = Column(String(100), ForeignKey('intersections.id', ondelete='CASCADE'), nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    current_phase = Column(Integer, default=0)
    phase_name = Column(String(50), nullable=True)
    phase_duration = Column(Integer, default=30)
    remaining_time = Column(Integer, default=30)
    signal_state = Column(String(20), nullable=True)  # GREEN, YELLOW, RED
    vehicles_ns = Column(Integer, default=0)
    vehicles_ew = Column(Integer, default=0)
    queue_length_ns = Column(Integer, default=0)
    queue_length_ew = Column(Integer, default=0)
    throughput_ns = Column(Integer, default=0)
    throughput_ew = Column(Integer, default=0)
    
    simulation = relationship("Simulation", back_populates="signal_states")
    intersection = relationship("Intersection", back_populates="signal_states")
    
    __table_args__ = (
        Index('idx_simulation_intersection_step', 'simulation_id', 'intersection_id', 'step'),
        Index('idx_signal_step', 'step'),
    )


class V2XCommunication(Base):
    """Vehicle-to-Infrastructure communication events"""
    __tablename__ = 'v2x_communications'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    vehicle_id = Column(String(36), ForeignKey('vehicles.id', ondelete='CASCADE'), nullable=False)
    intersection_id = Column(String(100), ForeignKey('intersections.id', ondelete='CASCADE'), nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String(50), nullable=True)
    data = Column(JSON, nullable=True)
    latency_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    
    simulation = relationship("Simulation", back_populates="v2x_communications")
    
    __table_args__ = (
        Index('idx_v2x_simulation_step', 'simulation_id', 'step'),
        Index('idx_v2x_vehicle', 'vehicle_id'),
    )


class V2VCommunication(Base):
    """Vehicle-to-Vehicle communication events"""
    __tablename__ = 'v2v_communications'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    vehicle_id_from = Column(String(36), nullable=False)
    vehicle_id_to = Column(String(36), nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String(50), nullable=True)
    data = Column(JSON, nullable=True)
    distance = Column(Float, default=0.0, index=True)
    latency_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    
    simulation = relationship("Simulation", back_populates="v2v_communications")
    
    __table_args__ = (
        Index('idx_v2v_simulation_step', 'simulation_id', 'step'),
        Index('idx_v2v_vehicle_from', 'vehicle_id_from'),
    )


class Alert(Base):
    """Events, warnings, and anomalies"""
    __tablename__ = 'alerts'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    location_id = Column(String(100), nullable=True)
    message = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    data = Column(JSON, nullable=True)
    
    simulation = relationship("Simulation", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alerts_simulation_type', 'simulation_id', 'alert_type'),
        Index('idx_alerts_severity', 'severity'),
    )


class SimulationExport(Base):
    """Data exports and result files"""
    __tablename__ = 'simulation_exports'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False, index=True)
    export_type = Column(String(50), nullable=True, index=True)
    export_path = Column(String(512), nullable=False)
    file_size = Column(BigInteger, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    exported_by = Column(String(100), nullable=True)
    
    simulation = relationship("Simulation", back_populates="exports")


class MetricsHourlySummary(Base):
    """Hourly aggregated metrics for analytics dashboards"""
    __tablename__ = 'metrics_hourly_summary'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), ForeignKey('simulations.id', ondelete='CASCADE'), nullable=False)
    hour_start = Column(DateTime, nullable=True)
    total_vehicles_peak = Column(Integer, nullable=True)
    avg_congestion = Column(Float, nullable=True)
    total_vehicles_completed = Column(Integer, nullable=True)
    total_wait_time = Column(Float, nullable=True)
    avg_speed = Column(Float, nullable=True)
    throughput_efficiency = Column(Float, nullable=True)
    
    simulation = relationship("Simulation")
    
    __table_args__ = (
        Index('idx_sim_hour', 'simulation_id', 'hour_start', unique=True),
    )
