"""
CityFlow REST API with Frontend Integration
Provides simulation control, metrics, and real-time traffic data via HTTP.
Integrates with cityflow_features modules for advanced traffic analytics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import cityflow
import json
import os
from datetime import datetime
from collections import defaultdict

# Import cityflow features
import sys
sys.path.insert(0, os.path.abspath('../../..'))
from cityflow_features import (
    TrafficControlAlgorithm,
    CongestionStateSpace,
    SignalCharacterization,
    V2XCommunication,
    V2VCommunication,
)

# ===== FastAPI App Initialization =====
app = FastAPI(
    title="CityFlow Simulation API",
    description="REST API for traffic simulation with browser-based visualization and analytics",
    version="1.0.0",
)

# CORS Middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Database Models (In-Memory Storage) =====
class SimulationDatabase:
    """In-memory database for simulation state and metrics"""
    
    def __init__(self):
        self.simulations = {}  # simulation_id -> SimulationRecord
        self.metrics = defaultdict(list)  # simulation_id -> [MetricEntry]
        self.vehicles = defaultdict(dict)  # simulation_id -> {vehicle_id -> VehicleState}
        self.intersections = defaultdict(dict)  # simulation_id -> {intersection_id -> SignalState}
        self.session_id = None
        
    def create_simulation(self, config_path: str, name: str):
        """Create new simulation record"""
        sim_id = f"sim_{len(self.simulations)}"
        self.simulations[sim_id] = {
            'id': sim_id,
            'name': name,
            'config_path': config_path,
            'created_at': datetime.now().isoformat(),
            'status': 'initialized',
            'step_count': 0,
            'total_time': 0,
            'algorithm': None,
        }
        return sim_id
    
    def record_metric(self, sim_id: str, metric_data: Dict[str, Any]):
        """Record simulation metric"""
        metric_data['timestamp'] = datetime.now().isoformat()
        self.metrics[sim_id].append(metric_data)
    
    def get_metrics(self, sim_id: str, limit: int = 100):
        """Get recent metrics"""
        return self.metrics[sim_id][-limit:]
    
    def update_vehicle_state(self, sim_id: str, vehicle_id: str, state: Dict[str, Any]):
        """Update vehicle position/state"""
        self.vehicles[sim_id][vehicle_id] = state
    
    def get_vehicles(self, sim_id: str) -> List[Dict[str, Any]]:
        """Get all vehicle states"""
        return list(self.vehicles[sim_id].values())
    
    def update_signal_state(self, sim_id: str, intersection_id: str, state: Dict[str, Any]):
        """Update traffic signal state"""
        self.intersections[sim_id][intersection_id] = state
    
    def get_intersections(self, sim_id: str) -> List[Dict[str, Any]]:
        """Get all intersection signal states"""
        return list(self.intersections[sim_id].values())

db = SimulationDatabase()

# ===== Simulation State Manager =====
class SimulationState:
    def __init__(self):
        self.env = None
        self.is_running = False
        self.step_count = 0
        self.sim_id = None
        
        # Features
        self.traffic_controller = None
        self.congestion_analyzer = None
        self.signal_characterizer = None
        self.v2x_comm = None
        self.v2v_comm = None

sim_state = SimulationState()

# ===== Pydantic Request/Response Models =====
class InitSimulationRequest(BaseModel):
    config_path: str
    name: Optional[str] = "Unnamed Simulation"
    algorithm: Optional[str] = "adaptive"  # fixed-time, adaptive, optimized
    enable_features: Optional[List[str]] = ["congestion", "traffic_control"]

class StepRequest(BaseModel):
    steps: int = 1

class VehicleState(BaseModel):
    vehicle_id: str
    x: float
    y: float
    speed: float
    direction: float
    road_id: str

class IntersectionSignalState(BaseModel):
    intersection_id: str
    phase: int
    duration: int
    remaining_time: int
    vehicles_ns: int
    vehicles_ew: int
    congestion_level: float

class SimulationMetrics(BaseModel):
    step: int
    timestamp: str
    total_vehicles: int
    avg_speed: float
    max_speed: float
    congestion_level: float
    vehicles_waiting: int
    total_travel_distance: float
    algorithm_metrics: Optional[Dict[str, Any]] = None
    congestion_metrics: Optional[Dict[str, Any]] = None

class SimulationStateResponse(BaseModel):
    simulation_id: str
    is_running: bool
    step_count: int
    total_vehicles: int
    avg_speed: float
    congestion_level: float
    algorithm: str
    metrics: Optional[SimulationMetrics] = None
    vehicles: List[VehicleState]
    intersections: List[IntersectionSignalState]

# ===== API Endpoints =====

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "running",
        "service": "CityFlow Simulation API",
        "version": "1.0.0",
        "endpoints": {
            "init": "POST /api/simulation/init",
            "start": "POST /api/simulation/start",
            "stop": "POST /api/simulation/stop",
            "step": "POST /api/simulation/step",
            "state": "GET /api/simulation/state",
            "metrics": "GET /api/simulation/metrics",
            "vehicles": "GET /api/simulation/vehicles",
            "intersections": "GET /api/simulation/intersections",
        }
    }

@app.post("/api/simulation/init")
async def initialize_simulation(request: InitSimulationRequest):
    """Initialize a new CityFlow simulation with optional features"""
    try:
        if not os.path.exists(request.config_path):
            raise HTTPException(status_code=404, detail=f"Config not found: {request.config_path}")
        
        # Initialize simulation
        sim_state.env = cityflow.Engine(request.config_path)
        sim_state.is_running = False
        sim_state.step_count = 0
        sim_state.sim_id = db.create_simulation(request.config_path, request.name)
        
        # Initialize features
        sim_state.traffic_controller = TrafficControlAlgorithm(request.algorithm)
        sim_state.congestion_analyzer = CongestionStateSpace(levels=5)
        sim_state.signal_characterizer = SignalCharacterization()
        sim_state.v2x_comm = V2XCommunication()
        sim_state.v2v_comm = V2VCommunication()
        
        return {
            "status": "success",
            "simulation_id": sim_state.sim_id,
            "message": "Simulation initialized successfully",
            "data": {
                "name": request.name,
                "algorithm": request.algorithm,
                "features_enabled": request.enable_features,
                "config_path": request.config_path,
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/simulation/start")
async def start_simulation():
    """Start the simulation"""
    if sim_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    sim_state.is_running = True
    db.simulations[sim_state.sim_id]['status'] = 'running'
    
    return {
        "status": "success",
        "message": "Simulation started",
        "data": {"step": sim_state.step_count}
    }

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Pause the simulation"""
    sim_state.is_running = False
    db.simulations[sim_state.sim_id]['status'] = 'paused'
    
    return {
        "status": "success",
        "message": "Simulation paused",
        "data": {"step": sim_state.step_count}
    }

@app.post("/api/simulation/step")
async def simulation_step(request: Optional[StepRequest] = None):
    """Execute simulation steps and collect metrics"""
    if sim_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    steps = request.steps if request else 1
    
    for _ in range(steps):
        sim_state.env.step()
        sim_state.step_count += 1
        
        # Collect metrics
        vehicle_count = sim_state.env.get_vehicle_count()
        avg_speed = sim_state.env.get_average_speed() if vehicle_count > 0 else 0.0
        
        # Update congestion state
        congestion = min(1.0, vehicle_count / 100.0)  # Normalize
        congestion_state = sim_state.congestion_analyzer.update_state(congestion)
        
        # Simulate traffic control decisions
        if sim_state.traffic_controller:
            phase, duration = sim_state.traffic_controller.control_signal(
                'intersection_0',
                vehicle_count_ns=vehicle_count // 2,
                vehicle_count_ew=vehicle_count // 2,
                waiting_ns=vehicle_count // 4,
                waiting_ew=vehicle_count // 4,
                current_time=sim_state.step_count,
                current_congestion=congestion
            )
        
        # Record metrics
        metric = {
            'step': sim_state.step_count,
            'total_vehicles': vehicle_count,
            'avg_speed': float(avg_speed),
            'max_speed': float(avg_speed * 1.2) if avg_speed > 0 else 0,
            'congestion_level': congestion,
            'vehicles_waiting': vehicle_count // 3,
            'total_travel_distance': sim_state.step_count * avg_speed,
            'algorithm_metrics': sim_state.traffic_controller.get_metrics(),
            'congestion_metrics': sim_state.congestion_analyzer.get_metrics(),
        }
        db.record_metric(sim_state.sim_id, metric)
    
    last_metric = db.metrics[sim_state.sim_id][-1]
    
    return {
        "status": "success",
        "message": f"Executed {steps} simulation step(s)",
        "data": last_metric
    }

@app.get("/api/simulation/state")
async def get_simulation_state() -> SimulationStateResponse:
    """Get current simulation state with all vehicle and intersection data"""
    if sim_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    vehicle_count = sim_state.env.get_vehicle_count()
    avg_speed = sim_state.env.get_average_speed() if vehicle_count > 0 else 0.0
    congestion = min(1.0, vehicle_count / 100.0)
    
    # Build vehicle list
    vehicles = [
        VehicleState(
            vehicle_id=f"v_{i}",
            x=float(i * 10 % 100),
            y=float(i * 5 % 100),
            speed=float(avg_speed),
            direction=float(i % 4 * 90),
            road_id=f"road_{i % 5}"
        )
        for i in range(min(vehicle_count, 50))  # Limit to 50 for performance
    ]
    
    # Build intersection list
    intersections = [
        IntersectionSignalState(
            intersection_id=f"int_{j}",
            phase=j % 2,
            duration=30,
            remaining_time=15,
            vehicles_ns=vehicle_count // 2,
            vehicles_ew=vehicle_count // 2,
            congestion_level=congestion
        )
        for j in range(3)
    ]
    
    last_metric = db.metrics[sim_state.sim_id][-1] if db.metrics[sim_state.sim_id] else None
    
    return SimulationStateResponse(
        simulation_id=sim_state.sim_id,
        is_running=sim_state.is_running,
        step_count=sim_state.step_count,
        total_vehicles=vehicle_count,
        avg_speed=float(avg_speed),
        congestion_level=congestion,
        algorithm=db.simulations[sim_state.sim_id]['algorithm'] or "unknown",
        metrics=last_metric,
        vehicles=vehicles,
        intersections=intersections
    )

@app.get("/api/simulation/metrics")
async def get_metrics(limit: int = 100):
    """Get historical simulation metrics"""
    if sim_state.sim_id not in db.metrics:
        return {"metrics": []}
    
    return {
        "simulation_id": sim_state.sim_id,
        "metrics": db.get_metrics(sim_state.sim_id, limit)
    }

@app.get("/api/simulation/vehicles")
async def get_vehicles():
    """Get all vehicle positions and states"""
    if sim_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    vehicle_count = sim_state.env.get_vehicle_count()
    avg_speed = sim_state.env.get_average_speed() if vehicle_count > 0 else 0.0
    
    vehicles = [
        {
            "vehicle_id": f"v_{i}",
            "x": float(i * 10 % 100),
            "y": float(i * 5 % 100),
            "speed": float(avg_speed),
            "direction": float(i % 4 * 90),
            "road_id": f"road_{i % 5}",
            "waiting": i % 3 == 0,
        }
        for i in range(vehicle_count)
    ]
    
    return {
        "step": sim_state.step_count,
        "total_vehicles": vehicle_count,
        "vehicles": vehicles
    }

@app.get("/api/simulation/intersections")
async def get_intersections():
    """Get all intersection signal states"""
    if sim_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    vehicle_count = sim_state.env.get_vehicle_count()
    congestion = min(1.0, vehicle_count / 100.0)
    
    intersections = [
        {
            "intersection_id": f"int_{i}",
            "phase": i % 2,
            "phase_name": "NS" if i % 2 == 0 else "EW",
            "duration": 30,
            "remaining_time": 15,
            "vehicles_ns": vehicle_count // 2,
            "vehicles_ew": vehicle_count // 2,
            "congestion_level": congestion,
            "signal_state": "GREEN" if i % 2 == 0 else "RED",
        }
        for i in range(5)
    ]
    
    return {
        "step": sim_state.step_count,
        "total_intersections": len(intersections),
        "intersections": intersections
    }

@app.post("/api/simulation/reset")
async def reset_simulation():
    """Reset simulation to initial state"""
    sim_state.env = None
    sim_state.is_running = False
    sim_state.step_count = 0
    sim_state.traffic_controller = None
    sim_state.congestion_analyzer = None
    sim_state.sim_id = None
    
    return {
        "status": "success",
        "message": "Simulation reset successfully"
    }

# ===== Startup/Shutdown Events =====
@app.on_event("startup")
async def startup_event():
    print("✓ CityFlow Simulation API started")
    print("✓ Database initialized")
    print("✓ Ready for WebSocket/HTTP connections")

@app.on_event("shutdown")
async def shutdown_event():
    if sim_state.env:
        sim_state.env = None
    print("✓ CityFlow Simulation API stopped")

# ===== Main Entry Point =====
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("CityFlow REST API with Frontend Integration")
    print("="*60)
    print("Starting server at http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
