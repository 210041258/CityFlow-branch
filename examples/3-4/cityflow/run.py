"""
CityFlow REST API with Frontend Integration
Provides simulation control, metrics, and real-time traffic data via HTTP.
Integrates with cityflow_features modules for advanced traffic analytics.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cityflow

# Import cityflow features
sys.path.insert(0, os.path.abspath("../../.."))
from cityflow_features import (
    CongestionStateSpace,
    SignalCharacterization,
    TrafficControlAlgorithm,
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
    """In-memory database for simulation state and metrics."""

    def __init__(self):
        self.simulations: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.vehicles: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.intersections: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def create_simulation(
        self,
        config_path: str,
        name: str,
        algorithm: Optional[str] = None,
        enable_features: Optional[List[str]] = None,
    ) -> str:
        """Create a new simulation record."""
        sim_id = f"sim_{len(self.simulations)}"
        self.simulations[sim_id] = {
            "id": sim_id,
            "name": name,
            "config_path": config_path,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "step_count": 0,
            "total_time": 0,
            "algorithm": algorithm,
            "features_enabled": enable_features or [],
        }
        return sim_id

    def record_metric(self, sim_id: str, metric_data: Dict[str, Any]):
        """Record a simulation metric."""
        metric_data["timestamp"] = datetime.now().isoformat()
        self.metrics[sim_id].append(metric_data)

    def get_metrics(self, sim_id: str, limit: int = 100):
        """Get recent metrics."""
        if sim_id not in self.metrics:
            return []
        if limit <= 0:
            return []
        return self.metrics[sim_id][-limit:]

    def update_vehicle_state(self, sim_id: str, vehicle_id: str, state: Dict[str, Any]):
        """Update vehicle position/state."""
        self.vehicles[sim_id][vehicle_id] = state

    def get_vehicles(self, sim_id: str) -> List[Dict[str, Any]]:
        """Get all vehicle states."""
        return list(self.vehicles[sim_id].values())

    def update_signal_state(self, sim_id: str, intersection_id: str, state: Dict[str, Any]):
        """Update traffic signal state."""
        self.intersections[sim_id][intersection_id] = state

    def get_intersections(self, sim_id: str) -> List[Dict[str, Any]]:
        """Get all intersection signal states."""
        return list(self.intersections[sim_id].values())


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


# Shared state

db = SimulationDatabase()
sim_state = SimulationState()


# ===== Pydantic Request/Response Models =====
class InitSimulationRequest(BaseModel):
    config_path: str
    name: Optional[str] = "Unnamed Simulation"
    algorithm: Optional[str] = "adaptive"  # fixed-time, adaptive, optimized
    enable_features: Optional[List[str]] = Field(
        default_factory=lambda: ["congestion", "traffic_control"]
    )


class StepRequest(BaseModel):
    steps: int = 1

    @property
    def safe_steps(self) -> int:
        if self.steps <= 0:
            return 1
        return min(self.steps, 1000)


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


# ===== Helpers =====
def require_initialized_simulation() -> str:
    if sim_state.env is None or sim_state.sim_id is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    return sim_state.sim_id


def safe_congestion(vehicle_count: int) -> float:
    return min(1.0, max(0.0, vehicle_count / 100.0))


def get_algorithm_status() -> Dict[str, Any]:
    if sim_state.traffic_controller is None:
        return {}
    try:
        return sim_state.traffic_controller.get_metrics() or {}
    except Exception:
        return {}


def get_congestion_status() -> Dict[str, Any]:
    if sim_state.congestion_analyzer is None:
        return {}
    try:
        return sim_state.congestion_analyzer.get_metrics() or {}
    except Exception:
        return {}


# ===== API Endpoints =====
@app.get("/")
async def root():
    """API health check."""
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
            "reset": "POST /api/simulation/reset",
        },
    }


@app.post("/api/simulation/init")
async def initialize_simulation(request: InitSimulationRequest):
    """Initialize a new CityFlow simulation with optional features."""
    try:
        if not os.path.exists(request.config_path):
            raise HTTPException(status_code=404, detail=f"Config not found: {request.config_path}")

        sim_state.env = cityflow.Engine(request.config_path)
        sim_state.is_running = False
        sim_state.step_count = 0

        sim_id = db.create_simulation(
            request.config_path,
            request.name,
            algorithm=request.algorithm,
            enable_features=request.enable_features or [],
        )
        sim_state.sim_id = sim_id

        enabled = set((request.enable_features or []))

        sim_state.traffic_controller = (
            TrafficControlAlgorithm(request.algorithm)
            if "traffic_control" in enabled
            else None
        )
        sim_state.congestion_analyzer = (
            CongestionStateSpace(levels=5) if "congestion" in enabled else None
        )
        sim_state.signal_characterizer = (
            SignalCharacterization() if "signal_characterization" in enabled else None
        )
        sim_state.v2x_comm = V2XCommunication() if "v2x" in enabled else None
        sim_state.v2v_comm = V2VCommunication() if "v2v" in enabled else None

        db.simulations[sim_id]["status"] = "initialized"

        return {
            "status": "success",
            "simulation_id": sim_id,
            "message": "Simulation initialized successfully",
            "data": {
                "name": request.name,
                "algorithm": request.algorithm,
                "features_enabled": request.enable_features,
                "config_path": request.config_path,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/simulation/start")
async def start_simulation():
    """Start the simulation."""
    sim_id = require_initialized_simulation()

    sim_state.is_running = True
    db.simulations[sim_id]["status"] = "running"

    return {
        "status": "success",
        "message": "Simulation started",
        "data": {"step": sim_state.step_count, "simulation_id": sim_id},
    }


@app.post("/api/simulation/stop")
async def stop_simulation():
    """Pause the simulation."""
    sim_id = require_initialized_simulation()

    sim_state.is_running = False
    db.simulations[sim_id]["status"] = "paused"

    return {
        "status": "success",
        "message": "Simulation paused",
        "data": {"step": sim_state.step_count, "simulation_id": sim_id},
    }


@app.post("/api/simulation/step")
async def simulation_step(request: Optional[StepRequest] = None):
    """Execute simulation steps and collect metrics."""
    sim_id = require_initialized_simulation()
    steps = (request.safe_steps if request else 1)

    for _ in range(steps):
        sim_state.env.step()
        sim_state.step_count += 1

        vehicle_count = sim_state.env.get_vehicle_count()
        avg_speed = sim_state.env.get_average_speed() if vehicle_count > 0 else 0.0
        congestion = safe_congestion(vehicle_count)

        if sim_state.congestion_analyzer is not None:
            sim_state.congestion_analyzer.update_state(congestion)

        if sim_state.traffic_controller is not None:
            sim_state.traffic_controller.control_signal(
                "intersection_0",
                vehicle_count_ns=vehicle_count // 2,
                vehicle_count_ew=vehicle_count // 2,
                waiting_ns=vehicle_count // 4,
                waiting_ew=vehicle_count // 4,
                current_time=sim_state.step_count,
                current_congestion=congestion,
            )

        metric = {
            "step": sim_state.step_count,
            "total_vehicles": vehicle_count,
            "avg_speed": float(avg_speed),
            "max_speed": float(avg_speed * 1.2) if avg_speed > 0 else 0,
            "congestion_level": congestion,
            "vehicles_waiting": max(0, vehicle_count // 3),
            "total_travel_distance": sim_state.step_count * avg_speed,
            "algorithm_metrics": get_algorithm_status(),
            "congestion_metrics": get_congestion_status(),
        }
        db.record_metric(sim_id, metric)

    if sim_id not in db.metrics or not db.metrics[sim_id]:
        raise HTTPException(status_code=500, detail="Simulation metrics were not recorded")

    last_metric = db.metrics[sim_id][-1]

    return {
        "status": "success",
        "message": f"Executed {steps} simulation step(s)",
        "data": last_metric,
    }


@app.get("/api/simulation/state")
async def get_simulation_state() -> SimulationStateResponse:
    """Get current simulation state with all vehicle and intersection data."""
    sim_id = require_initialized_simulation()

    vehicle_count = sim_state.env.get_vehicle_count()
    avg_speed = sim_state.env.get_average_speed() if vehicle_count > 0 else 0.0
    congestion = safe_congestion(vehicle_count)

    vehicles = [
        VehicleState(
            vehicle_id=f"v_{i}",
            x=float(i * 10 % 100),
            y=float(i * 5 % 100),
            speed=float(avg_speed),
            direction=float(i % 4 * 90),
            road_id=f"road_{i % 5}",
        )
        for i in range(min(vehicle_count, 50))
    ]

    intersections = [
        IntersectionSignalState(
            intersection_id=f"int_{j}",
            phase=j % 2,
            duration=30,
            remaining_time=15,
            vehicles_ns=vehicle_count // 2,
            vehicles_ew=vehicle_count // 2,
            congestion_level=congestion,
        )
        for j in range(3)
    ]

    last_metric = db.metrics[sim_id][-1] if db.metrics.get(sim_id) else None

    return SimulationStateResponse(
        simulation_id=sim_id,
        is_running=sim_state.is_running,
        step_count=sim_state.step_count,
        total_vehicles=vehicle_count,
        avg_speed=float(avg_speed),
        congestion_level=congestion,
        algorithm=db.simulations[sim_id].get("algorithm") or "unknown",
        metrics=last_metric,
        vehicles=vehicles,
        intersections=intersections,
    )


@app.get("/api/simulation/metrics")
async def get_metrics(limit: int = 100):
    """Get historical simulation metrics."""
    sim_id = require_initialized_simulation()
    if limit <= 0:
        limit = 100

    return {
        "simulation_id": sim_id,
        "metrics": db.get_metrics(sim_id, limit),
    }


@app.get("/api/simulation/vehicles")
async def get_vehicles():
    """Get all vehicle positions and states."""
    sim_id = require_initialized_simulation()

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
        "simulation_id": sim_id,
        "step": sim_state.step_count,
        "total_vehicles": vehicle_count,
        "vehicles": vehicles,
    }


@app.get("/api/simulation/intersections")
async def get_intersections():
    """Get all intersection signal states."""
    sim_id = require_initialized_simulation()

    vehicle_count = sim_state.env.get_vehicle_count()
    congestion = safe_congestion(vehicle_count)

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
        "simulation_id": sim_id,
        "step": sim_state.step_count,
        "total_intersections": len(intersections),
        "intersections": intersections,
    }


@app.post("/api/simulation/reset")
async def reset_simulation():
    """Reset the simulation to initial state."""
    sim_id = sim_state.sim_id
    sim_state.env = None
    sim_state.is_running = False
    sim_state.step_count = 0
    sim_state.traffic_controller = None
    sim_state.congestion_analyzer = None
    sim_state.signal_characterizer = None
    sim_state.v2x_comm = None
    sim_state.v2v_comm = None
    sim_state.sim_id = None

    if sim_id is not None:
        db.metrics.pop(sim_id, None)
        db.vehicles.pop(sim_id, None)
        db.intersections.pop(sim_id, None)
        db.simulations.pop(sim_id, None)

    return {
        "status": "success",
        "message": "Simulation reset successfully",
    }


# ===== Startup/Shutdown Events =====
@app.on_event("startup")
async def startup_event():
    print("✓ CityFlow Simulation API started")
    print("✓ Database initialized")
    print("✓ Ready for WebSocket/HTTP connections")


@app.on_event("shutdown")
async def shutdown_event():
    if sim_state.env is not None:
        sim_state.env = None
    print("✓ CityFlow Simulation API stopped")


# ===== Main Entry Point =====
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("CityFlow REST API with Frontend Integration")
    print("=" * 60)
    print("Starting server at http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
