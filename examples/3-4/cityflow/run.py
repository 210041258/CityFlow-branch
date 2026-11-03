"""
CityFlow REST API with Frontend Integration
Provides simulation control, metrics, and real-time traffic data via HTTP.
Includes multifactor analysis for observed traffic counts associated with a map.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import sys
import math

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


app = FastAPI(
    title="CityFlow Simulation API",
    description=(
        "REST API for traffic simulation, live metrics, and multifactor analysis "
        "of observed traffic data associated with a map scenario."
    ),
    version="1.1.0",
)

# Configure explicit origins for production deployments. Wildcard origins are
# retained here for compatibility with the example's local frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationDatabase:
    """In-memory store for simulation state and metrics."""

    def __init__(self):
        self.simulations: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.vehicles: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.intersections: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._next_id = 0

    def create_simulation(
        self,
        config_path: str,
        name: str,
        algorithm: Optional[str] = None,
        enable_features: Optional[List[str]] = None,
    ) -> str:
        """Create a simulation record with a collision-safe in-process ID."""
        sim_id = f"sim_{self._next_id}"
        self._next_id += 1
        self.simulations[sim_id] = {
            "id": sim_id,
            "name": name,
            "config_path": config_path,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "step_count": 0,
            "algorithm": algorithm,
            "features_enabled": enable_features or [],
        }
        return sim_id

    def record_metric(self, sim_id: str, metric_data: Dict[str, Any]):
        """Record a simulation metric with its wall-clock timestamp."""
        metric_data["timestamp"] = datetime.now().isoformat()
        self.metrics[sim_id].append(metric_data)

    def get_metrics(self, sim_id: str, limit: int = 100):
        """Return the most recent metrics."""
        if sim_id not in self.metrics or limit <= 0:
            return []
        return self.metrics[sim_id][-limit:]


db = SimulationDatabase()


class SimulationState:
    def __init__(self):
        self.env = None
        self.is_running = False
        self.step_count = 0
        self.sim_id: Optional[str] = None
        self.traffic_controller = None
        self.congestion_analyzer = None
        self.signal_characterizer = None
        self.v2x_comm = None
        self.v2v_comm = None


sim_state = SimulationState()


# ===== Request/Response Models =====
class InitSimulationRequest(BaseModel):
    config_path: str
    name: Optional[str] = "Unnamed Simulation"
    algorithm: Optional[str] = "adaptive"
    enable_features: Optional[List[str]] = Field(
        default_factory=lambda: ["congestion", "traffic_control"]
    )


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=1000)


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


class TrafficObservation(BaseModel):
    """One measured traffic sample; factors should be numeric and consistently scaled."""

    frame: int = Field(ge=0)
    vehicle_count: int = Field(ge=0)
    demand_multiplier: float = Field(default=1.0, gt=0)
    weather_factor: float = Field(default=1.0, gt=0)
    incident_factor: float = Field(default=1.0, gt=0)
    time_of_day_factor: float = Field(default=1.0, gt=0)


class TrafficScenario(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    map_name: Optional[str] = None
    observations: List[TrafficObservation] = Field(min_length=6)


class MultifactorAnalysisRequest(BaseModel):
    scenarios: List[TrafficScenario] = Field(min_length=1, max_length=20)


FACTOR_NAMES = [
    "demand_multiplier",
    "weather_factor",
    "incident_factor",
    "time_of_day_factor",
]


def require_initialized_simulation() -> str:
    if sim_state.env is None or sim_state.sim_id is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    return sim_state.sim_id


def safe_congestion(vehicle_count: int) -> float:
    return min(1.0, max(0.0, vehicle_count / 100.0))


def get_feature_metrics(feature: Any) -> Dict[str, Any]:
    """Read optional feature metrics without failing the simulation response."""
    if feature is None:
        return {}
    try:
        result = feature.get_metrics()
        return result if isinstance(result, dict) else {}
    except (AttributeError, TypeError, ValueError):
        return {}


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Solve a small dense system using Gaussian elimination with pivoting."""
    size = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError(
                "Factor data is rank-deficient; provide observations with independent variation."
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]

        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]

    return [augmented[row][-1] for row in range(size)]


def analyze_scenario(scenario: TrafficScenario) -> Dict[str, Any]:
    """Fit an OLS multifactor model and calculate in-sample fit criteria."""
    observations = scenario.observations
    parameter_count = len(FACTOR_NAMES) + 1  # intercept + four factors
    sample_count = len(observations)
    if sample_count <= parameter_count:
        raise ValueError(
            f"Scenario '{scenario.name}' needs at least {parameter_count + 1} observations."
        )

    rows = [[1.0] + [float(getattr(item, factor)) for factor in FACTOR_NAMES]
            for item in observations]
    targets = [float(item.vehicle_count) for item in observations]
    columns = len(rows[0])
    xtx = [
        [sum(row[left] * row[right] for row in rows) for right in range(columns)]
        for left in range(columns)
    ]
    xty = [
        sum(row[column] * target for row, target in zip(rows, targets))
        for column in range(columns)
    ]
    coefficients = solve_linear_system(xtx, xty)
    predictions = [sum(a * b for a, b in zip(row, coefficients)) for row in rows]

    mean_target = sum(targets) / sample_count
    residual_sum_squares = sum(
        (observed - predicted) ** 2
        for observed, predicted in zip(targets, predictions)
    )
    total_sum_squares = sum((observed - mean_target) ** 2 for observed in targets)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else (1.0 if residual_sum_squares < 1e-12 else 0.0)
    )

    # Gaussian-error information criteria, using the same parameter count for
    # all scenarios so AIC/BIC can be compared on the same measurement basis.
    safe_rss = max(residual_sum_squares, 1e-12)
    log_likelihood_term = sample_count * math.log(safe_rss / sample_count)
    aic = log_likelihood_term + 2 * parameter_count
    bic = log_likelihood_term + parameter_count * math.log(sample_count)

    return {
        "name": scenario.name,
        "map_name": scenario.map_name,
        "observations": sample_count,
        "model": "vehicle_count ~ intercept + demand + weather + incident + time_of_day",
        "factor_names": ["intercept"] + FACTOR_NAMES,
        "coefficients": dict(zip(["intercept"] + FACTOR_NAMES, coefficients)),
        "r_squared": r_squared,
        "aic": aic,
        "bic": bic,
        "rmse": math.sqrt(residual_sum_squares / sample_count),
        "observed_range": {
            "min": min(targets),
            "max": max(targets),
        },
    }


# ===== API Endpoints =====
@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "running",
        "service": "CityFlow Simulation API",
        "version": app.version,
        "endpoints": {
            "init": "POST /api/simulation/init",
            "start": "POST /api/simulation/start",
            "stop": "POST /api/simulation/stop",
            "step": "POST /api/simulation/step",
            "state": "GET /api/simulation/state",
            "metrics": "GET /api/simulation/metrics",
            "vehicles": "GET /api/simulation/vehicles",
            "intersections": "GET /api/simulation/intersections",
            "analyze_scenarios": "POST /api/simulation/scenarios/analyze",
            "reset": "POST /api/simulation/reset",
        },
    }


@app.post("/api/simulation/init")
async def initialize_simulation(request: InitSimulationRequest):
    """Initialize a CityFlow simulation and requested analytics features."""
    if not os.path.isfile(request.config_path):
        raise HTTPException(status_code=404, detail=f"Config not found: {request.config_path}")

    try:
        new_engine = cityflow.Engine(request.config_path)
        enabled = set(request.enable_features or [])
        new_traffic_controller = (
            TrafficControlAlgorithm(request.algorithm)
            if "traffic_control" in enabled
            else None
        )
        new_congestion_analyzer = (
            CongestionStateSpace(levels=5) if "congestion" in enabled else None
        )
        sim_id = db.create_simulation(
            request.config_path,
            request.name or "Unnamed Simulation",
            algorithm=request.algorithm,
            enable_features=request.enable_features or [],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not initialize simulation: {exc}")

    sim_state.env = new_engine
    sim_state.is_running = False
    sim_state.step_count = 0
    sim_state.sim_id = sim_id
    sim_state.traffic_controller = new_traffic_controller
    sim_state.congestion_analyzer = new_congestion_analyzer
    sim_state.signal_characterizer = (
        SignalCharacterization() if "signal_characterization" in enabled else None
    )
    sim_state.v2x_comm = V2XCommunication() if "v2x" in enabled else None
    sim_state.v2v_comm = V2VCommunication() if "v2v" in enabled else None

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


@app.post("/api/simulation/start")
async def start_simulation():
    """Mark the simulation as running."""
    sim_id = require_initialized_simulation()
    sim_state.is_running = True
    db.simulations[sim_id]["status"] = "running"
    return {"status": "success", "message": "Simulation started", "data": {"step": sim_state.step_count}}


@app.post("/api/simulation/stop")
async def stop_simulation():
    """Pause the simulation."""
    sim_id = require_initialized_simulation()
    sim_state.is_running = False
    db.simulations[sim_id]["status"] = "paused"
    return {"status": "success", "message": "Simulation paused", "data": {"step": sim_state.step_count}}


@app.post("/api/simulation/step")
async def simulation_step(request: Optional[StepRequest] = None):
    """Advance the engine and record its observed aggregate metrics."""
    sim_id = require_initialized_simulation()
    steps = request.steps if request else 1

    try:
        for _ in range(steps):
            sim_state.env.next_step()
            sim_state.step_count += 1
            vehicle_count = sim_state.env.get_vehicle_count()
            avg_speed = sim_state.env.get_average_speed() if vehicle_count else 0.0
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

            db.record_metric(
                sim_id,
                {
                    "step": sim_state.step_count,
                    "total_vehicles": vehicle_count,
                    "avg_speed": float(avg_speed),
                    "max_speed": float(avg_speed * 1.2) if avg_speed > 0 else 0.0,
                    "congestion_level": congestion,
                    "vehicles_waiting": max(0, vehicle_count // 3),
                    "total_travel_distance": sim_state.step_count * avg_speed,
                    "algorithm_metrics": get_feature_metrics(sim_state.traffic_controller),
                    "congestion_metrics": get_feature_metrics(sim_state.congestion_analyzer),
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation step failed: {exc}")

    return {
        "status": "success",
        "message": f"Executed {steps} simulation step(s)",
        "data": db.metrics[sim_id][-1],
    }


@app.get("/api/simulation/state", response_model=SimulationStateResponse)
async def get_simulation_state():
    """Get current simulation aggregates and visualization placeholders."""
    sim_id = require_initialized_simulation()
    vehicle_count = sim_state.env.get_vehicle_count()
    avg_speed = sim_state.env.get_average_speed() if vehicle_count else 0.0
    congestion = safe_congestion(vehicle_count)

    # CityFlow's public engine interface does not expose map coordinates for
    # each vehicle here; these remain visualization placeholders, not measured
    # real-world GPS positions.
    vehicles = [
        VehicleState(
            vehicle_id=f"v_{index}",
            x=float(index * 10 % 100),
            y=float(index * 5 % 100),
            speed=float(avg_speed),
            direction=float(index % 4 * 90),
            road_id=f"road_{index % 5}",
        )
        for index in range(min(vehicle_count, 50))
    ]
    intersections = [
        IntersectionSignalState(
            intersection_id=f"int_{index}",
            phase=index % 2,
            duration=30,
            remaining_time=15,
            vehicles_ns=vehicle_count // 2,
            vehicles_ew=vehicle_count // 2,
            congestion_level=congestion,
        )
        for index in range(3)
    ]
    latest = db.metrics[sim_id][-1] if db.metrics.get(sim_id) else None

    return SimulationStateResponse(
        simulation_id=sim_id,
        is_running=sim_state.is_running,
        step_count=sim_state.step_count,
        total_vehicles=vehicle_count,
        avg_speed=float(avg_speed),
        congestion_level=congestion,
        algorithm=db.simulations[sim_id].get("algorithm") or "unknown",
        metrics=latest,
        vehicles=vehicles,
        intersections=intersections,
    )


@app.get("/api/simulation/metrics")
async def get_metrics(limit: int = 100):
    """Get recent metrics for the active simulation."""
    sim_id = require_initialized_simulation()
    if limit < 1 or limit > 10000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 10000")
    return {"simulation_id": sim_id, "metrics": db.get_metrics(sim_id, limit)}


@app.get("/api/simulation/vehicles")
async def get_vehicles():
    """Return the active engine's vehicle count and aggregate speed."""
    sim_id = require_initialized_simulation()
    vehicle_count = sim_state.env.get_vehicle_count()
    avg_speed = sim_state.env.get_average_speed() if vehicle_count else 0.0
    return {
        "simulation_id": sim_id,
        "step": sim_state.step_count,
        "total_vehicles": vehicle_count,
        "average_speed": float(avg_speed),
        "note": "Per-vehicle map coordinates are not available through this endpoint.",
        "vehicles": [],
    }


@app.get("/api/simulation/intersections")
async def get_intersections():
    """Return map-level intersection placeholders and current aggregate load."""
    sim_id = require_initialized_simulation()
    vehicle_count = sim_state.env.get_vehicle_count()
    congestion = safe_congestion(vehicle_count)
    intersections = [
        {
            "intersection_id": f"int_{index}",
            "phase": index % 2,
            "phase_name": "NS" if index % 2 == 0 else "EW",
            "duration": 30,
            "remaining_time": 15,
            "vehicles_ns": vehicle_count // 2,
            "vehicles_ew": vehicle_count // 2,
            "congestion_level": congestion,
            "signal_state": "GREEN" if index % 2 == 0 else "RED",
        }
        for index in range(5)
    ]
    return {
        "simulation_id": sim_id,
        "step": sim_state.step_count,
        "total_intersections": len(intersections),
        "intersections": intersections,
        "note": "Intersection values are aggregate placeholders, not per-junction detector counts.",
    }


@app.post("/api/simulation/scenarios/analyze")
async def analyze_traffic_scenarios(request: MultifactorAnalysisRequest):
    """Analyze observed counts across map scenarios using a four-factor OLS model.

    Submit measured vehicle counts (for example, counts extracted from a real
    detector or replay log) and consistently encoded demand, weather, incident,
    and time-of-day factors. This analyzes supplied observations; it does not
    claim that the example's generated CityFlow grid is a surveyed real-world map.
    """
    try:
        results = [analyze_scenario(scenario) for scenario in request.scenarios]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    active_simulation = db.simulations.get(sim_state.sim_id, {}) if sim_state.sim_id else {}
    return {
        "analysis": "multifactor_ordinary_least_squares",
        "target": "observed vehicle_count",
        "factors": FACTOR_NAMES,
        "simulation_map_config": active_simulation.get("config_path"),
        "results": results,
        "interpretation": (
            "R-squared is in-sample fit; AIC/BIC are comparable only for scenarios "
            "using the same target definition and observation process. Correlation "
            "does not establish causal effects."
        ),
    }


@app.post("/api/simulation/reset")
async def reset_simulation():
    """Reset the active simulation and remove its in-memory records."""
    sim_id = sim_state.sim_id
    sim_state.env = None
    sim_state.is_running = False
    sim_state.step_count = 0
    sim_state.sim_id = None
    sim_state.traffic_controller = None
    sim_state.congestion_analyzer = None
    sim_state.signal_characterizer = None
    sim_state.v2x_comm = None
    sim_state.v2v_comm = None

    if sim_id is not None:
        db.metrics.pop(sim_id, None)
        db.vehicles.pop(sim_id, None)
        db.intersections.pop(sim_id, None)
        db.simulations.pop(sim_id, None)

    return {"status": "success", "message": "Simulation reset successfully"}


@app.on_event("startup")
async def startup_event():
    print("CityFlow Simulation API started")
    print("In-memory database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    sim_state.env = None
    print("CityFlow Simulation API stopped")


if __name__ == "__main__":
    import uvicorn

    print("Starting CityFlow API at http://0.0.0.0:8000")
    print("API documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
