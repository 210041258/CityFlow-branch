from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cityflow
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

# FastAPI app initialization
app = FastAPI(
    title="CityFlow Simulation API",
    description="API for traffic flow simulation with browser-based visualization",
    version="1.0.0",
)

# CORS middleware for browser frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    config_path: str
    name: Optional[str] = None


class SimulationState:
    def __init__(self):
        self.env = None
        self.is_running = False
        self.step_count = 0
        self.metrics = {}


simulation_state = SimulationState()


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "CityFlow Simulation API",
        "version": "1.0.0",
    }


@app.post("/api/simulation/init")
async def initialize_simulation(payload: SimulationRequest):
    """Initialize a CityFlow simulation from a config file."""
    try:
        if not os.path.exists(payload.config_path):
            raise HTTPException(status_code=404, detail="Config file not found")

        simulation_state.env = cityflow.Engine(payload.config_path)
        simulation_state.is_running = False
        simulation_state.step_count = 0
        simulation_state.metrics = {}

        return {
            "status": "success",
            "message": "Simulation initialized successfully",
            "data": {
                "config_path": payload.config_path,
                "name": payload.name or "unnamed",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/simulation/start")
async def start_simulation():
    """Start or resume the simulation."""
    if simulation_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    simulation_state.is_running = True
    return {
        "status": "success",
        "message": "Simulation started",
        "data": {"step": simulation_state.step_count},
    }


@app.post("/api/simulation/stop")
async def stop_simulation():
    """Pause the simulation."""
    simulation_state.is_running = False
    return {
        "status": "success",
        "message": "Simulation stopped",
        "data": {"step": simulation_state.step_count},
    }


@app.post("/api/simulation/step")
async def simulation_step():
    """Run a single simulation step."""
    if simulation_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    simulation_state.env.step()
    simulation_state.step_count += 1

    vehicles = simulation_state.env.get_vehicle_count()
    simulation_state.metrics = {
        "vehicles": vehicles,
        "avg_speed": simulation_state.env.get_average_speed() if vehicles > 0 else 0,
    }

    return {
        "status": "success",
        "message": "Simulation step completed",
        "data": {
            "step": simulation_state.step_count,
            "vehicles": vehicles,
            "metrics": simulation_state.metrics,
        },
    }


@app.get("/api/simulation/state")
async def get_simulation_state():
    """Return the current state for frontend polling."""
    if simulation_state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    return {
        "is_running": simulation_state.is_running,
        "step_count": simulation_state.step_count,
        "vehicles": simulation_state.env.get_vehicle_count(),
        "metrics": simulation_state.metrics,
    }


@app.post("/api/simulation/reset")
async def reset_simulation():
    """Reset simulation state."""
    simulation_state.env = None
    simulation_state.is_running = False
    simulation_state.step_count = 0
    simulation_state.metrics = {}
    return {
        "status": "success",
        "message": "Simulation reset successfully",
    }


# Optional browser frontend integration
# If a frontend is served from a folder such as 'frontend/dist', mount it like:
# app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
# @app.get("/{full_path:path}")
# async def serve_frontend(full_path: str):
#     return FileResponse("frontend/dist/index.html")


@app.get("/dashboard")
async def dashboard():
    return {
        "message": "Dashboard endpoint ready for browser frontend integration",
        "api_base": "/api",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
