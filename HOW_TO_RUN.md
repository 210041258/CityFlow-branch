# How to Run CityFlow

CityFlow is a Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Simulation. This guide provides step-by-step instructions to get it running on your system.

## Quick Start

### Option 1: Docker (Easiest)
```bash
# Pull the Docker image
docker pull cityflowproject/cityflow:latest

# Run the container
docker run -it cityflowproject/cityflow:latest

# Inside the container, test the installation
python3 -c "import cityflow; print('CityFlow installed successfully!')"
```

### Option 2: Build from Source

#### Prerequisites
- **Python 3.5+** (Python 3.6+ recommended)
- **C++ compiler** (g++/clang)
- **CMake 3.0+** (CMake 3.1+ on Windows)
- **Git** (for cloning and submodules)

#### Installation Steps

**1. Install system dependencies (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install -y build-essential cmake git python3-dev
```

**2. Clone the repository**
```bash
git clone https://github.com/210041258/CityFlow-branch.git
cd CityFlow-branch
```

**3. Install CityFlow**
```bash
pip install .
```

This command will:
- Run CMake to configure the build
- Compile the C++ extension module
- Install the Python package

**4. Verify installation**
```bash
python3 -c "import cityflow; print('CityFlow installed successfully!')"
```

---

## Running Your First Simulation

### 1. Frontend Tool (Visualization)

CityFlow includes a web-based visualization tool to replay traffic scenarios:

```bash
# Navigate to the frontend directory
cd frontend

# Download example replay files
python3 download_replay.py

# Open the visualization
# Simply open index.html in your browser to view the replay
```

Then open `frontend/index.html` in a web browser to visualize the traffic simulation.

For detailed instructions, see the [Replay Documentation](https://cityflow.readthedocs.io/en/latest/replay.html)

### 2. Python API

Create a Python script to run a simulation:

```python
import cityflow

# Create a traffic engine
# Replace 'path/to/config.json' with your config file
eng = cityflow.Engine("path/to/config.json", 
                      thread_num=4,           # Number of threads
                      verbose=False)          # Verbose output

# Run simulation
for step in range(3600):  # 3600 timesteps (1 hour at 1 step/second)
    eng.next_step()

eng.close()
```

### 3. Running Tests

If you have Google Test (GTest) installed:

```bash
# GTest is optional; install it if needed
git clone https://github.com/google/googletest.git
cd googletest && mkdir build && cd build
cmake ..
sudo make install

# Build and run tests
cd /path/to/CityFlow-branch
mkdir build && cd build
cmake ..
cmake --build . --target test
ctest
```

---

## Project Structure

```
CityFlow-branch/
├── src/              # C++ source code (simulation engine)
│   ├── engine/       # Core engine
│   ├── roadnet/      # Road network representation
│   ├── vehicle/      # Vehicle simulation
│   └── flow/         # Traffic flow
├── frontend/         # JavaScript/HTML visualization tool
├── tests/            # C++ unit tests
├── docs/             # Documentation
├── setup.py          # Python package setup
├── CMakeLists.txt    # CMake build configuration
└── Dockerfile        # Docker configuration
```

---

## Language Composition

- **Python (52.5%)** — High-level API, experimentation, tools, scripting
- **C++ (39.4%)** — High-performance simulation engine
- **JavaScript/HTML (6%)** — Visualization and replay tool
- **CMake/Other (0.8%)** — Build configuration

---

## Common Issues & Troubleshooting

### Issue: CMake not found
**Solution:**
```bash
sudo apt install cmake
```

### Issue: Python development headers missing
**Solution:**
```bash
sudo apt install python3-dev
```

### Issue: Build fails with "pybind11 not found"
**Solution:** CMake automatically downloads pybind11 via git submodule. Ensure:
```bash
git submodule update --init --recursive
```

### Issue: Permission denied on Windows
**Solution:** Use Windows Subsystem for Linux (WSL) or Docker instead. Native Windows builds are not officially supported.

---

## Documentation

For detailed documentation, visit:
- **Installation Guide:** [CityFlow Docs - Install](https://cityflow.readthedocs.io/en/latest/install.html)
- **Replay Tool:** [CityFlow Docs - Replay](https://cityflow.readthedocs.io/en/latest/replay.html)
- **API Reference:** https://cityflow.readthedocs.io/

---

## What to Run Next

1. **Download example scenarios** using `frontend/download_replay.py`
2. **Visualize traffic** by opening `frontend/index.html`
3. **Write a simulation script** using the Python API
4. **Explore the documentation** for advanced features (RL, custom configs, etc.)

---

## Environment Details

- **Official Project:** [CityFlow Project](https://github.com/cityflow-project/CityFlow)
- **Branch:** CityFlow-branch (user fork)
- **Tested on:** Ubuntu 16.04+, Python 3.6+, CMake 3.0+
