# Mechanics Simulator

An interactive 2D physics sandbox built with [Dear PyGui](https://github.com/hoffstadt/DearPyGui). Pick a simulation, tweak its parameters live, and watch it evolve. All physics is integrated from scratch (no physics engine).

## Simulations

- Projectile motion
- Simple pendulum
- Block on an incline
- Spring–mass oscillator
- Double pendulum
- Orbital motion

## Features

- Live parameter controls with named presets
- Euler and RK4 integrators
- Pan/zoom canvas and time-series analysis plots
- Layout persisted between sessions (`layout.ini`)

## Requirements

- Python 3.10+
- `dearpygui>=1.11.0`, `numpy>=1.24.0`

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Build a standalone executable

Runs PyInstaller to produce `dist/MechanicsSimulator.exe`:

```bash
build.bat
```

## Project layout

```
core/    integrators, camera, recorder, simulation base class
sims/    individual simulations
ui/      Dear PyGui application shell
main.py  entry point
```
