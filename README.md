# Dynamic Pathfinding Agent

A Python application implementing informed search algorithms (GBFS and A*) for grid-based navigation with dynamic obstacle spawning and real-time re-planning.

## Features

- **Dynamic Grid Sizing**: Adjust rows and columns via +/- buttons
- **Fixed Start & Goal**: Drag to reposition; always clearly identified
- **Random Map Generation**: Generate mazes with configurable obstacle density (10–90%)
- **Interactive Map Editor**: Click cells to add/remove walls
- **Search Algorithms**: Greedy Best-First Search (GBFS) and A*
- **Heuristics**: Manhattan (L1), Euclidean (L2), Chebyshev (L∞)
- **Dynamic Mode**: Obstacles spawn randomly during agent movement; automatic re-planning when path is blocked
- **Visualization**: Frontier (yellow), Visited (blue), Path (green)
- **Metrics**: Nodes visited, path cost, execution time (ms), re-plans (in dynamic mode)

## Installation

```bash
pip install pygame
```

## Usage

```bash
python main.py
```

### Controls

- **Click** on empty cells to add walls; click walls to remove them
- **Drag** Start (green) or Goal (red) to reposition
- **Run Search**: Compute path with selected algorithm and heuristic
- **Dynamic Mode**: Run with random obstacle spawning; agent re-plans when blocked
- **Random Map**: Generate a new maze with current obstacle density
- **Clear & Reset**: Clear all obstacles and reset state

### Algorithm Selection

- **A***: f(n) = g(n) + h(n) — optimal with admissible heuristic
- **GBFS**: f(n) = h(n) — fast but not always optimal

### Heuristic Selection

- **Manhattan**: |Δx| + |Δy| — 4-connected grids
- **Euclidean**: √((Δx)² + (Δy)²) — diagonal-aware
- **Chebyshev**: max(|Δx|, |Δy|) — 8-connected approximation

## Project Structure

```
pathfinding_agent/
├── main.py          # Pygame GUI and application logic
├── pathfinding.py   # Grid environment, GBFS, A*, heuristics
├── requirements.txt
└── README.md
```
