"""
Dynamic Pathfinding Agent - Core Algorithms and Environment
Implements GBFS, A* with Manhattan, Euclidean, and Chebyshev heuristics.
"""

import heapq
import time
from enum import Enum
from typing import Callable, List, Optional, Tuple


class Heuristic(Enum):
    """Available heuristic functions for informed search."""
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    CHEBYSHEV = "chebyshev"


def manhattan_distance(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """L1 norm: |x1-x2| + |y1-y2|"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def euclidean_distance(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """L2 norm: sqrt((x1-x2)^2 + (y1-y2)^2)"""
    return ((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) ** 0.5


def chebyshev_distance(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """L∞ norm: max(|x1-x2|, |y1-y2|)"""
    return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))


HEURISTIC_FUNCTIONS = {
    Heuristic.MANHATTAN: manhattan_distance,
    Heuristic.EUCLIDEAN: euclidean_distance,
    Heuristic.CHEBYSHEV: chebyshev_distance,
}


class GridEnvironment:
    """Grid-based environment with dynamic obstacle support."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid: List[List[int]] = [[0] * cols for _ in range(rows)]
        self.start: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (rows - 1, cols - 1)

    def set_start(self, row: int, col: int) -> bool:
        """Set start position. Returns False if position is blocked."""
        if self.is_valid(row, col) and not self.is_obstacle(row, col):
            self.start = (row, col)
            return True
        return False

    def set_goal(self, row: int, col: int) -> bool:
        """Set goal position. Returns False if position is blocked."""
        if self.is_valid(row, col) and not self.is_obstacle(row, col):
            self.goal = (row, col)
            return True
        return False

    def is_valid(self, row: int, col: int) -> bool:
        """Check if (row, col) is within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_obstacle(self, row: int, col: int) -> bool:
        """Check if cell is an obstacle."""
        return self.grid[row][col] == 1

    def toggle_obstacle(self, row: int, col: int) -> bool:
        """Toggle obstacle at position. Returns True if toggled."""
        if self.is_valid(row, col):
            pos = (row, col)
            if pos != self.start and pos != self.goal:
                self.grid[row][col] = 1 - self.grid[row][col]
                return True
        return False

    def set_obstacle(self, row: int, col: int, value: bool) -> bool:
        """Set obstacle at position. value=True for wall, False for empty."""
        if self.is_valid(row, col):
            pos = (row, col)
            if pos != self.start and pos != self.goal:
                self.grid[row][col] = 1 if value else 0
                return True
        return False

    def resize(self, rows: int, cols: int) -> None:
        """Resize grid, preserving start/goal if possible."""
        self.rows = rows
        self.cols = cols
        self.grid = [[0] * cols for _ in range(rows)]
        self.start = (min(self.start[0], rows - 1), min(self.start[1], cols - 1))
        self.goal = (min(self.goal[0], rows - 1), min(self.goal[1], cols - 1))

    def generate_random_map(self, obstacle_density: float) -> None:
        """Generate random obstacles. density in [0, 1] (e.g., 0.3 = 30% walls)."""
        import random
        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                if pos != self.start and pos != self.goal:
                    self.grid[r][c] = 1 if random.random() < obstacle_density else 0

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid 4-connected neighbors (up, down, left, right)."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_valid(nr, nc) and not self.is_obstacle(nr, nc):
                neighbors.append((nr, nc))
        return neighbors


def reconstruct_path(
    came_from: dict,
    current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Reconstruct path from start to current using came_from map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(
    env: GridEnvironment,
    heuristic: Heuristic,
    callback_visited: Optional[Callable[[Tuple[int, int]], None]] = None,
    callback_frontier: Optional[Callable[[Tuple[int, int]], None]] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], int, float]:
    """
    A* search: f(n) = g(n) + h(n).
    Returns (path, nodes_visited, execution_time_ms).
    """
    h_func = HEURISTIC_FUNCTIONS[heuristic]
    start_time = time.perf_counter()
    nodes_visited = 0

    open_set = []
    g_score = {env.start: 0}
    h_val = h_func(env.start, env.goal)
    heapq.heappush(open_set, (g_score[env.start] + h_val, env.start))
    came_from: dict = {}
    in_open = {env.start}

    while open_set:
        _, current = heapq.heappop(open_set)
        in_open.discard(current)

        if current == env.goal:
            path = reconstruct_path(came_from, current)
            elapsed = (time.perf_counter() - start_time) * 1000
            return path, nodes_visited, elapsed

        nodes_visited += 1
        if callback_visited:
            callback_visited(current)

        for neighbor in env.get_neighbors(current[0], current[1]):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_val = tentative_g + h_func(neighbor, env.goal)
                heapq.heappush(open_set, (f_val, neighbor))
                in_open.add(neighbor)
                if callback_frontier:
                    callback_frontier(neighbor)

    elapsed = (time.perf_counter() - start_time) * 1000
    return None, nodes_visited, elapsed


def greedy_best_first(
    env: GridEnvironment,
    heuristic: Heuristic,
    callback_visited: Optional[Callable[[Tuple[int, int]], None]] = None,
    callback_frontier: Optional[Callable[[Tuple[int, int]], None]] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], int, float]:
    """
    Greedy Best-First Search: f(n) = h(n) only.
    Returns (path, nodes_visited, execution_time_ms).
    """
    h_func = HEURISTIC_FUNCTIONS[heuristic]
    start_time = time.perf_counter()
    nodes_visited = 0

    open_set = []
    heapq.heappush(open_set, (h_func(env.start, env.goal), env.start))
    came_from: dict = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1
        if callback_visited:
            callback_visited(current)

        if current == env.goal:
            path = reconstruct_path(came_from, current)
            elapsed = (time.perf_counter() - start_time) * 1000
            return path, nodes_visited, elapsed

        for neighbor in env.get_neighbors(current[0], current[1]):
            if neighbor not in visited:
                came_from[neighbor] = current
                h_val = h_func(neighbor, env.goal)
                heapq.heappush(open_set, (h_val, neighbor))
                if callback_frontier:
                    callback_frontier(neighbor)

    elapsed = (time.perf_counter() - start_time) * 1000
    return None, nodes_visited, elapsed


def is_obstacle_on_path(
    path: List[Tuple[int, int]],
    env: GridEnvironment
) -> bool:
    """Check if any obstacle blocks the given path."""
    for pos in path:
        if env.is_obstacle(pos[0], pos[1]):
            return True
    return False


def get_blocked_path_index(path: List[Tuple[int, int]], env: GridEnvironment) -> int:
    """Return index of first blocked cell in path, or -1 if none."""
    for i, pos in enumerate(path):
        if env.is_obstacle(pos[0], pos[1]):
            return i
    return -1
