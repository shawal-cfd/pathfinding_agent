"""
Dynamic Pathfinding Agent - Pygame GUI
Visualization, metrics dashboard, dynamic obstacles, and re-planning.
"""

import pygame
import random
import sys
from pathfinding import (
    GridEnvironment,
    Heuristic,
    a_star,
    greedy_best_first,
    get_blocked_path_index,
)

# Colors
BACKGROUND = (30, 30, 40)
GRID_LINE = (60, 60, 70)
EMPTY = (45, 45, 55)
WALL = (80, 80, 90)
START = (46, 204, 113)   # Green
GOAL = (231, 76, 60)     # Red
FRONTIER = (241, 196, 15)   # Yellow
VISITED = (52, 152, 219)    # Blue
PATH = (39, 174, 96)       # Bright green
TEXT = (236, 240, 241)
TEXT_DIM = (149, 165, 166)
PANEL_BG = (40, 40, 52)
BUTTON_BG = (52, 73, 94)
BUTTON_HOVER = (72, 93, 114)
BUTTON_ACTIVE = (39, 174, 96)


class PathfindingApp:
    def __init__(self):
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Grid config
        self.cell_size = 24
        self.margin = 2
        self.panel_width = 300
        self.grid_offset_x = 20
        self.grid_offset_y = 120
        self.obstacle_density = 0.3  # 30% for random map

        # Environment
        self.rows = 20
        self.cols = 30
        self.env = GridEnvironment(self.rows, self.cols)
        self.env.start = (0, 0)
        self.env.goal = (self.rows - 1, self.cols - 1)

        # State
        self.algorithm = "A*"
        self.heuristic = Heuristic.MANHATTAN
        self.dynamic_mode = False
        self.spawn_probability = 0.02
        self.agent_position = None
        self.current_path = []
        self.path_cells = set()
        self.path_index = 0
        self.is_running = False
        self.is_paused = False
        self.editing = True
        self.dragging_start = False
        self.dragging_goal = False

        # Visualization
        self.visited_set = set()
        self.frontier_set = set()
        self.animate_search = False
        self.search_frames = []
        self.frame_index = 0

        # Metrics
        self.nodes_visited = 0
        self.path_cost = 0
        self.execution_time_ms = 0.0
        self.last_plan_time = 0.0
        self.replans = 0

        # Fonts
        self.font_large = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

    def set_current_path(self, path: list | None):
        """
        Store the current planned path and precompute drawable cells.
        We keep Start/Goal markers distinct, so exclude them from the path fill.
        Also avoid rendering any cells that are currently obstacles.
        """
        if not path:
            self.current_path = []
            self.path_cells = set()
            return
        self.current_path = path
        drawable = []
        for pos in path[1:-1]:
            if not self.env.is_obstacle(pos[0], pos[1]):
                drawable.append(pos)
        self.path_cells = set(drawable)

    def get_cell_rect(self, row: int, col: int):
        x = self.grid_offset_x + col * (self.cell_size + self.margin)
        y = self.grid_offset_y + row * (self.cell_size + self.margin)
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def grid_to_cell(self, screen_x: int, screen_y: int) -> tuple | None:
        x = screen_x - self.grid_offset_x
        y = screen_y - self.grid_offset_y
        if x < 0 or y < 0:
            return None
        col = x // (self.cell_size + self.margin)
        row = y // (self.cell_size + self.margin)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    def draw_cell(self, surface, row: int, col: int, color: tuple, border=True):
        rect = self.get_cell_rect(row, col)
        pygame.draw.rect(surface, color, rect)
        if border:
            pygame.draw.rect(surface, GRID_LINE, rect, 1)

    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                if self.env.is_obstacle(r, c):
                    color = WALL
                elif pos == self.env.start:
                    color = START
                elif pos == self.env.goal:
                    color = GOAL
                elif pos == self.agent_position:
                    color = (255, 255, 100)
                elif pos in self.path_cells:
                    color = PATH
                elif pos in self.visited_set:
                    color = VISITED
                elif pos in self.frontier_set:
                    color = FRONTIER
                else:
                    color = EMPTY
                # Render final path as solid green boxes.
                border = not (pos in self.path_cells)
                self.draw_cell(self.screen, r, c, color, border=border)

    def draw_panel(self):
        panel = pygame.Rect(self.screen_width - self.panel_width, 0, self.panel_width, self.screen_height)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        pygame.draw.line(self.screen, GRID_LINE, (self.screen_width - self.panel_width, 0),
                         (self.screen_width - self.panel_width, self.screen_height), 2)

        y = 20
        self.draw_text("Pathfinding Agent", self.font_large, TEXT, self.screen_width - self.panel_width + 15, y)
        y += 45

        # Metrics
        self.draw_text("Metrics", self.font_large, TEXT, self.screen_width - self.panel_width + 15, y)
        y += 28
        self.draw_text(f"Nodes Visited: {self.nodes_visited}", self.font_small, TEXT_DIM,
                       self.screen_width - self.panel_width + 15, y)
        y += 24
        self.draw_text(f"Path Cost: {self.path_cost}", self.font_small, TEXT_DIM,
                       self.screen_width - self.panel_width + 15, y)
        y += 24
        self.draw_text(f"Time: {self.execution_time_ms:.2f} ms", self.font_small, TEXT_DIM,
                       self.screen_width - self.panel_width + 15, y)
        y += 24
        if self.dynamic_mode:
            self.draw_text(f"Re-plans: {self.replans}", self.font_small, TEXT_DIM,
                           self.screen_width - self.panel_width + 15, y)
        y += 40

        # Algorithm
        self.draw_text("Algorithm", self.font_large, TEXT, self.screen_width - self.panel_width + 15, y)
        y += 28
        btn_a = self.draw_button("A*", self.screen_width - self.panel_width + 15, y, 75, 28,
                                  self.algorithm == "A*")
        btn_g = self.draw_button("GBFS", self.screen_width - self.panel_width + 95, y, 75, 28,
                                  self.algorithm == "GBFS")
        y += 40

        # Heuristic
        self.draw_text("Heuristic", self.font_large, TEXT, self.screen_width - self.panel_width + 15, y)
        y += 28
        btn_m = self.draw_button("Manhattan", self.screen_width - self.panel_width + 15, y, 80, 28,
                                  self.heuristic == Heuristic.MANHATTAN)
        y += 32
        btn_e = self.draw_button("Euclidean", self.screen_width - self.panel_width + 15, y, 80, 28,
                                  self.heuristic == Heuristic.EUCLIDEAN)
        y += 32
        btn_c = self.draw_button("Chebyshev", self.screen_width - self.panel_width + 15, y, 80, 28,
                                  self.heuristic == Heuristic.CHEBYSHEV)
        y += 45

        # Actions
        self.draw_text("Actions", self.font_large, TEXT, self.screen_width - self.panel_width + 15, y)
        y += 28
        btn_run = self.draw_button("Run Search", self.screen_width - self.panel_width + 15, y, 115, 28,
                                    self.is_running and not self.dynamic_mode)
        y += 32
        btn_dyn = self.draw_button("Dynamic Mode", self.screen_width - self.panel_width + 15, y, 115, 28,
                                   self.dynamic_mode)
        y += 32
        btn_rand = self.draw_button("Random Map", self.screen_width - self.panel_width + 15, y, 115, 28, False)
        y += 32
        btn_clear = self.draw_button("Clear & Reset", self.screen_width - self.panel_width + 15, y, 115, 28, False)
        y += 45

        # Grid size
        self.draw_text("Grid Size", self.font_large, TEXT, self.screen_width - self.panel_width + 15, y)
        y += 28
        self.draw_text(f"Rows: {self.rows}", self.font_small, TEXT_DIM, self.screen_width - self.panel_width + 15, y)
        btn_r_plus = self.draw_button("+", self.screen_width - self.panel_width + 120, y - 2, 28, 24, False)
        btn_r_minus = self.draw_button("-", self.screen_width - self.panel_width + 150, y - 2, 28, 24, False)
        y += 28
        self.draw_text(f"Cols: {self.cols}", self.font_small, TEXT_DIM, self.screen_width - self.panel_width + 15, y)
        btn_c_plus = self.draw_button("+", self.screen_width - self.panel_width + 120, y - 2, 28, 24, False)
        btn_c_minus = self.draw_button("-", self.screen_width - self.panel_width + 150, y - 2, 28, 24, False)
        y += 35

        # Obstacle density for random map
        self.draw_text("Obstacle Density", self.font_large, TEXT, self.screen_width - self.panel_width + 15, y)
        y += 28
        self.draw_text(f"{int(self.obstacle_density * 100)}%", self.font_small, TEXT_DIM,
                       self.screen_width - self.panel_width + 15, y)
        btn_d_plus = self.draw_button("+", self.screen_width - self.panel_width + 90, y - 2, 28, 24, False)
        btn_d_minus = self.draw_button("-", self.screen_width - self.panel_width + 120, y - 2, 28, 24, False)

        return {
            "a_star": btn_a, "gbfs": btn_g,
            "manhattan": btn_m, "euclidean": btn_e, "chebyshev": btn_c,
            "run": btn_run, "dynamic": btn_dyn, "random": btn_rand, "clear": btn_clear,
            "rows_plus": btn_r_plus, "rows_minus": btn_r_minus,
            "cols_plus": btn_c_plus, "cols_minus": btn_c_minus,
            "density_plus": btn_d_plus, "density_minus": btn_d_minus,
        }

    def draw_button(self, text: str, x: int, y: int, w: int, h: int, active: bool) -> pygame.Rect:
        rect = pygame.Rect(x, y, w, h)
        color = BUTTON_ACTIVE if active else BUTTON_BG
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, GRID_LINE, rect, 1)
        surf = self.font_small.render(text, True, TEXT)
        tw, th = surf.get_size()
        self.screen.blit(surf, (x + (w - tw) // 2, y + (h - th) // 2))
        return rect

    def draw_text(self, text: str, font, color, x: int, y: int):
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def run_search(self, animate: bool = False):
        self.visited_set.clear()
        self.frontier_set.clear()
        self.set_current_path(None)
        self.path_index = 0
        self.agent_position = None
        self.nodes_visited = 0
        self.path_cost = 0
        self.execution_time_ms = 0.0

        def on_visited(pos):
            self.visited_set.add(pos)
            self.frontier_set.discard(pos)

        def on_frontier(pos):
            if pos not in self.visited_set:
                self.frontier_set.add(pos)

        if self.algorithm == "A*":
            path, nv, et = a_star(
                self.env, self.heuristic,
                callback_visited=on_visited, callback_frontier=on_frontier
            )
        else:
            path, nv, et = greedy_best_first(
                self.env, self.heuristic,
                callback_visited=on_visited, callback_frontier=on_frontier
            )

        self.nodes_visited = nv
        self.execution_time_ms = et
        self.last_plan_time = et
        if path:
            self.set_current_path(path)
            self.path_cost = len(path) - 1
            self.path_index = 0
            self.agent_position = self.env.start
        return path is not None

    def run_search_sync(self):
        """Run search without animation callbacks for speed."""
        self.visited_set.clear()
        self.frontier_set.clear()
        self.set_current_path(None)
        self.path_index = 0
        self.agent_position = None
        self.nodes_visited = 0
        self.path_cost = 0
        self.execution_time_ms = 0.0

        if self.algorithm == "A*":
            path, nv, et = a_star(self.env, self.heuristic)
        else:
            path, nv, et = greedy_best_first(self.env, self.heuristic)

        self.nodes_visited = nv
        self.execution_time_ms = et
        self.last_plan_time = et
        if path:
            self.set_current_path(path)
            self.path_cost = len(path) - 1
            self.path_index = 0
            self.agent_position = self.env.start
            for p in path:
                self.visited_set.add(p)
        return path is not None

    def replan_from_current(self) -> bool:
        """Re-plan from agent's current position. Returns True if new path found."""
        if self.agent_position is None:
            self.agent_position = self.env.start
        old_start = self.env.start
        self.env.start = self.agent_position
        self.replans += 1

        if self.algorithm == "A*":
            path, nv, et = a_star(self.env, self.heuristic)
        else:
            path, nv, et = greedy_best_first(self.env, self.heuristic)
        self.env.start = old_start

        self.nodes_visited += nv
        self.execution_time_ms += et
        self.last_plan_time = et

        if path:
            self.set_current_path(path)
            self.path_cost = len(path) - 1
            self.path_index = 0
            return True
        self.set_current_path(None)
        return False

    def spawn_random_obstacle(self):
        """Spawn one random obstacle (not on start/goal/agent)."""
        attempts = 0
        while attempts < 50:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            pos = (r, c)
            if pos != self.env.start and pos != self.env.goal and pos != self.agent_position:
                if not self.env.is_obstacle(r, c):
                    self.env.set_obstacle(r, c, True)
                    return
            attempts += 1

    def handle_click(self, pos: tuple):
        if pos[0] >= self.screen_width - self.panel_width:
            return
        cell = self.grid_to_cell(pos[0], pos[1])
        if cell is None:
            return
        r, c = cell
        if self.dragging_start:
            if self.env.set_start(r, c):
                self.dragging_start = False
            return
        if self.dragging_goal:
            if self.env.set_goal(r, c):
                self.dragging_goal = False
            return
        if self.editing and not self.is_running:
            self.env.toggle_obstacle(r, c)

    def handle_mouse_down(self, pos: tuple):
        if pos[0] >= self.screen_width - self.panel_width:
            return
        cell = self.grid_to_cell(pos[0], pos[1])
        if cell is None:
            return
        r, c = cell
        if cell == self.env.start:
            self.dragging_start = True
        elif cell == self.env.goal:
            self.dragging_goal = True

    def handle_button_click(self, mouse_pos: tuple, buttons: dict):
        for name, rect in buttons.items():
            if rect.collidepoint(mouse_pos):
                if name == "a_star":
                    self.algorithm = "A*"
                elif name == "gbfs":
                    self.algorithm = "GBFS"
                elif name == "manhattan":
                    self.heuristic = Heuristic.MANHATTAN
                elif name == "euclidean":
                    self.heuristic = Heuristic.EUCLIDEAN
                elif name == "chebyshev":
                    self.heuristic = Heuristic.CHEBYSHEV
                elif name == "run" and not self.dynamic_mode:
                    self.is_running = True
                    self.run_search(animate=True)
                    self.is_running = False
                elif name == "dynamic":
                    self.dynamic_mode = not self.dynamic_mode
                    if self.dynamic_mode:
                        self.is_running = True
                        self.replans = 0
                        self.run_search_sync()
                    else:
                        self.is_running = False
                        self.agent_position = None
                        self.set_current_path(None)
                elif name == "random":
                    self.env.generate_random_map(self.obstacle_density)
                    self.visited_set.clear()
                    self.frontier_set.clear()
                    self.set_current_path(None)
                    self.agent_position = None
                elif name == "rows_plus":
                    if self.rows < 50 and not self.is_running:
                        self.rows += 1
                        self.env.resize(self.rows, self.cols)
                        self.env.start = (0, 0)
                        self.env.goal = (self.rows - 1, self.cols - 1)
                elif name == "rows_minus":
                    if self.rows > 5 and not self.is_running:
                        self.rows -= 1
                        self.env.resize(self.rows, self.cols)
                        self.env.start = (0, 0)
                        self.env.goal = (self.rows - 1, self.cols - 1)
                elif name == "cols_plus":
                    if self.cols < 60 and not self.is_running:
                        self.cols += 1
                        self.env.resize(self.rows, self.cols)
                        self.env.start = (0, 0)
                        self.env.goal = (self.rows - 1, self.cols - 1)
                elif name == "cols_minus":
                    if self.cols > 5 and not self.is_running:
                        self.cols -= 1
                        self.env.resize(self.rows, self.cols)
                        self.env.start = (0, 0)
                        self.env.goal = (self.rows - 1, self.cols - 1)
                elif name == "density_plus":
                    self.obstacle_density = min(0.9, self.obstacle_density + 0.1)
                elif name == "density_minus":
                    self.obstacle_density = max(0.1, self.obstacle_density - 0.1)
                elif name == "clear":
                    self.env = GridEnvironment(self.rows, self.cols)
                    self.env.start = (0, 0)
                    self.env.goal = (self.rows - 1, self.cols - 1)
                    self.visited_set.clear()
                    self.frontier_set.clear()
                    self.set_current_path(None)
                    self.agent_position = None
                    self.is_running = False
                    self.dynamic_mode = False
                    self.nodes_visited = 0
                    self.path_cost = 0
                    self.execution_time_ms = 0.0
                    self.replans = 0
                return

    def update_dynamic(self, dt_ms: int, move_interval_ms: int, move_timer_ms: int) -> int:
        """
        Dynamic mode update.
        - Obstacles may spawn each frame (rate scaled to dt).
        - If any obstacle blocks the planned path, immediately clear the green path and re-plan.
        - Agent movement is rate-limited by move_interval_ms.
        Returns updated move_timer_ms.
        """
        if not self.dynamic_mode or not self.is_running:
            return 0

        # Keep agent position initialized for rendering and replanning.
        if self.agent_position is None:
            self.agent_position = self.env.start

        # Spawn obstacles with probability scaled to frame time to keep behavior consistent.
        if move_interval_ms <= 0:
            move_interval_ms = 1
        p_frame = 1.0 - (1.0 - self.spawn_probability) ** (dt_ms / move_interval_ms)
        if random.random() < p_frame:
            self.spawn_random_obstacle()

        # If current path is blocked, clear it immediately and re-plan.
        if self.current_path:
            old_path = self.current_path
            blocked = get_blocked_path_index(old_path, self.env)
            if blocked >= 0:
                # Remove old green boxes right away.
                self.set_current_path(None)
                # Place agent just before the blocked cell (if possible) and re-plan.
                if blocked > 0:
                    self.agent_position = old_path[max(0, blocked - 1)]
                if self.env.is_obstacle(self.agent_position[0], self.agent_position[1]):
                    self.agent_position = self.env.start
                success = self.replan_from_current()
                if not success:
                    self.is_running = False
                    return 0

        # Move agent along the (possibly new) path at a fixed interval.
        move_timer_ms += dt_ms
        if self.current_path and move_timer_ms >= move_interval_ms:
            move_timer_ms = 0
            if self.path_index < len(self.current_path) - 1:
                next_pos = self.current_path[self.path_index + 1]
                if not self.env.is_obstacle(next_pos[0], next_pos[1]):
                    self.path_index += 1
                    self.agent_position = self.current_path[self.path_index]
                else:
                    success = self.replan_from_current()
                    if not success:
                        self.is_running = False
                        return 0
        return move_timer_ms

    def run(self):
        running = True
        step_interval = 80  # ms between agent steps in dynamic mode
        move_timer = 0

        while running:
            dt = self.clock.tick(self.fps)
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if mouse_pos[0] < self.screen_width - self.panel_width:
                            self.handle_mouse_down(mouse_pos)
                            self.handle_click(mouse_pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_start = False
                        self.dragging_goal = False
                        buttons = self.draw_panel()
                        self.handle_button_click(event.pos, buttons)
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_width = event.w
                    self.screen_height = event.h
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)

            # Dynamic mode update (checks every frame; movement is rate-limited internally)
            move_timer = self.update_dynamic(dt, step_interval, move_timer)

            self.screen.fill(BACKGROUND)
            self.draw_grid()
            buttons = self.draw_panel()

            # Instructions
            inst = "Click cells to add/remove walls | Drag Start/Goal to move"
            if self.dynamic_mode:
                inst = "Dynamic Mode: Obstacles spawn randomly. Agent re-plans when blocked."
            self.draw_text(inst, self.font_small, TEXT_DIM, self.grid_offset_x,
                           self.grid_offset_y - 25)

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = PathfindingApp()
    app.run()
