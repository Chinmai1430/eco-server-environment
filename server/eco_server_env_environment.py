# server/eco_server_env_environment.py
from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ── Action ────────────────────────────────────────────────────────────────────
@dataclass
class EcoServerAction:
    action_type: str          # See ACTION_TYPES below
    x: Optional[int] = None  # Grid coordinate
    y: Optional[int] = None  # Grid coordinate

    ACTION_TYPES = [
        "plant_tree",         # Increases green cover, reduces heat
        "remove_pollution",   # Cleans a polluted cell
        "monitor",            # Observe without side effects
        "develop",            # Adds infrastructure, increases load & heat
        "install_solar",      # Adds renewable energy to a cell
        "cool_server",        # Reduces temperature of a hot server cell
        "upgrade_efficiency", # Improves server efficiency, reduces energy use
        "decommission",       # Shuts down an overloaded server cell
    ]


# ── Cell Types ────────────────────────────────────────────────────────────────
CELL_EMPTY      = 0
CELL_SERVER     = 1
CELL_TREE       = 2
CELL_POLLUTED   = 3
CELL_SOLAR      = 4
CELL_HOT_SERVER = 5  # Overheating server
CELL_EFFICIENT  = 6  # Upgraded efficient server


# ── Observation ───────────────────────────────────────────────────────────────
@dataclass
class EcoServerObservation:
    # Grid state
    grid: List[List[int]] = field(default_factory=list)

    # Ecological metrics (0.0 - 1.0)
    eco_score: float = 0.0       # Overall ecological health
    pollution: float = 0.0       # Pollution level (lower = better)
    green_cover: float = 0.0     # % of grid with trees/solar
    temperature: float = 0.0     # Average server temperature (lower = better)

    # Infrastructure metrics
    energy_usage: float = 0.0    # Total energy consumption (lower = better)
    renewable_ratio: float = 0.0 # % energy from renewables (higher = better)
    server_load: float = 0.0     # Average server load (balanced = better)
    uptime: float = 1.0          # System uptime (higher = better)

    # Episode info
    step: int = 0
    done: bool = False
    reward: float = 0.0

    # Event log
    last_event: str = ""


# ── Environment ───────────────────────────────────────────────────────────────
class EcoServerEnv:
    """
    Realistic server ecosystem management environment.

    The agent manages a W×H data center grid where each cell represents
    a physical unit (server rack, green space, solar panel, etc.).

    Goals:
    - Reduce pollution and heat
    - Increase renewable energy ratio
    - Maintain server uptime and balanced load
    - Maximize overall eco_score
    """

    def __init__(self, width: int = 15, height: int = 15, max_steps: int = 50):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.grid: List[List[int]] = []
        self._step = 0
        self._obs: Optional[EcoServerObservation] = None
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> EcoServerObservation:
        """Reset to a degraded starting state — agent must improve it."""
        self._step = 0
        self.grid = self._generate_degraded_grid()
        self._obs = self._compute_observation("Environment initialized.")
        return self._obs

    def _generate_degraded_grid(self) -> List[List[int]]:
        """
        Generate a realistic degraded data center grid:
        - Many hot/overloaded servers
        - Some polluted zones
        - Few trees and no solar panels
        """
        grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                r = random.random()
                if r < 0.40:
                    row.append(CELL_SERVER)       # 40% normal servers
                elif r < 0.60:
                    row.append(CELL_HOT_SERVER)   # 20% overheating servers
                elif r < 0.75:
                    row.append(CELL_POLLUTED)     # 15% polluted
                elif r < 0.85:
                    row.append(CELL_TREE)         # 10% trees
                elif r < 0.90:
                    row.append(CELL_EMPTY)        # 5% empty
                else:
                    row.append(CELL_EFFICIENT)    # 10% efficient servers
            grid.append(row)
        return grid

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: EcoServerAction) -> EcoServerObservation:
        """Execute an action and return the new observation."""
        self._step += 1

        x = action.x if action.x is not None else self.width // 2
        y = action.y if action.y is not None else self.height // 2

        # Clamp coordinates to grid bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))

        event = self._apply_action(action.action_type, x, y)
        self._apply_environment_dynamics()

        done = self._step >= self.max_steps or self._check_win_condition()
        self._obs = self._compute_observation(event, done=done)
        return self._obs

    def _apply_action(self, action_type: str, x: int, y: int) -> str:
        """Apply action to the grid and return event description."""
        cell = self.grid[y][x]

        if action_type == "plant_tree":
            if cell in (CELL_EMPTY, CELL_POLLUTED):
                self.grid[y][x] = CELL_TREE
                return f"🌳 Planted tree at ({x},{y})"
            return f"⚠️ Cannot plant tree at ({x},{y}) - cell occupied"

        elif action_type == "remove_pollution":
            if cell == CELL_POLLUTED:
                self.grid[y][x] = CELL_EMPTY
                return f"✨ Removed pollution at ({x},{y})"
            # Spread cleanup to neighbors
            cleaned = self._clean_neighbors(x, y)
            return f"🧹 Cleaned {cleaned} neighboring cells around ({x},{y})"

        elif action_type == "monitor":
            stats = self._get_local_stats(x, y)
            return f"📊 Monitored ({x},{y}): {stats}"

        elif action_type == "develop":
            if cell == CELL_EMPTY:
                self.grid[y][x] = CELL_SERVER
                return f"🏗️ Developed server at ({x},{y})"
            elif cell == CELL_SERVER:
                self.grid[y][x] = CELL_HOT_SERVER
                return f"⚠️ Overloaded server at ({x},{y})"
            return f"⚠️ Cannot develop at ({x},{y})"

        elif action_type == "install_solar":
            if cell in (CELL_EMPTY, CELL_TREE):
                self.grid[y][x] = CELL_SOLAR
                return f"☀️ Installed solar panel at ({x},{y})"
            return f"⚠️ Cannot install solar at ({x},{y})"

        elif action_type == "cool_server":
            if cell == CELL_HOT_SERVER:
                self.grid[y][x] = CELL_SERVER
                return f"❄️ Cooled server at ({x},{y})"
            elif cell == CELL_SERVER:
                return f"ℹ️ Server at ({x},{y}) already cool"
            return f"⚠️ No server to cool at ({x},{y})"

        elif action_type == "upgrade_efficiency":
            if cell == CELL_SERVER:
                self.grid[y][x] = CELL_EFFICIENT
                return f"⚡ Upgraded server efficiency at ({x},{y})"
            return f"⚠️ Cannot upgrade at ({x},{y})"

        elif action_type == "decommission":
            if cell in (CELL_SERVER, CELL_HOT_SERVER):
                self.grid[y][x] = CELL_EMPTY
                return f"🔌 Decommissioned server at ({x},{y})"
            return f"⚠️ Nothing to decommission at ({x},{y})"

        return f"❓ Unknown action: {action_type}"

    def _apply_environment_dynamics(self):
        """
        Simulate realistic environment changes each step:
        - Hot servers spread heat to neighbors
        - Pollution spreads slowly
        - Trees reduce nearby pollution
        - Solar panels reduce energy demand
        """
        new_grid = [row[:] for row in self.grid]

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]

                # Hot servers have a chance to spread heat
                if cell == CELL_HOT_SERVER:
                    for nx, ny in self._neighbors(x, y):
                        if self.grid[ny][nx] == CELL_SERVER and random.random() < 0.05:
                            new_grid[ny][nx] = CELL_HOT_SERVER

                # Servers occasionally overheat under load
                if cell == CELL_SERVER and random.random() < 0.02:
                    new_grid[y][x] = CELL_HOT_SERVER

                # Pollution spreads slowly
                if cell == CELL_POLLUTED and random.random() < 0.03:
                    for nx, ny in self._neighbors(x, y):
                        if self.grid[ny][nx] == CELL_EMPTY:
                            new_grid[ny][nx] = CELL_POLLUTED

                # Trees naturally clean neighboring pollution
                if cell == CELL_TREE:
                    for nx, ny in self._neighbors(x, y):
                        if self.grid[ny][nx] == CELL_POLLUTED and random.random() < 0.08:
                            new_grid[ny][nx] = CELL_EMPTY

        self.grid = new_grid

    def _clean_neighbors(self, x: int, y: int) -> int:
        """Clean polluted cells around (x, y). Returns count cleaned."""
        cleaned = 0
        for nx, ny in self._neighbors(x, y):
            if self.grid[ny][nx] == CELL_POLLUTED:
                self.grid[ny][nx] = CELL_EMPTY
                cleaned += 1
        return cleaned

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Return valid neighboring coordinates (up/down/left/right)."""
        result = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                result.append((nx, ny))
        return result

    def _get_local_stats(self, x: int, y: int) -> str:
        """Get stats for a local area around (x, y)."""
        cells = [self.grid[y][x]]
        for nx, ny in self._neighbors(x, y):
            cells.append(self.grid[ny][nx])
        hot = cells.count(CELL_HOT_SERVER)
        trees = cells.count(CELL_TREE)
        polluted = cells.count(CELL_POLLUTED)
        return f"hot={hot} trees={trees} polluted={polluted}"

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _compute_observation(self, event: str = "", done: bool = False) -> EcoServerObservation:
        """Compute all metrics from current grid state."""
        total = self.width * self.height
        counts = {i: 0 for i in range(7)}
        for row in self.grid:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1

        # Cell counts
        n_server    = counts[CELL_SERVER]
        n_tree      = counts[CELL_TREE]
        n_polluted  = counts[CELL_POLLUTED]
        n_solar     = counts[CELL_SOLAR]
        n_hot       = counts[CELL_HOT_SERVER]
        n_efficient = counts[CELL_EFFICIENT]
        n_servers_total = n_server + n_hot + n_efficient

        # Metrics
        pollution       = round(n_polluted / total, 4)
        green_cover     = round((n_tree + n_solar) / total, 4)
        temperature     = round(n_hot / max(n_servers_total, 1), 4)
        energy_usage    = round((n_server * 1.0 + n_hot * 1.5 + n_efficient * 0.6) / max(total, 1), 4)
        renewable_ratio = round(n_solar / max(n_servers_total, 1), 4)
        server_load     = round(n_servers_total / total, 4)
        uptime          = round(1.0 - (n_hot / max(n_servers_total, 1)) * 0.5, 4)

        # Eco score — weighted combination of all metrics
        eco_score = round(
            green_cover     * 0.25 +
            (1 - pollution) * 0.25 +
            (1 - temperature) * 0.20 +
            renewable_ratio * 0.15 +
            uptime          * 0.15,
            4
        )

        # Reward with partial progress signals
        reward = round(min(1.0, max(0.0,
            eco_score * 0.5 +
            (1 - pollution) * 0.2 +
            renewable_ratio * 0.2 +
            uptime * 0.1
        )), 4)

        return EcoServerObservation(
            grid=self.grid,
            eco_score=eco_score,
            pollution=pollution,
            green_cover=green_cover,
            temperature=temperature,
            energy_usage=energy_usage,
            renewable_ratio=renewable_ratio,
            server_load=server_load,
            uptime=uptime,
            step=self._step,
            done=done,
            reward=reward,
            last_event=event,
        )

    def _check_win_condition(self) -> bool:
        """Episode ends early if agent achieves excellent eco score."""
        if self._obs and self._obs.eco_score >= 0.90:
            return True
        return False

    @property
    def state(self) -> Optional[EcoServerObservation]:
        return self._obs


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = EcoServerEnv(width=15, height=15)
    obs = env.reset()
    print(f"Initial eco_score: {obs.eco_score}")
    print(f"Initial pollution:  {obs.pollution}")
    print(f"Initial temperature: {obs.temperature}")
    print(f"Initial renewable_ratio: {obs.renewable_ratio}")

    actions = [
        EcoServerAction("cool_server", 3, 3),
        EcoServerAction("install_solar", 5, 5),
        EcoServerAction("plant_tree", 7, 7),
        EcoServerAction("remove_pollution", 2, 2),
        EcoServerAction("upgrade_efficiency", 4, 4),
        EcoServerAction("monitor", 7, 7),
    ]

    for action in actions:
        result = env.step(action)
        print(f"\n{result.last_event}")
        print(f"  eco_score={result.eco_score} pollution={result.pollution} "
              f"temp={result.temperature} renewable={result.renewable_ratio}")
