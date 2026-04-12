# server/eco_server_env_environment.py
"""
EcoServer Environment — Core Logic
Real-world data center ecological management simulation.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ── Cell Types ────────────────────────────────────────────────────────────────
CELL_EMPTY      = 0
CELL_SERVER     = 1
CELL_TREE       = 2
CELL_POLLUTED   = 3
CELL_SOLAR      = 4
CELL_HOT_SERVER = 5
CELL_EFFICIENT  = 6

CELL_NAMES = {
    CELL_EMPTY:      "empty",
    CELL_SERVER:     "server",
    CELL_TREE:       "tree",
    CELL_POLLUTED:   "polluted",
    CELL_SOLAR:      "solar",
    CELL_HOT_SERVER: "hot_server",
    CELL_EFFICIENT:  "efficient_server",
}

# ── Action ────────────────────────────────────────────────────────────────────
@dataclass
class EcoServerAction:
    """
    Action for EcoServer environment.
    action_type: one of plant_tree, remove_pollution, monitor, develop,
                 install_solar, cool_server, upgrade_efficiency, decommission
    x, y: grid coordinates (0-14)
    """
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None

    VALID_ACTIONS: List[str] = field(default_factory=lambda: [
        "plant_tree",
        "remove_pollution",
        "monitor",
        "develop",
        "install_solar",
        "cool_server",
        "upgrade_efficiency",
        "decommission",
    ])


# ── Observation ───────────────────────────────────────────────────────────────
@dataclass
class EcoServerObservation:
    """Full observation returned by reset() and step()."""
    # Grid
    grid: List[List[int]] = field(default_factory=list)

    # Ecological metrics (all 0.0–1.0)
    eco_score:        float = 0.0
    pollution:        float = 0.0
    green_cover:      float = 0.0
    temperature:      float = 0.0
    energy_usage:     float = 0.0
    renewable_ratio:  float = 0.0
    server_load:      float = 0.0
    uptime:           float = 1.0

    # Episode
    step:    int   = 0
    done:    bool  = False
    reward:  float = 0.0

    # Diagnostics
    last_event:   str   = ""
    win:          bool  = False
    lose:         bool  = False
    episode_score:float = 0.0


# ── Environment ───────────────────────────────────────────────────────────────
class EcoServerEnv:
    """
    Real-world ecological data center management environment.

    The agent manages a W×H grid of server racks, green spaces,
    and solar panels. It must balance uptime, pollution, temperature,
    and renewable energy to maximize eco_score.

    Episode ends when:
      - eco_score >= 0.90 (win)
      - pollution >= 0.95 (catastrophic pollution)
      - uptime <= 0.10 (server collapse)
      - step >= max_steps (time limit)
    """

    def __init__(self, width: int = 15, height: int = 15, max_steps: int = 50):
        self.width     = width
        self.height    = height
        self.max_steps = max_steps
        self._step     = 0
        self._obs:     Optional[EcoServerObservation] = None
        self.grid:     List[List[int]] = []
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> EcoServerObservation:
        self._step = 0
        self.grid  = self._generate_degraded_grid()
        self._obs  = self._compute_obs("Environment initialized.")
        return self._obs

    def _generate_degraded_grid(self) -> List[List[int]]:
        """Start in a degraded state — agent must improve it."""
        grid = []
        for _ in range(self.height):
            row = []
            for _ in range(self.width):
                r = random.random()
                if   r < 0.35: row.append(CELL_SERVER)
                elif r < 0.55: row.append(CELL_HOT_SERVER)
                elif r < 0.70: row.append(CELL_POLLUTED)
                elif r < 0.80: row.append(CELL_TREE)
                elif r < 0.88: row.append(CELL_EMPTY)
                else:          row.append(CELL_EFFICIENT)
            grid.append(row)
        return grid

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: EcoServerAction) -> EcoServerObservation:
        self._step += 1

        # Clamp coordinates
        x = max(0, min(int(action.x or self.width  // 2), self.width  - 1))
        y = max(0, min(int(action.y or self.height // 2), self.height - 1))

        event = self._apply_action(action.action_type, x, y)
        self._apply_dynamics()

        done, win, lose = self._check_terminal()
        self._obs = self._compute_obs(event, done=done, win=win, lose=lose)
        return self._obs

    # ── Action Logic ──────────────────────────────────────────────────────────
    def _apply_action(self, action_type: str, x: int, y: int) -> str:
        cell = self.grid[y][x]

        if action_type == "plant_tree":
            if cell in (CELL_EMPTY, CELL_POLLUTED):
                self.grid[y][x] = CELL_TREE
                return f"🌳 Planted tree at ({x},{y})"
            return f"⚠️ Cannot plant tree at ({x},{y})"

        elif action_type == "remove_pollution":
            cleaned = 0
            if cell == CELL_POLLUTED:
                self.grid[y][x] = CELL_EMPTY
                cleaned += 1
            for nx, ny in self._neighbors(x, y):
                if self.grid[ny][nx] == CELL_POLLUTED:
                    self.grid[ny][nx] = CELL_EMPTY
                    cleaned += 1
            return f"🧹 Cleaned {cleaned} polluted cells around ({x},{y})"

        elif action_type == "monitor":
            n = self._neighbors(x, y)
            hot  = sum(1 for nx,ny in n if self.grid[ny][nx] == CELL_HOT_SERVER)
            pol  = sum(1 for nx,ny in n if self.grid[ny][nx] == CELL_POLLUTED)
            tree = sum(1 for nx,ny in n if self.grid[ny][nx] == CELL_TREE)
            return f"📊 ({x},{y}): hot={hot} polluted={pol} trees={tree}"

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
                return f"☀️ Installed solar at ({x},{y})"
            return f"⚠️ Cannot install solar at ({x},{y})"

        elif action_type == "cool_server":
            if cell == CELL_HOT_SERVER:
                self.grid[y][x] = CELL_SERVER
                return f"❄️ Cooled server at ({x},{y})"
            return f"ℹ️ No hot server at ({x},{y})"

        elif action_type == "upgrade_efficiency":
            if cell == CELL_SERVER:
                self.grid[y][x] = CELL_EFFICIENT
                return f"⚡ Upgraded server at ({x},{y})"
            return f"⚠️ Cannot upgrade at ({x},{y})"

        elif action_type == "decommission":
            if cell in (CELL_SERVER, CELL_HOT_SERVER):
                self.grid[y][x] = CELL_EMPTY
                return f"🔌 Decommissioned server at ({x},{y})"
            return f"⚠️ Nothing to decommission at ({x},{y})"

        return f"❓ Unknown action: {action_type}"

    # ── Dynamics ──────────────────────────────────────────────────────────────
    def _apply_dynamics(self):
        """Realistic per-step dynamics."""
        new_grid = [row[:] for row in self.grid]
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell == CELL_HOT_SERVER:
                    for nx,ny in self._neighbors(x, y):
                        if self.grid[ny][nx] == CELL_SERVER and random.random() < 0.05:
                            new_grid[ny][nx] = CELL_HOT_SERVER
                if cell == CELL_SERVER and random.random() < 0.02:
                    new_grid[y][x] = CELL_HOT_SERVER
                if cell == CELL_POLLUTED and random.random() < 0.03:
                    for nx,ny in self._neighbors(x, y):
                        if self.grid[ny][nx] == CELL_EMPTY:
                            new_grid[ny][nx] = CELL_POLLUTED
                if cell == CELL_TREE:
                    for nx,ny in self._neighbors(x, y):
                        if self.grid[ny][nx] == CELL_POLLUTED and random.random() < 0.08:
                            new_grid[ny][nx] = CELL_EMPTY
        self.grid = new_grid

    # ── Terminal ──────────────────────────────────────────────────────────────
    def _check_terminal(self) -> Tuple[bool, bool, bool]:
        """Returns (done, win, lose)."""
        if self._obs is None:
            return False, False, False
        win  = self._obs.eco_score  >= 0.90
        lose = self._obs.pollution  >= 0.95 or self._obs.uptime <= 0.10
        done = win or lose or self._step >= self.max_steps
        return done, win, lose

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _compute_obs(self, event: str = "", done: bool = False,
                     win: bool = False, lose: bool = False) -> EcoServerObservation:
        total  = self.width * self.height
        counts = {}
        for row in self.grid:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1

        n_server    = counts.get(CELL_SERVER,     0)
        n_tree      = counts.get(CELL_TREE,       0)
        n_polluted  = counts.get(CELL_POLLUTED,   0)
        n_solar     = counts.get(CELL_SOLAR,      0)
        n_hot       = counts.get(CELL_HOT_SERVER, 0)
        n_efficient = counts.get(CELL_EFFICIENT,  0)
        n_servers   = n_server + n_hot + n_efficient

        pollution       = round(n_polluted / total, 4)
        green_cover     = round((n_tree + n_solar) / total, 4)
        temperature     = round(n_hot / max(n_servers, 1), 4)
        energy_usage    = round((n_server*1.0 + n_hot*1.5 + n_efficient*0.6) / max(total,1), 4)
        renewable_ratio = round(n_solar / max(n_servers, 1), 4)
        server_load     = round(n_servers / total, 4)
        uptime          = round(1.0 - (n_hot / max(n_servers, 1)) * 0.5, 4)

        eco_score = round(
            green_cover      * 0.25 +
            (1 - pollution)  * 0.25 +
            (1 - temperature)* 0.20 +
            renewable_ratio  * 0.15 +
            uptime           * 0.15,
        4)

        reward = round(min(1.0, max(0.0,
            eco_score       * 0.50 +
            (1 - pollution) * 0.20 +
            renewable_ratio * 0.20 +
            uptime          * 0.10
        )), 4)

        if win:  reward = 1.0
        if lose: reward = 0.0

        return EcoServerObservation(
            grid            = self.grid,
            eco_score       = eco_score,
            pollution       = pollution,
            green_cover     = green_cover,
            temperature     = temperature,
            energy_usage    = energy_usage,
            renewable_ratio = renewable_ratio,
            server_load     = server_load,
            uptime          = uptime,
            step            = self._step,
            done            = done,
            reward          = reward,
            last_event      = event,
            win             = win,
            lose            = lose,
            episode_score   = eco_score,
        )

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        return [
            (x+dx, y+dy)
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]
            if 0 <= x+dx < self.width and 0 <= y+dy < self.height
        ]

    @property
    def state(self) -> Optional[EcoServerObservation]:
        return self._obs
