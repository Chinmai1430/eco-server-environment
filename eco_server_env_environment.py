# eco_server_env_environment.py
import random
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Define data classes for the environment
@dataclass
class EcoServerAction:
    """Action that can be taken in the environment"""
    action_type: str  # "plant_tree", "remove_pollution", "monitor", "develop"
    x: Optional[int] = None
    y: Optional[int] = None

@dataclass
class EcoServerObservation:
    """What the agent observes from the environment"""
    grid: List[List[int]]  # 2D grid representing ecosystem state
    width: int
    height: int
    step: int
    pollution_level: float
    biodiversity_score: float
    resources_available: int
    total_carbon_captured: int
    done: bool
    reward: float = 0.0

@dataclass
class EcoServerState:
    """Internal state of the environment"""
    episode_id: str
    step_count: int
    total_pollution_removed: int
    total_trees_planted: int
    grid: List[List[int]]

class EcoServerEnv:
    """
    Ecosystem Server Environment for reinforcement learning
    Grid cell values:
    0 = Empty land
    1 = Healthy forest
    2 = Polluted area
    3 = Industrial zone
    4 = Protected area
    5 = Water body
    """
    
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.max_steps = 100
        self.current_step = 0
        
        # Initialize grid
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        
        # Environment state
        self.pollution_level = 100.0  # Percentage
        self.biodiversity_score = 50.0  # Percentage
        self.resources = 100  # Resource points
        self.carbon_captured = 0
        self.trees_planted = 0
        self.pollution_removed = 0
        
        # Episode tracking
        self.episode_id = f"eco_{random.randint(1000, 9999)}"
        self.done = False
        
        # Initialize environment
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize the environment with realistic ecosystem elements"""
        # Add some initial forests
        for _ in range(15):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y][x] = 1  # Healthy forest
        
        # Add some polluted areas
        for _ in range(10):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y][x] = 2  # Polluted area
        
        # Add some industrial zones
        for _ in range(5):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y][x] = 3  # Industrial zone
        
        # Add some water bodies
        for _ in range(3):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y][x] = 5  # Water body
    
    def reset(self) -> EcoServerObservation:
        """Reset the environment to initial state"""
        self.__init__(self.width, self.height)
        return self._get_observation()
    
    def step(self, action: EcoServerAction) -> EcoServerObservation:
        """Execute one step in the environment"""
        if self.done:
            raise Exception("Environment is already done. Call reset() to restart.")
        
        self.current_step += 1
        reward = 0.0
        
        # Execute action based on type
        if action.action_type == "plant_tree" and action.x is not None and action.y is not None:
            reward = self._plant_tree(action.x, action.y)
        elif action.action_type == "remove_pollution" and action.x is not None and action.y is not None:
            reward = self._remove_pollution(action.x, action.y)
        elif action.action_type == "monitor":
            reward = self._monitor_environment()
        elif action.action_type == "develop" and action.x is not None and action.y is not None:
            reward = self._develop_area(action.x, action.y)
        
        # Natural processes (environment evolves)
        self._evolve_environment()
        
        # Check termination conditions
        if self.current_step >= self.max_steps:
            self.done = True
            # Bonus for good performance
            if self.biodiversity_score > 70:
                reward += 50
            if self.pollution_level < 30:
                reward += 50
        
        # Update observation
        observation = self._get_observation()
        observation.reward = reward
        return observation
    
    def _plant_tree(self, x: int, y: int) -> float:
        """Plant a tree at given coordinates"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -10  # Invalid coordinates penalty
        
        if self.resources < 5:
            return -5  # Not enough resources penalty
        
        cell_type = self.grid[y][x]
        if cell_type == 0:  # Empty land
            self.grid[y][x] = 1  # Plant tree
            self.resources -= 5
            self.trees_planted += 1
            self.biodiversity_score = min(100, self.biodiversity_score + 0.5)
            return 10  # Positive reward for successful planting
        elif cell_type == 2:  # Polluted area
            self.grid[y][x] = 1  # Clean and plant
            self.resources -= 8
            self.trees_planted += 1
            self.pollution_level = max(0, self.pollution_level - 2)
            self.biodiversity_score = min(100, self.biodiversity_score + 1)
            return 15  # Extra reward for cleaning polluted area
        else:
            return -5  # Can't plant here penalty
    
    def _remove_pollution(self, x: int, y: int) -> float:
        """Remove pollution from given coordinates"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -10  # Invalid coordinates penalty
        
        if self.resources < 3:
            return -5  # Not enough resources penalty
        
        cell_type = self.grid[y][x]
        if cell_type == 2:  # Polluted area
            self.grid[y][x] = 0  # Clean the area
            self.resources -= 3
            self.pollution_removed += 1
            self.pollution_level = max(0, self.pollution_level - 3)
            self.carbon_captured += 2
            return 8  # Reward for pollution removal
        else:
            return -2  # No pollution to remove penalty
    
    def _monitor_environment(self) -> float:
        """Monitor the environment (passive action)"""
        # Monitoring gives information but costs resources
        if self.resources >= 1:
            self.resources -= 1
            # Small positive reward for gathering information
            return 1
        return -1  # Not enough resources
    
    def _develop_area(self, x: int, y: int) -> float:
        """Develop an area (could be good or bad)"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -10  # Invalid coordinates penalty
        
        if self.resources < 10:
            return -5  # Not enough resources penalty
        
        cell_type = self.grid[y][x]
        if cell_type == 0:  # Empty land
            # 70% chance of sustainable development, 30% chance of pollution
            if random.random() < 0.7:
                self.grid[y][x] = 4  # Protected area
                self.resources -= 10
                self.biodiversity_score = min(100, self.biodiversity_score + 2)
                return 15  # Sustainable development reward
            else:
                self.grid[y][x] = 3  # Industrial zone (polluting)
                self.resources -= 10
                self.pollution_level = min(100, self.pollution_level + 5)
                return -10  # Polluting development penalty
        else:
            return -5  # Can't develop here penalty
    
    def _evolve_environment(self):
        """Natural evolution of the environment"""
        # Pollution spreads naturally
        if self.pollution_level > 20:
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == 1 and random.random() < 0.1:  # Forest near pollution
                        self.grid[y][x] = 2  # Becomes polluted
                        self.pollution_level = min(100, self.pollution_level + 0.5)
        
        # Trees capture carbon over time
        healthy_forests = sum(row.count(1) for row in self.grid)
        self.carbon_captured += healthy_forests * 0.5
        
        # Biodiversity fluctuates naturally
        self.biodiversity_score += random.uniform(-1, 1)
        self.biodiversity_score = max(0, min(100, self.biodiversity_score))
        
        # Pollution reduction from carbon capture
        if self.carbon_captured > 0:
            self.pollution_level = max(0, self.pollution_level - (self.carbon_captured * 0.01))
    
    def _get_observation(self) -> EcoServerObservation:
        """Get current observation from the environment"""
        return EcoServerObservation(
            grid=[row[:] for row in self.grid],  # Deep copy
            width=self.width,
            height=self.height,
            step=self.current_step,
            pollution_level=self.pollution_level,
            biodiversity_score=self.biodiversity_score,
            resources_available=self.resources,
            total_carbon_captured=int(self.carbon_captured),
            done=self.done
        )
    
    @property
    def state(self) -> EcoServerState:
        """Get internal state of the environment"""
        return EcoServerState(
            episode_id=self.episode_id,
            step_count=self.current_step,
            total_pollution_removed=self.pollution_removed,
            total_trees_planted=self.trees_planted,
            grid=[row[:] for row in self.grid]  # Deep copy
        )

# Example usage:
if __name__ == "__main__":
    # Test the environment
    env = EcoServerEnv()
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    # Take a sample action
    action = EcoServerAction(action_type="plant_tree", x=5, y=5)
    result = env.step(action)
    print(f"Action result: {result}")
