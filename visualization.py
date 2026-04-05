# visualization.py
from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

def visualize_grid(grid):
    """Simple text visualization of the ecosystem"""
    symbols = {
        0: '🌱',  # Empty land
        1: '🌳',  # Healthy forest
        2: '🏭',  # Polluted area
        3: '🏢',  # Industrial zone
        4: '保护区', # Protected area
        5: '💧'   # Water body
    }
    
    print("🌍 Ecosystem Map:")
    print("-" * 30)
    for row in grid:
        print(' '.join(symbols.get(cell, '?') for cell in row))
    print("-" * 30)

def visualize_detailed_stats(obs):
    """Show detailed environment statistics"""
    print(f"📊 Environment Status:")
    print(f"   Pollution Level: {obs.pollution_level:.1f}%")
    print(f"   Biodiversity: {obs.biodiversity_score:.1f}%")
    print(f"   Resources Available: {obs.resources_available}")
    print(f"   Carbon Captured: {obs.total_carbon_captured}")
    print(f"   Step: {obs.step}")

# Test visualization
if __name__ == "__main__":
    env = EcoServerEnv()
    obs = env.reset()
    visualize_grid(obs.grid)
    visualize_detailed_stats(obs)
