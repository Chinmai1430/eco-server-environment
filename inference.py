# inference.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.eco_server_env_environment import EcoServerEnv, EcoServerAction
from visualization import visualize_grid, visualize_detailed_stats

def main():
    print("🌍 Initializing EcoServer Environment...")
    
    # Create environment
    env = EcoServerEnv(width=15, height=15)
    
    # Reset environment
    observation = env.reset()
    print(f"✅ Environment initialized!")
    
    # Show initial state
    visualize_grid(observation.grid)
    visualize_detailed_stats(observation)
    
    # Run a few test steps with different actions
    actions = [
        EcoServerAction(action_type="plant_tree", x=7, y=7),
        EcoServerAction(action_type="remove_pollution", x=5, y=5),
        EcoServerAction(action_type="monitor"),
        EcoServerAction(action_type="develop", x=3, y=3),
        EcoServerAction(action_type="plant_tree", x=10, y=10)
    ]
    
    for i, action in enumerate(actions):
        print(f"\n📍 Step {i + 1} - Action: {action.action_type}")
        
        # Execute action
        result = env.step(action)
        
        # Show results
        visualize_grid(result.grid)
        visualize_detailed_stats(result)
        print(f"💰 Reward: {result.reward:.1f}")
        
        if result.done:
            print("🏁 Episode completed!")
            break
    
    print("\n✅ Inference completed successfully!")

if __name__ == "__main__":
    main()
