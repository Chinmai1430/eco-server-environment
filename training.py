# training.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

def smart_eco_agent(observation, env):
    """Smarter agent that checks cell states before acting"""
    
    # Look for empty spots or polluted areas to plant trees
    for y in range(env.height):
        for x in range(env.width):
            cell_type = observation.grid[y][x]
            # 0 = empty land, 2 = polluted area
            if cell_type in [0, 2] and env.resources >= 5:
                return EcoServerAction(action_type="plant_tree", x=x, y=y)
    
    # If no good planting spots, try removing pollution
    for y in range(env.height):
        for x in range(env.width):
            cell_type = observation.grid[y][x]
            if cell_type == 2 and env.resources >= 3:  # Polluted area
                return EcoServerAction(action_type="remove_pollution", x=x, y=y)
    
    # If no good actions, monitor environment
    return EcoServerAction(action_type="monitor")

def aggressive_development_agent(observation, env):
    """Agent that focuses on development and economic growth"""
    # Try development first (economic growth)
    if env.resources >= 10:
        for y in range(env.height):
            for x in range(env.width):
                cell_type = observation.grid[y][x]
                if cell_type == 0 and env.resources >= 10:  # Empty land
                    return EcoServerAction(action_type="develop", x=x, y=y)
    
    # If development fails, plant trees
    for y in range(env.height):
        for x in range(env.width):
            cell_type = observation.grid[y][x]
            if cell_type in [0, 2] and env.resources >= 5:
                return EcoServerAction(action_type="plant_tree", x=x, y=y)
    
    # Monitor if nothing else
    return EcoServerAction(action_type="monitor")

def conservation_focused_agent(observation, env):
    """Agent that prioritizes environmental protection"""
    # Remove pollution first
    for y in range(env.height):
        for x in range(env.width):
            cell_type = observation.grid[y][x]
            if cell_type == 2 and env.resources >= 3:
                return EcoServerAction(action_type="remove_pollution", x=x, y=y)
    
    # Plant trees in empty areas
    for y in range(env.height):
        for x in range(env.width):
            cell_type = observation.grid[y][x]
            if cell_type == 0 and env.resources >= 5:
                return EcoServerAction(action_type="plant_tree", x=x, y=y)
    
    # Monitor environment
    return EcoServerAction(action_type="monitor")

def balanced_approach_agent(observation, env):
    """Balanced agent that considers multiple factors"""
    # Critical situation: high pollution, act immediately
    if observation.pollution_level > 80 and env.resources >= 3:
        for y in range(env.height):
            for x in range(env.width):
                if observation.grid[y][x] == 2:
                    return EcoServerAction(action_type="remove_pollution", x=x, y=y)
    
    # Good resources, moderate pollution - plant trees
    elif env.resources >= 5 and observation.pollution_level < 70:
        for y in range(env.height):
            for x in range(env.width):
                cell_type = observation.grid[y][x]
                if cell_type in [0, 2]:
                    return EcoServerAction(action_type="plant_tree", x=x, y=y)
    
    # Low resources - monitor and conserve
    else:
        return EcoServerAction(action_type="monitor")

def train_single_agent(agent_func, agent_name, steps=50, width=20, height=20):
    """Train a single agent and return results"""
    env = EcoServerEnv(width=width, height=height)
    total_reward = 0
    actions_taken = {"plant_tree": 0, "remove_pollution": 0, "develop": 0, "monitor": 0}
    
    # Reset environment
    obs = env.reset()
    
    print(f"🌱 Starting Training: {agent_name}")
    print("=" * 60)
    
    # Run for specified steps
    for step in range(steps):
        # Use agent function
        action = agent_func(obs, env)
        actions_taken[action.action_type] += 1
        
        result = env.step(action)
        total_reward += result.reward
        
        # Print progress every 10 steps
        if step % 10 == 0 or step == steps - 1:
            print(f"Step {step+1:2d}: Action={action.action_type:15} "
                  f"Reward={result.reward:5.1f}, Total={total_reward:6.1f}")
            print(f"  Stats - Pollution: {result.pollution_level:5.1f}%, "
                  f"Biodiversity: {result.biodiversity_score:5.1f}%, "
                  f"Resources: {result.resources_available:3d}")
        
        if result.done:
            print(f"🏁 Episode completed early at step {step+1}!")
            break
            
        obs = result  # Update observation
    
    print(f"\n🏆 {agent_name} Final Results:")
    print(f"   Total Score: {total_reward:.1f}")
    print(f"   Pollution Reduction: {obs.pollution_level:.1f}%")
    print(f"   Biodiversity Increase: {obs.biodiversity_score:.1f}%")
    print(f"   Actions Taken: {actions_taken}")
    print("-" * 60)
    
    return {
        "score": total_reward,
        "pollution": obs.pollution_level,
        "biodiversity": obs.biodiversity_score,
        "resources": obs.resources_available,
        "actions": actions_taken
    }

def compare_all_strategies():
    """Compare all AI strategies head-to-head"""
    strategies = {
        "🌱 Smart Eco Agent": smart_eco_agent,
        "🏙️ Development Focused": aggressive_development_agent,
        "🌿 Conservation First": conservation_focused_agent,
        "⚖️ Balanced Approach": balanced_approach_agent
    }
    
    print("🤖 COMPARING ALL AI STRATEGIES")
    print("=" * 80)
    
    results = {}
    
    for name, strategy in strategies.items():
        result = train_single_agent(strategy, name, steps=30, width=15, height=15)
        results[name] = result
        print()
    
    # Show comparison summary
    print("📊 STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Score':<8} {'Pollution':<10} {'Biodiversity':<12} {'Best For':<15}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for name, result in sorted_results:
        best_for = ""
        if result['score'] == max(r['score'] for r in results.values()):
            best_for = "🏆 Overall"
        elif result['pollution'] == min(r['pollution'] for r in results.values()):
            best_for = "💨 Pollution"
        elif result['biodiversity'] == max(r['biodiversity'] for r in results.values()):
            best_for = "🌳 Biodiversity"
            
        print(f"{name:<25} {result['score']:<8.1f} {result['pollution']:<10.1f} "
              f"{result['biodiversity']:<12.1f} {best_for:<15}")
    
    winner = sorted_results[0][0]
    print(f"\n🎉 WINNER: {winner}")
    return results

def demonstrate_learning_progression():
    """Show how the environment improves over time with good policies"""
    print("📈 DEMONSTRATING LEARNING PROGRESSION")
    print("=" * 60)
    
    env = EcoServerEnv(width=20, height=20)
    obs = env.reset()
    
    print("Initial State:")
    print(f"  Pollution: {obs.pollution_level:.1f}%")
    print(f"  Biodiversity: {obs.biodiversity_score:.1f}%")
    print(f"  Resources: {obs.resources_available}")
    
    # Run 3 phases of training
    phases = [
        ("Phase 1 - Initial Cleanup", 15),
        ("Phase 2 - Active Restoration", 20),
        ("Phase 3 - Maintenance", 15)
    ]
    
    total_reward = 0
    
    for phase_name, steps in phases:
        print(f"\n🎯 {phase_name}")
        print("-" * 30)
        
        phase_reward = 0
        for step in range(steps):
            action = smart_eco_agent(obs, env)
            result = env.step(action)
            phase_reward += result.reward
            total_reward += result.reward
            obs = result
            
            if step % 5 == 0:
                print(f"  Step {step+1}: Pollution={result.pollution_level:5.1f}%, "
                      f"Biodiversity={result.biodiversity_score:5.1f}%, "
                      f"Reward={result.reward:5.1f}")
        
        print(f"  {phase_name} Total Reward: {phase_reward:.1f}")
    
    print(f"\n🏁 FINAL RESULTS AFTER LEARNING:")
    print(f"  Total Score: {total_reward:.1f}")
    print(f"  Pollution Reduction: {(100 - obs.pollution_level):.1f}%")
    print(f"  Biodiversity Improvement: {(obs.biodiversity_score - 50):.1f}%")
    print(f"  Carbon Captured: {obs.total_carbon_captured}")

def main():
    """Main training function"""
    print("🌍 ECOGUARDIAN AI - INTELLIGENT ECOSYSTEM MANAGEMENT")
    print("=" * 80)
    print("Training advanced AI agents for environmental restoration...")
    print()
    
    # Run comprehensive comparison
    compare_all_strategies()
    
    print("\n" + "=" * 80)
    
    # Show learning progression
    demonstrate_learning_progression()
    
    print("\n✅ Training completed successfully!")
    print("🏆 Your EcoGuardian AI is ready for the Meta Hackathon!")

if __name__ == "__main__":
    main()
