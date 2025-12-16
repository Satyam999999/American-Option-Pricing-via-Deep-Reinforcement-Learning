import yaml
import matplotlib.pyplot as plt
from src.environment.american_option_env import AmericanOptionEnv

# 1. Load Config
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Initialize Environment
env = AmericanOptionEnv(config)

# 3. Test Reset
state, info = env.reset()
print(f"Initial State: {state}")  # Should be [100.0, 1.0]

# 4. Run a random loop
print("Running random policy...")
rewards = []
done = False

while not done:
    # Random action: 0 (Hold) or 1 (Exercise)
    # We force 0 mostly to see the path length
    action = 0 if state[1] > 0.1 else 1 
    
    state, reward, done, truncated, _ = env.step(action)
    print(f"Step: {env.current_step}, Price: {state[0]:.2f}, Time: {state[1]:.2f}")

print(f"Final Reward: {reward}")

# 5. Visual Check (Optional)
plt.plot(env.price_path)
plt.title("Generated GBM Price Path")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.savefig("debug_path.png")
print("Saved debug plot to debug_path.png")