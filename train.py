import yaml
import numpy as np
import torch
import os  # Added this to auto-create folders
from tqdm import tqdm
import wandb # <--- NEW: Import W&B
from src.environment.american_option_env import AmericanOptionEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.replay_buffer import ReplayBuffer

def train():
    # 1. Load Config
    with open("experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # <--- NEW: Initialize W&B Project
    wandb.init(
        project=config['project_name'], 
        config=config,
        name="DQN_Run_1"
    )

    # 2. Setup
    env = AmericanOptionEnv(config)
    agent = DQNAgent(state_dim=2, action_dim=2, config=config)
    memory = ReplayBuffer(
        capacity=config['agent']['memory_size'],
        batch_size=config['agent']['batch_size'],
        device=agent.device
    )

    # 3. Training Loop
    num_episodes = config['training']['episodes']
    epsilon = config['agent']['epsilon_start']
    epsilon_decay = config['agent']['epsilon_decay']
    target_update = config['agent']['target_update']

    # Ensure checkpoint folder exists
    os.makedirs("checkpoints", exist_ok=True) 

    pbar = tqdm(range(num_episodes))
    
    for episode in pbar:
        state, _ = env.reset()
        total_reward = 0
        done = False
        loss_val = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.optimize_model(memory)
            if loss: loss_val = loss # Capture loss for logging

        # Update Target Network
        if episode % target_update == 0:
            agent.update_target_network()

        # Decay Epsilon
        epsilon = max(config['agent']['epsilon_end'], epsilon * epsilon_decay)

        # <--- NEW: Log metrics to W&B
        wandb.log({
            "reward": total_reward,
            "loss": loss_val,
            "epsilon": epsilon
        })

        if episode % 10 == 0:
            pbar.set_description(f"Ep {episode} | Reward: {total_reward:.2f}")

    # Save Model
    torch.save(agent.policy_net.state_dict(), "checkpoints/dqn_option.pth")
    print("✅ Model saved.")
    wandb.finish()

if __name__ == "__main__":
    train()