import yaml
import numpy as np
import torch
import os
import wandb
from tqdm import tqdm
from src.environment.american_option_env import AmericanOptionEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.replay_buffer import ReplayBuffer

def train():
    with open("experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # wandb.init(project="DeepOptions-RL", config=config, name="DQN_Production") # Uncomment if using wandb

    env = AmericanOptionEnv(config)
    agent = DQNAgent(state_dim=2, action_dim=2, config=config)
    memory = ReplayBuffer(config['agent']['memory_size'], config['agent']['batch_size'], agent.device)

    num_episodes = config['training']['episodes']
    epsilon = config['agent']['epsilon_start']
    epsilon_decay = config['agent']['epsilon_decay']
    target_update = config['agent']['target_update']

    pbar = tqdm(range(num_episodes))
    for episode in pbar:
        state, _ = env.reset()
        done = False
        total_reward = 0
        loss_val = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.optimize_model(memory)
            if loss: loss_val = loss

        if episode % target_update == 0:
            agent.update_target_network()

        epsilon = max(config['agent']['epsilon_end'], epsilon * epsilon_decay)
        
        # wandb.log({"reward": total_reward, "loss": loss_val, "epsilon": epsilon})
        if episode % 10 == 0:
            pbar.set_description(f"Ep {episode} | Reward: {total_reward:.2f} | Loss: {loss_val:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.policy_net.state_dict(), "checkpoints/dqn_option.pth")
    print("✅ DQN Model saved to checkpoints/dqn_option.pth")

if __name__ == "__main__":
    train()