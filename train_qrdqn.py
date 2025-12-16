import yaml
import numpy as np
import torch
import os
from tqdm import tqdm
# import wandb
from src.environment.american_option_env import AmericanOptionEnv
from src.agents.qrdqn_agent import QRDQNAgent
from src.utils.replay_buffer import ReplayBuffer

def train_qrdqn():
    with open("experiments/config_qrdqn.yaml", "r") as f:
        config = yaml.safe_load(f)

    # wandb.init(project="DeepOptions-RL", config=config, name="QR-DQN_SoftUpdate")

    env = AmericanOptionEnv(config)
    agent = QRDQNAgent(state_dim=2, action_dim=2, config=config)
    memory = ReplayBuffer(config['agent']['memory_size'], config['agent']['batch_size'], agent.device)

    num_episodes = config['training']['episodes']
    epsilon = config['agent']['epsilon_start']
    epsilon_decay = config['agent']['epsilon_decay']

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
            if loss: 
                loss_val = loss
                agent.update_target_network(tau=0.005)

        epsilon = max(config['agent']['epsilon_end'], epsilon * epsilon_decay)
        # wandb.log({"reward": total_reward, "loss": loss_val, "epsilon": epsilon})

        if episode % 100 == 0:
            pbar.set_description(f"Ep {episode} | Reward: {total_reward:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.policy_net.state_dict(), "checkpoints/qrdqn_option.pth")
    print("✅ QR-DQN Model saved to checkpoints/qrdqn_option.pth")

if __name__ == "__main__":
    train_qrdqn()