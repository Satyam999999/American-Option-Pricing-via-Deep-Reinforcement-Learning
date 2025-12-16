import yaml
import numpy as np
import torch
import sys
import os

# Fix imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.gbm import GBMGenerator
from src.baselines.lsm import longstaff_schwartz
from src.baselines.european_mc import EuropeanOptionPricing
from src.agents.qrdqn_agent import QRDQNAgent # <--- NEW AGENT

def compare_qrdqn():
    # 1. Load QR-DQN Config
    with open("experiments/config_qrdqn.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"🎲 Generating 10,000 paths for QR-DQN Evaluation...")
    config['simulation']['n_sims'] = 10000 
    
    gen = GBMGenerator(config)
    paths = []
    for _ in range(config['simulation']['n_sims']):
        paths.append(gen.generate_path())
    paths = np.array(paths)

    # 2. Baselines (The Truth)
    print("🧮 Calculating LSM Baseline (Truth)...")
    lsm_price = longstaff_schwartz(paths, config['simulation']['k'], config['simulation']['r'], config['simulation']['t'])
    
    eu_pricer = EuropeanOptionPricing(config)
    bs_price = eu_pricer.black_scholes_price()

    # 3. AI Agent (QR-DQN)
    print("🤖 Calculating QR-DQN Price (Averaging Distributions)...")
    agent = QRDQNAgent(state_dim=2, action_dim=2, config=config)
    try:
        agent.policy_net.load_state_dict(torch.load("checkpoints/qrdqn_option.pth", map_location=agent.device))
        agent.policy_net.eval()
    except:
        print("⚠️ Model not found! Train it with train_qrdqn.py first.")
        return

    ai_payoffs = []
    for i in range(len(paths)):
        path = paths[i]
        curr_step = 0
        done = False
        payoff = 0
        
        while not done:
            # Prepare State
            price = path[curr_step]
            norm_price = price / config['simulation']['k']
            time_left = config['simulation']['t'] - (curr_step * (config['simulation']['t'] / config['simulation']['n_steps']))
            state = np.array([norm_price, time_left], dtype=np.float32)
            
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                # Get Full Distribution: (1, Actions, Quantiles)
                dist = agent.policy_net(state_t)
                
                # KEY DIFFERENCE: We take the MEAN of the quantiles to make a decision
                # This collapses the "Risk Curve" into a single "Price"
                mean_q = dist.mean(dim=2)
                action = mean_q.argmax().item()
            
            if action == 1:
                payoff = max(config['simulation']['k'] - price, 0) * np.exp(-config['simulation']['r'] * (curr_step/52))
                done = True
            elif curr_step == len(path) - 1:
                payoff = max(config['simulation']['k'] - price, 0) * np.exp(-config['simulation']['r'] * 1.0)
                done = True
            
            curr_step += 1
            if curr_step >= len(path): done = True

        ai_payoffs.append(payoff)

    ai_price = np.mean(ai_payoffs)

    # 4. METRICS
    accuracy = 100 * (1 - abs(lsm_price - ai_price) / lsm_price)
    
    premium_available = lsm_price - bs_price
    premium_captured = ai_price - bs_price
    if premium_available > 0:
        capture_score = 100 * (premium_captured / premium_available)
    else:
        capture_score = 0.0

    print("\n" + "="*50)
    print(f"📊 QR-DQN DISTRIBUTIONAL REPORT")
    print("="*50)
    print(f"📉 European Floor:           ${bs_price:.4f}")
    print(f"🌲 American Ceiling (LSM):   ${lsm_price:.4f}")
    print(f"🧠 QR-DQN Mean Price:        ${ai_price:.4f}")
    print("-" * 50)
    print(f"✅ Distributional Accuracy:   {accuracy:.2f}%")
    print(f"💰 Risk-Adjusted Capture:     {capture_score:.2f}%")
    print("="*50)

if __name__ == "__main__":
    compare_qrdqn()