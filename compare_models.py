import yaml
import numpy as np
import torch
from src.environment.gbm import GBMGenerator
from src.baselines.lsm import longstaff_schwartz
from src.baselines.european_mc import EuropeanOptionPricing
from src.agents.dqn_agent import DQNAgent

def compare():
    with open("experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Generate Evaluation Data
    print("🎲 Generating 10,000 paths for evaluation...")
    config['simulation']['n_sims'] = 100
    
    gen = GBMGenerator(config)
    paths = []
    for _ in range(config['simulation']['n_sims']):
        paths.append(gen.generate_path())
    paths = np.array(paths)

    # 2. Calculate LSM Baseline
    print("🧮 Calculating Longstaff-Schwartz (LSM) Baseline...")
    lsm_price = longstaff_schwartz(
        paths, 
        config['simulation']['k'], 
        config['simulation']['r'], 
        config['simulation']['t']
    )

    # 3. Calculate AI Price
    print("🤖 Calculating AI Agent Price...")
    agent = DQNAgent(2, 2, config)
    try:
        agent.policy_net.load_state_dict(torch.load("checkpoints/dqn_option.pth"))
        agent.policy_net.eval()
    except:
        print("⚠️ Model not found, using random agent.")

    ai_payoffs = []
    
    for i in range(len(paths)):
        path = paths[i]
        curr_step = 0
        done = False
        payoff = 0
        
        while not done:
            # Prepare State
            price = path[curr_step]
            norm_price = price / config['simulation']['k'] # Normalize Price
            
            T_max = config['simulation']['t']
            dt = T_max / config['simulation']['n_steps']
            time_left = T_max - (curr_step * dt)
            norm_time = time_left / T_max # Normalize Time
            
            state = np.array([norm_price, norm_time], dtype=np.float32)
            
            # Action
            with torch.no_grad():
                st_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action = agent.policy_net(st_tensor).argmax().item()
            
            # LOGIC FIX: Check Maturity vs Exercise
            if action == 1: # Exercise Early
                payoff = max(config['simulation']['k'] - price, 0) * np.exp(-config['simulation']['r'] * (curr_step/252))
                done = True
            elif curr_step == len(path) - 1: # Reached Maturity (Force Exercise)
                payoff = max(config['simulation']['k'] - price, 0) * np.exp(-config['simulation']['r'] * 1.0)
                done = True
            
            curr_step += 1
            if curr_step >= len(path):
                done = True

        ai_payoffs.append(payoff)

    ai_price = np.mean(ai_payoffs)

    # 4. European Price
    eu_pricer = EuropeanOptionPricing(config)
    bs_price = eu_pricer.black_scholes_price()

    print("\n" + "="*40)
    print(f"📉 European (Black-Scholes): {bs_price:.4f} (Lower Bound)")
    print(f"🌲 American (LSM Baseline):   {lsm_price:.4f} (Target)")
    print(f"🤖 American (AI Agent):       {ai_price:.4f}")
    print("="*40)

if __name__ == "__main__":
    compare()