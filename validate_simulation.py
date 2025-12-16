import yaml
import numpy as np
from src.baselines.european_mc import EuropeanOptionPricing

def validate():
    # 1. Load Config
    with open("experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize Pricer
    pricer = EuropeanOptionPricing(config)

    # 3. Calculate Prices
    bs_price = pricer.black_scholes_price()
    mc_price = pricer.monte_carlo_price()

    # 4. Compare
    error = abs(bs_price - mc_price)
    percent_error = (error / bs_price) * 100

    print("-" * 40)
    print(f"📉 Black-Scholes Price (Theoretical): {bs_price:.4f}")
    print(f"🎲 Monte Carlo Price (Simulated):    {mc_price:.4f}")
    print(f"❌ Difference: {error:.4f} ({percent_error:.2f}%)")
    print("-" * 40)

    if percent_error < 1.0:
        print("✅ SUCCESS: Simulation is accurate within 1%.")
    else:
        print("⚠️ WARNING: Simulation drift is high. Check random seed or increase n_sims.")

if __name__ == "__main__":
    validate()