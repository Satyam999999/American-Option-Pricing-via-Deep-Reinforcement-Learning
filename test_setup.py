import torch
import numpy as np
import yaml
import gymnasium as gym

def test_environment():
    print("✅ Libraries imported successfully.")
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Running on device: {device}")

    # Load config
    try:
        with open("experiments/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✅ Config file loaded.")
        print(f"   Target Volatility: {config['simulation']['sigma']}")
    except Exception as e:
        print(f"❌ Config error: {e}")

if __name__ == "__main__":
    test_environment()