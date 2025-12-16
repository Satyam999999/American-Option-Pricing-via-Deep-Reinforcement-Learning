# DeepOptions: American Option Pricing via Deep Reinforcement Learning

## 📌 Overview
This project implements a **Deep Q-Network (DQN)** agent to solve the **American Option Pricing** problem (Optimal Stopping). It compares the RL approach against traditional quantitative finance baselines, including the **Longstaff-Schwartz (LSM)** algorithm and **Binomial Trees**.

## 🚀 Features
- **Custom Gym Environment:** Simulates Geometric Brownian Motion (GBM) market dynamics.
- **Deep Reinforcement Learning:** DQN with Experience Replay and Target Networks.
- **Quantitative Baselines:** Validated against LSM and Binomial Tree pricing models.
- **MLOps:** Dockerized environment, CI/CD with GitHub Actions, and experiment tracking via Weights & Biases.

## 🛠️ Tech Stack
- **Languages:** Python 3.10+, C++ (planned for optimization)
- **ML:** PyTorch, Gymnasium
- **Data:** NumPy, Pandas
- **Ops:** Docker, GitHub Actions, WandB

## 📂 Structure
- `src/agents`: Neural network architectures and RL logic.
- `src/environment`: Market simulator and Gym environment.
- `src/baselines`: Mathematical pricing models (LSM, Binomial).

## 🏃‍♂️ Quick Start
1. Clone the repo
2. `pip install -r requirements.txt`
3. `python src/main.py` (Coming soon)