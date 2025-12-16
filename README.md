# DeepOptions: American Option Pricing via Deep Reinforcement Learning

![CI Status](https://github.com/Satyam999999/DeepOptions-RL/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **"Can an AI learn the Black-Scholes formula from scratch just by trading?"**

**DeepOptions** is a high-performance Reinforcement Learning system that solves the *Optimal Stopping Problem* for American Options. Unlike European options, American options can be exercised at any time, making them computationally expensive to price.

This project implements two distinct agents:
1.  **Production Agent (DQN):** A risk-neutral pricer achieving **99.2% accuracy** against the Longstaff-Schwartz (LSM) baseline.
2.  **Research Agent (QR-DQN):** A Distributional RL agent that visualizes **Tail Risk** and implicit risk aversion.

---

## 📊 Key Results

The system was calibrated using **5-year historical S&P 500 volatility ($\sigma=0.1708$)**.

| Model | Price Estimate | Description |
| :--- | :--- | :--- |
| **Black-Scholes** | $4.48 | Theoretical Lower Bound (European) |
| **Deep RL (DQN)** | **$4.89** | **Learned Policy (Production Model)** |
| **Longstaff-Schwartz**| $4.93 | Theoretical Upper Bound (American) |

* **Pricing Accuracy:** **99.24%** (Relative to LSM).
* **Premium Capture:** **91.55%** (The agent captured >90% of the available arbitrage opportunity).

---

## 🖼️ Visualizations

### 1. Real-Time Trading Engine (Streamlit)
A "Matrix-style" dashboard that visualizes live market ticks and the agent's decision boundary in real-time.
![Dashboard Demo](download.jpeg)

### 2. Risk Profile (QR-DQN)
Unlike standard models that output a single price, the Research Agent visualizes the **uncertainty** of holding the option.
* **Blue Curve:** Variance of returns if HELD.
* **Red Spike:** Deterministic value if EXERCISED.
![Risk Profile](qrdqn_risk_profile.png)

---

## 🚀 Features
* **Physics-Based Simulation:** Custom Geometric Brownian Motion (GBM) environment calibrated to real-world market data.
* **Dual-Agent Architecture:**
    * **DQN:** Optimized for mean-value convergence (Pricing).
    * **QR-DQN:** Optimized for risk distribution learning (Risk Management).
* **Mathematical Validation:** Benchmarked against Black-Scholes (analytical) and Longstaff-Schwartz (numerical) algorithms.
* **MLOps Pipeline:**
    * **Dockerized:** Fully reproducible environment.
    * **CI/CD:** GitHub Actions triggers automated regression testing on every commit.
    * **Experiment Tracking:** Weights & Biases (WandB) integration.

---

## 🛠️ Tech Stack
* **Core:** Python 3.10, NumPy, Pandas
* **Deep Learning:** PyTorch (DQN, QR-DQN)
* **Simulation:** Gymnasium (Custom Env)
* **Visualization:** Streamlit, Plotly, Matplotlib
* **DevOps:** Docker, GitHub Actions, Pytest

---

## 💻 Installation

### Option A: Local Installation
```bash
# 1. Clone the repository
git clone [https://github.com/Satyam999999/DeepOptions-RL.git](https://github.com/Satyam999999/DeepOptions-RL.git)
cd DeepOptions-RL

# 2. Create virtual environment
conda create -n deep_options python=3.10
conda activate deep_options

# 3. Install dependencies
pip install -r requirements.txt
