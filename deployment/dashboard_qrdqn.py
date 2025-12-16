import streamlit as st
import time
import yaml
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import sys
import os

# Fix import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.agents.qrdqn_agent import QRDQNAgent

# 1. Page Config
st.set_page_config(page_title="DeepOptions QR-DQN Risk Engine", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# 2. Load QR-DQN System
@st.cache_resource
def load_system():
    try:
        # Load the QR-DQN Config
        with open("experiments/config_qrdqn.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        agent = QRDQNAgent(state_dim=2, action_dim=2, config=config)
        model_path = "checkpoints/qrdqn_option.pth"
        
        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
            agent.policy_net.eval()
        else:
            st.error(f"❌ Model not found at {model_path}")
            return None, None
            
        return agent, config
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None

agent, config = load_system()

# 3. Sidebar
with st.sidebar:
    st.title("🧠 Risk Engine")
    simulation_speed = st.slider("Speed (sec)", 0.1, 2.0, 0.5)
    start_btn = st.button("🚀 Start Risk Analysis", type="primary")
    st.info("Visualizing Distributional RL (QR-DQN)")

# 4. Main Layout
st.title("🧠 DeepOptions: Distributional Risk Analysis")
st.markdown("Unlike standard RL, QR-DQN predicts the **full probability distribution** of future returns.")

# Layout: Top Row (Metrics), Bottom Row (Charts)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1: price_placeholder = st.empty()
with kpi2: strike_placeholder = st.empty()
with kpi3: time_placeholder = st.empty()
with kpi4: signal_placeholder = st.empty()

col_price, col_risk = st.columns([1, 1])
with col_price:
    st.subheader("📉 Market Feed")
    price_chart = st.empty()
with col_risk:
    st.subheader("📊 Live Risk Profile (Agent's Brain)")
    risk_chart = st.empty()

# 5. Trading Loop
if start_btn:
    if 'history' not in st.session_state: st.session_state.history = []
    st.session_state.history = []

    while True:
        # A. Simulate Data
        live_price = np.random.uniform(85, 105)
        live_time = np.random.uniform(0.01, 1.0)
        
        # B. QR-DQN Inference
        norm_price = live_price / config['simulation']['k']
        state = np.array([norm_price, live_time], dtype=np.float32)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            # Get Full Distribution: (1, Actions, Quantiles)
            dist_tensor = agent.policy_net(state_t)
            
            # Convert to numpy for plotting
            # dist[0, 0] = Hold Distribution (51 quantiles)
            # dist[0, 1] = Exercise Distribution (51 quantiles)
            hold_dist = dist_tensor[0, 0, :].cpu().numpy()
            ex_dist = dist_tensor[0, 1, :].cpu().numpy()
            
            # Decision based on MEAN
            action = dist_tensor.mean(dim=2).argmax().item()

        # C. Visual Updates - Metrics
        price_placeholder.metric("Live Price", f"${live_price:.2f}", f"{live_price - 100:.2f}")
        strike_placeholder.metric("Strike", f"${config['simulation']['k']:.0f}")
        time_placeholder.metric("Maturity", f"{live_time:.2f} yr")
        
        if action == 1:
            signal_placeholder.markdown(
                """<div style="background-color: #ff4b4b; padding: 10px; border-radius: 5px; text-align: center;">
                <h3 style="color: white; margin:0;">🔴 EXERCISE</h3></div>""", unsafe_allow_html=True
            )
        else:
            signal_placeholder.markdown(
                """<div style="background-color: #1f2937; padding: 10px; border-radius: 5px; text-align: center;">
                <h3 style="color: #a0aec0; margin:0;">🔵 HOLD</h3></div>""", unsafe_allow_html=True
            )

        # D. Chart 1: Price History
        st.session_state.history.append({"Tick": len(st.session_state.history), "Price": live_price, "Action": action})
        df = pd.DataFrame(st.session_state.history[-30:])
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df['Tick'], y=df['Price'], mode='lines+markers', line=dict(color='gray')))
        ex_pts = df[df['Action'] == 1]
        if not ex_pts.empty:
            fig_price.add_trace(go.Scatter(x=ex_pts['Tick'], y=ex_pts['Price'], mode='markers', marker=dict(color='red', size=12, symbol='x')))
        fig_price.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark", yaxis_title="Price")
        price_chart.plotly_chart(fig_price, use_container_width=True)

        # E. Chart 2: RISK PROFILE (The Flex)
        # We plot the histograms of the predicted values
        fig_risk = go.Figure()
        
        # Plot Hold Distribution (Blue)
        fig_risk.add_trace(go.Histogram(
            x=hold_dist, 
            name='Value if HOLD', 
            marker_color='blue', 
            opacity=0.6,
            nbinsx=30
        ))
        
        # Plot Exercise Distribution (Red)
        fig_risk.add_trace(go.Histogram(
            x=ex_dist, 
            name='Value if EXERCISE', 
            marker_color='red', 
            opacity=0.6,
            nbinsx=30
        ))
        
        fig_risk.update_layout(
            title="Forecasted Return Distribution",
            xaxis_title="Predicted Payoff ($)",
            yaxis_title="Probability Density",
            barmode='overlay',
            height=350,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        risk_chart.plotly_chart(fig_risk, use_container_width=True)

        time.sleep(simulation_speed)