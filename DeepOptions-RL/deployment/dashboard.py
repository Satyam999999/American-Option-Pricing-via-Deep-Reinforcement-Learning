import streamlit as st
import time
import yaml
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.agents.dqn_agent import DQNAgent

st.set_page_config(page_title="DeepOptions DQN", layout="wide")

@st.cache_resource
def load_agent():
    with open("experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    agent = DQNAgent(2, 2, config)
    agent.policy_net.load_state_dict(torch.load("checkpoints/dqn_option.pth", map_location=agent.device))
    agent.policy_net.eval()
    return agent, config

agent, config = load_agent()
st.title("🚀 DeepOptions: AI Option Pricing (DQN)")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Start Simulation"):
        st.session_state.run = True
    if st.button("Stop"):
        st.session_state.run = False

chart = col1.empty()
price_history = []

if 'run' in st.session_state and st.session_state.run:
    while True:
        price = np.random.uniform(80, 120)
        time_left = np.random.uniform(0.1, 1.0)
        norm_price = price / config['simulation']['k']
        state = np.array([norm_price, time_left], dtype=np.float32)
        
        with torch.no_grad():
            q = agent.policy_net(torch.FloatTensor(state).unsqueeze(0).to(agent.device))
            action = q.argmax().item()
        
        price_history.append({"Price": price, "Action": action})
        if len(price_history) > 50: price_history.pop(0)
        
        df = pd.DataFrame(price_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['Price'], mode='lines+markers'))
        
        # Mark Exercises
        exercises = df[df['Action'] == 1]
        if not exercises.empty:
            fig.add_trace(go.Scatter(x=exercises.index, y=exercises['Price'], mode='markers', marker=dict(color='red', size=12)))
            
        chart.plotly_chart(fig)
        time.sleep(0.5)