import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, batch_size, device):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(state)).to(self.device),
            torch.LongTensor(np.array(action)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_state)).to(self.device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)