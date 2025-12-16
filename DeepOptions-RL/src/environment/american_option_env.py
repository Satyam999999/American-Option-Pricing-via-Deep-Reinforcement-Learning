import gymnasium as gym
import numpy as np
from src.environment.gbm import GBMGenerator

class AmericanOptionEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.generator = GBMGenerator(config)
        self.action_space = gym.spaces.Discrete(2) # 0: Hold, 1: Exercise
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.price_path = self.generator.generate_path()
        self.current_step = 0
        self.done = False
        return self._get_state(), {}

    def _get_state(self):
        price = self.price_path[self.current_step]
        time_left = self.config['simulation']['t'] - (self.current_step * (self.config['simulation']['t'] / self.config['simulation']['n_steps']))
        
        # Normalize: Moneyness (S/K) and Time
        norm_price = price / self.config['simulation']['k']
        return np.array([norm_price, time_left], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, False, {}

        current_price = self.price_path[self.current_step]
        reward = 0
        
        # Action 1: Exercise
        if action == 1:
            payoff = max(self.config['simulation']['k'] - current_price, 0)
            # Discount reward back to time 0 logic handled by agent/Bellman, 
            # but environment returns immediate payoff.
            reward = payoff 
            self.done = True
            
        # Action 0: Hold
        elif action == 0:
            reward = 0
            self.current_step += 1
            if self.current_step >= len(self.price_path) - 1:
                # Forced exercise at maturity
                final_price = self.price_path[-1]
                reward = max(self.config['simulation']['k'] - final_price, 0)
                self.done = True

        return self._get_state(), reward, self.done, False, {}