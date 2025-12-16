import numpy as np

class GBMGenerator:
    def __init__(self, config):
        self.s0 = config['simulation']['s0']
        self.t = config['simulation']['t']
        self.r = config['simulation']['r']
        self.sigma = config['simulation']['sigma']
        self.n_steps = config['simulation']['n_steps']
        self.dt = self.t / self.n_steps

    def generate_path(self):
        """Generates a single price path using Geometric Brownian Motion."""
        path = np.zeros(self.n_steps + 1)
        path[0] = self.s0
        
        for i in range(1, self.n_steps + 1):
            z = np.random.standard_normal()
            path[i] = path[i-1] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt + 
                                         self.sigma * np.sqrt(self.dt) * z)
        return path