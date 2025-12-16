import numpy as np
from scipy.stats import norm

class EuropeanOptionPricing:
    def __init__(self, config):
        self.S = config['simulation']['s0']
        self.K = config['simulation']['k']
        self.T = config['simulation']['t']
        self.r = config['simulation']['r']
        self.sigma = config['simulation']['sigma']

    def black_scholes_price(self):
        """Calculates exact European Put price using Black-Scholes formula."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        put_price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) - (self.S * norm.cdf(-d1))
        return put_price