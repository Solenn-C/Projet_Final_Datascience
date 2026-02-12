import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        # Actions : 0 = Rien, 1 = Achat, 2 = Vente
        self.action_space = spaces.Discrete(3)
        # Observation : Open, High, Low, Close + Position Actuelle (5 valeurs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # On prend les 4 colonnes de prix à l'étape actuelle
        obs = self.df.iloc[self.current_step].values[:4].astype(np.float32)
        # On ajoute la position actuelle
        return np.append(obs, [float(self.position)]).astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        
        # Calcul du rendement (variation du prix)
        returns = (next_price - current_price) / (current_price + 1e-7)
        
        # Sécurité anti-NaN
        if np.isnan(returns) or np.isinf(returns):
            returns = 0.0

        reward = 0
        if action == 1:   # Achat
            reward = returns
        elif action == 2: # Vente
            reward = -returns
            
        # Amplification du signal pour l'IA
        reward *= 100.0
        
        # Clip pour éviter l'explosion des gradients
        reward = np.clip(reward, -10, 10)
        
        self.position = action
        self.current_step += 1
        
        # Fin si on arrive au bout des données
        terminated = self.current_step >= len(self.df) - 2
        return self._get_obs(), float(reward), terminated, False, {}