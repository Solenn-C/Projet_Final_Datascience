import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from stable_baselines3 import PPO # Choix de l'algorithme (Point 7)

# --- CONFIGURATION (Point 9.2) ---
CONFIG = {
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "batch_size": 64,
    "n_epochs": 10,
    "total_timesteps": 50000,
    "transaction_fee": 0.0001, # Point 6 : Coûts
    "seed": 42
}

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        
        # Actions : 0=Flat, 1=Long, 2=Short (Point 4)
        self.action_space = spaces.Discrete(3)
        
        # State : OHLC + Position (Point 3)
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
        obs = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close']].values
        return np.append(obs, [self.position]).astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        
        # Calcul du rendement (Point 5 : Reward)
        returns = (next_price - current_price) / current_price
        
        reward = 0
        if action == 1: # Long
            reward = returns - CONFIG["transaction_fee"]
        elif action == 2: # Short
            reward = -returns - CONFIG["transaction_fee"]
        
        self.position = action
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 2
        
        return self._get_obs(), reward, terminated, False, {}

# --- RÉALISATION ---

# 1. Chargement des données (Assure-toi que le chemin est correct)
df = pd.read_csv('data/gbpusd_m15.csv')

# 2. Initialisation de l'environnement
env = TradingEnv(df)

# 3. Création de l'Agent PPO (Point 7 : Justification - Stabilité)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    gamma=CONFIG["gamma"],
    learning_rate=CONFIG["learning_rate"],
    batch_size=CONFIG["batch_size"],
    n_epochs=CONFIG["n_epochs"],
    seed=CONFIG["seed"]
)

# 4. Entraînement
print("🚀 Début de l'entraînement de l'agent RL...")
model.learn(total_timesteps=CONFIG["total_timesteps"])

# 5. Sauvegarde
model.save("models/ppo_gbpusd_agent")
print("✅ Modèle sauvegardé dans models/ppo_gbpusd_agent")