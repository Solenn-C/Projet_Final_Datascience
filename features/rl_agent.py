import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from rl_env import TradingEnv

# 1. Création du dossier modèle
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Chargement et Nettoyage des données
print("📊 Chargement des données...")
df_raw = pd.read_csv('data/GBPUSD_M15_CLEANED.csv')
features = ['Open', 'High', 'Low', 'Close']
df = df_raw[features].copy()

# Suppression des valeurs aberrantes et remplissage des vides
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

# Normalisation Min-Max robuste
df_norm = (df - df.min()) / (df.max() - df.min() + 1e-5)

print(f"✅ Données prêtes ({len(df_norm)} lignes).")

# 3. Configuration de l'environnement et du modèle
env = TradingEnv(df_norm)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    gamma=0.99,
    ent_coef=0.02, # Un peu de curiosité pour éviter le plat
    seed=42
)

# 4. Entraînement
print("🚀 Lancement de l'entraînement...")
model.learn(total_timesteps=100000)

# 5. Sauvegarde
model.save("models/ppo_gbpusd_agent")
print("✅ Modèle sauvegardé : models/ppo_gbpusd_agent.zip")