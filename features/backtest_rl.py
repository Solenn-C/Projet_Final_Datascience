import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from rl_env import TradingEnv

# 1. Préparation des données (Identique à l'entraînement)
df_raw = pd.read_csv('data/GBPUSD_M15_CLEANED.csv')
features = ['Open', 'High', 'Low', 'Close']
df = df_raw[features].copy()
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
df_norm = (df - df.min()) / (df.max() - df.min() + 1e-5)

# 2. Chargement du modèle
if not os.path.exists("models/ppo_gbpusd_agent.zip"):
    print("❌ Erreur : Le fichier models/ppo_gbpusd_agent.zip n'existe pas !")
else:
    print("🤖 Chargement du modèle ppo_gbpusd_agent...")
    model = PPO.load("models/ppo_gbpusd_agent")
    env = TradingEnv(df_norm)

    # 3. Simulation
    obs, _ = env.reset()
    history = []
    actions = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        history.append(reward)
        actions.append(action)
        done = terminated or truncated

    # 4. Calcul performance (On divise par 100 car reward était multiplié par 100)
    cumulative_return = np.cumsum(history) / 100.0

    # 5. Affichage
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_return, label='Performance Agent V1', color='green')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title(f'Backtest RL - Actions: {np.count_nonzero(actions)}')
    plt.xlabel('Bougies M15')
    plt.ylabel('Profit Cumulé')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if not os.path.exists('analyse'):
        os.makedirs('analyse')
    plt.savefig('analyse/performance_v1_stable.png')
    print("📈 Graphique généré dans analyse/performance_v1_stable.png")
    plt.show()