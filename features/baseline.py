import pandas as pd
import numpy as np

# 1. Chargement des données
df = pd.read_csv('data/gbpusd_m15.csv', parse_dates=['Datetime'], index_col='Datetime')

# Calcul du rendement simple (log returns pour la précision financière)
df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# --- STRATÉGIE 1 : Buy & Hold ---
# On achète au début, on vend à la fin
df['Strategy_BuyHold'] = df['Returns'].cumsum()

# --- STRATÉGIE 2 : Aléatoire (Random) ---
# On pile ou face à chaque bougie (1 = Achat, -1 = Vente)
np.random.seed(42) # Pour que le résultat soit reproductible
df['Random_Signal'] = np.random.choice([1, -1], size=len(df))
df['Strategy_Random'] = (df['Random_Signal'] * df['Returns']).cumsum()

# --- STRATÉGIE 3 : Règles fixes (Exemple : Moyenne Mobile Rapide) ---
# Si le prix est > à la moyenne 20, on achète, sinon on vend (Trend Following basique)
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['Rule_Signal'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
df['Strategy_Rules'] = (df['Rule_Signal'].shift(1) * df['Returns']).cumsum()

# 3. Affichage des résultats finaux
print("--- PERFORMANCES DES BASELINES (Rendements cumulés) ---")
print(f"Buy & Hold      : {df['Strategy_BuyHold'].iloc[-1]:.4f}")
print(f"Aléatoire       : {df['Strategy_Random'].iloc[-1]:.4f}")
print(f"Règles (SMA20)  : {df['Strategy_Rules'].iloc[-1]:.4f}")

# Optionnel : Sauvegarder un graphique pour comparer
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(df['Strategy_BuyHold'], label='Buy & Hold')
plt.plot(df['Strategy_Random'], label='Aléatoire')
plt.plot(df['Strategy_Rules'], label='Règles Fixes (SMA)')
plt.legend()
plt.title('Comparaison des Baselines - GBP/USD')
plt.savefig('analyse/baselines_comparison.png')
print("✅ Graphique de baseline enregistré dans analyse/")