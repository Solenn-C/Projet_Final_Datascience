import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# 1. Chargement et préparation
df = pd.read_csv('features/gbpusd_m15.csv', parse_dates=['Datetime'], index_col='Datetime')

# Calcul des rendements log (plus robustes pour l'analyse statistique)
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# --- A. DISTRIBUTION DES RENDEMENTS ---
plt.figure(figsize=(10, 5))
sns.histplot(df['returns'], kde=True, bins=100)
plt.title('Distribution des Rendements (GBP/USD M15)')
plt.show()
# Note : Cherche si la distribution est "normale" ou si elle a des "queues épaisses" (fat tails)

# --- B. VOLATILITÉ DANS LE TEMPS ---
# Calcul de la volatilité roulante sur 24h (96 bougies de 15 min)
df['volatility'] = df['returns'].rolling(window=96).std()
plt.figure(figsize=(12, 5))
plt.plot(df['volatility'], color='orange')
plt.title('Volatilité Roulante (Fenêtre de 24h)')
plt.show()

# --- C. ANALYSE HORAIRE (Saisonnalité) ---
# On regarde à quel moment de la journée la volatilité est la plus forte
df['hour'] = df.index.hour
plt.figure(figsize=(10, 5))
df.groupby('hour')['returns'].std().plot(kind='bar', color='skyblue')
plt.title('Volatilité moyenne par heure de la journée')
plt.ylabel('Écart-type des rendements')
plt.show()
# Note : Tu devrais voir des pics lors de l'ouverture de Londres (8h-9h) et New York (14h-15h)

# --- D. AUTOCORRÉLATION (ACF) ---
plt.figure(figsize=(10, 5))
plot_acf(df['returns'], lags=50)
plt.title('Autocorrélation des Rendements')
plt.show()
# Note : Si les barres dépassent la zone bleue, il y a une corrélation temporelle exploitable

# --- E. TEST ADF (Stationnarité) ---
print("\n--- TEST DE STATIONNARITÉ (ADF) ---")
result = adfuller(df['returns'])
print(f'Statistique ADF : {result[0]:.4f}')
print(f'p-value : {result[1]:.4e}')

if result[1] < 0.05:
    print("Résultat : La série est STATIONNAIRE (H0 rejetée). Prête pour le ML.")
else:
    print("Résultat : La série n'est pas stationnaire. Vérifie tes calculs.")