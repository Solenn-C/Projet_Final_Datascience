import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# --- 0. CRÉATION ET VÉRIFICATION DU DOSSIER ---
output_dir = 'analyse'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ Dossier '{output_dir}' créé.")

# 1. Chargement
try:
    df = pd.read_csv('data/GBPUSD_M15_CLEANED.csv', parse_dates=['Datetime'], index_col='Datetime')
    print("✅ Fichier CSV chargé avec succès.")
except FileNotFoundError:
    print("❌ Erreur : Le fichier 'data/GBPUSD_M15_CLEANED.csv' est introuvable !")
    exit()

df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# --- A. DISTRIBUTION ---
plt.figure(figsize=(10, 5))
sns.histplot(df['returns'], kde=True, bins=100)
plt.title('Distribution des Rendements (GBP/USD M15)')
path_a = f'{output_dir}/distribution_rendements.png'
plt.savefig(path_a)
plt.close()
print(f"📊 Graphique Distribution enregistré : {path_a}")

# --- B. VOLATILITÉ ---
df['volatility'] = df['returns'].rolling(window=96).std()
plt.figure(figsize=(12, 5))
plt.plot(df['volatility'], color='orange')
plt.title('Volatilité Roulante (Fenêtre de 24h)')
path_b = f'{output_dir}/volatilit_temporelle.png'
plt.savefig(path_b)
plt.close()
print(f"📊 Graphique Volatilité enregistré : {path_b}")

# --- C. SAISONNALITÉ ---
df['hour'] = df.index.hour
plt.figure(figsize=(10, 5))
df.groupby('hour')['returns'].std().plot(kind='bar', color='skyblue')
plt.title('Volatilité moyenne par heure de la journée')
path_c = f'{output_dir}/saisonnalite_horaire.png'
plt.savefig(path_c)
plt.close()
print(f"📊 Graphique Saisonnalité enregistré : {path_c}")

# --- D. AUTOCORRÉLATION ---
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(df['returns'], lags=50, ax=ax)
plt.title('Autocorrélation des Rendements')
path_d = f'{output_dir}/autocorrelation.png'
plt.savefig(path_d)
plt.close()
print(f"📊 Graphique ACF enregistré : {path_d}")

# --- E. TEST ADF (Stationnarité) ---
result = adfuller(df['returns'])
path_e = f'{output_dir}/test_stationnarite.txt'
with open(path_e, 'w') as f:
    f.write(f'Statistique ADF : {result[0]:.4f}\n')
    f.write(f'p-value : {result[1]:.4e}\n')
print(f"📄 Résultats du test ADF enregistrés : {path_e}")

# --- RÉCAPITULATIF FINAL ---
print("\n" + "="*30)
files_to_check = [path_a, path_b, path_c, path_d, path_e]
missing_files = [f for f in files_to_check if not os.path.exists(f)]

if not missing_files:
    print("✨ TOUT EST BON ! Tous les fichiers sont dans le dossier /analyse.")
    print("Tu peux maintenant faire : git add analyse/ ")
else:
    print(f"⚠️ Attention, fichiers manquants : {missing_files}")
print("="*30)