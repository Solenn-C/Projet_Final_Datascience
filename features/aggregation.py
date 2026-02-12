import pandas as pd

# 1. Chargement des données
# On utilise 'Datetime' comme index temporel
path_m1 = 'data/gbpusd_m1_cleaned.csv'
df = pd.read_csv(path_m1, parse_dates=['Datetime'], index_col='Datetime')

# 2. Phase 2 – Agrégation M1 → M15
df_m15 = df.resample('15Min').agg({
    'Open': 'first',   # open 1ère minute du bloc
    'High': 'max',     # max des High sur 15 min
    'Low': 'min',      # min des Low sur 15 min
    'Close': 'last',   # close dernière minute du bloc
    'Volume': 'sum'    # (Optionnel) Somme des volumes sur 15 min
})

# 3. Sauvegarde
# Le fichier final sera enregistré dans le dossier features/
df_m15.to_csv('data/gbpusd_m15.csv')

print("Phase 2 terminée avec succès !")
print(df_m15.head())