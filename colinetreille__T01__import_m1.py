import pandas as pd
import numpy as np

def phase_1_import_and_clean(file_paths):
    all_dfs = []
    columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for path in file_paths:
        df = pd.read_csv(path, names=columns, header=None)
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        all_dfs.append(df)
    
    df = pd.concat(all_dfs)
    
    # 1. TRI CHRONOLOGIQUE
    # Indispensable avant de vérifier la régularité
    df = df.sort_values('Datetime').reset_index(drop=True)
    df.set_index('Datetime', inplace=True)
    
    print(f"--- Phase 1 : Rapport de Qualité GBP/USD ---")
    
    # 2. DÉTECTION DES INCOHÉRENCES (Prix)
    # Le High doit être le max, le Low le min.
    mask_incoherence = (
        (df['High'] < df['Low']) | 
        (df['Open'] > df['High']) | 
        (df['Open'] < df['Low']) |
        (df['Close'] > df['High']) | 
        (df['Close'] < df['Low'])
    )
    nb_incoherences = mask_incoherence.sum()
    print(f"[!] Incohérences de prix détectées : {nb_incoherences}")
    
    # Correction : on force le High/Low si l'erreur est minime, ou on supprime
    if nb_incoherences > 0:
        # Ici on fait le choix de supprimer les lignes aberrantes pour ne pas polluer le ML
        df = df[~mask_incoherence]
        print("    -> Lignes incohérentes supprimées.")

    # 3. VÉRIFICATION RÉGULARITÉ (1 minute)
    # On calcule la différence de temps entre chaque ligne
    time_diffs = df.index.to_series().diff().dt.total_seconds()
    
    # Une série M1 parfaite a un écart de 60s (sauf week-end)
    gaps = time_diffs[time_diffs > 60]
    
    print(f"[!] Trous (gaps) dans les données : {len(gaps)} interruptions détectées.")
    print(f"    -> Saut maximal : {gaps.max()/3600:.2f} heures (probablement des week-ends).")
    
    # Vérification des doublons de temps
    duplicates = df.index.duplicated().sum()
    print(f"[!] Doublons de timestamps : {duplicates}")
    if duplicates > 0:
        df = df[~df.index.duplicated(keep='first')]

    return df

# --- Exécution ---
files = [
    'C:/Users/cocop/Desktop/SUP_DE_VINCI/DataScience/Projet Final/data/DAT_MT_GBPUSD_M1_2022.csv',
    'C:/Users/cocop/Desktop/SUP_DE_VINCI/DataScience/Projet Final/data/DAT_MT_GBPUSD_M1_2023.csv',
    'C:/Users/cocop/Desktop/SUP_DE_VINCI/DataScience/Projet Final/data/DAT_MT_GBPUSD_M1_2024.csv'
]
clean_df = phase_1_import_and_clean(files)

# Sauvegarde pour l'Étudiant B
clean_df.to_csv('C:/Users/cocop/Desktop/SUP_DE_VINCI/DataScience/Projet Final/GBPUSD_M1_cleaned.csv')