import os
import pandas as pd

def nettoyage_complet_m15(path_in):
    print(f"Tentative d'ouverture du fichier : {path_in}")
    if not os.path.exists(path_in):
        print(f"ERREUR : Le fichier {path_in} n'existe pas !") # <-- AJOUTE ÇA
        return
    # Chargement
    df = pd.read_csv(path_in, parse_dates=['Datetime'], index_col='Datetime')
    print("Fichier chargé avec succès, début du nettoyage...")
    print(f"--- Analyse Phase 3 : {path_in} ---")


    
    # 1. Suppression des bougies incomplètes (si 'count' < 15)
    if 'count' in df.columns:
        nb_avant = len(df)
        df = df[df['count'] == 15]
        print(f"[*] Bougies incomplètes supprimées : {nb_avant - len(df)}")
    
    # 2. Contrôle prix négatifs
    cols = ['Open', 'High', 'Low', 'Close']
    mask_neg = (df[cols] <= 0).any(axis=1)
    if mask_neg.any():
        print(f"[*] Suppression de {mask_neg.sum()} lignes avec prix <= 0")
        df = df[~mask_neg]

    # 3. Détection gaps anormaux (plus de 15 min en semaine)
    diffs = df.index.to_series().diff().dt.total_seconds()
    # On filtre : écart > 15min ET jour de semaine (0=Lundi, 4=Vendredi)
    gaps = df[(diffs > 900) & (df.index.weekday < 5)]
    print(f"[*] Nombre de gaps anormaux détectés : {len(gaps)}")
    
    # Sauvegarde finale
    output = "data/GBPUSD_M15_CLEANED.csv"
    df.to_csv(output)
    print(f"✅ Fichier final sauvegardé : {output}")

if __name__ == "__main__":
    nettoyage_complet_m15("data/gbpusd_m15.csv")