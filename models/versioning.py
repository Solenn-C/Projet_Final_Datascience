import joblib
import json
import os
from xgboost import XGBClassifier

def save_model_to_registry(model, version, metrics):
    """Sauvegarde le modèle et met à jour le registre JSON."""
    folder = f"models/{version}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 1. Sauvegarde du fichier modèle
    model_path = f"{folder}/model.pkl"
    joblib.dump(model, model_path)
    
    # 2. Mise à jour du registre
    registry_path = "models/registry.json"
    registry = {}
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    
    registry[version] = {
        "path": model_path,
        "metrics": metrics,
        "status": "production" if version == "v2" else "archive"
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    
    print(f"Modèle {version} enregistré dans le registry.")

if __name__ == "__main__":
    # Exemple : On enregistre ton XGBoost comme la V2 (Production)
    # Dans un vrai workflow, tu passerais ton objet modèle entraîné ici
    print("Initialisation du versioning...")
    # On simule un dictionnaire de métriques basées sur tes résultats 2024
    metrics_v2 = {
        "profit": "2.06%",
        "sharpe": 0.63,
        "profit_factor": 1.02
    }
    # Ici on crée un modèle vide juste pour tester la structure, 
    # mais en pratique tu utiliseras ton modèle entraîné.
    dummy_model = XGBClassifier() 
    save_model_to_registry(dummy_model, "v2", metrics_v2)


registry_path = "models/registry.json"

if os.path.exists(registry_path):
    with open(registry_path, 'r') as f:
        registry = json.load(f)

# Ajout de la V1 (RSI)
registry["v1"] = {
    "path": "models/v1/rsi_logic.py", # Ou juste un indicateur
    "metrics": {
        "profit": "4.81%",
        "sharpe": 1.33,
        "profit_factor": 1.04
    },
    "status": "archive"
}

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=4)

print("✅ v1 (RSI) ajoutée au registre en tant qu'archive.")