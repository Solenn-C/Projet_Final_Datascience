class ModelService:
    def __init__(self):
        # On définit la version du modèle ici pour le suivi
        self.model_version = "v1.0.0_rsi_threshold"
        self.is_ready = True
        print(f"✅ Service chargé : Utilisation du modèle {self.model_version}")

    def predict(self, data):
        try:
            rsi_value = data.rsi
            
            # --- LOGIQUE AVEC POSITION HOLD ---
            if rsi_value < 40:
                return 1  # BUY
            elif rsi_value > 60:
                return 0  # SELL
            else:
                return 2  # HOLD (Zone entre 40 et 60)
            # ----------------------------------
            
        except Exception as e:
            print(f"❌ Erreur : {e}")
            return None

# On crée l'instance prête à l'emploi
model_service = ModelService()