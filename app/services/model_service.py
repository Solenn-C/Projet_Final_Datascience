class ModelService:
    def __init__(self):
        # On définit la version du modèle ici pour le suivi
        self.model_version = "v1.0.0_rsi_threshold"
        self.is_ready = True
        print(f"✅ Service chargé : Utilisation du modèle {self.model_version}")

    def predict(self, data):
        """
        Logique extraite de models/evaluation.py :
        La règle validée est : si RSI < 50 alors ACHAT (1), sinon VENTE (0).
        """
        try:
            rsi_value = data.rsi
            
            # Application de la règle du binôme
            if rsi_value < 50:
                action = 1  # Signal d'Achat
            else:
                action = 0  # Signal de Vente
            
            return action
        except Exception as e:
            print(f"❌ Erreur lors du calcul : {e}")
            return None

# On crée l'instance prête à l'emploi
model_service = ModelService()