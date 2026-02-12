from fastapi import APIRouter, HTTPException
from app.schemas.trading import PredictionRequest
from app.services.model_service import model_service

router = APIRouter()

@router.post("/predict")
async def get_prediction(data: PredictionRequest):
    # 1. On passe les données au service
    prediction = model_service.predict(data)
    
    # 2. Sécurité
    if prediction is None:
        raise HTTPException(status_code=500, detail="Erreur de calcul interne")
    
    # 3. Mapping des labels pour gérer le HOLD
    labels = {
        0: "SELL",
        1: "BUY",
        2: "HOLD"
    }
    
    return {
        "status": "success",
        "model_version": model_service.model_version,
        "prediction": {
            "action_code": prediction,
            "label": labels.get(prediction, "UNKNOWN")
        },
        "input_received": {
            "rsi": data.rsi, 
            "close": data.close
        }
    }