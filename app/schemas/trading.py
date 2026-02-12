from pydantic import BaseModel

class PredictionRequest(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: float  # C'est cette valeur qui sera utilisée par le modèle