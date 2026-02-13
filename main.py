from fastapi import FastAPI
from app.routers import predict

app = FastAPI(
    title="Trading API - Projet Final Solenn & Coline",
    version="1.0.0",
    description="API de prédiction GBP/USD avec PPO et RSI"
)

# Ton préfixe ici crée l'URL : http://127.0.0.1:8000/api/v1/...
app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "message": "API opérationnelle",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)