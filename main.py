from fastapi import FastAPI
from app.routers import predict

app = FastAPI(
    title="Trading API - Projet Final Solenn & Coline",
    version="1.0.0",
    description="API de prédiction GBP/USD avec PPO et RSI"
)

# Inclusion des routes avec le préfixe de versioning
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
    # Lancement du serveur
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)