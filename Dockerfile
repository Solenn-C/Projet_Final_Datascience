FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour le calcul scientifique
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le code source
COPY . .

# Création des dossiers de sortie pour éviter les erreurs d'écriture
RUN mkdir -p analyse models data

# Par défaut, on lance le backtest, mais on pourra surcharger cette commande
CMD ["python", "features/backtest_rl.py"]