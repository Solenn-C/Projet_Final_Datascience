# Conception Reinforcement Learning - GBP/USD

## 9.1 Conception Stratégique
1. **Problème métier** : Maximiser le rendement net sur le GBP/USD (M15) en gérant le risque de drawdown.
2. **Données** : Prix OHLC + Indicateurs (T03/T04). Qualité vérifiée lors de l'EDA.
3. **State (État)** : Fenêtre de 10 bougies (Close, RSI, Volatilité) + Position actuelle (0 ou 1).
4. **Action** : Discret (0: Flat, 1: Long, 2: Short).
5. **Reward** : Rendement logarithmique de la période, pénalité de -0.01 pour chaque transaction (coût).
6. **Environnement** : Simulateur sur mesure avec gestion des spreads et frais.
7. **Algorithme** : **DQN** (Deep Q-Network). *Justification : Adapté aux espaces d'actions discrets et permet de modéliser des décisions basées sur l'historique récent.*

## 9.2 Paramètres clés
- **Entraînement** : Gamma = 0.99, LR = 0.001, Epsilon-decay (1.0 -> 0.01).
- **Évaluation** : Split temporel (80% Train / 20% Test), Ratio de Sharpe, Max Drawdown.