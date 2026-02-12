Élément,Proposition de configuration
1. Problème métier,Prédire la direction du GBP/USD à 15min pour maximiser le gain net.
2. Données,"M15 nettoyées, alignées (T03), avec coûts de transaction inclus."
3. State (État),"Fenêtre glissante de prix + indicateurs (RSI, Volatilité) + Position actuelle."
4. Action,"Discret : [0: Rester flat, 1: Acheter (Long), 2: Vendre (Short)]."
5. Reward,"PnL de la bougie, ajusté par une pénalité si l'agent change trop souvent d'avis."
6. Environnement,Simulateur custom incluant le Slippage et le Spread (coûts).
7. Algorithme,DQN (Deep Q-Network) ou PPO (justifié par la nature temporelle).