import numpy as np
import pandas as pd

class TradingEnv:
    def __init__(self, df, initial_balance=1000, transaction_fee=0.0001):
        self.df = df.reset_index()
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        self.fee = transaction_fee
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0  # 0: Flat, 1: Long, 2: Short
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        # On donne à l'agent les prix (Point 3 de ta conception : State)
        return self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close']].values

    def step(self, action):
        # Calcul du rendement (Point 5 : Reward)
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        step_return = (next_price - current_price) / current_price

        reward = 0
        if action == 1: # Long
            reward = step_return - self.fee
        elif action == 2: # Short
            reward = -step_return - self.fee
        
        self.current_step += 1
        if self.current_step >= self.n_steps - 2:
            self.done = True

        return self._get_observation(), reward, self.done

print("✅ Structure de l'environnement RL prête.")