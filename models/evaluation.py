import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# Import pour le modèle de Solenn
try:
    from stable_baselines3 import PPO
except ImportError:
    print("SB3 non installé. Tapez 'pip install stable-baselines3'")

def calculate_metrics(returns):
    if returns.empty or returns.std() == 0: return 0, 0, 0, 0
    cum_ret = (1 + returns.fillna(0)).cumprod()
    total_profit = cum_ret.iloc[-1] - 1
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret / peak) - 1
    max_dd = drawdown.min()
    sharpe = np.sqrt(25000) * (returns.mean() / returns.std())
    gains = returns[returns > 0].sum()
    pertes = abs(returns[returns < 0].sum())
    pf = gains / pertes if pertes != 0 else gains
    return total_profit, max_dd, sharpe, pf

def run_final_evaluation(input_path):
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier {input_path} est introuvable.")
        return

    df = pd.read_csv(input_path, parse_dates=['Datetime'], index_col='Datetime').sort_index()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df.dropna(inplace=True)

    train_df = df[df.index.year == 2022].copy()
    test_df  = df[df.index.year == 2024].copy()

    # 1. RANDOM
    np.random.seed(42)
    test_df['pred_random'] = np.random.randint(0, 2, size=len(test_df))

    # 2. RÈGLES (RSI)
    test_df['pred_rules'] = (test_df['rsi_14'] < 50).astype(int)

    # 3. ML (XGBoost)
    cols_to_exclude = ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'return_1']
    X_train = train_df.drop(columns=cols_to_exclude)
    y_train = train_df['target']
    X_test = test_df.drop(columns=cols_to_exclude + ['pred_random', 'pred_rules'])
    
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    test_df['pred_ml'] = xgb.predict(X_test)

   # --- 4. RL (Modèle de Solenn) ---
    test_df['pred_rl'] = 0 
    model_file = "models/ppo_gbpusd_agent.zip"
    
    if os.path.exists(model_file):
        try:
            # 1. Préparation des 5 features spécifiques demandées par son modèle
            # On les calcule sur test_df pour être sûr d'avoir les bonnes valeurs en 2024
            rl_features = pd.DataFrame(index=test_df.index)
            rl_features['log_return'] = np.log(test_df['Close'] / test_df['Close'].shift(1))
            rl_features['volatility'] = rl_features['log_return'].rolling(window=20).std()
            
            roll_min = test_df['Close'].rolling(window=20).min()
            roll_max = test_df['Close'].rolling(window=20).max()
            rl_features['relative_close'] = (test_df['Close'] - roll_min) / (roll_max - roll_min)
            
            rl_features['ma_ratio'] = test_df['Close'] / test_df['Close'].rolling(window=20).mean()
            rl_features['hour'] = test_df.index.hour / 24.0
            
            # On remplit les quelques NaN du début (dus au rolling 20) pour ne pas faire planter le RL
            rl_features = rl_features.ffill().bfill() 

            # 2. Chargement et Prédiction
            rl_model = PPO.load(model_file.replace(".zip", ""))
            
            # On envoie EXACTEMENT les 5 colonnes dans l'ordre de son code
            obs_rl = rl_features[['log_return', 'volatility', 'relative_close', 'ma_ratio', 'hour']].values
            
            preds, _ = rl_model.predict(obs_rl, deterministic=True)
            test_df['pred_rl'] = preds
            print(f" RL appliqué avec succès sur les features de Solenn.")
            
        except Exception as e:
            print(f" Erreur lors de l'application du RL : {e}")

    # Calcul et affichage
    strategies = {'Random': 'pred_random', 'RSI_Rules': 'pred_rules', 'XGBoost_ML': 'pred_ml', 'PPO_RL': 'pred_rl'}
    summary = []
    plt.figure(figsize=(10, 6))

    for name, col in strategies.items():
        ret_col = f'ret_{name}'
        test_df[ret_col] = test_df[col] * test_df['return_1'].shift(-1)
        p, m, s, pf = calculate_metrics(test_df[ret_col])
        summary.append([name, f"{p:.2%}", f"{m:.2%}", f"{s:.2f}", f"{pf:.2f}"])
        plt.plot((1 + test_df[ret_col].fillna(0)).cumprod(), label=f"{name} (PF: {pf:.2f})")

    print("\n--- RÉSULTATS PHASES 9 (TEST 2024) ---")
    print(pd.DataFrame(summary, columns=['Modèle', 'Profit', 'MaxDD', 'Sharpe', 'PF']))
    
    plt.plot((1 + test_df['return_1'].fillna(0)).cumprod(), color='black', linestyle='--', label='Market')
    plt.legend(); plt.grid(True); plt.show()

if __name__ == "__main__":
    run_final_evaluation("data/gbpusd_m15_features_v2.csv")