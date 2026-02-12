import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def calculate_financial_metrics(returns):
    """Calcule le Rendement, le Max Drawdown et le Ratio de Sharpe."""
    # Rendement cumulé
    cum_ret = (1 + returns.fillna(0)).cumprod()
    total_return = cum_ret.iloc[-1]
    
    # Max Drawdown (Pire chute du capital)
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret / peak) - 1
    max_dd = drawdown.min()
    
    # Ratio de Sharpe (Annualisé pour du M15 : env. 25000 bougies de trading par an)
    # Formule : Moyenne des rendements / Ecart-type des rendements
    if returns.std() != 0:
        sharpe = np.sqrt(25000) * (returns.mean() / returns.std())
    else:
        sharpe = 0
        
    return total_return, max_dd, sharpe


def run_ml_pipeline(input_path):
    # 1. Chargement et tri
    df = pd.read_csv(input_path, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()

    # --- CRÉATION DE LA CIBLE (y) ---
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # --- AJOUT DES TIME FEATURES ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    df.dropna(inplace=True)

    # --- 2. SPLIT TEMPOREL STRICT (2022 Train, 2023 Val, 2024 Test) ---
    train_df = df[df.index.year == 2022].copy()
    val_df   = df[df.index.year == 2023].copy()
    test_df  = df[df.index.year == 2024].copy()

    cols_to_exclude = ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'return_1']
    
    # Sets d'entraînement
    X_train = train_df.drop(columns=cols_to_exclude)
    y_train = train_df['target']
    
    # Sets de Validation (2023)
    X_val = val_df.drop(columns=cols_to_exclude)
    y_val = val_df['target']
    
    # Sets de Test (2024)
    X_test = test_df.drop(columns=cols_to_exclude)
    y_test = test_df['target']

    print(f"Structure des données :")
    print(f"- Entraînement (2022) : {len(X_train)} lignes")
    print(f"- Validation    (2023) : {len(X_val)} lignes")
    print(f"- Test Final    (2024) : {len(X_test)} lignes")

    # --- 3. MODÈLES ---
    models = {
        "Baseline (LogReg)": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost_Tuned": XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    }

    # --- 4. ÉVALUATION ET MÉTRIQUES ---
    plt.figure(figsize=(12, 8))
    

    for name, model in models.items():
        # Entraînement sur 2022 uniquement
        model.fit(X_train, y_train)
        
        # Validation sur 2023 (pour vérifier si le modèle généralise bien)
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Test sur 2024 (Le verdict final)
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        
        # Calcul des métriques statistiques
        print(f"========================================")
        print(f" MODÈLE : {name}")
        print(f"========================================")
        print(f"Accuracy VAL (2023) : {val_acc:.4f}")
        print(f"Accuracy TEST (2024): {test_acc:.4f}")
        print("\nClassification Report (Test 2024) :")
        print(classification_report(y_test, y_test_pred))
        
        # Calcul des métriques financières sur 2024
        # On multiplie la prédiction (0 ou 1) par le rendement réel de la bougie suivante
        test_df[f'strat_{name}'] = y_test_pred * test_df['return_1'].shift(-1)
        
        total_ret, mdd, sharpe = calculate_financial_metrics(test_df[f'strat_{name}'])
        
        print(f"--- MÉTRIQUES FINANCIÈRES (2024) ---")
        print(f"Rendement Cumulé : {total_ret:.4f}")
        print(f"Max Drawdown     : {mdd:.2%}")
        print(f"Ratio de Sharpe  : {sharpe:.2f}\n")
        
        # Plot de la stratégie
        cum_ret_curve = (1 + test_df[f'strat_{name}'].fillna(0)).cumprod()
        plt.plot(cum_ret_curve, label=f'{name} (Ret: {total_ret:.3f}, Sharpe: {sharpe:.2f})')

    # Ajout du Buy & Hold (Marché) pour comparaison
    test_df['buy_and_hold'] = (1 + test_df['return_1'].fillna(0)).cumprod()
    plt.plot(test_df['buy_and_hold'], label=f'Buy & Hold ({test_df["buy_and_hold"].iloc[-1]:.3f})', color='black', linestyle='--')

    # --- 5. FINALISATION DU GRAPHIQUE ---
    plt.title("Phase 7 : Comparaison Finale des Modèles sur 2024", fontsize=14)
    plt.xlabel("Datetime")
    plt.ylabel("Rendement Cumulé")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ml_pipeline("data/gbpusd_m15_features_v2.csv")


        
        
        