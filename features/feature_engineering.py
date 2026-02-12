import pandas as pd
import pandas_ta as ta

def generate_features_v2(input_path):
    # Chargement du fichier nettoyé en Phase 3
    df = pd.read_csv(input_path, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()

    print(f"--- Calcul des Features V2 sur {len(df)} lignes ---")

    # --- 6.1 BLOC COURT TERME ---
    # Rendements (log returns pour plus de stabilité en ML)
    df['return_1'] = df['Close'].pct_change(1)
    df['return_4'] = df['Close'].pct_change(4)
    
    # EMAs
    df['ema_20'] = ta.ema(df['Close'], length=20)
    df['ema_50'] = ta.ema(df['Close'], length=50)
    df['ema_diff'] = df['ema_20'] - df['ema_50']
    
    # RSI & Volatilité
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()
    
    # Bougies (Price Action)
    df['range_15m'] = df['High'] - df['Low']
    df['body'] = (df['Close'] - df['Open']).abs()
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # --- 6.2 BLOC CONTEXTE & RÉGIME ---
    # Tendance Long Terme
    df['ema_200'] = ta.ema(df['Close'], length=200)
    df['distance_to_ema200'] = (df['Close'] - df['ema_200']) / df['ema_200']
    df['slope_ema50'] = df['ema_50'].diff(3) # Pente sur les 3 dernières bougies

    # Régime de Volatilité
    df['atr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['rolling_std_100'] = df['Close'].rolling(window=100).std()
    df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_100']

    # Force Directionnelle
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['adx_14'] = adx['ADX_14']
    
    macd = ta.macd(df['Close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    # NETTOYAGE : On supprime les premières lignes qui ont des NaN à cause des calculs (ex: ema 200)
    df.dropna(inplace=True)

    # Sauvegarde
    output_path = "data/gbpusd_m15_features_v2.csv"
    df.to_csv(output_path)
    print(f"✅ Phase 6 terminée. {df.shape[1]} colonnes générées.")
    return df

if __name__ == "__main__":
    generate_features_v2("data/GBPUSD_M15_CLEANED.csv")