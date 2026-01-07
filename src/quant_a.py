import pandas as pd
import numpy as np

class QuantAAnalyzer:
    def __init__(self, data):
        # On s'assure que les données sont propres dès le départ
        self.data = data.copy()
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)

    def apply_strategy(self, strategy_type="SMA Crossover", **params):
        """
        Applique une stratégie (SMA, RSI, Bollinger) et calcule la Equity Curve.
        """
        if self.data.empty:
            return self.data
            
        df = self.data.copy()
        df['Signal'] = 0 # Par défaut, pas de position
        
        # --- 1. STRATEGIE: SMA CROSSOVER ---
        if strategy_type == "SMA Crossover":
            short_window = params.get('short_window', 20)
            long_window = params.get('long_window', 50)
            
            df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
            df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
            
            # Signal: 1 si Court > Long, sinon 0
            df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1.0, 0.0)

        # --- 2. STRATEGIE: RSI MEAN REVERSION ---
        elif strategy_type == "RSI Mean Reversion":
            rsi_window = params.get('rsi_window', 14)
            rsi_lower = params.get('rsi_lower', 30)
            rsi_upper = params.get('rsi_upper', 70)
            
            # Calcul RSI Manuel
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Logique: Achat si survendu (<30), Vente si suracheté (>70)
            # On utilise une boucle simple pour simuler la tenue de position
            signals = np.zeros(len(df))
            position = 0
            for i in range(rsi_window, len(df)):
                if df['RSI'].iloc[i] < rsi_lower:
                    position = 1 # Buy
                elif df['RSI'].iloc[i] > rsi_upper:
                    position = 0 # Sell
                signals[i] = position
            df['Signal'] = signals

        # --- 3. STRATEGIE: BOLLINGER BANDS ---
        elif strategy_type == "Bollinger Bands":
            window = params.get('bb_window', 20)
            std_dev = params.get('bb_std', 2.0)
            
            sma = df['Close'].rolling(window=window).mean()
            std = df['Close'].rolling(window=window).std()
            
            df['BB_Upper'] = sma + (std * std_dev)
            df['BB_Lower'] = sma - (std * std_dev)
            
            # Logique: Achat si prix < Bas, Vente si prix > Haut
            signals = np.zeros(len(df))
            position = 0
            for i in range(window, len(df)):
                if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
                    position = 1
                elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
                    position = 0
                signals[i] = position
            df['Signal'] = signals

        # --- 4. STRATEGIE: BUY & HOLD ---
        elif strategy_type == "Buy & Hold":
            df['Signal'] = 1.0

        # --- CALCUL PERFORMANCE (Commun à toutes) ---
        # Strategy Return = Signal de la veille * Rendement du jour
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Market_Return']
        df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
        
        # Base 100 pour l'Equity Curve
        df['Equity_Curve'] = (1 + df['Strategy_Return']).cumprod() * 100
        
        self.data = df
        return df

    def get_metrics(self):
        """
        Calcule les métriques de la STRATEGIE (pas juste du marché).
        """
        if self.data.empty or 'Strategy_Return' not in self.data.columns:
            return {
                "Total Return": 0.0, "Volatility": 0.0,
                "Sharpe Ratio": 0.0, "Max Drawdown": 0.0
            }
            
        # On travaille sur les retours de la stratégie
        returns = self.data['Strategy_Return']
        equity = self.data['Equity_Curve']
        
        # 1. Total Return
        total_return = (equity.iloc[-1] / 100) - 1
        
        # 2. Max Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 3. Volatilité Annualisée (252 jours de trading)
        volatility = returns.std() * np.sqrt(252)
        
        # 4. Sharpe Ratio (Annualisé, sans taux sans risque pour simplifier)
        if volatility == 0:
            sharpe = 0.0
        else:
            sharpe = (returns.mean() * 252) / volatility
            
        return {
            "Total Return": round(total_return, 4),
            "Volatility": round(volatility, 4),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown": round(max_drawdown, 4)
        }