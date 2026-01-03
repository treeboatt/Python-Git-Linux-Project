import pandas as pd
import numpy as np

class QuantAAnalyzer:
    def __init__(self, data):
        self.data = data.copy()

    def apply_strategy(self, short_window=20, long_window=50):
        if self.data.empty:
            return self.data
            
        df = self.data.copy()
        
        # Moyennes mobiles
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        
        # Signal d'achat
        df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1.0, 0.0)
        df['Position'] = df['Signal'].diff()
        
        self.data = df
        return df

    def get_metrics(self):
        if self.data.empty:
            return {"Max Drawdown": 0.0, "Sharpe Ratio": 0.0}
            
        returns = self.data['Close'].pct_change().dropna()
        
        # Max Drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio
        annual_factor = np.sqrt(252 * 78)
        if returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = (returns.mean() / returns.std()) * annual_factor
            
        return {
            "Max Drawdown": round(max_drawdown, 4),
            "Sharpe Ratio": round(sharpe, 4)
        }