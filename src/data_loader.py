import yfinance as yf
import pandas as pd

def fetch_data(ticker, period="5d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()]

        if df.empty or 'Close' not in df.columns:
            return pd.DataFrame()
            
        return df
        
    except Exception:
        return pd.DataFrame()