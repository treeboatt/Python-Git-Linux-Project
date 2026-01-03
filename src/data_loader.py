import yfinance as yf
import pandas as pd

def fetch_data(ticker, period="5d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            return pd.DataFrame()
            
        return df
        
    except Exception:
        return pd.DataFrame()