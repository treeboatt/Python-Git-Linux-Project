"""
Data loader module for fetching financial data.
Handles yfinance API calls with caching, retry logic, and error handling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import time

from src.utils import clean_dataframe, is_valid_combination, suggest_valid_combination, VALID_COMBINATIONS


CACHE_TTL = 300  # 5 minutes
MAX_RETRIES = 3
RETRY_DELAY = 1


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_data(ticker: str, period: str = "5d", interval: str = "5m", auto_adjust: bool = True) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker with caching."""
    if not ticker or not isinstance(ticker, str):
        return pd.DataFrame()
    
    ticker = ticker.strip().upper()
    
    if not is_valid_combination(period, interval):
        period, interval = suggest_valid_combination(period, interval)
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=auto_adjust, threads=False)
            
            if df is None or df.empty:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return pd.DataFrame()
            
            df = clean_dataframe(df)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            df = df.dropna(subset=['Close'])
            
            if df.empty:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return pd.DataFrame()
            
            df['Ticker'] = ticker
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            return df
            
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
    
    return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_multiple(tickers: List[str], period: str = "5d", interval: str = "5m") -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers."""
    results = {}
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        df = fetch_data(ticker, period=period, interval=interval)
        if not df.empty:
            results[ticker] = df
    return results


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_price_matrix(tickers: List[str], period: str = "5d", interval: str = "5m", column: str = "Close") -> pd.DataFrame:
    """Fetch data and return a single DataFrame with one column per ticker."""
    data = fetch_multiple(tickers, period=period, interval=interval)
    
    if not data:
        return pd.DataFrame()
    
    frames = []
    for ticker, df in data.items():
        if column in df.columns:
            series = df[column].rename(ticker)
            frames.append(series)
    
    if not frames:
        return pd.DataFrame()
    
    prices = pd.concat(frames, axis=1)
    prices = prices.sort_index()
    prices = prices.dropna(how='all')
    prices = prices.ffill()
    prices = prices.dropna(how='any')
    prices = prices[~prices.index.duplicated(keep='last')]
    
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_info(ticker: str) -> Dict:
    """Get basic info about a ticker."""
    try:
        ticker_obj = yf.Ticker(ticker.strip().upper())
        info = ticker_obj.info
        return {
            "symbol": info.get("symbol", ticker),
            "name": info.get("shortName", info.get("longName", ticker)),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
        }
    except Exception:
        return {"symbol": ticker, "name": ticker, "currency": "USD", "exchange": "Unknown", "market_cap": 0, "sector": "N/A", "industry": "N/A"}


def get_latest_price(ticker: str) -> Optional[Dict]:
    """Get the most recent price data for a ticker."""
    try:
        df = fetch_data(ticker, period="1d", interval="1m")
        
        if df.empty:
            df = fetch_data(ticker, period="5d", interval="1d")
        
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest['Close']
        
        change = latest['Close'] - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        return {
            "ticker": ticker.upper(),
            "price": float(latest['Close']),
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            "change": float(change),
            "change_pct": float(change_pct),
            "timestamp": df.index[-1],
            "is_positive": change >= 0,
        }
    except Exception:
        return None


def get_latest_prices(tickers: List[str]) -> Dict[str, Dict]:
    """Get latest prices for multiple tickers."""
    results = {}
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if ticker:
            data = get_latest_price(ticker)
            if data:
                results[ticker] = data
    return results


def get_daily_summary(ticker: str) -> Optional[Dict]:
    """Get a daily summary for today."""
    try:
        df = fetch_data(ticker, period="5d", interval="1d")
        
        if df.empty:
            return None
        
        today = df.iloc[-1]
        yesterday = df.iloc[-2] if len(df) > 1 else today
        
        return {
            "ticker": ticker.upper(),
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "open": float(today['Open']),
            "high": float(today['High']),
            "low": float(today['Low']),
            "close": float(today['Close']),
            "volume": int(today['Volume']) if pd.notna(today['Volume']) else 0,
            "prev_close": float(yesterday['Close']),
            "change": float(today['Close'] - yesterday['Close']),
            "change_pct": float((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if yesterday['Close'] != 0 else 0,
        }
    except Exception:
        return None


def get_period_stats(ticker: str, period: str = "1mo") -> Optional[Dict]:
    """Get statistical summary for a given period."""
    try:
        df = fetch_data(ticker, period=period, interval="1d")
        
        if df.empty or len(df) < 2:
            return None
        
        returns = df['Close'].pct_change().dropna()
        
        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": df.index[0].strftime("%Y-%m-%d"),
            "end_date": df.index[-1].strftime("%Y-%m-%d"),
            "start_price": float(df['Close'].iloc[0]),
            "end_price": float(df['Close'].iloc[-1]),
            "total_return": float((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1),
            "high": float(df['High'].max()),
            "low": float(df['Low'].min()),
            "volatility": float(returns.std() * np.sqrt(252)),
            "avg_daily_return": float(returns.mean()),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
        }
    except Exception:
        return None


def clear_cache():
    """Clear all cached data."""
    fetch_data.clear()
    fetch_multiple.clear()
    fetch_price_matrix.clear()
    get_ticker_info.clear()


def get_available_intervals() -> List[str]:
    return ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo"]


def get_available_periods() -> List[str]:
    return ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]


def get_valid_periods_for_interval(interval: str) -> List[str]:
    interval = interval.lower().strip()
    if interval in VALID_COMBINATIONS:
        return VALID_COMBINATIONS[interval]
    return get_available_periods()