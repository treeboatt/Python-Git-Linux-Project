"""
Data loader module for fetching financial data.
Handles yfinance API calls with caching, retry logic, and proper error handling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import time

from src.utils import (
    clean_dataframe,
    is_valid_combination,
    suggest_valid_combination,
    VALID_COMBINATIONS,
)


# ============================================================
# CACHING CONFIGURATION
# ============================================================

# Cache TTL in seconds (5 minutes as required by the project)
CACHE_TTL = 300

# Max retries for failed API calls
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds between retries


# ============================================================
# MAIN DATA FETCHING FUNCTIONS
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_data(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker.
    Uses Streamlit cache with 5-minute TTL for auto-refresh.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'BTC-USD', 'EURUSD=X')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '1wk', '1mo')
        auto_adjust: Whether to adjust prices for splits/dividends
    
    Returns:
        DataFrame with OHLCV data, empty DataFrame on failure
    """
    if not ticker or not isinstance(ticker, str):
        return pd.DataFrame()
    
    ticker = ticker.strip().upper()
    
    # validate and fix period/interval combo if needed
    if not is_valid_combination(period, interval):
        period, interval = suggest_valid_combination(period, interval)
    
    # retry logic for resilience
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=auto_adjust,
                threads=False,  # more stable
            )
            
            if df is None or df.empty:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return pd.DataFrame()
            
            # clean up the dataframe
            df = clean_dataframe(df)
            
            # make sure we have the essential columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            # drop rows where Close is missing (useless data)
            df = df.dropna(subset=['Close'])
            
            if df.empty:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return pd.DataFrame()
            
            # add some useful derived columns
            df['Ticker'] = ticker
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            return df
            
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
    
    # all retries failed
    return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_multiple(
    tickers: List[str],
    period: str = "5d",
    interval: str = "5m"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple tickers at once.
    Returns a dict mapping ticker -> DataFrame.
    """
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
def fetch_price_matrix(
    tickers: List[str],
    period: str = "5d",
    interval: str = "5m",
    column: str = "Close"
) -> pd.DataFrame:
    """
    Fetch data and return a single DataFrame with one column per ticker.
    Useful for portfolio analysis where you need aligned price series.
    
    Args:
        tickers: List of ticker symbols
        period: Data period
        interval: Data interval
        column: Which column to extract ('Close', 'Open', etc.)
    
    Returns:
        DataFrame with tickers as columns, dates as index
    """
    data = fetch_multiple(tickers, period=period, interval=interval)
    
    if not data:
        return pd.DataFrame()
    
    # extract the requested column from each ticker
    frames = []
    for ticker, df in data.items():
        if column in df.columns:
            series = df[column].rename(ticker)
            frames.append(series)
    
    if not frames:
        return pd.DataFrame()
    
    # combine into single dataframe
    prices = pd.concat(frames, axis=1)
    prices = prices.sort_index()
    
    # handle missing data
    prices = prices.dropna(how='all')  # remove rows where everything is NaN
    prices = prices.ffill()  # forward fill gaps
    prices = prices.dropna(how='any')  # remove any remaining NaN rows
    
    # remove duplicate indices
    prices = prices[~prices.index.duplicated(keep='last')]
    
    return prices


# ============================================================
# TICKER INFO & VALIDATION
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)  # cache for 1 hour
def get_ticker_info(ticker: str) -> Dict:
    """
    Get basic info about a ticker (name, currency, exchange, etc.).
    """
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
            "country": info.get("country", "N/A"),
            "website": info.get("website", ""),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {
            "symbol": ticker,
            "name": ticker,
            "currency": "USD",
            "exchange": "Unknown",
            "market_cap": 0,
            "sector": "N/A",
            "industry": "N/A",
            "country": "N/A",
            "website": "",
            "description": "",
        }


def validate_ticker(ticker: str) -> bool:
    """
    Check if a ticker is valid by trying to fetch minimal data.
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    try:
        df = yf.download(
            ticker.strip().upper(),
            period="1d",
            interval="1d",
            progress=False,
        )
        return df is not None and not df.empty
    except Exception:
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of tickers.
    Returns (valid_tickers, invalid_tickers).
    """
    valid = []
    invalid = []
    
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        
        if validate_ticker(ticker):
            valid.append(ticker)
        else:
            invalid.append(ticker)
    
    return valid, invalid


# ============================================================
# REAL-TIME / LATEST PRICE
# ============================================================

def get_latest_price(ticker: str) -> Optional[Dict]:
    """
    Get the most recent price data for a ticker.
    Returns dict with price, change, volume, etc.
    """
    try:
        df = fetch_data(ticker, period="1d", interval="1m")
        
        if df.empty:
            # fallback to daily data
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
    """
    Get latest prices for multiple tickers.
    """
    results = {}
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if ticker:
            data = get_latest_price(ticker)
            if data:
                results[ticker] = data
    return results


# ============================================================
# HISTORICAL DATA HELPERS
# ============================================================

def get_daily_summary(ticker: str) -> Optional[Dict]:
    """
    Get a daily summary (open, high, low, close, volume) for today.
    Useful for the daily report generation.
    """
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
            "intraday_range": float(today['High'] - today['Low']),
            "intraday_range_pct": float((today['High'] - today['Low']) / today['Low'] * 100) if today['Low'] != 0 else 0,
        }
    except Exception:
        return None


def get_period_stats(ticker: str, period: str = "1mo") -> Optional[Dict]:
    """
    Get statistical summary for a given period.
    """
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
            "avg_volume": float(df['Volume'].mean()) if 'Volume' in df.columns else 0,
            "volatility": float(returns.std() * np.sqrt(252)),  # annualized
            "avg_daily_return": float(returns.mean()),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "positive_days": int((returns > 0).sum()),
            "negative_days": int((returns < 0).sum()),
            "total_days": len(returns),
        }
    except Exception:
        return None


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Run quality checks on a DataFrame.
    Returns a report dict.
    """
    if df.empty:
        return {
            "is_valid": False,
            "rows": 0,
            "issues": ["DataFrame is empty"],
        }
    
    issues = []
    
    # check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col, pct in missing_pct.items():
        if pct > 10:
            issues.append(f"{col} has {pct:.1f}% missing values")
    
    # check for duplicate indices
    dup_idx = df.index.duplicated().sum()
    if dup_idx > 0:
        issues.append(f"{dup_idx} duplicate timestamps found")
    
    # check for suspicious price movements (> 50% in one period)
    if 'Close' in df.columns:
        returns = df['Close'].pct_change().abs()
        extreme = (returns > 0.5).sum()
        if extreme > 0:
            issues.append(f"{extreme} extreme price movements (>50%) detected")
    
    # check date range
    if len(df) > 0:
        date_range = (df.index[-1] - df.index[0]).days
        if date_range == 0 and len(df) < 10:
            issues.append("Very limited data available")
    
    return {
        "is_valid": len(issues) == 0,
        "rows": len(df),
        "columns": list(df.columns),
        "start_date": df.index[0] if len(df) > 0 else None,
        "end_date": df.index[-1] if len(df) > 0 else None,
        "missing_values": missing.to_dict(),
        "issues": issues,
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clear_cache():
    """Clear all cached data. Useful for forcing a refresh."""
    fetch_data.clear()
    fetch_multiple.clear()
    fetch_price_matrix.clear()
    get_ticker_info.clear()
    validate_tickers.clear()


def get_available_intervals() -> List[str]:
    """Return list of available intervals."""
    return ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo"]


def get_available_periods() -> List[str]:
    """Return list of available periods."""
    return ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]


def get_valid_periods_for_interval(interval: str) -> List[str]:
    """Get list of valid periods for a given interval."""
    interval = interval.lower().strip()
    if interval in VALID_COMBINATIONS:
        return VALID_COMBINATIONS[interval]
    return get_available_periods()