"""
Shared utilities for the Quant Terminal platform.
Contains metrics calculations, formatting helpers, and constants.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import pytz


# ============================================================
# CONSTANTS
# ============================================================

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04

COLORS = {
    "primary": "#f97316",
    "secondary": "#ea580c",
    "success": "#3fb950",
    "danger": "#f85149",
    "warning": "#d29922",
    "info": "#3b82f6",
    "dark": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "muted": "#8b949e",
}

POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX",
    "JPM", "BAC", "V", "MA", "JNJ", "PFE", "UNH", "XOM", "CVX",
    "KO", "PEP", "MCD", "NKE", "DIS", "BA", "CAT",
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "GC=F", "SI=F", "CL=F",
    "MC.PA", "OR.PA", "AIR.PA", "BNP.PA",
]

TICKER_CATEGORIES = {
    "US Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "ORCL", "CRM", "ADBE", "INTC"],
    "US Finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "PYPL", "AXP", "BLK"],
    "US Healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABBV", "LLY", "BMY"],
    "US Consumer": ["KO", "PEP", "MCD", "SBUX", "NKE", "DIS", "WMT", "HD"],
    "US Energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "Indices": ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "European": ["MC.PA", "OR.PA", "SAN.PA", "AIR.PA", "BNP.PA", "TTE.PA"],
}

PERIOD_OPTIONS = {
    "1d": "1 Day",
    "5d": "5 Days",
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y": "1 Year",
    "2y": "2 Years",
    "5y": "5 Years",
}

PERIOD_INTERVAL_MAP = {
    "1d": "5m",
    "5d": "15m",
    "1mo": "1h",
    "3mo": "1d",
    "6mo": "1d",
    "1y": "1d",
    "2y": "1d",
    "5y": "1wk",
}

VALID_COMBINATIONS = {
    "1m": ["1d", "5d", "7d"],
    "2m": ["1d", "5d", "7d", "1mo"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "30m": ["1d", "5d", "7d", "1mo"],
    "60m": ["1d", "5d", "7d", "1mo", "3mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo"],
    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    "1mo": ["3mo", "6mo", "1y", "2y", "5y", "max"],
}


def get_interval_for_period(period: str) -> str:
    return PERIOD_INTERVAL_MAP.get(period, "1d")


def is_valid_combination(period: str, interval: str) -> bool:
    interval = interval.lower().strip()
    period = period.lower().strip()
    if interval not in VALID_COMBINATIONS:
        return True
    return period in VALID_COMBINATIONS[interval]


def suggest_valid_combination(period: str, interval: str) -> Tuple[str, str]:
    if is_valid_combination(period, interval):
        return period, interval
    interval = interval.lower().strip()
    if interval in VALID_COMBINATIONS and VALID_COMBINATIONS[interval]:
        return VALID_COMBINATIONS[interval][-1], interval
    return "1mo", "1d"


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return (1 + returns).cumprod()


def compute_equity_curve(returns: pd.Series, initial_value: float = 100.0) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return compute_cumulative_returns(returns) * initial_value


def compute_volatility(returns: pd.Series, annualize: bool = True, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if returns.empty or len(returns) < 2:
        return 0.0
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(periods_per_year)
    return safe_float(vol)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return safe_float(drawdown.min())


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    if equity_curve.empty:
        return pd.Series(dtype=float)
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if returns.empty or len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = excess_returns.mean() * periods_per_year
    vol = returns.std() * np.sqrt(periods_per_year)
    if vol == 0:
        return 0.0
    return safe_float(mean_excess / vol)


def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if returns.empty or len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    if downside_returns.empty or len(downside_returns) < 2:
        return 0.0
    mean_excess = excess_returns.mean() * periods_per_year
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0
    return safe_float(mean_excess / downside_vol)


def compute_calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if returns.empty or equity_curve.empty:
        return 0.0
    annual_return = returns.mean() * periods_per_year
    max_dd = abs(compute_max_drawdown(equity_curve))
    if max_dd == 0:
        return 0.0
    return safe_float(annual_return / max_dd)


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return safe_float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    var = compute_var(returns, confidence)
    tail_returns = returns[returns <= var]
    if tail_returns.empty:
        return var
    return safe_float(tail_returns.mean())


def compute_win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    wins = (returns > 0).sum()
    total = len(returns)
    return safe_float(wins / total) if total > 0 else 0.0


def compute_profit_factor(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return safe_float(gains / losses)


def compute_all_metrics(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR, risk_free_rate: float = RISK_FREE_RATE) -> Dict[str, float]:
    if returns.empty or equity_curve.empty:
        return {"total_return": 0.0, "annualized_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0, "max_drawdown": 0.0, "var_95": 0.0, "cvar_95": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0}
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else 0.0
    
    return {
        "total_return": safe_float(total_return),
        "annualized_return": safe_float(annualized_return),
        "volatility": compute_volatility(returns, True, periods_per_year),
        "sharpe_ratio": compute_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": compute_sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": compute_calmar_ratio(returns, equity_curve, periods_per_year),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "var_95": compute_var(returns, 0.95),
        "cvar_95": compute_cvar(returns, 0.95),
        "win_rate": compute_win_rate(returns),
        "profit_factor": compute_profit_factor(returns),
        "total_trades": len(returns),
    }


def get_periods_per_year(interval: str) -> int:
    interval = interval.lower().strip()
    if interval.endswith("m"):
        try:
            mins = int(interval[:-1])
        except ValueError:
            mins = 5
        bars_per_day = 390 / max(mins, 1)
        return int(TRADING_DAYS_PER_YEAR * bars_per_day)
    if interval.endswith("h"):
        try:
            hours = int(interval[:-1])
        except ValueError:
            hours = 1
        bars_per_day = 6.5 / max(hours, 0.5)
        return int(TRADING_DAYS_PER_YEAR * bars_per_day)
    if interval in ["1d", "d", "daily"]:
        return TRADING_DAYS_PER_YEAR
    if interval in ["1wk", "wk", "weekly"]:
        return 52
    if interval in ["1mo", "mo", "monthly"]:
        return 12
    return TRADING_DAYS_PER_YEAR


def format_percentage(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def format_currency(value: float, currency: str = "$", decimals: int = 2) -> str:
    return f"{currency}{value:,.{decimals}f}"


def get_paris_time() -> datetime:
    paris_tz = pytz.timezone("Europe/Paris")
    return datetime.now(paris_tz)


def is_market_open() -> bool:
    now = datetime.now(pytz.timezone("America/New_York"))
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df


def count_trades(signals: pd.Series) -> int:
    if signals.empty:
        return 0
    changes = signals.diff().fillna(0)
    return int((changes != 0).sum())