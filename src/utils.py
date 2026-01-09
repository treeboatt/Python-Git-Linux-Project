"""
Shared utilities for the quant platform.
Contains metrics calculations, formatting helpers, and constants used across modules.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import pytz


# ============================================================
# CONSTANTS
# ============================================================

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # 4% annual, adjust as needed

# Color palette for consistent styling across the app
COLORS = {
    "primary": "#6366f1",      # indigo
    "secondary": "#8b5cf6",    # violet
    "success": "#10b981",      # emerald
    "danger": "#ef4444",       # red
    "warning": "#f59e0b",      # amber
    "info": "#3b82f6",         # blue
    "dark": "#1e1e2e",         # background
    "light": "#f8fafc",        # text
    "muted": "#64748b",        # gray
}

# Asset categories for quick selection
ASSET_PRESETS = {
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "Indices": ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "French Stocks": ["MC.PA", "OR.PA", "SAN.PA", "AIR.PA", "BNP.PA"],
}

# Period/interval compatibility for yfinance
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


# ============================================================
# VALIDATION HELPERS
# ============================================================

def is_valid_combination(period: str, interval: str) -> bool:
    """Check if a period/interval combo works with yfinance."""
    interval = interval.lower().strip()
    period = period.lower().strip()
    
    if interval not in VALID_COMBINATIONS:
        return True  # unknown interval, let yfinance handle it
    
    return period in VALID_COMBINATIONS[interval]


def suggest_valid_combination(period: str, interval: str) -> Tuple[str, str]:
    """Suggest a working period/interval if the current one doesn't work."""
    if is_valid_combination(period, interval):
        return period, interval
    
    interval = interval.lower().strip()
    
    # try to keep the interval, change period
    if interval in VALID_COMBINATIONS and VALID_COMBINATIONS[interval]:
        return VALID_COMBINATIONS[interval][-1], interval
    
    # fallback to daily
    return "1mo", "1d"


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert anything to float."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# RETURN CALCULATIONS
# ============================================================

def compute_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Compute returns from a price series.
    method: 'simple' for arithmetic returns, 'log' for log returns
    """
    if prices.empty:
        return pd.Series(dtype=float)
    
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert returns series to cumulative returns (base 1)."""
    if returns.empty:
        return pd.Series(dtype=float)
    return (1 + returns).cumprod()


def compute_equity_curve(returns: pd.Series, initial_value: float = 100.0) -> pd.Series:
    """Convert returns to equity curve starting at initial_value."""
    if returns.empty:
        return pd.Series(dtype=float)
    return compute_cumulative_returns(returns) * initial_value


# ============================================================
# RISK METRICS
# ============================================================

def compute_volatility(returns: pd.Series, annualize: bool = True, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compute volatility (standard deviation of returns)."""
    if returns.empty or len(returns) < 2:
        return 0.0
    
    vol = returns.std()
    
    if annualize:
        vol *= np.sqrt(periods_per_year)
    
    return safe_float(vol)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown from an equity curve.
    Returns a negative number (e.g., -0.15 means -15% drawdown).
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    return safe_float(drawdown.min())


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Get the full drawdown series for plotting."""
    if equity_curve.empty:
        return pd.Series(dtype=float)
    
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Compute annualized Sharpe ratio.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    mean_excess = excess_returns.mean() * periods_per_year
    vol = returns.std() * np.sqrt(periods_per_year)
    
    if vol == 0:
        return 0.0
    
    return safe_float(mean_excess / vol)


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Compute Sortino ratio (like Sharpe but only considers downside volatility).
    """
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


def compute_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Compute Calmar ratio (annual return / max drawdown).
    """
    if returns.empty or equity_curve.empty:
        return 0.0
    
    annual_return = returns.mean() * periods_per_year
    max_dd = abs(compute_max_drawdown(equity_curve))
    
    if max_dd == 0:
        return 0.0
    
    return safe_float(annual_return / max_dd)


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute Value at Risk at given confidence level.
    Returns a negative number representing the loss threshold.
    """
    if returns.empty:
        return 0.0
    
    return safe_float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute Conditional VaR (Expected Shortfall).
    Average of returns below the VaR threshold.
    """
    if returns.empty:
        return 0.0
    
    var = compute_var(returns, confidence)
    tail_returns = returns[returns <= var]
    
    if tail_returns.empty:
        return var
    
    return safe_float(tail_returns.mean())


def compute_win_rate(returns: pd.Series) -> float:
    """Percentage of positive returns."""
    if returns.empty:
        return 0.0
    
    wins = (returns > 0).sum()
    total = len(returns)
    
    return safe_float(wins / total) if total > 0 else 0.0


def compute_profit_factor(returns: pd.Series) -> float:
    """Ratio of gross profits to gross losses."""
    if returns.empty:
        return 0.0
    
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return safe_float(gains / losses)


# ============================================================
# COMPLETE METRICS CALCULATION
# ============================================================

def compute_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = RISK_FREE_RATE
) -> Dict[str, float]:
    """
    Compute all performance metrics at once.
    Returns a dict with all the key metrics.
    """
    if returns.empty or equity_curve.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
        }
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # annualized return approximation
    n_periods = len(returns)
    years = n_periods / periods_per_year
    if years > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0.0
    
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


# ============================================================
# ANNUALIZATION HELPERS
# ============================================================

def get_periods_per_year(interval: str) -> int:
    """
    Estimate periods per year based on interval string.
    Used for annualizing returns and volatility.
    """
    interval = interval.lower().strip()
    
    # minutes
    if interval.endswith("m"):
        try:
            mins = int(interval[:-1])
        except ValueError:
            mins = 5
        bars_per_day = 390 / max(mins, 1)  # 6.5 hours of trading
        return int(TRADING_DAYS_PER_YEAR * bars_per_day)
    
    # hours
    if interval.endswith("h"):
        try:
            hours = int(interval[:-1])
        except ValueError:
            hours = 1
        bars_per_day = 6.5 / max(hours, 0.5)
        return int(TRADING_DAYS_PER_YEAR * bars_per_day)
    
    # daily
    if interval in ["1d", "d", "daily"]:
        return TRADING_DAYS_PER_YEAR
    
    # weekly
    if interval in ["1wk", "wk", "weekly"]:
        return 52
    
    # monthly
    if interval in ["1mo", "mo", "monthly"]:
        return 12
    
    # default to daily
    return TRADING_DAYS_PER_YEAR


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousand separators."""
    return f"{value:,.{decimals}f}"


def format_currency(value: float, currency: str = "$", decimals: int = 2) -> str:
    """Format as currency."""
    return f"{currency}{value:,.{decimals}f}"


def format_metric_delta(current: float, previous: float) -> Tuple[str, str]:
    """
    Format a metric change for display.
    Returns (formatted_delta, color).
    """
    if previous == 0:
        return "N/A", "gray"
    
    change = (current - previous) / abs(previous)
    color = "green" if change >= 0 else "red"
    sign = "+" if change >= 0 else ""
    
    return f"{sign}{format_percentage(change)}", color


# ============================================================
# DATE/TIME HELPERS
# ============================================================

def get_paris_time() -> datetime:
    """Get current time in Paris timezone."""
    paris_tz = pytz.timezone("Europe/Paris")
    return datetime.now(paris_tz)


def is_market_open() -> bool:
    """
    Check if US markets are roughly open.
    Simple heuristic, doesn't account for holidays.
    """
    now = datetime.now(pytz.timezone("America/New_York"))
    
    # weekends
    if now.weekday() >= 5:
        return False
    
    # market hours 9:30 - 16:00 ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def get_last_trading_day() -> datetime:
    """Get the most recent trading day."""
    now = datetime.now()
    
    # if weekend, go back to Friday
    while now.weekday() >= 5:
        now -= timedelta(days=1)
    
    return now


# ============================================================
# DATA CLEANING HELPERS
# ============================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for price dataframes."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # handle multiindex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # sort by date
    df = df.sort_index()
    
    # remove duplicate indices
    df = df[~df.index.duplicated(keep='last')]
    
    return df


def align_dataframes(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Align multiple dataframes to common dates."""
    if not dfs:
        return ()
    
    # find common index
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)
    
    return tuple(df.loc[common_idx] for df in dfs)


# ============================================================
# SIGNAL HELPERS
# ============================================================

def generate_signal_changes(signals: pd.Series) -> pd.Series:
    """
    Detect where signals change (for trade counting).
    Returns 1 at each change point, 0 otherwise.
    """
    if signals.empty:
        return pd.Series(dtype=float)
    
    changes = signals.diff().fillna(0)
    return (changes != 0).astype(int)


def count_trades(signals: pd.Series) -> int:
    """Count number of trades from a signal series."""
    changes = generate_signal_changes(signals)
    return int(changes.sum())