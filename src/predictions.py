"""
Prediction module for forecasting asset prices.
Simple, robust models that actually work: Momentum, Linear Trend, Moving Average.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from src.utils import safe_float


@dataclass
class PredictionResult:
    """Container for prediction results."""
    model_name: str
    forecast_dates: pd.DatetimeIndex
    predictions: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    confidence_level: float
    metrics: Dict[str, float]
    historical_fitted: Optional[pd.Series] = None


def _generate_future_dates(prices: pd.Series, periods: int) -> pd.DatetimeIndex:
    """Generate future dates based on the data frequency."""
    if len(prices) < 2:
        return pd.date_range(start=datetime.now(), periods=periods, freq='D')
    
    # Estimate frequency from data
    time_diffs = pd.Series(prices.index).diff().dropna()
    if len(time_diffs) == 0:
        freq = 'D'
    else:
        median_diff = time_diffs.median()
        if median_diff < timedelta(hours=1):
            freq = '5min'
        elif median_diff < timedelta(days=1):
            freq = 'H'
        elif median_diff < timedelta(days=6):
            freq = 'D'
        else:
            freq = 'W'
    
    try:
        last_date = prices.index[-1]
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
    except:
        future_dates = pd.date_range(start=datetime.now(), periods=periods, freq='D')
    
    return future_dates


def predict_momentum(prices: pd.Series, periods: int = 30, confidence: float = 0.95, lookback: int = 20) -> Optional[PredictionResult]:
    """
    Momentum-based forecast.
    Projects recent trend forward with volatility-based confidence intervals.
    Simple but effective for trending markets.
    """
    if prices.empty or len(prices) < lookback + 5:
        return None
    
    try:
        prices = prices.dropna()
        returns = prices.pct_change().dropna()
        
        if len(returns) < lookback:
            lookback = max(5, len(returns) // 2)
        
        recent_returns = returns.iloc[-lookback:]
        avg_return = recent_returns.mean()
        volatility = recent_returns.std()
        
        if volatility == 0:
            volatility = 0.01
        
        # Z-score for confidence interval
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        future_dates = _generate_future_dates(prices, periods)
        last_price = prices.iloc[-1]
        
        predictions_list = []
        lower_list = []
        upper_list = []
        
        for i in range(periods):
            # Compound the expected return with decay
            decay = 0.98 ** i  # Slight mean reversion
            expected_return = avg_return * decay
            expected_price = last_price * (1 + expected_return) ** (i + 1)
            
            # Uncertainty grows with square root of time
            cumulative_std = volatility * np.sqrt(i + 1)
            
            predictions_list.append(expected_price)
            lower_list.append(max(0.01, expected_price * (1 - z * cumulative_std)))
            upper_list.append(expected_price * (1 + z * cumulative_std))
        
        return PredictionResult(
            model_name="Momentum",
            forecast_dates=future_dates,
            predictions=pd.Series(predictions_list, index=future_dates),
            lower_bound=pd.Series(lower_list, index=future_dates),
            upper_bound=pd.Series(upper_list, index=future_dates),
            confidence_level=confidence,
            metrics={
                "avg_return": safe_float(avg_return * 100),
                "volatility": safe_float(volatility * 100),
                "lookback": lookback,
            },
            historical_fitted=prices.copy(),
        )
    except Exception as e:
        return None


def predict_linear_trend(prices: pd.Series, periods: int = 30, confidence: float = 0.95) -> Optional[PredictionResult]:
    """
    Linear trend extrapolation.
    Fits a simple linear regression and projects forward.
    Good for assets with clear directional trends.
    """
    if prices.empty or len(prices) < 10:
        return None
    
    try:
        prices = prices.dropna()
        
        # Prepare data for linear regression
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        
        # Simple linear regression
        X_mean = X.mean()
        y_mean = y.mean()
        
        numerator = ((X.flatten() - X_mean) * (y - y_mean)).sum()
        denominator = ((X.flatten() - X_mean) ** 2).sum()
        
        if denominator == 0:
            return None
        
        slope = numerator / denominator
        intercept = y_mean - slope * X_mean
        
        # Fitted values and residuals
        fitted = slope * X.flatten() + intercept
        residuals = y - fitted
        std_residual = residuals.std()
        
        if std_residual == 0:
            std_residual = y.std() * 0.1
        
        # Z-score for CI
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        future_dates = _generate_future_dates(prices, periods)
        
        predictions_list = []
        lower_list = []
        upper_list = []
        
        for i in range(periods):
            future_x = len(prices) + i
            pred = slope * future_x + intercept
            
            # Widen CI over time
            uncertainty = std_residual * np.sqrt(1 + (i + 1) / len(prices))
            
            predictions_list.append(max(0.01, pred))
            lower_list.append(max(0.01, pred - z * uncertainty))
            upper_list.append(pred + z * uncertainty)
        
        # Calculate R-squared
        ss_res = (residuals ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return PredictionResult(
            model_name="Linear Trend",
            forecast_dates=future_dates,
            predictions=pd.Series(predictions_list, index=future_dates),
            lower_bound=pd.Series(lower_list, index=future_dates),
            upper_bound=pd.Series(upper_list, index=future_dates),
            confidence_level=confidence,
            metrics={
                "slope": safe_float(slope),
                "r_squared": safe_float(r_squared),
                "daily_change": safe_float(slope),
            },
            historical_fitted=pd.Series(fitted, index=prices.index),
        )
    except Exception as e:
        return None


def predict_moving_average(prices: pd.Series, periods: int = 30, confidence: float = 0.95, window: int = 20) -> Optional[PredictionResult]:
    """
    Moving average forecast.
    Uses exponential smoothing for prediction.
    Good for mean-reverting or range-bound assets.
    """
    if prices.empty or len(prices) < window:
        return None
    
    try:
        prices = prices.dropna()
        
        # Calculate EMA
        alpha = 2 / (window + 1)
        ema = prices.ewm(alpha=alpha, adjust=False).mean()
        
        # Residuals for CI
        residuals = prices - ema
        std_residual = residuals.std()
        
        if std_residual == 0:
            std_residual = prices.std() * 0.1
        
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        future_dates = _generate_future_dates(prices, periods)
        last_ema = ema.iloc[-1]
        last_price = prices.iloc[-1]
        
        # Trend component
        recent_trend = (prices.iloc[-1] - prices.iloc[-window]) / window if len(prices) >= window else 0
        
        predictions_list = []
        lower_list = []
        upper_list = []
        
        current_pred = last_ema
        
        for i in range(periods):
            # EMA gradually moves toward recent price level with diminishing trend
            trend_decay = 0.9 ** i
            current_pred = current_pred + recent_trend * trend_decay
            
            # Widen CI over time
            uncertainty = std_residual * np.sqrt(1 + i * 0.15)
            
            predictions_list.append(max(0.01, current_pred))
            lower_list.append(max(0.01, current_pred - z * uncertainty))
            upper_list.append(current_pred + z * uncertainty)
        
        return PredictionResult(
            model_name="Moving Average",
            forecast_dates=future_dates,
            predictions=pd.Series(predictions_list, index=future_dates),
            lower_bound=pd.Series(lower_list, index=future_dates),
            upper_bound=pd.Series(upper_list, index=future_dates),
            confidence_level=confidence,
            metrics={
                "window": window,
                "last_ema": safe_float(last_ema),
                "trend": safe_float(recent_trend),
            },
            historical_fitted=ema,
        )
    except Exception as e:
        return None


def predict_mean_reversion(prices: pd.Series, periods: int = 30, confidence: float = 0.95, window: int = 50) -> Optional[PredictionResult]:
    """
    Mean reversion forecast.
    Assumes price will revert to historical mean.
    Good for range-bound assets or after extreme moves.
    """
    if prices.empty or len(prices) < window:
        return None
    
    try:
        prices = prices.dropna()
        
        # Calculate rolling mean and std
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]
        last_price = prices.iloc[-1]
        
        if current_std == 0:
            current_std = prices.std() * 0.1
        
        # Z-score of current price
        z_score = (last_price - current_mean) / current_std
        
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        future_dates = _generate_future_dates(prices, periods)
        
        predictions_list = []
        lower_list = []
        upper_list = []
        
        for i in range(periods):
            # Price reverts to mean with half-life
            half_life = window / 4
            reversion_speed = 1 - np.exp(-np.log(2) * (i + 1) / half_life)
            
            pred = last_price + (current_mean - last_price) * reversion_speed
            
            # Uncertainty
            uncertainty = current_std * np.sqrt(1 + i * 0.1)
            
            predictions_list.append(max(0.01, pred))
            lower_list.append(max(0.01, pred - z * uncertainty))
            upper_list.append(pred + z * uncertainty)
        
        return PredictionResult(
            model_name="Mean Reversion",
            forecast_dates=future_dates,
            predictions=pd.Series(predictions_list, index=future_dates),
            lower_bound=pd.Series(lower_list, index=future_dates),
            upper_bound=pd.Series(upper_list, index=future_dates),
            confidence_level=confidence,
            metrics={
                "current_mean": safe_float(current_mean),
                "z_score": safe_float(z_score),
                "half_life": safe_float(window / 4),
            },
            historical_fitted=rolling_mean,
        )
    except Exception as e:
        return None


# Model registry
PREDICTION_MODELS = {
    "Momentum": predict_momentum,
    "Linear Trend": predict_linear_trend,
    "Moving Average": predict_moving_average,
    "Mean Reversion": predict_mean_reversion,
}


def run_prediction(prices: pd.Series, model: str = "Momentum", periods: int = 30, confidence: float = 0.95, **kwargs) -> Optional[PredictionResult]:
    """Main interface to run predictions."""
    if prices.empty:
        return None
    
    prices = prices.dropna()
    if len(prices) < 10:
        return None
    
    # Normalize model name
    model_lower = model.lower().strip()
    
    # Match model
    if "momentum" in model_lower:
        return predict_momentum(prices, periods, confidence, **kwargs)
    elif "linear" in model_lower or "trend" in model_lower:
        return predict_linear_trend(prices, periods, confidence)
    elif "moving" in model_lower or "average" in model_lower or "ma" in model_lower:
        return predict_moving_average(prices, periods, confidence, **kwargs)
    elif "mean" in model_lower or "reversion" in model_lower:
        return predict_mean_reversion(prices, periods, confidence, **kwargs)
    
    # Default to momentum
    return predict_momentum(prices, periods, confidence)


def get_available_models() -> List[str]:
    """Return list of available prediction models."""
    return list(PREDICTION_MODELS.keys())


def get_model_description(model: str) -> str:
    """Get a short description of each model."""
    descriptions = {
        "Momentum": "Projects recent price momentum forward. Best for trending markets.",
        "Linear Trend": "Fits a linear trend line and extrapolates. Good for steady trends.",
        "Moving Average": "Uses exponential smoothing. Best for stable, less volatile assets.",
        "Mean Reversion": "Assumes price reverts to historical average. Good after extreme moves.",
    }
    return descriptions.get(model, "Forecasting model for price prediction.")