"""
Prediction module for forecasting asset prices.
Implements multiple models: Prophet, Linear Regression, and simple momentum-based forecasts.
Each model provides point estimates and confidence intervals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

# suppress prophet and sklearn warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from src.utils import safe_float, compute_volatility, get_periods_per_year


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PredictionResult:
    """Container for prediction results."""
    model_name: str
    forecast_dates: pd.DatetimeIndex
    predictions: pd.Series
    lower_bound: pd.Series  # confidence interval lower
    upper_bound: pd.Series  # confidence interval upper
    confidence_level: float  # e.g., 0.95 for 95% CI
    metrics: Dict[str, float]  # model performance metrics
    historical_fitted: Optional[pd.Series] = None  # fitted values on training data


# ============================================================
# PROPHET MODEL
# ============================================================

def predict_with_prophet(
    prices: pd.Series,
    periods: int = 30,
    confidence: float = 0.95,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
) -> Optional[PredictionResult]:
    """
    Forecast using Facebook Prophet.
    Great for capturing trends and seasonality.
    
    Args:
        prices: Historical price series with datetime index
        periods: Number of periods to forecast
        confidence: Confidence interval level (0.95 = 95%)
        yearly_seasonality: Include yearly patterns
        weekly_seasonality: Include weekly patterns
        daily_seasonality: Include daily patterns (use for intraday data)
    
    Returns:
        PredictionResult or None if prediction fails
    """
    try:
        from prophet import Prophet
    except ImportError:
        return None
    
    if prices.empty or len(prices) < 10:
        return None
    
    try:
        # prophet needs a dataframe with 'ds' and 'y' columns
        df = pd.DataFrame({
            'ds': prices.index,
            'y': prices.values
        })
        
        # remove timezone info if present (prophet doesn't like it)
        if df['ds'].dt.tz is not None:
            df['ds'] = df['ds'].dt.tz_localize(None)
        
        # configure prophet
        model = Prophet(
            interval_width=confidence,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=0.05,  # regularization
        )
        
        # fit the model
        model.fit(df)
        
        # figure out the frequency for future dates
        freq = pd.infer_freq(prices.index)
        if freq is None:
            # estimate from data
            avg_diff = (prices.index[-1] - prices.index[0]) / len(prices)
            if avg_diff < timedelta(hours=1):
                freq = 'T'  # minutes
            elif avg_diff < timedelta(days=1):
                freq = 'H'  # hours
            else:
                freq = 'D'  # days
        
        # create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # predict
        forecast = model.predict(future)
        
        # extract results (only future predictions)
        future_mask = forecast['ds'] > df['ds'].max()
        future_forecast = forecast[future_mask]
        
        if future_forecast.empty:
            return None
        
        forecast_dates = pd.DatetimeIndex(future_forecast['ds'])
        predictions = pd.Series(future_forecast['yhat'].values, index=forecast_dates)
        lower = pd.Series(future_forecast['yhat_lower'].values, index=forecast_dates)
        upper = pd.Series(future_forecast['yhat_upper'].values, index=forecast_dates)
        
        # fitted values on historical data
        hist_forecast = forecast[~future_mask]
        historical_fitted = pd.Series(
            hist_forecast['yhat'].values,
            index=pd.DatetimeIndex(hist_forecast['ds'])
        )
        
        # calculate some metrics
        mae = np.mean(np.abs(historical_fitted.values - df['y'].values[:len(historical_fitted)]))
        mape = np.mean(np.abs((historical_fitted.values - df['y'].values[:len(historical_fitted)]) / df['y'].values[:len(historical_fitted)])) * 100
        
        return PredictionResult(
            model_name="Prophet",
            forecast_dates=forecast_dates,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            metrics={
                "mae": safe_float(mae),
                "mape": safe_float(mape),
                "periods_forecast": periods,
            },
            historical_fitted=historical_fitted,
        )
        
    except Exception as e:
        return None


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

def predict_with_linear_regression(
    prices: pd.Series,
    periods: int = 30,
    confidence: float = 0.95,
    features: List[str] = None,
) -> Optional[PredictionResult]:
    """
    Forecast using Linear Regression with technical features.
    Simple but interpretable model.
    
    Args:
        prices: Historical price series
        periods: Number of periods to forecast
        confidence: Confidence interval level
        features: List of features to use ('trend', 'momentum', 'volatility', 'ma')
    
    Returns:
        PredictionResult or None if prediction fails
    """
    if prices.empty or len(prices) < 30:
        return None
    
    if features is None:
        features = ['trend', 'momentum', 'ma_ratio']
    
    try:
        df = pd.DataFrame({'price': prices})
        
        # create features
        df['trend'] = np.arange(len(df))  # simple time trend
        df['momentum'] = df['price'].pct_change(5).fillna(0)  # 5-period momentum
        df['ma_short'] = df['price'].rolling(5).mean()
        df['ma_long'] = df['price'].rolling(20).mean()
        df['ma_ratio'] = (df['ma_short'] / df['ma_long']).fillna(1)
        df['volatility'] = df['price'].pct_change().rolling(10).std().fillna(0)
        df['log_price'] = np.log(df['price'].clip(lower=1e-10))
        
        # drop NaN rows
        df = df.dropna()
        
        if len(df) < 20:
            return None
        
        # prepare X and y
        feature_cols = ['trend', 'momentum', 'ma_ratio', 'volatility']
        X = df[feature_cols].values
        y = df['log_price'].values  # predict log price for stability
        
        # scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # fit model
        model = Ridge(alpha=1.0)  # ridge regression for regularization
        model.fit(X_scaled, y)
        
        # fitted values
        y_fitted = model.predict(X_scaled)
        fitted_prices = np.exp(y_fitted)
        historical_fitted = pd.Series(fitted_prices, index=df.index)
        
        # calculate residuals for confidence intervals
        residuals = y - y_fitted
        std_residual = np.std(residuals)
        
        # z-score for confidence interval
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # forecast future
        last_trend = df['trend'].iloc[-1]
        last_momentum = df['momentum'].iloc[-1]
        last_ma_ratio = df['ma_ratio'].iloc[-1]
        last_volatility = df['volatility'].iloc[-1]
        last_price = df['price'].iloc[-1]
        
        # generate future dates
        freq = pd.infer_freq(prices.index)
        if freq is None:
            avg_diff = (prices.index[-1] - prices.index[0]) / len(prices)
            if avg_diff < timedelta(hours=1):
                freq = '5T'
            elif avg_diff < timedelta(days=1):
                freq = 'H'
            else:
                freq = 'D'
        
        future_dates = pd.date_range(
            start=prices.index[-1] + pd.Timedelta(freq),
            periods=periods,
            freq=freq
        )
        
        # simple forecast: extrapolate trend with decay
        predictions_list = []
        lower_list = []
        upper_list = []
        
        for i in range(periods):
            # features for future point
            future_trend = last_trend + i + 1
            # decay momentum and volatility towards mean
            decay = 0.95 ** (i + 1)
            future_momentum = last_momentum * decay
            future_ma_ratio = 1 + (last_ma_ratio - 1) * decay
            future_volatility = last_volatility * decay + 0.01 * (1 - decay)
            
            X_future = np.array([[future_trend, future_momentum, future_ma_ratio, future_volatility]])
            X_future_scaled = scaler.transform(X_future)
            
            log_pred = model.predict(X_future_scaled)[0]
            
            # widen confidence interval over time
            uncertainty = std_residual * np.sqrt(1 + (i + 1) / len(df))
            
            predictions_list.append(np.exp(log_pred))
            lower_list.append(np.exp(log_pred - z * uncertainty))
            upper_list.append(np.exp(log_pred + z * uncertainty))
        
        forecast_dates = pd.DatetimeIndex(future_dates)
        predictions = pd.Series(predictions_list, index=forecast_dates)
        lower = pd.Series(lower_list, index=forecast_dates)
        upper = pd.Series(upper_list, index=forecast_dates)
        
        # metrics
        mae = np.mean(np.abs(fitted_prices - df['price'].values))
        mape = np.mean(np.abs((fitted_prices - df['price'].values) / df['price'].values)) * 100
        
        return PredictionResult(
            model_name="Linear Regression",
            forecast_dates=forecast_dates,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            metrics={
                "mae": safe_float(mae),
                "mape": safe_float(mape),
                "r_squared": safe_float(model.score(X_scaled, y)),
                "periods_forecast": periods,
            },
            historical_fitted=historical_fitted,
        )
        
    except Exception as e:
        return None


# ============================================================
# MOMENTUM-BASED FORECAST
# ============================================================

def predict_with_momentum(
    prices: pd.Series,
    periods: int = 30,
    confidence: float = 0.95,
    lookback: int = 20,
) -> Optional[PredictionResult]:
    """
    Simple momentum-based forecast.
    Projects recent trend forward with volatility-based confidence intervals.
    Fast and doesn't require external libraries.
    
    Args:
        prices: Historical price series
        periods: Number of periods to forecast
        confidence: Confidence interval level
        lookback: Periods to use for momentum calculation
    
    Returns:
        PredictionResult or None if prediction fails
    """
    if prices.empty or len(prices) < lookback + 5:
        return None
    
    try:
        # calculate momentum (average return over lookback period)
        returns = prices.pct_change().dropna()
        recent_returns = returns.iloc[-lookback:]
        
        avg_return = recent_returns.mean()
        volatility = recent_returns.std()
        
        # z-score for confidence interval
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # generate future dates
        freq = pd.infer_freq(prices.index)
        if freq is None:
            avg_diff = (prices.index[-1] - prices.index[0]) / len(prices)
            if avg_diff < timedelta(hours=1):
                freq = '5T'
            elif avg_diff < timedelta(days=1):
                freq = 'H'
            else:
                freq = 'D'
        
        future_dates = pd.date_range(
            start=prices.index[-1] + pd.Timedelta(freq),
            periods=periods,
            freq=freq
        )
        
        # project forward
        last_price = prices.iloc[-1]
        predictions_list = []
        lower_list = []
        upper_list = []
        
        for i in range(periods):
            # compound the expected return
            expected_price = last_price * (1 + avg_return) ** (i + 1)
            
            # uncertainty grows with square root of time
            cumulative_std = volatility * np.sqrt(i + 1)
            
            predictions_list.append(expected_price)
            lower_list.append(expected_price * (1 - z * cumulative_std))
            upper_list.append(expected_price * (1 + z * cumulative_std))
        
        forecast_dates = pd.DatetimeIndex(future_dates)
        predictions = pd.Series(predictions_list, index=forecast_dates)
        lower = pd.Series(lower_list, index=forecast_dates)
        upper = pd.Series(upper_list, index=forecast_dates)
        
        # simple historical fitted (just the actual prices)
        historical_fitted = prices.copy()
        
        return PredictionResult(
            model_name="Momentum",
            forecast_dates=forecast_dates,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            metrics={
                "avg_daily_return": safe_float(avg_return),
                "volatility": safe_float(volatility),
                "lookback_periods": lookback,
                "periods_forecast": periods,
            },
            historical_fitted=historical_fitted,
        )
        
    except Exception as e:
        return None


# ============================================================
# EXPONENTIAL SMOOTHING FORECAST
# ============================================================

def predict_with_exp_smoothing(
    prices: pd.Series,
    periods: int = 30,
    confidence: float = 0.95,
    alpha: float = 0.3,
) -> Optional[PredictionResult]:
    """
    Simple exponential smoothing forecast.
    Good for data without strong trend or seasonality.
    
    Args:
        prices: Historical price series
        periods: Number of periods to forecast
        confidence: Confidence interval level
        alpha: Smoothing parameter (0-1), higher = more weight on recent data
    """
    if prices.empty or len(prices) < 10:
        return None
    
    try:
        # calculate exponential smoothing
        smoothed = prices.ewm(alpha=alpha, adjust=False).mean()
        
        # calculate residuals for CI
        residuals = prices - smoothed
        std_residual = residuals.std()
        
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # generate future dates
        freq = pd.infer_freq(prices.index)
        if freq is None:
            avg_diff = (prices.index[-1] - prices.index[0]) / len(prices)
            if avg_diff < timedelta(hours=1):
                freq = '5T'
            elif avg_diff < timedelta(days=1):
                freq = 'H'
            else:
                freq = 'D'
        
        future_dates = pd.date_range(
            start=prices.index[-1] + pd.Timedelta(freq),
            periods=periods,
            freq=freq
        )
        
        # forecast is just the last smoothed value
        last_smoothed = smoothed.iloc[-1]
        
        predictions_list = []
        lower_list = []
        upper_list = []
        
        for i in range(periods):
            # constant forecast with widening CI
            pred = last_smoothed
            uncertainty = std_residual * np.sqrt(1 + i * 0.1)  # CI widens slowly
            
            predictions_list.append(pred)
            lower_list.append(pred - z * uncertainty)
            upper_list.append(pred + z * uncertainty)
        
        forecast_dates = pd.DatetimeIndex(future_dates)
        predictions = pd.Series(predictions_list, index=forecast_dates)
        lower = pd.Series(lower_list, index=forecast_dates)
        upper = pd.Series(upper_list, index=forecast_dates)
        
        # metrics
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / prices)) * 100
        
        return PredictionResult(
            model_name="Exponential Smoothing",
            forecast_dates=forecast_dates,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            metrics={
                "mae": safe_float(mae),
                "mape": safe_float(mape),
                "alpha": alpha,
                "periods_forecast": periods,
            },
            historical_fitted=smoothed,
        )
        
    except Exception as e:
        return None


# ============================================================
# MAIN PREDICTION INTERFACE
# ============================================================

def run_prediction(
    prices: pd.Series,
    model: str = "Prophet",
    periods: int = 30,
    confidence: float = 0.95,
    **kwargs
) -> Optional[PredictionResult]:
    """
    Main interface to run predictions.
    Tries the requested model, falls back to simpler ones if it fails.
    
    Args:
        prices: Historical price series with datetime index
        model: Model to use ('Prophet', 'Linear Regression', 'Momentum', 'Exponential Smoothing')
        periods: Number of periods to forecast
        confidence: Confidence interval level
        **kwargs: Additional model-specific parameters
    
    Returns:
        PredictionResult or None
    """
    if prices.empty:
        return None
    
    # clean the series
    prices = prices.dropna()
    if len(prices) < 10:
        return None
    
    model = model.lower().strip()
    
    # try requested model
    if model == "prophet":
        result = predict_with_prophet(prices, periods, confidence, **kwargs)
        if result:
            return result
    
    if model in ["linear regression", "linear", "lr"]:
        result = predict_with_linear_regression(prices, periods, confidence, **kwargs)
        if result:
            return result
    
    if model == "momentum":
        result = predict_with_momentum(prices, periods, confidence, **kwargs)
        if result:
            return result
    
    if model in ["exponential smoothing", "exp smoothing", "ets"]:
        result = predict_with_exp_smoothing(prices, periods, confidence, **kwargs)
        if result:
            return result
    
    # fallback chain: try each model until one works
    for fallback_model in [predict_with_momentum, predict_with_exp_smoothing, predict_with_linear_regression]:
        result = fallback_model(prices, periods, confidence)
        if result:
            return result
    
    return None


def run_all_predictions(
    prices: pd.Series,
    periods: int = 30,
    confidence: float = 0.95,
) -> Dict[str, PredictionResult]:
    """
    Run all available models and return results.
    Useful for model comparison.
    """
    results = {}
    
    models = [
        ("Prophet", predict_with_prophet),
        ("Linear Regression", predict_with_linear_regression),
        ("Momentum", predict_with_momentum),
        ("Exponential Smoothing", predict_with_exp_smoothing),
    ]
    
    for name, func in models:
        try:
            result = func(prices, periods, confidence)
            if result:
                results[name] = result
        except Exception:
            continue
    
    return results


# ============================================================
# PREDICTION UTILITIES
# ============================================================

def evaluate_prediction(
    actual: pd.Series,
    predicted: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate prediction accuracy given actual values.
    """
    # align the series
    common_idx = actual.index.intersection(predicted.index)
    
    if len(common_idx) == 0:
        return {
            "mae": np.nan,
            "mape": np.nan,
            "rmse": np.nan,
            "direction_accuracy": np.nan,
        }
    
    actual = actual.loc[common_idx]
    predicted = predicted.loc[common_idx]
    
    errors = actual - predicted
    
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actual)) * 100
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # direction accuracy: did we predict the right direction of movement?
    actual_direction = np.sign(actual.diff().dropna())
    predicted_direction = np.sign(predicted.diff().dropna())
    common_dir_idx = actual_direction.index.intersection(predicted_direction.index)
    
    if len(common_dir_idx) > 0:
        direction_accuracy = (actual_direction.loc[common_dir_idx] == predicted_direction.loc[common_dir_idx]).mean()
    else:
        direction_accuracy = np.nan
    
    return {
        "mae": safe_float(mae),
        "mape": safe_float(mape),
        "rmse": safe_float(rmse),
        "direction_accuracy": safe_float(direction_accuracy),
    }


def get_available_models() -> List[str]:
    """Return list of available prediction models."""
    return ["Prophet", "Linear Regression", "Momentum", "Exponential Smoothing"]


def get_model_description(model: str) -> str:
    """Get a short description of each model."""
    descriptions = {
        "Prophet": "Facebook's Prophet model. Great for capturing trends and seasonality. Works best with daily data over longer periods.",
        "Linear Regression": "Ridge regression with technical features. Simple, interpretable, and fast.",
        "Momentum": "Projects recent momentum forward. Fast and intuitive but assumes trends continue.",
        "Exponential Smoothing": "Weighted average favoring recent data. Good for stable series without strong trends.",
    }
    return descriptions.get(model, "No description available.")