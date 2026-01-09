"""
Unit tests for Quant Terminal strategies and utilities.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quant_a import (
    apply_sma_crossover,
    apply_ema_crossover,
    apply_rsi_strategy,
    apply_bollinger_bands,
    apply_macd_strategy,
    apply_momentum_strategy,
    apply_buy_and_hold,
    run_strategy,
    STRATEGIES,
)
from src.utils import (
    compute_returns,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_volatility,
    safe_float,
    is_valid_combination,
)


def create_mock_data(n=100, trend=True):
    """Create mock price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    
    if trend:
        prices = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 2
    else:
        prices = 100 + np.random.randn(n) * 5
    
    prices = np.maximum(prices, 1)
    
    df = pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.02,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1000000, 10000000, n),
    }, index=dates)
    
    return df


class TestStrategies:
    """Test trading strategy implementations."""
    
    def test_sma_crossover(self):
        df = create_mock_data(100)
        result = apply_sma_crossover(df, short_window=10, long_window=20)
        
        assert "SMA_Short" in result.columns
        assert "SMA_Long" in result.columns
        assert "Signal" in result.columns
        assert result["Signal"].isin([0.0, 1.0]).all()
    
    def test_ema_crossover(self):
        df = create_mock_data(100)
        result = apply_ema_crossover(df, short_window=12, long_window=26)
        
        assert "EMA_Short" in result.columns
        assert "EMA_Long" in result.columns
        assert "Signal" in result.columns
    
    def test_rsi_strategy(self):
        df = create_mock_data(100)
        result = apply_rsi_strategy(df, window=14, oversold=30, overbought=70)
        
        assert "RSI" in result.columns
        assert "Signal" in result.columns
        assert result["RSI"].min() >= 0
        assert result["RSI"].max() <= 100
    
    def test_bollinger_bands(self):
        df = create_mock_data(100)
        result = apply_bollinger_bands(df, window=20, std_dev=2.0)
        
        assert "BB_Upper" in result.columns
        assert "BB_Middle" in result.columns
        assert "BB_Lower" in result.columns
        assert "Signal" in result.columns
    
    def test_macd_strategy(self):
        df = create_mock_data(100)
        result = apply_macd_strategy(df, fast=12, slow=26, signal_period=9)
        
        assert "MACD" in result.columns
        assert "MACD_Signal" in result.columns
        assert "MACD_Histogram" in result.columns
    
    def test_momentum_strategy(self):
        df = create_mock_data(100)
        result = apply_momentum_strategy(df, lookback=20)
        
        assert "Momentum" in result.columns
        assert "Signal" in result.columns
    
    def test_buy_and_hold(self):
        df = create_mock_data(100)
        result = apply_buy_and_hold(df)
        
        assert "Signal" in result.columns
        assert (result["Signal"] == 1.0).all()
    
    def test_run_strategy(self):
        df = create_mock_data(100, trend=True)
        result = run_strategy(df, "SMA Crossover", interval="1d")
        
        assert result is not None
        assert result.name == "SMA Crossover"
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
    
    def test_all_strategies_run(self):
        df = create_mock_data(200)
        
        for name in STRATEGIES.keys():
            result = run_strategy(df, name, interval="1d")
            assert result is not None, f"Strategy {name} failed"


class TestUtils:
    """Test utility functions."""
    
    def test_compute_returns(self):
        prices = pd.Series([100, 101, 102, 101, 103])
        returns = compute_returns(prices)
        
        assert len(returns) == 4
        assert not returns.isnull().any()
    
    def test_compute_max_drawdown(self):
        equity = pd.Series([100, 110, 105, 115, 100, 120])
        mdd = compute_max_drawdown(equity)
        
        assert mdd < 0
        assert mdd >= -1
    
    def test_compute_sharpe_ratio(self):
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
        sharpe = compute_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_compute_volatility(self):
        returns = pd.Series(np.random.randn(100) * 0.02)
        vol = compute_volatility(returns, annualize=True)
        
        assert vol > 0
        assert vol < 2  # Reasonable annualized vol
    
    def test_safe_float(self):
        assert safe_float(1.5) == 1.5
        assert safe_float(None) == 0.0
        assert safe_float(np.nan) == 0.0
        assert safe_float(np.inf) == 0.0
        assert safe_float("invalid") == 0.0
    
    def test_is_valid_combination(self):
        assert is_valid_combination("1d", "5m") == True
        assert is_valid_combination("1y", "1m") == False
        assert is_valid_combination("1y", "1d") == True


class TestEmptyData:
    """Test handling of edge cases."""
    
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = run_strategy(df, "SMA Crossover")
        assert result is None
    
    def test_insufficient_data(self):
        df = create_mock_data(5)
        result = run_strategy(df, "SMA Crossover", short_window=10, long_window=20)
        # Should still work but with limited signals
        assert result is not None
    
    def test_empty_returns(self):
        returns = pd.Series(dtype=float)
        
        assert compute_volatility(returns) == 0.0
        assert compute_sharpe_ratio(returns) == 0.0
        assert compute_max_drawdown(returns) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])