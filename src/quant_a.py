"""
Quant A - Single Asset Analysis Module
Professional backtesting strategies, performance metrics, and ML predictions for individual assets.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from src.data_loader import fetch_data, get_ticker_info, get_latest_price
from src.predictions import run_prediction, get_available_models, get_model_description, PredictionResult
from src.utils import (
    compute_all_metrics,
    compute_returns,
    compute_equity_curve,
    compute_drawdown_series,
    get_periods_per_year,
    format_percentage,
    format_number,
    safe_float,
    is_valid_combination,
    suggest_valid_combination,
    count_trades,
    COLORS,
    POPULAR_TICKERS,
    TICKER_CATEGORIES,
    PERIOD_OPTIONS,
    get_interval_for_period,
)


# ============================================================
# DESIGN SYSTEM CONSTANTS
# ============================================================
THEME = {
    "bg_primary": "#0a0e14",
    "bg_secondary": "#111820",
    "bg_tertiary": "#1a2332",
    "bg_hover": "#242d3d",
    "accent_primary": "#00d4aa",
    "accent_secondary": "#6366f1",
    "accent_gold": "#f59e0b",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "border": "#1e293b",
    "border_light": "#334155",
}


# ============================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================
@dataclass
class StrategyResult:
    """Container for strategy backtest results."""
    name: str
    data: pd.DataFrame
    signals: pd.Series
    returns: pd.Series
    equity_curve: pd.Series
    metrics: Dict[str, float]
    params: Dict


def apply_sma_crossover(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """SMA Crossover Strategy - Buy when short MA crosses above long MA."""
    df = df.copy()
    df['SMA_Short'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1.0, 0.0)
    return df


def apply_ema_crossover(df: pd.DataFrame, short_window: int = 12, long_window: int = 26) -> pd.DataFrame:
    """EMA Crossover Strategy - Uses exponential moving averages for faster response."""
    df = df.copy()
    df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['Signal'] = np.where(df['EMA_Short'] > df['EMA_Long'], 1.0, 0.0)
    return df


def apply_rsi_strategy(df: pd.DataFrame, window: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """RSI Mean Reversion Strategy - Buy when RSI is oversold, sell when overbought."""
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    signals = np.zeros(len(df))
    position = 0
    for i in range(len(df)):
        if df['RSI'].iloc[i] < oversold:
            position = 1
        elif df['RSI'].iloc[i] > overbought:
            position = 0
        signals[i] = position
    
    df['Signal'] = signals
    df['RSI_Oversold'] = oversold
    df['RSI_Overbought'] = overbought
    return df


def apply_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands Strategy - Buy at lower band, sell at upper band."""
    df = df.copy()
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    
    signals = np.zeros(len(df))
    position = 0
    for i in range(window, len(df)):
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            position = 1
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            position = 0
        signals[i] = position
    
    df['Signal'] = signals
    return df


def apply_macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """MACD Strategy - Buy when MACD crosses above signal line."""
    df = df.copy()
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    df['Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1.0, 0.0)
    return df


def apply_momentum_strategy(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.0) -> pd.DataFrame:
    """Momentum Strategy - Buy when momentum is positive."""
    df = df.copy()
    df['Momentum'] = df['Close'].pct_change(periods=lookback)
    df['Signal'] = np.where(df['Momentum'] > threshold, 1.0, 0.0)
    return df


def apply_mean_reversion(df: pd.DataFrame, window: int = 20, z_threshold: float = -1.5) -> pd.DataFrame:
    """Mean Reversion Strategy - Buy when z-score indicates oversold."""
    df = df.copy()
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['Z_Score'] = (df['Close'] - rolling_mean) / rolling_std
    
    signals = np.zeros(len(df))
    position = 0
    for i in range(window, len(df)):
        if df['Z_Score'].iloc[i] < z_threshold:
            position = 1
        elif df['Z_Score'].iloc[i] > 0:
            position = 0
        signals[i] = position
    
    df['Signal'] = signals
    return df


def apply_buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Buy & Hold Strategy - Simply hold the asset for the entire period."""
    df = df.copy()
    df['Signal'] = 1.0
    return df


# ============================================================
# STRATEGY RUNNER
# ============================================================
STRATEGIES = {
    "SMA Crossover": apply_sma_crossover,
    "EMA Crossover": apply_ema_crossover,
    "RSI Mean Reversion": apply_rsi_strategy,
    "Bollinger Bands": apply_bollinger_bands,
    "MACD": apply_macd_strategy,
    "Momentum": apply_momentum_strategy,
    "Mean Reversion": apply_mean_reversion,
    "Buy & Hold": apply_buy_and_hold,
}


def run_strategy(df: pd.DataFrame, strategy_name: str, interval: str = "1d", **params) -> Optional[StrategyResult]:
    """Run a strategy and compute performance metrics."""
    if df.empty or 'Close' not in df.columns:
        return None
    
    if strategy_name not in STRATEGIES:
        return None
    
    try:
        result_df = STRATEGIES[strategy_name](df.copy(), **params)
        
        result_df['Market_Return'] = result_df['Close'].pct_change()
        result_df['Strategy_Return'] = result_df['Signal'].shift(1) * result_df['Market_Return']
        result_df['Strategy_Return'] = result_df['Strategy_Return'].fillna(0)
        
        result_df['Market_Equity'] = (1 + result_df['Market_Return'].fillna(0)).cumprod() * 100
        result_df['Strategy_Equity'] = (1 + result_df['Strategy_Return']).cumprod() * 100
        
        signals = result_df['Signal']
        strategy_returns = result_df['Strategy_Return'].dropna()
        equity_curve = result_df['Strategy_Equity']
        
        periods_per_year = get_periods_per_year(interval)
        metrics = compute_all_metrics(strategy_returns, equity_curve, periods_per_year)
        metrics['num_trades'] = count_trades(signals)
        
        return StrategyResult(
            name=strategy_name,
            data=result_df,
            signals=signals,
            returns=strategy_returns,
            equity_curve=equity_curve,
            metrics=metrics,
            params=params,
        )
        
    except Exception as e:
        return None


def get_current_signal(df: pd.DataFrame, strategy_name: str, **params) -> str:
    """Get the current trading signal for display."""
    if df.empty:
        return "HOLD"
    
    try:
        result_df = STRATEGIES[strategy_name](df.copy(), **params)
        last_signal = result_df['Signal'].iloc[-1]
        
        if len(result_df) > 1:
            prev_signal = result_df['Signal'].iloc[-2]
            if last_signal > prev_signal:
                return "BUY"
            elif last_signal < prev_signal:
                return "SELL"
        
        return "LONG" if last_signal == 1 else "FLAT"
    except:
        return "HOLD"


# ============================================================
# PLOTLY CHARTS - FIXED COLORS (using rgba format)
# ============================================================
def create_main_chart(df: pd.DataFrame, strategy_result: StrategyResult, ticker: str, show_signals: bool = True) -> go.Figure:
    """Create the main chart showing price and strategy equity curve."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=(None, None)
    )
    
    data = strategy_result.data
    
    # Price line (accent primary - teal)
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data['Close'],
            name="Price",
            line=dict(color=THEME['accent_primary'], width=2),
            hovertemplate="%{y:.2f}<extra>Price</extra>"
        ), row=1, col=1
    )
    
    # Normalize equity to price scale
    price_start = data['Close'].iloc[0]
    equity_normalized = (data['Strategy_Equity'] / 100) * price_start
    
    # Strategy line (accent secondary - indigo)
    fig.add_trace(
        go.Scatter(
            x=data.index, y=equity_normalized,
            name="Strategy",
            line=dict(color=THEME['accent_secondary'], width=2),
            hovertemplate="%{y:.2f}<extra>Strategy</extra>"
        ), row=1, col=1
    )
    
    # Buy/sell signals
    if show_signals:
        signal_changes = data['Signal'].diff().fillna(0)
        
        buy_mask = signal_changes == 1
        if buy_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data.index[buy_mask], y=data['Close'][buy_mask],
                    mode='markers', name='Buy',
                    marker=dict(color=THEME['success'], size=10, symbol='triangle-up'),
                ), row=1, col=1
            )
        
        sell_mask = signal_changes == -1
        if sell_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data.index[sell_mask], y=data['Close'][sell_mask],
                    mode='markers', name='Sell',
                    marker=dict(color=THEME['danger'], size=10, symbol='triangle-down'),
                ), row=1, col=1
            )
    
    # Drawdown subplot - FIXED: using rgba format
    drawdown = compute_drawdown_series(data['Strategy_Equity'])
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown * 100,
            name='Drawdown', fill='tozeroy',
            line=dict(color=THEME['danger'], width=1),
            fillcolor='rgba(239, 68, 68, 0.2)',
        ), row=2, col=1
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="left", 
            x=0,
            font=dict(size=11)
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=30, b=50),
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(30, 41, 59, 0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(30, 41, 59, 0.5)')
    fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=11))
    fig.update_yaxes(title_text="DD %", row=2, col=1, title_font=dict(size=11))
    
    return fig


def create_indicator_chart(df: pd.DataFrame, strategy_name: str) -> Optional[go.Figure]:
    """Create a chart showing the strategy's technical indicators."""
    data = df.copy()
    
    fig = go.Figure()
    
    if strategy_name == "RSI Mean Reversion" and 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'], 
            name='RSI', 
            line=dict(color=THEME['accent_primary'], width=2)
        ))
        fig.add_hline(y=70, line_dash="dash", line_color=THEME['danger'], annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color=THEME['success'], annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dot", line_color=THEME['text_muted'], opacity=0.5)
        fig.update_layout(yaxis=dict(range=[0, 100]))
        
    elif strategy_name == "MACD" and 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'], 
            name='MACD', 
            line=dict(color=THEME['info'], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD_Signal'], 
            name='Signal', 
            line=dict(color=THEME['accent_gold'], width=2)
        ))
        colors = [THEME['success'] if v >= 0 else THEME['danger'] for v in data['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=data.index, y=data['MACD_Histogram'], 
            name='Histogram', 
            marker_color=colors, 
            opacity=0.6
        ))
        
    elif strategy_name == "Bollinger Bands" and 'BB_Upper' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'], 
            name='Upper', 
            line=dict(color=THEME['danger'], width=1)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'], 
            name='Lower', 
            line=dict(color=THEME['success'], width=1), 
            fill='tonexty', 
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], 
            name='Price', 
            line=dict(color=THEME['accent_primary'], width=2)
        ))
        
    elif "SMA" in strategy_name or "EMA" in strategy_name:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], 
            name='Price', 
            line=dict(color=THEME['accent_primary'], width=2)
        ))
        if 'SMA_Short' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_Short'], 
                name='Short MA', 
                line=dict(color=THEME['success'], width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_Long'], 
                name='Long MA', 
                line=dict(color=THEME['danger'], width=1.5)
            ))
        elif 'EMA_Short' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['EMA_Short'], 
                name='Short EMA', 
                line=dict(color=THEME['success'], width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['EMA_Long'], 
                name='Long EMA', 
                line=dict(color=THEME['danger'], width=1.5)
            ))
    else:
        return None
    
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=60, r=40, t=30, b=30),
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(30, 41, 59, 0.5)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(30, 41, 59, 0.5)')
    
    return fig


def create_prediction_chart(historical_prices: pd.Series, prediction: PredictionResult, ticker: str) -> go.Figure:
    """Create chart showing historical prices and future predictions."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_prices.index, y=historical_prices.values,
        name='Historical', 
        line=dict(color=THEME['accent_primary'], width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction.forecast_dates, y=prediction.predictions.values,
        name='Forecast', 
        line=dict(color=THEME['accent_gold'], width=2, dash='dash'),
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction.forecast_dates, y=prediction.upper_bound.values,
        name='Upper Bound',
        line=dict(color=THEME['text_muted'], width=0),
        showlegend=False,
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction.forecast_dates, y=prediction.lower_bound.values,
        name=f'{int(prediction.confidence_level*100)}% CI',
        fill='tonexty',
        fillcolor='rgba(245, 158, 11, 0.15)',
        line=dict(color=THEME['text_muted'], width=0),
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=40, t=30, b=50),
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        hovermode="x unified",
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(30, 41, 59, 0.5)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(30, 41, 59, 0.5)', title_text="Price")
    
    return fig


# ============================================================
# MAIN RENDER FUNCTION
# ============================================================
def render_quant_a(show_header: bool = False) -> None:
    """Main render function for Quant A module."""
    
    # Custom CSS for this module
    st.markdown(f"""
        <style>
        .section-title {{
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {THEME['text_muted']};
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-title::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, {THEME['border']} 0%, transparent 100%);
        }}
        .metric-box {{
            background: {THEME['bg_tertiary']};
            border: 1px solid {THEME['border']};
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            transition: all 0.2s ease;
        }}
        .metric-box:hover {{
            border-color: {THEME['border_light']};
            background: {THEME['bg_hover']};
        }}
        .metric-label {{
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {THEME['text_muted']};
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .v-positive {{ color: {THEME['success']}; }}
        .v-negative {{ color: {THEME['danger']}; }}
        .v-neutral {{ color: {THEME['text_primary']}; }}
        .v-accent {{ color: {THEME['accent_primary']}; }}
        </style>
    """, unsafe_allow_html=True)
    
    # Layout: Settings | Main Content
    col_settings, col_main = st.columns([1, 3.5], gap="large")
    
    with col_settings:
        with st.container(border=True):
            st.markdown("### ⚙️ Settings")
            st.markdown("---")
            
            # Asset Selection
            st.caption("ASSET SELECTION")
            
            category = st.selectbox(
                "Category",
                options=["All"] + list(TICKER_CATEGORIES.keys()),
                index=0,
            )
            
            if category == "All":
                ticker_options = POPULAR_TICKERS
            else:
                ticker_options = TICKER_CATEGORIES[category]
            
            ticker = st.selectbox("Ticker", options=ticker_options, index=0)
            
            st.markdown("---")
            
            # Time Period
            st.caption("TIME PERIOD")
            period = st.selectbox(
                "Period",
                options=list(PERIOD_OPTIONS.keys()),
                index=4,
                format_func=lambda x: PERIOD_OPTIONS[x],
            )
            interval = get_interval_for_period(period)
            st.caption(f"📊 Interval: {interval}")
            
            st.markdown("---")
            
            # Strategy Selection
            st.caption("STRATEGY")
            strategy = st.selectbox("Strategy", options=list(STRATEGIES.keys()), index=0)
            
            # Strategy Parameters
            strategy_params = {}
            
            if strategy == "SMA Crossover":
                strategy_params['short_window'] = st.slider("Short Window", 5, 50, 20)
                strategy_params['long_window'] = st.slider("Long Window", 20, 200, 50)
            elif strategy == "EMA Crossover":
                strategy_params['short_window'] = st.slider("Short Window", 5, 30, 12)
                strategy_params['long_window'] = st.slider("Long Window", 15, 50, 26)
            elif strategy == "RSI Mean Reversion":
                strategy_params['window'] = st.slider("RSI Period", 5, 30, 14)
                strategy_params['oversold'] = st.slider("Oversold Level", 10, 40, 30)
                strategy_params['overbought'] = st.slider("Overbought Level", 60, 90, 70)
            elif strategy == "Bollinger Bands":
                strategy_params['window'] = st.slider("Window", 10, 50, 20)
                strategy_params['std_dev'] = st.slider("Std Dev", 1.0, 3.0, 2.0, 0.1)
            elif strategy == "MACD":
                strategy_params['fast'] = st.slider("Fast Period", 5, 20, 12)
                strategy_params['slow'] = st.slider("Slow Period", 15, 40, 26)
                strategy_params['signal_period'] = st.slider("Signal Period", 5, 15, 9)
            elif strategy == "Momentum":
                strategy_params['lookback'] = st.slider("Lookback", 5, 60, 20)
                strategy_params['threshold'] = st.slider("Threshold", -0.1, 0.1, 0.0, 0.01)
            elif strategy == "Mean Reversion":
                strategy_params['window'] = st.slider("Window", 10, 60, 20)
                strategy_params['z_threshold'] = st.slider("Z-Score Threshold", -3.0, -0.5, -1.5, 0.1)
    
    with col_main:
        # Fetch Data
        with st.spinner("Loading market data..."):
            df = fetch_data(ticker, period=period, interval=interval)
        
        if df.empty:
            st.error(f"Could not load data for {ticker}. Please try another ticker.")
            return
        
        # Run Strategy
        result = run_strategy(df, strategy, interval=interval, **strategy_params)
        
        if result is None:
            st.error("Strategy calculation failed. Please adjust parameters.")
            return
        
        # Current Signal
        current_signal = get_current_signal(df, strategy, **strategy_params)
        
        # Top Metrics Row
        st.markdown('<div class="section-title"> Performance Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        price_data = get_latest_price(ticker)
        
        with col1:
            if price_data:
                change_class = "v-positive" if price_data['is_positive'] else "v-negative"
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value v-neutral">${price_data['price']:,.2f}</div>
                        <div style="font-size: 11px;" class="{change_class}">{price_data['change_pct']:+.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            ret_class = "v-positive" if result.metrics['total_return'] >= 0 else "v-negative"
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {ret_class}">{result.metrics['total_return']*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value v-accent">{result.metrics['sharpe_ratio']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value v-negative">{result.metrics['max_drawdown']*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            signal_color = THEME['success'] if "BUY" in current_signal or "LONG" in current_signal else THEME['danger']
            if current_signal in ["HOLD", "FLAT"]:
                signal_color = THEME['text_muted']
            st.markdown(f"""
                <div class="metric-box" style="border-left: 3px solid {signal_color};">
                    <div class="metric-label">Signal</div>
                    <div class="metric-value" style="color: {signal_color};">{current_signal}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Main Chart
        st.markdown("<br>", unsafe_allow_html=True)
        show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
        fig = create_main_chart(df, result, ticker, show_signals)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabs for Additional Analysis
        tab1, tab2, tab3, tab4 = st.tabs([" Indicators", " Predictions", " Metrics", " Data"])
        
        with tab1:
            indicator_chart = create_indicator_chart(result.data, strategy)
            if indicator_chart:
                st.plotly_chart(indicator_chart, use_container_width=True)
            else:
                st.info("No specific indicator chart for this strategy.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Signal Distribution
            signal_counts = result.signals.value_counts()
            total = len(result.signals)
            long_pct = signal_counts.get(1.0, 0) / total * 100
            cash_pct = signal_counts.get(0.0, 0) / total * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Periods", f"{total:,}")
            col2.metric("Time in Market", f"{long_pct:.1f}%")
            col3.metric("Time in Cash", f"{cash_pct:.1f}%")
        
        with tab2:
            st.markdown("#### 🔮 Price Prediction")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                pred_model = st.selectbox("Model", get_available_models(), index=2)
            with col2:
                pred_periods = st.slider("Forecast Periods", 5, 60, 20)
            with col3:
                pred_confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            
            st.caption(f"ℹ️ {get_model_description(pred_model)}")
            
            if st.button("Run Prediction", type="primary"):
                with st.spinner(f"Running {pred_model} model..."):
                    prediction = run_prediction(
                        df['Close'], model=pred_model,
                        periods=pred_periods, confidence=pred_confidence
                    )
                
                if prediction:
                    fig = create_prediction_chart(df['Close'], prediction, ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        last_price = df['Close'].iloc[-1]
                        pred_price = prediction.predictions.iloc[-1]
                        change = (pred_price - last_price) / last_price
                        st.metric(
                            f"Predicted ({pred_periods} periods)",
                            f"${pred_price:.2f}",
                            f"{change*100:+.2f}%"
                        )
                    with col2:
                        st.metric(
                            "Prediction Range",
                            f"${prediction.lower_bound.iloc[-1]:.2f} - ${prediction.upper_bound.iloc[-1]:.2f}"
                        )
                else:
                    st.error("Prediction failed. Try a different model or check data availability.")
        
        with tab3:
            st.markdown("#### 📊 Performance Analysis")
            
            # Performance Metrics
            perf_metrics = [
                ("Total Return", result.metrics['total_return'] * 100, "%", result.metrics['total_return'] >= 0),
                ("Annualized Return", result.metrics['annualized_return'] * 100, "%", result.metrics['annualized_return'] >= 0),
                ("Volatility", result.metrics['volatility'] * 100, "%", None),
                ("Max Drawdown", result.metrics['max_drawdown'] * 100, "%", False),
            ]
            
            cols = st.columns(4)
            for col, (label, val, unit, is_positive) in zip(cols, perf_metrics):
                if is_positive is None:
                    val_class = "v-neutral"
                elif is_positive:
                    val_class = "v-positive"
                else:
                    val_class = "v-negative"
                
                with col:
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value {val_class}">{val:.2f}{unit}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📈 Risk & Ratios")
            
            risk_metrics = [
                ("Sharpe Ratio", result.metrics['sharpe_ratio'], "", result.metrics['sharpe_ratio'] > 1),
                ("Sortino Ratio", result.metrics['sortino_ratio'], "", result.metrics['sortino_ratio'] > 1),
                ("Calmar Ratio", result.metrics['calmar_ratio'], "", result.metrics['calmar_ratio'] > 1),
                ("VaR (95%)", result.metrics['var_95'] * 100, "%", False),
            ]
            
            cols = st.columns(4)
            for col, (label, val, unit, is_good) in zip(cols, risk_metrics):
                val_class = "v-positive" if is_good else "v-neutral" if unit == "" else "v-negative"
                with col:
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value {val_class}">{val:.2f}{unit}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Execution Stats")
            
            cols = st.columns(3)
            with cols[0]:
                wr = result.metrics['win_rate'] * 100
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value v-accent">{wr:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                pf = result.metrics['profit_factor']
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value v-neutral">{pf:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                trades = result.metrics['total_trades']
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value v-neutral">{trades}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown(f"**Data Points:** {len(df):,} | **Period:** {period} | **Interval:** {interval}")
            
            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Signal']
            display_df = result.data[display_cols].tail(50)
            
            st.dataframe(display_df, use_container_width=True)
            
            csv = result.data.to_csv()
            st.download_button(
                "📥 Download CSV",
                csv,
                f"{ticker}_{strategy}_{period}.csv",
                "text/csv"
            )


def render() -> None:
    """Entry point for the module."""
    render_quant_a(show_header=False)