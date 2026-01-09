"""
Quant A - Single Asset Analysis Module
Implements backtesting strategies, performance metrics, and ML predictions for individual assets.
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
    ASSET_PRESETS,
)


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


def apply_sma_crossover(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    """
    SMA Crossover Strategy.
    Buy when short MA crosses above long MA, sell when it crosses below.
    """
    df = df.copy()
    
    df['SMA_Short'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # signal: 1 when short > long, 0 otherwise
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1.0, 0.0)
    
    return df


def apply_ema_crossover(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26
) -> pd.DataFrame:
    """
    EMA Crossover Strategy.
    Similar to SMA but uses exponential moving averages for faster response.
    """
    df = df.copy()
    
    df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    
    df['Signal'] = np.where(df['EMA_Short'] > df['EMA_Long'], 1.0, 0.0)
    
    return df


def apply_rsi_strategy(
    df: pd.DataFrame,
    window: int = 14,
    oversold: int = 30,
    overbought: int = 70
) -> pd.DataFrame:
    """
    RSI Mean Reversion Strategy.
    Buy when RSI is oversold, sell when overbought.
    """
    df = df.copy()
    
    # calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # generate signals with position holding
    signals = np.zeros(len(df))
    position = 0
    
    for i in range(len(df)):
        if df['RSI'].iloc[i] < oversold:
            position = 1  # buy
        elif df['RSI'].iloc[i] > overbought:
            position = 0  # sell
        signals[i] = position
    
    df['Signal'] = signals
    df['RSI_Oversold'] = oversold
    df['RSI_Overbought'] = overbought
    
    return df


def apply_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands Strategy.
    Buy when price touches lower band, sell when it touches upper band.
    """
    df = df.copy()
    
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    
    # generate signals
    signals = np.zeros(len(df))
    position = 0
    
    for i in range(window, len(df)):
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            position = 1  # buy at lower band
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            position = 0  # sell at upper band
        signals[i] = position
    
    df['Signal'] = signals
    
    return df


def apply_macd_strategy(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    MACD Strategy.
    Buy when MACD crosses above signal line, sell when it crosses below.
    """
    df = df.copy()
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # signal: 1 when MACD > signal line
    df['Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1.0, 0.0)
    
    return df


def apply_momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Momentum Strategy.
    Buy when momentum (return over lookback period) is positive.
    """
    df = df.copy()
    
    df['Momentum'] = df['Close'].pct_change(periods=lookback)
    df['Signal'] = np.where(df['Momentum'] > threshold, 1.0, 0.0)
    
    return df


def apply_buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buy & Hold Strategy.
    Simply hold the asset for the entire period.
    """
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
    "Buy & Hold": apply_buy_and_hold,
}


def run_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    interval: str = "1d",
    **params
) -> Optional[StrategyResult]:
    """
    Run a strategy and compute performance metrics.
    """
    if df.empty or 'Close' not in df.columns:
        return None
    
    if strategy_name not in STRATEGIES:
        return None
    
    try:
        # apply strategy
        result_df = STRATEGIES[strategy_name](df.copy(), **params)
        
        # compute returns
        result_df['Market_Return'] = result_df['Close'].pct_change()
        result_df['Strategy_Return'] = result_df['Signal'].shift(1) * result_df['Market_Return']
        result_df['Strategy_Return'] = result_df['Strategy_Return'].fillna(0)
        
        # equity curves (base 100)
        result_df['Market_Equity'] = (1 + result_df['Market_Return'].fillna(0)).cumprod() * 100
        result_df['Strategy_Equity'] = (1 + result_df['Strategy_Return']).cumprod() * 100
        
        # get clean series
        signals = result_df['Signal']
        strategy_returns = result_df['Strategy_Return'].dropna()
        equity_curve = result_df['Strategy_Equity']
        
        # compute metrics
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


def compare_strategies(
    df: pd.DataFrame,
    strategies: List[str],
    interval: str = "1d",
    params_dict: Dict[str, Dict] = None
) -> Dict[str, StrategyResult]:
    """
    Run multiple strategies and return results for comparison.
    """
    results = {}
    
    if params_dict is None:
        params_dict = {}
    
    for strategy in strategies:
        params = params_dict.get(strategy, {})
        result = run_strategy(df, strategy, interval, **params)
        if result:
            results[strategy] = result
    
    return results


# ============================================================
# PLOTLY CHARTS
# ============================================================

def create_main_chart(
    df: pd.DataFrame,
    strategy_result: StrategyResult,
    ticker: str,
    show_signals: bool = True
) -> go.Figure:
    """
    Create the main chart showing price and strategy equity curve.
    As required: raw asset price + cumulative strategy value on same graph.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Price & Strategy Performance", "Drawdown")
    )
    
    data = strategy_result.data
    
    # price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name=f"{ticker} Price",
            line=dict(color=COLORS['info'], width=2),
            hovertemplate="%{y:.2f}<extra>Price</extra>"
        ),
        row=1, col=1
    )
    
    # normalize equity to price scale for visual comparison
    price_start = data['Close'].iloc[0]
    equity_normalized = (data['Strategy_Equity'] / 100) * price_start
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=equity_normalized,
            name=f"Strategy ({strategy_result.name})",
            line=dict(color=COLORS['success'], width=2),
            hovertemplate="%{y:.2f}<extra>Strategy</extra>"
        ),
        row=1, col=1
    )
    
    # buy & hold comparison
    market_equity_normalized = (data['Market_Equity'] / 100) * price_start
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=market_equity_normalized,
            name="Buy & Hold",
            line=dict(color=COLORS['muted'], width=1, dash='dot'),
            hovertemplate="%{y:.2f}<extra>Buy & Hold</extra>"
        ),
        row=1, col=1
    )
    
    # show buy/sell signals if requested
    if show_signals:
        signal_changes = data['Signal'].diff().fillna(0)
        
        # buy signals
        buy_mask = signal_changes == 1
        if buy_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data.index[buy_mask],
                    y=data['Close'][buy_mask],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color=COLORS['success'], size=10, symbol='triangle-up'),
                    hovertemplate="%{y:.2f}<extra>BUY</extra>"
                ),
                row=1, col=1
            )
        
        # sell signals
        sell_mask = signal_changes == -1
        if sell_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data.index[sell_mask],
                    y=data['Close'][sell_mask],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color=COLORS['danger'], size=10, symbol='triangle-down'),
                    hovertemplate="%{y:.2f}<extra>SELL</extra>"
                ),
                row=1, col=1
            )
    
    # drawdown chart
    drawdown = compute_drawdown_series(data['Strategy_Equity'])
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=COLORS['danger'], width=1),
            hovertemplate="%{y:.2f}%<extra>Drawdown</extra>"
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    
    return fig


def create_indicator_chart(
    df: pd.DataFrame,
    strategy_name: str
) -> Optional[go.Figure]:
    """
    Create a chart showing the strategy's technical indicators.
    """
    data = df.copy()
    
    if strategy_name == "RSI Mean Reversion" and 'RSI' in data.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            name='RSI', line=dict(color=COLORS['primary'], width=2)
        ))
        
        # overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'], annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS['success'], annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color=COLORS['muted'])
        
        fig.update_layout(
            template="plotly_dark",
            height=250,
            title="RSI Indicator",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=50, r=50, t=50, b=30),
        )
        return fig
    
    elif strategy_name == "MACD" and 'MACD' in data.columns:
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'],
            name='MACD', line=dict(color=COLORS['info'], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD_Signal'],
            name='Signal', line=dict(color=COLORS['warning'], width=2)
        ))
        
        # histogram
        colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in data['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=data.index, y=data['MACD_Histogram'],
            name='Histogram', marker_color=colors, opacity=0.5
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=250,
            title="MACD Indicator",
            margin=dict(l=50, r=50, t=50, b=30),
        )
        return fig
    
    elif strategy_name == "Bollinger Bands" and 'BB_Upper' in data.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'],
            name='Upper Band', line=dict(color=COLORS['danger'], width=1)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Middle'],
            name='Middle Band', line=dict(color=COLORS['warning'], width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'],
            name='Lower Band', line=dict(color=COLORS['success'], width=1),
            fill='tonexty', fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name='Price', line=dict(color=COLORS['info'], width=2)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300,
            title="Bollinger Bands",
            margin=dict(l=50, r=50, t=50, b=30),
        )
        return fig
    
    elif "SMA" in strategy_name or "EMA" in strategy_name:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name='Price', line=dict(color=COLORS['info'], width=2)
        ))
        
        if 'SMA_Short' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_Short'],
                name='Short MA', line=dict(color=COLORS['success'], width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_Long'],
                name='Long MA', line=dict(color=COLORS['danger'], width=1)
            ))
        elif 'EMA_Short' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['EMA_Short'],
                name='Short EMA', line=dict(color=COLORS['success'], width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['EMA_Long'],
                name='Long EMA', line=dict(color=COLORS['danger'], width=1)
            ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300,
            title="Moving Averages",
            margin=dict(l=50, r=50, t=50, b=30),
        )
        return fig
    
    return None


def create_prediction_chart(
    historical_prices: pd.Series,
    prediction: PredictionResult,
    ticker: str
) -> go.Figure:
    """
    Create chart showing historical prices and future predictions with confidence intervals.
    """
    fig = go.Figure()
    
    # historical prices
    fig.add_trace(go.Scatter(
        x=historical_prices.index,
        y=historical_prices.values,
        name='Historical',
        line=dict(color=COLORS['info'], width=2),
    ))
    
    # prediction
    fig.add_trace(go.Scatter(
        x=prediction.forecast_dates,
        y=prediction.predictions.values,
        name=f'Forecast ({prediction.model_name})',
        line=dict(color=COLORS['warning'], width=2, dash='dash'),
    ))
    
    # confidence interval
    fig.add_trace(go.Scatter(
        x=prediction.forecast_dates,
        y=prediction.upper_bound.values,
        name=f'{int(prediction.confidence_level*100)}% CI Upper',
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=prediction.forecast_dates,
        y=prediction.lower_bound.values,
        name=f'{int(prediction.confidence_level*100)}% CI Lower',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(245, 158, 11, 0.2)',
        showlegend=True,
    ))
    
    # vertical line at forecast start
    fig.add_vline(
        x=historical_prices.index[-1],
        line_dash="dot",
        line_color=COLORS['muted'],
        annotation_text="Forecast Start"
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title=f"{ticker} Price Forecast",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_xaxes(title_text="Date", showgrid=True)
    fig.update_yaxes(title_text="Price", showgrid=True)
    
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================

def render_quant_a(show_header: bool = False) -> None:
    """
    Main render function for Quant A module.
    """
    if show_header:
        st.subheader("📈 Quant A - Single Asset Analysis")
        st.caption("Backtest strategies, analyze performance, and forecast prices for individual assets.")
    
    # sidebar-style settings in expander
    with st.expander("⚙️ Settings", expanded=True):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            # preset selector
            preset = st.selectbox(
                "Quick Select",
                ["Custom"] + list(ASSET_PRESETS.keys()),
                index=0,
                help="Choose a preset or enter custom ticker"
            )
            
            if preset != "Custom":
                default_ticker = ASSET_PRESETS[preset][0]
            else:
                default_ticker = "AAPL"
            
            ticker = st.text_input(
                "Ticker Symbol",
                value=default_ticker,
                help="e.g., AAPL, BTC-USD, EURUSD=X, ^GSPC"
            ).strip().upper()
        
        with col2:
            period = st.selectbox(
                "Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                index=3,
                help="Historical data period"
            )
        
        with col3:
            interval = st.selectbox(
                "Interval",
                ["1m", "5m", "15m", "30m", "1h", "1d"],
                index=5,
                help="Data granularity"
            )
        
        with col4:
            strategy = st.selectbox(
                "Strategy",
                list(STRATEGIES.keys()),
                index=0,
                help="Trading strategy to backtest"
            )
    
    # validate period/interval combo
    if not is_valid_combination(period, interval):
        suggested_period, suggested_interval = suggest_valid_combination(period, interval)
        st.warning(f"⚠️ Invalid combo. Using {suggested_period}/{suggested_interval} instead.")
        period, interval = suggested_period, suggested_interval
    
    # strategy parameters
    params = {}
    with st.expander("📊 Strategy Parameters", expanded=False):
        if strategy == "SMA Crossover":
            col1, col2 = st.columns(2)
            params['short_window'] = col1.slider("Short Window", 5, 50, 20)
            params['long_window'] = col2.slider("Long Window", 20, 200, 50)
        
        elif strategy == "EMA Crossover":
            col1, col2 = st.columns(2)
            params['short_window'] = col1.slider("Short EMA", 5, 30, 12)
            params['long_window'] = col2.slider("Long EMA", 15, 50, 26)
        
        elif strategy == "RSI Mean Reversion":
            col1, col2, col3 = st.columns(3)
            params['window'] = col1.slider("RSI Period", 5, 30, 14)
            params['oversold'] = col2.slider("Oversold Level", 10, 40, 30)
            params['overbought'] = col3.slider("Overbought Level", 60, 90, 70)
        
        elif strategy == "Bollinger Bands":
            col1, col2 = st.columns(2)
            params['window'] = col1.slider("Window", 10, 50, 20)
            params['std_dev'] = col2.slider("Std Deviations", 1.0, 3.0, 2.0, 0.1)
        
        elif strategy == "MACD":
            col1, col2, col3 = st.columns(3)
            params['fast'] = col1.slider("Fast Period", 5, 20, 12)
            params['slow'] = col2.slider("Slow Period", 15, 40, 26)
            params['signal_period'] = col3.slider("Signal Period", 5, 15, 9)
        
        elif strategy == "Momentum":
            col1, col2 = st.columns(2)
            params['lookback'] = col1.slider("Lookback Period", 5, 60, 20)
            params['threshold'] = col2.slider("Threshold", -0.05, 0.05, 0.0, 0.01)
    
    # fetch data
    if not ticker:
        st.error("Please enter a ticker symbol.")
        return
    
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_data(ticker, period=period, interval=interval)
    
    if df.empty:
        st.error(f"❌ No data found for {ticker}. Check the symbol or try different period/interval.")
        return
    
    # run strategy
    result = run_strategy(df, strategy, interval, **params)
    
    if result is None:
        st.error("Strategy execution failed.")
        return
    
    # display current price info
    latest = get_latest_price(ticker)
    if latest:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${latest['price']:.2f}",
                f"{latest['change_pct']:.2f}%",
                delta_color="normal" if latest['is_positive'] else "inverse"
            )
        with col2:
            st.metric("Day High", f"${latest['high']:.2f}")
        with col3:
            st.metric("Day Low", f"${latest['low']:.2f}")
        with col4:
            st.metric("Volume", format_number(latest['volume'], 0))
    
    # tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance",
        "📊 Indicators", 
        "🔮 Predictions",
        "📋 Metrics",
        "📁 Data"
    ])
    
    with tab1:
        st.markdown("### Strategy Performance")
        st.caption(f"Comparing **{strategy}** vs Buy & Hold for **{ticker}**")
        
        show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
        
        fig = create_main_chart(df, result, ticker, show_signals)
        st.plotly_chart(fig, use_container_width=True)
        
        # performance summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Strategy Performance")
            m = result.metrics
            
            perf_data = {
                "Total Return": format_percentage(m['total_return']),
                "Annualized Return": format_percentage(m['annualized_return']),
                "Volatility": format_percentage(m['volatility']),
                "Max Drawdown": format_percentage(m['max_drawdown']),
            }
            
            for k, v in perf_data.items():
                st.markdown(f"**{k}:** {v}")
        
        with col2:
            st.markdown("#### Risk Metrics")
            
            risk_data = {
                "Sharpe Ratio": f"{m['sharpe_ratio']:.2f}",
                "Sortino Ratio": f"{m['sortino_ratio']:.2f}",
                "Calmar Ratio": f"{m['calmar_ratio']:.2f}",
                "VaR (95%)": format_percentage(m['var_95']),
            }
            
            for k, v in risk_data.items():
                st.markdown(f"**{k}:** {v}")
    
    with tab2:
        st.markdown("### Technical Indicators")
        
        indicator_chart = create_indicator_chart(result.data, strategy)
        
        if indicator_chart:
            st.plotly_chart(indicator_chart, use_container_width=True)
        else:
            st.info("No specific indicator chart for this strategy.")
        
        # signal distribution
        st.markdown("#### Signal Distribution")
        
        signal_counts = result.signals.value_counts()
        col1, col2, col3 = st.columns(3)
        
        total_periods = len(result.signals)
        long_periods = int(signal_counts.get(1.0, 0))
        cash_periods = int(signal_counts.get(0.0, 0))
        
        col1.metric("Total Periods", total_periods)
        col2.metric("Long Periods", f"{long_periods} ({long_periods/total_periods*100:.1f}%)")
        col3.metric("Cash Periods", f"{cash_periods} ({cash_periods/total_periods*100:.1f}%)")
    
    with tab3:
        st.markdown("### Price Predictions")
        st.caption("ML-based forecasting with confidence intervals")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_model = st.selectbox(
                "Model",
                get_available_models(),
                index=0,
                help="Select prediction model"
            )
        
        with col2:
            pred_periods = st.slider(
                "Forecast Periods",
                min_value=5,
                max_value=100,
                value=30,
                help="Number of periods to forecast"
            )
        
        with col3:
            pred_confidence = st.slider(
                "Confidence Level",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Confidence interval level"
            )
        
        st.info(f"ℹ️ {get_model_description(pred_model)}")
        
        if st.button("🔮 Run Prediction", type="primary"):
            with st.spinner(f"Running {pred_model} prediction..."):
                prices = df['Close']
                prediction = run_prediction(
                    prices,
                    model=pred_model,
                    periods=pred_periods,
                    confidence=pred_confidence
                )
            
            if prediction:
                fig = create_prediction_chart(prices, prediction, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # prediction metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    last_price = prices.iloc[-1]
                    pred_price = prediction.predictions.iloc[-1]
                    change = (pred_price - last_price) / last_price
                    
                    st.metric(
                        f"Predicted Price ({pred_periods} periods)",
                        f"${pred_price:.2f}",
                        f"{change*100:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Prediction Range",
                        f"${prediction.lower_bound.iloc[-1]:.2f} - ${prediction.upper_bound.iloc[-1]:.2f}"
                    )
                
                with col3:
                    if 'mape' in prediction.metrics:
                        st.metric("Model MAPE", f"{prediction.metrics['mape']:.2f}%")
            else:
                st.error("Prediction failed. Try a different model or check if you have enough data.")
    
    with tab4:
        st.markdown("### Detailed Metrics")
        
        metrics_df = pd.DataFrame({
            "Metric": [
                "Total Return",
                "Annualized Return", 
                "Volatility (Ann.)",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "VaR (95%)",
                "CVaR (95%)",
                "Win Rate",
                "Profit Factor",
                "Number of Trades",
            ],
            "Value": [
                format_percentage(result.metrics['total_return']),
                format_percentage(result.metrics['annualized_return']),
                format_percentage(result.metrics['volatility']),
                f"{result.metrics['sharpe_ratio']:.3f}",
                f"{result.metrics['sortino_ratio']:.3f}",
                f"{result.metrics['calmar_ratio']:.3f}",
                format_percentage(result.metrics['max_drawdown']),
                format_percentage(result.metrics['var_95']),
                format_percentage(result.metrics['cvar_95']),
                format_percentage(result.metrics['win_rate']),
                f"{result.metrics['profit_factor']:.3f}",
                f"{result.metrics.get('num_trades', 0)}",
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # strategy comparison
        st.markdown("### Strategy Comparison")
        
        if st.button("Compare All Strategies"):
            with st.spinner("Running all strategies..."):
                all_results = compare_strategies(df, list(STRATEGIES.keys()), interval)
            
            if all_results:
                comparison_data = []
                for name, res in all_results.items():
                    comparison_data.append({
                        "Strategy": name,
                        "Total Return": format_percentage(res.metrics['total_return']),
                        "Sharpe": f"{res.metrics['sharpe_ratio']:.2f}",
                        "Max DD": format_percentage(res.metrics['max_drawdown']),
                        "Win Rate": format_percentage(res.metrics['win_rate']),
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with tab5:
        st.markdown("### Raw Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Data Points:** {len(df)}")
            st.markdown(f"**Date Range:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        with col2:
            st.markdown(f"**Interval:** {interval}")
            st.markdown(f"**Period:** {period}")
        
        # show latest data
        st.markdown("#### Recent Data (Last 50 rows)")
        display_df = result.data[['Open', 'High', 'Low', 'Close', 'Volume', 'Signal']].tail(50)
        st.dataframe(display_df.style.format({
            'Open': '{:.2f}',
            'High': '{:.2f}',
            'Low': '{:.2f}',
            'Close': '{:.2f}',
            'Volume': '{:,.0f}',
            'Signal': '{:.0f}'
        }), use_container_width=True)
        
        # download button
        csv = result.data.to_csv()
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv,
            file_name=f"{ticker}_{strategy.replace(' ', '_')}_{period}.csv",
            mime="text/csv"
        )


def render() -> None:
    """Entry point for the module."""
    render_quant_a(show_header=False)