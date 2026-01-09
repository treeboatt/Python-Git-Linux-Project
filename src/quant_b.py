"""
Quant B - Multi-Asset Portfolio Module
Implements portfolio simulation, allocation strategies, and risk analysis for multiple assets.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from src.data_loader import fetch_data, fetch_price_matrix, get_latest_prices
from src.utils import (
    compute_all_metrics,
    compute_returns,
    compute_drawdown_series,
    get_periods_per_year,
    format_percentage,
    format_number,
    safe_float,
    is_valid_combination,
    suggest_valid_combination,
    COLORS,
    ASSET_PRESETS,
)


# ============================================================
# PORTFOLIO DATA STRUCTURES
# ============================================================

@dataclass
class PortfolioResult:
    """Container for portfolio simulation results."""
    weights: pd.Series
    prices: pd.DataFrame
    returns: pd.Series
    equity_curve: pd.Series
    asset_returns: pd.DataFrame
    metrics: Dict[str, float]
    correlation_matrix: pd.DataFrame
    covariance_matrix: pd.DataFrame
    contribution_to_risk: pd.Series
    normalized_prices: pd.DataFrame


# ============================================================
# WEIGHT CALCULATION METHODS
# ============================================================

def calculate_equal_weights(n_assets: int) -> np.ndarray:
    """Equal weight allocation."""
    return np.ones(n_assets) / n_assets


def calculate_market_cap_weights(prices: pd.DataFrame) -> np.ndarray:
    """
    Weights proportional to last price (proxy for market cap).
    In reality you'd use actual market cap data.
    """
    last_prices = prices.iloc[-1].values
    weights = last_prices / last_prices.sum()
    return weights


def calculate_inverse_volatility_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Inverse volatility weighting.
    Lower volatility assets get higher weights.
    """
    vols = returns.std()
    inv_vols = 1 / vols.replace(0, np.nan).fillna(vols.mean())
    weights = inv_vols / inv_vols.sum()
    return weights.values


def calculate_min_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Minimum variance portfolio.
    Minimizes portfolio variance using closed-form solution.
    """
    n = cov_matrix.shape[0]
    
    # add small regularization for numerical stability
    cov_reg = cov_matrix + np.eye(n) * 1e-8
    
    try:
        inv_cov = np.linalg.inv(cov_reg)
        ones = np.ones(n)
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        
        # handle negative weights by setting floor at 0
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return weights
    except np.linalg.LinAlgError:
        return calculate_equal_weights(n)


def calculate_risk_parity_weights(
    cov_matrix: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Risk parity allocation.
    Each asset contributes equally to total portfolio risk.
    """
    n = cov_matrix.shape[0]
    weights = np.ones(n) / n
    
    # add regularization
    cov_reg = cov_matrix + np.eye(n) * 1e-8
    
    for _ in range(max_iter):
        port_var = weights @ cov_reg @ weights
        
        if port_var <= 0:
            break
        
        # marginal risk contribution
        mrc = cov_reg @ weights
        
        # risk contribution
        rc = (weights * mrc) / port_var
        
        # target: equal risk contribution
        target = 1.0 / n
        
        # check convergence
        if np.max(np.abs(rc - target)) < tol:
            break
        
        # update weights
        weights = weights * (target / np.clip(rc, 1e-10, None))
        weights = weights / weights.sum()
    
    return weights


def calculate_max_sharpe_weights(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    n_portfolios: int = 5000
) -> np.ndarray:
    """
    Maximum Sharpe ratio portfolio via Monte Carlo simulation.
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252  # annualized
    cov_matrix = returns.cov() * 252  # annualized
    
    best_sharpe = -np.inf
    best_weights = calculate_equal_weights(n_assets)
    
    for _ in range(n_portfolios):
        # random weights
        w = np.random.random(n_assets)
        w = w / w.sum()
        
        # portfolio return and volatility
        port_return = w @ mean_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        
        # sharpe ratio
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = w
    
    return best_weights


def calculate_hierarchical_risk_parity(returns: pd.DataFrame) -> np.ndarray:
    """
    Simplified Hierarchical Risk Parity (HRP).
    Uses correlation clustering for diversification.
    """
    n_assets = returns.shape[1]
    
    if n_assets <= 2:
        return calculate_equal_weights(n_assets)
    
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        
        # correlation to distance
        corr = returns.corr()
        dist = np.sqrt((1 - corr) / 2)
        
        # hierarchical clustering
        dist_condensed = squareform(dist.values, checks=False)
        link = linkage(dist_condensed, method='ward')
        sort_idx = leaves_list(link)
        
        # get inverse variance weights in clustered order
        cov = returns.cov()
        inv_var = 1 / np.diag(cov.values)
        inv_var = inv_var / inv_var.sum()
        
        # reorder by cluster
        weights = np.zeros(n_assets)
        weights[sort_idx] = inv_var[sort_idx]
        
        return weights / weights.sum()
        
    except ImportError:
        return calculate_inverse_volatility_weights(returns)


ALLOCATION_METHODS = {
    "Equal Weight": calculate_equal_weights,
    "Inverse Volatility": calculate_inverse_volatility_weights,
    "Min Variance": calculate_min_variance_weights,
    "Risk Parity": calculate_risk_parity_weights,
    "Max Sharpe": calculate_max_sharpe_weights,
    "HRP": calculate_hierarchical_risk_parity,
    "Custom": None,  # handled separately
}


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    rebalance_freq: str = "none",
    initial_value: float = 100.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate portfolio with optional rebalancing.
    
    Returns:
        Tuple of (portfolio_returns, equity_curve)
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    
    if returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # ensure sums to 1
    
    if rebalance_freq == "none":
        # buy and hold: weights drift with prices
        # simple approximation: use weighted returns
        port_returns = (returns * weights).sum(axis=1)
    
    elif rebalance_freq == "daily":
        # rebalance every period
        port_returns = (returns * weights).sum(axis=1)
    
    elif rebalance_freq == "weekly":
        # rebalance weekly
        port_returns = pd.Series(index=returns.index, dtype=float)
        
        current_weights = weights.copy()
        week_returns = []
        
        for i, (date, row) in enumerate(returns.iterrows()):
            period_return = (row * current_weights).sum()
            port_returns.loc[date] = period_return
            
            # update weights based on returns (drift)
            current_weights = current_weights * (1 + row.values)
            current_weights = current_weights / current_weights.sum()
            
            # rebalance on Fridays or every 5 periods
            if i > 0 and i % 5 == 0:
                current_weights = weights.copy()
    
    elif rebalance_freq == "monthly":
        # rebalance monthly
        port_returns = pd.Series(index=returns.index, dtype=float)
        
        current_weights = weights.copy()
        last_rebalance = returns.index[0]
        
        for date, row in returns.iterrows():
            period_return = (row * current_weights).sum()
            port_returns.loc[date] = period_return
            
            # update weights
            current_weights = current_weights * (1 + row.values)
            current_weights = current_weights / current_weights.sum()
            
            # rebalance monthly
            if (date - last_rebalance).days >= 21:
                current_weights = weights.copy()
                last_rebalance = date
    
    else:
        port_returns = (returns * weights).sum(axis=1)
    
    # compute equity curve
    equity = (1 + port_returns).cumprod() * initial_value
    
    return port_returns, equity


def calculate_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate each asset's contribution to portfolio risk.
    """
    port_var = weights @ cov_matrix @ weights
    
    if port_var <= 0:
        return np.zeros_like(weights)
    
    port_vol = np.sqrt(port_var)
    
    # marginal contribution
    mrc = cov_matrix @ weights
    
    # risk contribution (percentage)
    rc = (weights * mrc) / port_var
    
    return rc


def run_portfolio_simulation(
    prices: pd.DataFrame,
    method: str = "Equal Weight",
    custom_weights: Optional[Dict[str, float]] = None,
    rebalance_freq: str = "none",
    interval: str = "1d"
) -> Optional[PortfolioResult]:
    """
    Main function to run portfolio simulation.
    """
    if prices.empty or prices.shape[1] < 2:
        return None
    
    try:
        tickers = list(prices.columns)
        n_assets = len(tickers)
        
        # calculate returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        if returns.empty:
            return None
        
        # covariance and correlation
        cov_matrix = returns.cov()
        corr_matrix = returns.corr()
        
        # determine weights
        if method == "Custom" and custom_weights:
            weights = np.array([custom_weights.get(t, 1/n_assets) for t in tickers])
            weights = weights / weights.sum()
        elif method == "Equal Weight":
            weights = calculate_equal_weights(n_assets)
        elif method == "Inverse Volatility":
            weights = calculate_inverse_volatility_weights(returns)
        elif method == "Min Variance":
            weights = calculate_min_variance_weights(cov_matrix.values)
        elif method == "Risk Parity":
            weights = calculate_risk_parity_weights(cov_matrix.values)
        elif method == "Max Sharpe":
            weights = calculate_max_sharpe_weights(returns)
        elif method == "HRP":
            weights = calculate_hierarchical_risk_parity(returns)
        else:
            weights = calculate_equal_weights(n_assets)
        
        # simulate
        port_returns, equity = simulate_portfolio(prices, weights, rebalance_freq)
        
        if port_returns.empty:
            return None
        
        # calculate metrics
        periods_per_year = get_periods_per_year(interval)
        metrics = compute_all_metrics(port_returns, equity, periods_per_year)
        
        # additional portfolio-specific metrics
        cov_annual = cov_matrix * periods_per_year
        
        # diversification ratio
        port_vol = np.sqrt(weights @ cov_annual.values @ weights)
        asset_vols = np.sqrt(np.diag(cov_annual.values))
        weighted_avg_vol = weights @ asset_vols
        div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0
        
        # average correlation
        corr_values = corr_matrix.values
        n = corr_values.shape[0]
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            avg_corr = corr_values[mask].mean()
        else:
            avg_corr = 0.0
        
        # effective number of bets
        enb = 1 / (weights ** 2).sum()
        
        metrics['diversification_ratio'] = safe_float(div_ratio)
        metrics['avg_correlation'] = safe_float(avg_corr)
        metrics['effective_n_bets'] = safe_float(enb)
        
        # risk contribution
        risk_contrib = calculate_risk_contribution(weights, cov_matrix.values)
        
        # normalized prices for charting
        norm_prices = prices / prices.iloc[0] * 100
        norm_prices['Portfolio'] = equity / equity.iloc[0] * 100
        
        return PortfolioResult(
            weights=pd.Series(weights, index=tickers),
            prices=prices,
            returns=port_returns,
            equity_curve=equity,
            asset_returns=returns,
            metrics=metrics,
            correlation_matrix=corr_matrix,
            covariance_matrix=cov_matrix,
            contribution_to_risk=pd.Series(risk_contrib, index=tickers),
            normalized_prices=norm_prices,
        )
        
    except Exception as e:
        return None


# ============================================================
# EFFICIENT FRONTIER
# ============================================================

def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 1000,
    risk_free_rate: float = 0.04
) -> pd.DataFrame:
    """
    Generate random portfolios to plot efficient frontier.
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    results = []
    
    for _ in range(n_portfolios):
        # random weights
        w = np.random.random(n_assets)
        w = w / w.sum()
        
        # portfolio metrics
        port_return = w @ mean_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        results.append({
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe,
        })
    
    return pd.DataFrame(results)


# ============================================================
# PLOTLY CHARTS
# ============================================================

def create_portfolio_chart(result: PortfolioResult) -> go.Figure:
    """
    Create main chart showing all assets and portfolio performance.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Normalized Performance (Base 100)", "Portfolio Drawdown")
    )
    
    norm = result.normalized_prices
    
    # individual assets
    for col in norm.columns:
        if col == 'Portfolio':
            continue
        
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[col],
                name=col,
                mode='lines',
                line=dict(width=1),
                opacity=0.7,
            ),
            row=1, col=1
        )
    
    # portfolio (highlighted)
    if 'Portfolio' in norm.columns:
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm['Portfolio'],
                name='Portfolio',
                mode='lines',
                line=dict(color=COLORS['warning'], width=3),
            ),
            row=1, col=1
        )
    
    # drawdown
    drawdown = compute_drawdown_series(result.equity_curve)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=COLORS['danger'], width=1),
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
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    
    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Create correlation matrix heatmap.
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        zmin=-1,
        zmax=1,
        colorscale='RdBu_r',
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>',
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="Correlation Matrix",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_weights_chart(weights: pd.Series) -> go.Figure:
    """
    Create pie chart showing portfolio weights.
    """
    fig = go.Figure(data=[go.Pie(
        labels=weights.index,
        values=weights.values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=[
            COLORS['primary'], COLORS['secondary'], COLORS['success'],
            COLORS['warning'], COLORS['info'], COLORS['danger'],
            '#9333ea', '#06b6d4', '#84cc16', '#f97316'
        ])
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="Portfolio Weights",
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
    )
    
    return fig


def create_risk_contribution_chart(
    weights: pd.Series,
    risk_contrib: pd.Series
) -> go.Figure:
    """
    Create bar chart comparing weights vs risk contribution.
    """
    tickers = weights.index.tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Weight',
        x=tickers,
        y=weights.values * 100,
        marker_color=COLORS['info'],
    ))
    
    fig.add_trace(go.Bar(
        name='Risk Contribution',
        x=tickers,
        y=risk_contrib.values * 100,
        marker_color=COLORS['warning'],
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        title="Weights vs Risk Contribution",
        barmode='group',
        xaxis_title="Asset",
        yaxis_title="Percentage (%)",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_efficient_frontier_chart(
    frontier_data: pd.DataFrame,
    current_portfolio: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Create scatter plot of efficient frontier.
    """
    fig = go.Figure()
    
    # color by sharpe ratio
    fig.add_trace(go.Scatter(
        x=frontier_data['volatility'] * 100,
        y=frontier_data['return'] * 100,
        mode='markers',
        marker=dict(
            size=5,
            color=frontier_data['sharpe'],
            colorscale='Viridis',
            colorbar=dict(title='Sharpe'),
            showscale=True,
        ),
        name='Random Portfolios',
        hovertemplate='Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>',
    ))
    
    # current portfolio
    if current_portfolio:
        fig.add_trace(go.Scatter(
            x=[current_portfolio[1] * 100],
            y=[current_portfolio[0] * 100],
            mode='markers',
            marker=dict(size=15, color=COLORS['danger'], symbol='star'),
            name='Current Portfolio',
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title="Efficient Frontier",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_asset_comparison_chart(returns: pd.DataFrame) -> go.Figure:
    """
    Create box plot comparing asset return distributions.
    """
    fig = go.Figure()
    
    for col in returns.columns:
        fig.add_trace(go.Box(
            y=returns[col] * 100,
            name=col,
            boxpoints='outliers',
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        title="Return Distribution by Asset",
        yaxis_title="Daily Return (%)",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================

def render_quant_b(show_header: bool = False) -> None:
    """
    Main render function for Quant B module.
    """
    if show_header:
        st.subheader("📊 Quant B - Multi-Asset Portfolio")
        st.caption("Build and analyze portfolios with multiple allocation strategies.")
    
    # settings
    with st.expander("⚙️ Settings", expanded=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            preset = st.selectbox(
                "Quick Select Preset",
                ["Custom"] + list(ASSET_PRESETS.keys()),
                index=0,
            )
            
            if preset != "Custom":
                default_tickers = ", ".join(ASSET_PRESETS[preset][:5])
            else:
                default_tickers = "AAPL, MSFT, GOOGL, AMZN, META"
            
            tickers_input = st.text_input(
                "Tickers (comma-separated, min 3)",
                value=default_tickers,
                help="Enter at least 3 ticker symbols"
            )
        
        with col2:
            period = st.selectbox(
                "Period",
                ["5d", "1mo", "3mo", "6mo", "1y", "2y"],
                index=2,
            )
        
        with col3:
            interval = st.selectbox(
                "Interval",
                ["5m", "15m", "1h", "1d"],
                index=3,
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            method = st.selectbox(
                "Allocation Method",
                list(ALLOCATION_METHODS.keys()),
                index=0,
                help="How to determine portfolio weights"
            )
        
        with col5:
            rebalance = st.selectbox(
                "Rebalancing",
                ["none", "daily", "weekly", "monthly"],
                index=0,
                help="How often to rebalance to target weights"
            )
    
    # validate period/interval
    if not is_valid_combination(period, interval):
        period, interval = suggest_valid_combination(period, interval)
        st.warning(f"⚠️ Using {period}/{interval} instead (yfinance limitation)")
    
    # parse tickers
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))  # remove duplicates
    
    if len(tickers) < 3:
        st.error("Please provide at least 3 tickers for portfolio analysis.")
        return
    
    # custom weights UI
    custom_weights = None
    if method == "Custom":
        with st.expander("📊 Custom Weights", expanded=True):
            st.caption("Adjust weights (will be normalized to sum to 100%)")
            
            cols = st.columns(min(4, len(tickers)))
            custom_weights = {}
            
            for i, ticker in enumerate(tickers):
                with cols[i % len(cols)]:
                    custom_weights[ticker] = st.slider(
                        ticker,
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / len(tickers),
                        step=0.01,
                        format="%.2f"
                    )
    
    # fetch data
    with st.spinner(f"Fetching data for {len(tickers)} assets..."):
        prices = fetch_price_matrix(tickers, period=period, interval=interval)
    
    if prices.empty:
        st.error("No data retrieved. Check ticker symbols or try different settings.")
        return
    
    # check which tickers were actually loaded
    loaded_tickers = list(prices.columns)
    missing_tickers = set(tickers) - set(loaded_tickers)
    
    if missing_tickers:
        st.warning(f"Could not load data for: {', '.join(missing_tickers)}")
    
    if len(loaded_tickers) < 3:
        st.error("Need at least 3 assets with valid data.")
        return
    
    # run simulation
    result = run_portfolio_simulation(
        prices,
        method=method,
        custom_weights=custom_weights,
        rebalance_freq=rebalance,
        interval=interval
    )
    
    if result is None:
        st.error("Portfolio simulation failed.")
        return
    
    # display current prices
    latest_prices = get_latest_prices(loaded_tickers)
    if latest_prices:
        st.markdown("---")
        cols = st.columns(min(5, len(latest_prices)))
        
        for i, (ticker, data) in enumerate(latest_prices.items()):
            with cols[i % len(cols)]:
                st.metric(
                    ticker,
                    f"${data['price']:.2f}",
                    f"{data['change_pct']:.2f}%",
                    delta_color="normal" if data['is_positive'] else "inverse"
                )
    
    # main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance",
        "⚖️ Allocation",
        "🔗 Correlation",
        "📊 Analysis",
        "📁 Data"
    ])
    
    with tab1:
        st.markdown("### Portfolio Performance")
        st.caption(f"**{method}** allocation with **{rebalance}** rebalancing")
        
        fig = create_portfolio_chart(result)
        st.plotly_chart(fig, use_container_width=True)
        
        # key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        m = result.metrics
        
        col1.metric(
            "Total Return",
            format_percentage(m['total_return']),
            delta=None
        )
        col2.metric(
            "Volatility (Ann.)",
            format_percentage(m['volatility'])
        )
        col3.metric(
            "Sharpe Ratio",
            f"{m['sharpe_ratio']:.2f}"
        )
        col4.metric(
            "Max Drawdown",
            format_percentage(m['max_drawdown'])
        )
        
        # additional metrics row
        col5, col6, col7 = st.columns(3)
        
        col5.metric(
            "Diversification Ratio",
            f"{m['diversification_ratio']:.2f}",
            help="Higher is better. Ratio > 1 means diversification benefit."
        )
        col6.metric(
            "Avg Correlation",
            f"{m['avg_correlation']:.2f}",
            help="Average pairwise correlation between assets."
        )
        col7.metric(
            "Effective N Bets",
            f"{m['effective_n_bets']:.1f}",
            help="Effective number of independent bets in portfolio."
        )
    
    with tab2:
        st.markdown("### Portfolio Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_weights_chart(result.weights)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_risk_contribution_chart(
                result.weights,
                result.contribution_to_risk
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # weights table
        st.markdown("#### Weight Details")
        
        weights_df = pd.DataFrame({
            'Asset': result.weights.index,
            'Weight': result.weights.values,
            'Risk Contribution': result.contribution_to_risk.values,
        })
        weights_df['Weight %'] = weights_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
        weights_df['Risk %'] = weights_df['Risk Contribution'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(
            weights_df[['Asset', 'Weight %', 'Risk %']],
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.markdown("### Correlation Analysis")
        
        fig = create_correlation_heatmap(result.correlation_matrix)
        st.plotly_chart(fig, use_container_width=True)
        
        # correlation insights
        corr = result.correlation_matrix
        
        # find highest and lowest correlations
        corr_values = corr.values
        n = corr_values.shape[0]
        
        high_corr = []
        low_corr = []
        
        for i in range(n):
            for j in range(i+1, n):
                pair = (corr.index[i], corr.columns[j], corr_values[i, j])
                if corr_values[i, j] > 0.7:
                    high_corr.append(pair)
                elif corr_values[i, j] < 0.3:
                    low_corr.append(pair)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Highly Correlated Pairs")
            if high_corr:
                for a, b, c in sorted(high_corr, key=lambda x: -x[2])[:5]:
                    st.markdown(f"**{a}** & **{b}**: {c:.2f}")
            else:
                st.info("No highly correlated pairs (>0.7)")
        
        with col2:
            st.markdown("#### 🟢 Low Correlation Pairs")
            if low_corr:
                for a, b, c in sorted(low_corr, key=lambda x: x[2])[:5]:
                    st.markdown(f"**{a}** & **{b}**: {c:.2f}")
            else:
                st.info("No low correlation pairs (<0.3)")
    
    with tab4:
        st.markdown("### Portfolio Analysis")
        
        # return distribution
        st.markdown("#### Return Distribution")
        fig = create_asset_comparison_chart(result.asset_returns)
        st.plotly_chart(fig, use_container_width=True)
        
        # efficient frontier
        st.markdown("#### Efficient Frontier")
        
        if st.button("Generate Efficient Frontier", type="primary"):
            with st.spinner("Generating random portfolios..."):
                frontier = calculate_efficient_frontier(result.asset_returns)
            
            # current portfolio position
            ann_ret = result.metrics.get('annualized_return', 0)
            ann_vol = result.metrics.get('volatility', 0)
            
            fig = create_efficient_frontier_chart(frontier, (ann_ret, ann_vol))
            st.plotly_chart(fig, use_container_width=True)
        
        # detailed metrics
        st.markdown("#### All Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Volatility (Ann.)',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Calmar Ratio',
                'Max Drawdown',
                'VaR (95%)',
                'CVaR (95%)',
                'Win Rate',
                'Diversification Ratio',
                'Avg Correlation',
                'Effective N Bets',
            ],
            'Value': [
                format_percentage(m['total_return']),
                format_percentage(m['annualized_return']),
                format_percentage(m['volatility']),
                f"{m['sharpe_ratio']:.3f}",
                f"{m['sortino_ratio']:.3f}",
                f"{m['calmar_ratio']:.3f}",
                format_percentage(m['max_drawdown']),
                format_percentage(m['var_95']),
                format_percentage(m['cvar_95']),
                format_percentage(m['win_rate']),
                f"{m['diversification_ratio']:.3f}",
                f"{m['avg_correlation']:.3f}",
                f"{m['effective_n_bets']:.2f}",
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with tab5:
        st.markdown("### Data")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Assets:** {len(loaded_tickers)}")
            st.markdown(f"**Data Points:** {len(prices)}")
        with col2:
            st.markdown(f"**Period:** {period}")
            st.markdown(f"**Interval:** {interval}")
        
        st.markdown("#### Prices (Last 50)")
        st.dataframe(
            prices.tail(50).style.format("{:.2f}"),
            use_container_width=True
        )
        
        st.markdown("#### Normalized Performance")
        st.dataframe(
            result.normalized_prices.tail(50).style.format("{:.2f}"),
            use_container_width=True
        )
        
        # download
        csv = prices.to_csv()
        st.download_button(
            label="📥 Download Price Data (CSV)",
            data=csv,
            file_name=f"portfolio_{'_'.join(loaded_tickers[:3])}_{period}.csv",
            mime="text/csv"
        )


def render() -> None:
    """Entry point for the module."""
    render_quant_b(show_header=False)