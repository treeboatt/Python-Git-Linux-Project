"""
Quant B - Multi-Asset Portfolio Analysis Module 
Professional portfolio simulation with multiple allocation methods and risk analysis.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.data_loader import fetch_data
from src.utils import (
    TICKER_CATEGORIES, PERIOD_OPTIONS, get_interval_for_period,
    safe_float, format_percentage, COLORS, POPULAR_TICKERS
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

# Professional chart color palette
CHART_COLORS = [
    "#00d4aa",  # Teal (Primary - Portfolio)
    "#6366f1",  # Indigo
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#3b82f6",  # Blue
    "#8b5cf6",  # Purple
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#84cc16",  # Lime
    "#f97316",  # Orange
]


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class PortfolioResult:
    """Container for portfolio simulation results."""
    weights: pd.Series
    equity: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    corr: pd.DataFrame
    cov: pd.DataFrame
    normalized_prices: pd.DataFrame
    individual_returns: pd.DataFrame


# ============================================================
# PORTFOLIO OPTIMIZATION METHODS
# ============================================================
def _normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to 1."""
    w = np.array(w, dtype=float)
    w[~np.isfinite(w)] = 0.0
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Calculate minimum variance portfolio weights."""
    n = cov.shape[0]
    cov = cov.copy() + np.eye(n) * 1e-8
    ones = np.ones(n)
    try:
        inv = np.linalg.pinv(cov)
        w = inv @ ones
        return _normalize_weights(w)
    except:
        return np.ones(n) / n


def _risk_parity_weights(cov: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Calculate risk parity weights (equal risk contribution)."""
    n = cov.shape[0]
    w = np.ones(n) / n
    cov = cov.copy() + np.eye(n) * 1e-8
    target = 1.0 / n
    
    for _ in range(max_iter):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            break
        mrc = cov @ w
        rc = (w * mrc) / port_var
        err = rc - target
        if np.max(np.abs(err)) < 1e-6:
            break
        w = w * (target / np.clip(rc, 1e-10, None))
        w = _normalize_weights(w)
    
    return w


def _inverse_volatility_weights(cov: np.ndarray) -> np.ndarray:
    """Weights inversely proportional to volatility."""
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
    inv_vols = 1.0 / vols
    return _normalize_weights(inv_vols)


def _max_sharpe_weights(returns: pd.DataFrame, n_simulations: int = 5000) -> np.ndarray:
    """Monte Carlo optimization for maximum Sharpe ratio."""
    n = returns.shape[1]
    mean_returns = returns.mean().values
    cov = returns.cov().values
    
    best_sharpe = -np.inf
    best_weights = np.ones(n) / n
    
    for _ in range(n_simulations):
        w = np.random.random(n)
        w = _normalize_weights(w)
        port_return = np.sum(w * mean_returns) * 252
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        if port_vol > 0:
            sharpe = port_return / port_vol
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w.copy()
    
    return best_weights


def _to_price_matrix(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    """Fetch and align price data for multiple tickers."""
    frames = []
    for t in tickers:
        df = fetch_data(t, period=period, interval=interval)
        if df is not None and not df.empty and "Close" in df.columns:
            s = df["Close"].rename(t)
            frames.append(s)
    
    if not frames:
        return pd.DataFrame()
    
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna(how="any")
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from prices."""
    rets = np.log(prices).diff()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return rets


def _max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


# ============================================================
# PORTFOLIO ANALYZER CLASS
# ============================================================
class QuantBPortfolio:
    """Portfolio analyzer class."""
    
    def __init__(self, prices: pd.DataFrame, interval: str):
        self.prices = prices.copy()
        self.interval = interval
        self.returns = _compute_returns(self.prices)
        
        # Annualization factor
        if interval.endswith("m"):
            mins = int(interval[:-1]) if interval[:-1].isdigit() else 5
            bars_per_day = 390.0 / max(mins, 1)
            self.ann_factor = np.sqrt(252.0 * bars_per_day)
        elif interval.endswith("h"):
            hours = int(interval[:-1]) if interval[:-1].isdigit() else 1
            bars_per_day = 6.5 / max(hours, 0.5)
            self.ann_factor = np.sqrt(252.0 * bars_per_day)
        else:
            self.ann_factor = np.sqrt(252.0)
    
    def simulate(self, method: str, custom_weights: Optional[Dict[str, float]] = None, rebalance: str = "None") -> PortfolioResult:
        """Run portfolio simulation."""
        if self.prices.empty or self.returns.empty:
            return PortfolioResult(
                weights=pd.Series(dtype=float),
                equity=pd.Series(dtype=float),
                returns=pd.Series(dtype=float),
                metrics={},
                corr=pd.DataFrame(),
                cov=pd.DataFrame(),
                normalized_prices=pd.DataFrame(),
                individual_returns=pd.DataFrame(),
            )
        
        tickers = list(self.prices.columns)
        cov = self.returns.cov()
        corr = self.returns.corr()
        
        # Get weights based on method
        w = self._get_weights(method, tickers, cov.values, custom_weights)
        
        # Calculate portfolio returns
        port_rets = self._portfolio_returns(w, rebalance)
        
        # Equity curve
        equity = (1.0 + port_rets).cumprod()
        equity.name = "Portfolio"
        
        # Metrics
        metrics = self._compute_metrics(port_rets, equity, w)
        
        # Normalized prices for chart
        norm_prices = self.prices / self.prices.iloc[0]
        norm_prices["Portfolio"] = equity / equity.iloc[0]
        
        # Individual asset returns for analysis
        individual_rets = self.returns.copy()
        individual_rets['Portfolio'] = port_rets
        
        return PortfolioResult(
            weights=pd.Series(w, index=tickers, name="Weights"),
            equity=equity,
            returns=port_rets,
            metrics=metrics,
            corr=corr,
            cov=cov,
            normalized_prices=norm_prices,
            individual_returns=individual_rets,
        )
    
    def _get_weights(self, method: str, tickers: List[str], cov: np.ndarray, custom_weights: Optional[Dict[str, float]]) -> np.ndarray:
        """Calculate portfolio weights based on method."""
        n = len(tickers)
        
        if method == "Equal Weight":
            return np.ones(n) / n
        elif method == "Custom" and custom_weights:
            w = np.array([custom_weights.get(t, 0.0) for t in tickers])
            return _normalize_weights(w)
        elif method == "Min Variance":
            return _min_variance_weights(cov)
        elif method == "Risk Parity":
            return _risk_parity_weights(cov)
        elif method == "Inverse Volatility":
            return _inverse_volatility_weights(cov)
        elif method == "Max Sharpe":
            return _max_sharpe_weights(self.returns)
        
        return np.ones(n) / n
    
    def _portfolio_returns(self, w: np.ndarray, rebalance: str) -> pd.Series:
        """Calculate portfolio returns with optional rebalancing."""
        rets = self.returns.copy()
        
        if rebalance == "None" or rebalance == "No Rebalance":
            # Buy and hold - weights drift
            wealth = (1.0 + rets).cumprod()
            alloc = wealth.mul(w, axis=1)
            port_equity = alloc.sum(axis=1)
            port_rets = port_equity.pct_change().fillna(0.0)
        else:
            # Periodic rebalancing
            port_rets = (rets * w).sum(axis=1)
        
        port_rets.name = "Portfolio Returns"
        return port_rets
    
    def _compute_metrics(self, port_rets: pd.Series, equity: pd.Series, w: np.ndarray) -> Dict[str, float]:
        """Compute portfolio performance metrics."""
        if port_rets.empty:
            return {}
        
        mean_ret = port_rets.mean()
        std_ret = port_rets.std()
        
        ann_ret = mean_ret * (self.ann_factor ** 2)
        ann_vol = std_ret * self.ann_factor
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        mdd = _max_drawdown(equity)
        
        # Sortino Ratio
        downside_rets = port_rets[port_rets < 0]
        downside_vol = downside_rets.std() * self.ann_factor if len(downside_rets) > 0 else ann_vol
        sortino = ann_ret / downside_vol if downside_vol > 0 else 0.0
        
        # Calmar Ratio
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0
        
        # Diversification ratio
        cov = self.returns.cov().values
        port_vol = np.sqrt(w @ cov @ w)
        indiv_vols = np.sqrt(np.maximum(np.diag(cov), 0))
        weighted_vol = np.sum(w * indiv_vols)
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1.0
        
        # Average correlation
        corr = self.returns.corr().values
        n = corr.shape[0]
        if n >= 2:
            off_diag = corr[~np.eye(n, dtype=bool)]
            avg_corr = np.nanmean(off_diag)
        else:
            avg_corr = 0.0
        
        # Effective N bets
        enb = 1.0 / np.sum(w ** 2) if np.sum(w ** 2) > 0 else 0.0
        
        # VaR and CVaR
        var_95 = np.percentile(port_rets.dropna(), 5)
        cvar_95 = port_rets[port_rets <= var_95].mean() if len(port_rets[port_rets <= var_95]) > 0 else var_95
        
        return {
            "Annual Return": safe_float(ann_ret),
            "Annual Volatility": safe_float(ann_vol),
            "Sharpe Ratio": safe_float(sharpe),
            "Sortino Ratio": safe_float(sortino),
            "Calmar Ratio": safe_float(calmar),
            "Max Drawdown": safe_float(mdd),
            "Diversification Ratio": safe_float(div_ratio),
            "Avg Correlation": safe_float(avg_corr),
            "Effective N Bets": safe_float(enb),
            "VaR 95%": safe_float(var_95),
            "CVaR 95%": safe_float(cvar_95),
        }


# ============================================================
# PLOTLY CHARTS - PROFESSIONAL STYLING (FIXED COLORS)
# ============================================================
def _plot_portfolio_chart(norm_prices: pd.DataFrame) -> go.Figure:
    """Create the main portfolio performance chart."""
    fig = go.Figure()
    
    for i, col in enumerate(norm_prices.columns):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        is_portfolio = col == "Portfolio"
        width = 3 if is_portfolio else 1.5
        dash = None if is_portfolio else "dot"
        
        fig.add_trace(go.Scatter(
            x=norm_prices.index,
            y=norm_prices[col],
            mode="lines",
            name=col,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{col}</b><br>Value: %{{y:.3f}}<extra></extra>",
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=60, r=40, t=40, b=50),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            x=0,
            font=dict(size=10)
        ),
        hovermode="x unified",
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
        title=dict(
            text="📈 Portfolio vs Individual Assets",
            font=dict(size=14, color=THEME['text_secondary']),
            x=0.5,
            xanchor='center'
        ),
    )
    
    # FIXED: Using rgba format for grid colors
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='rgba(30, 41, 59, 0.5)',
        title_text="Date",
        title_font=dict(size=11)
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='rgba(30, 41, 59, 0.5)', 
        title_text="Normalized Value (Base = 1.0)",
        title_font=dict(size=11)
    )
    
    return fig


def _plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Create enhanced correlation heatmap with annotations."""
    
    # Create text annotations for the heatmap
    text_annotations = [[f"{val:.2f}" for val in row] for row in corr.values]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1, 
        zmax=1,
        colorscale=[
            [0.0, "#ef4444"],      # Strong negative - Red
            [0.25, "#f97316"],     # Weak negative - Orange
            [0.5, "#1e293b"],      # Zero - Dark
            [0.75, "#22c55e"],     # Weak positive - Green
            [1.0, "#10b981"],      # Strong positive - Emerald
        ],
        text=text_annotations,
        texttemplate="%{text}",
        textfont=dict(size=11, color='white'),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(size=11)),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
            len=0.8,
        ),
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=80, r=40, t=60, b=80),
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
        title=dict(
            text="🔗 Asset Correlation Matrix",
            font=dict(size=14, color=THEME['text_secondary']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            tickfont=dict(size=10),
        ),
    )
    
    return fig


def _plot_weights_pie(weights: pd.Series) -> go.Figure:
    """Create weights donut chart."""
    fig = go.Figure(data=[go.Pie(
        labels=weights.index,
        values=weights.values,
        hole=0.5,
        marker_colors=CHART_COLORS[:len(weights)],
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.2%}<extra></extra>",
        pull=[0.02] * len(weights),  # Slight separation
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
        showlegend=False,
        title=dict(
            text="⚖️ Portfolio Allocation",
            font=dict(size=14, color=THEME['text_secondary']),
            x=0.5,
            xanchor='center'
        ),
        annotations=[dict(
            text="Weights",
            x=0.5, y=0.5,
            font_size=12,
            font_color=THEME['text_muted'],
            showarrow=False
        )],
    )
    
    return fig


def _plot_weights_bar(weights: pd.Series) -> go.Figure:
    """Create horizontal bar chart for weights."""
    sorted_weights = weights.sort_values(ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        x=sorted_weights.values * 100,
        y=sorted_weights.index,
        orientation='h',
        marker_color=CHART_COLORS[:len(sorted_weights)],
        text=[f"{w:.1f}%" for w in sorted_weights.values * 100],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.2f}%<extra></extra>",
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=max(200, len(weights) * 35),
        margin=dict(l=80, r=60, t=40, b=40),
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
        title=dict(
            text="📊 Weight Distribution",
            font=dict(size=14, color=THEME['text_secondary']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Weight (%)",
            showgrid=True,
            gridcolor='rgba(30, 41, 59, 0.5)',
        ),
        yaxis=dict(
            tickfont=dict(size=10),
        ),
    )
    
    return fig


def _plot_risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> go.Figure:
    """Create risk contribution chart."""
    w = weights.values
    cov_vals = cov.values
    
    # Calculate marginal risk contribution
    port_vol = np.sqrt(w @ cov_vals @ w)
    mrc = (cov_vals @ w) / port_vol if port_vol > 0 else np.zeros(len(w))
    risk_contrib = w * mrc
    risk_contrib_pct = risk_contrib / risk_contrib.sum() if risk_contrib.sum() > 0 else w
    
    rc_series = pd.Series(risk_contrib_pct, index=weights.index).sort_values(ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        x=rc_series.values * 100,
        y=rc_series.index,
        orientation='h',
        marker_color=[THEME['danger'] if v > 1/len(w) * 1.5 else THEME['accent_primary'] for v in rc_series.values],
        text=[f"{v:.1f}%" for v in rc_series.values * 100],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Risk Contribution: %{x:.2f}%<extra></extra>",
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=max(200, len(weights) * 35),
        margin=dict(l=80, r=60, t=40, b=40),
        paper_bgcolor=THEME['bg_primary'],
        plot_bgcolor=THEME['bg_primary'],
        title=dict(
            text="⚠️ Risk Contribution",
            font=dict(size=14, color=THEME['text_secondary']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Risk Contribution (%)",
            showgrid=True,
            gridcolor='rgba(30, 41, 59, 0.5)',
        ),
    )
    
    # Add equal risk line
    equal_risk = 100 / len(weights)
    fig.add_vline(
        x=equal_risk, 
        line_dash="dash", 
        line_color=THEME['accent_gold'],
        annotation_text=f"Equal Risk ({equal_risk:.1f}%)",
        annotation_font_size=9,
    )
    
    return fig


# ============================================================
# MAIN RENDER FUNCTION
# ============================================================
def render_quant_b(show_header: bool = False) -> None:
    """Main render function for Quant B module."""
    
    # Custom CSS
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
            font-size: 22px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .v-positive {{ color: {THEME['success']}; }}
        .v-negative {{ color: {THEME['danger']}; }}
        .v-neutral {{ color: {THEME['text_primary']}; }}
        .v-accent {{ color: {THEME['accent_primary']}; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {THEME['bg_secondary']};
            border-radius: 12px;
        }}
        span[data-baseweb="tag"] {{
            background-color: {THEME['bg_tertiary']} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Layout: Settings | Main Content
    col_settings, col_main = st.columns([1, 3.5], gap="large")

    with col_settings:
        with st.container(border=True):
            st.markdown("### ⚙️ Settings")
            st.markdown("---")
            
            # Asset Selection
            st.caption("🎯 ASSET SELECTION")
            preset = st.selectbox(
                "Preset", 
                ["Custom"] + list(TICKER_CATEGORIES.keys()), 
                index=0
            )
            
            # Build options list
            all_options = set(POPULAR_TICKERS)
            for cat_tickers in TICKER_CATEGORIES.values():
                all_options.update(cat_tickers)
            all_options = sorted(list(all_options))
            
            # Default selection
            default_selection = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            if preset != "Custom":
                default_selection = TICKER_CATEGORIES[preset][:10]
                default_selection = [t for t in default_selection if t in all_options]
            
            tickers = st.multiselect(
                "Assets",
                options=all_options,
                default=default_selection,
            )
            st.caption(f" {len(tickers)} assets selected")
            
            st.markdown("---")
            
            # Time Period
            st.caption("📅 TIME PERIOD")
            period = st.selectbox(
                "Period",
                options=list(PERIOD_OPTIONS.keys()),
                index=4,
                format_func=lambda x: PERIOD_OPTIONS[x],
            )
            interval = get_interval_for_period(period)
            st.caption(f" Interval: {interval}")
            
            st.markdown("---")
            
            # Optimization
            st.caption("🎛️ OPTIMIZATION")
            method = st.selectbox(
                "Method",
                ["Equal Weight", "Inverse Volatility", "Min Variance", "Risk Parity", "Max Sharpe", "Custom"],
                index=0,
            )
            
            rebalance = st.selectbox(
                "Rebalancing",
                ["None", "Daily", "Weekly", "Monthly"],
                index=0,
            )
            
            # Method explanation
            method_info = {
                "Equal Weight": "🔄 Same allocation to each asset (1/n)",
                "Inverse Volatility": "📉 Higher weight to less volatile assets",
                "Min Variance": "🎯 Minimize portfolio variance",
                "Risk Parity": "⚖️ Equal risk contribution from each asset",
                "Max Sharpe": "🚀 Maximize risk-adjusted returns (Monte Carlo)",
                "Custom": "✏️ Define your own weights",
            }
            st.info(method_info.get(method, ""))

    with col_main:
        # Validation
        if len(tickers) < 3:
            st.error("⚠️ Please select at least 3 assets to build a diversified portfolio.")
            return

        # Custom weights UI
        custom_weights = None
        if method == "Custom":
            st.markdown("#### ✏️ Custom Weights")
            cols = st.columns(min(5, len(tickers)))
            cw = {}
            for i, t in enumerate(tickers):
                with cols[i % len(cols)]:
                    cw[t] = st.number_input(
                        t, 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=1.0/len(tickers), 
                        format="%.3f",
                        key=f"weight_{t}"
                    )
            custom_weights = cw
            total_weight = sum(cw.values())
            weight_color = THEME['success'] if 0.99 <= total_weight <= 1.01 else THEME['warning']
            st.markdown(f"<p style='color:{weight_color}'>📊 Total: {total_weight:.2f} (will be normalized to 1.0)</p>", unsafe_allow_html=True)

        # Fetch data
        with st.spinner("📡 Loading market data..."):
            prices = _to_price_matrix(tickers, period, interval)

        if prices.empty:
            st.error("❌ Could not load data. Please check ticker symbols.")
            return

        loaded_tickers = list(prices.columns)
        if len(loaded_tickers) < 3:
            st.error(f"❌ Not enough valid data. Only found: {', '.join(loaded_tickers)}")
            return

        # Missing tickers warning
        missing = set(tickers) - set(loaded_tickers)
        if missing:
            st.warning(f"⚠️ Could not load data for: {', '.join(missing)}")

        # Run simulation
        analyzer = QuantBPortfolio(prices, interval)
        result = analyzer.simulate(method, custom_weights, rebalance)

        if result.equity.empty:
            st.error("❌ Portfolio simulation failed.")
            return

        # Performance Metrics Header
        st.markdown('<div class="section-title"> Portfolio Performance</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ret_val = result.metrics.get("Annual Return", 0)
            ret_class = "v-positive" if ret_val >= 0 else "v-negative"
            st.markdown(f"""
                <div class="metric-box" style="border-left: 3px solid {THEME['accent_primary']};">
                    <div class="metric-label"> Annual Return</div>
                    <div class="metric-value {ret_class}">{ret_val*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vol_val = result.metrics.get("Annual Volatility", 0)
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label"> Volatility</div>
                    <div class="metric-value v-neutral">{vol_val*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            sharpe_val = result.metrics.get("Sharpe Ratio", 0)
            sharpe_class = "v-positive" if sharpe_val > 1 else "v-neutral"
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label"> Sharpe Ratio</div>
                    <div class="metric-value {sharpe_class}">{sharpe_val:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            mdd_val = result.metrics.get("Max Drawdown", 0)
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label"> Max Drawdown</div>
                    <div class="metric-value v-negative">{mdd_val*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

        # Main Chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig = _plot_portfolio_chart(result.normalized_prices)
        st.plotly_chart(fig, use_container_width=True)

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            " Allocation", 
            " Correlation", 
            " Metrics",
            " Data"
        ])

        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig_pie = _plot_weights_pie(result.weights)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = _plot_weights_bar(result.weights)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk Contribution Chart
            fig_risk = _plot_risk_contribution(result.weights, result.cov)
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Diversification Metrics
            st.markdown("#### 🌐 Diversification Analysis")
            m1, m2, m3 = st.columns(3)
            
            with m1:
                div_ratio = result.metrics.get("Diversification Ratio", 0)
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label"> Diversification Ratio</div>
                        <div class="metric-value v-accent">{div_ratio:.2f}</div>
                        <div style="font-size:10px; color:{THEME['text_muted']}">Higher = More Diversified</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with m2:
                avg_corr = result.metrics.get("Avg Correlation", 0)
                corr_class = "v-positive" if avg_corr < 0.5 else "v-negative"
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label"> Avg Correlation</div>
                        <div class="metric-value {corr_class}">{avg_corr:.2f}</div>
                        <div style="font-size:10px; color:{THEME['text_muted']}">Lower = Better Diversification</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with m3:
                enb = result.metrics.get("Effective N Bets", 0)
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label"> Effective N Bets</div>
                        <div class="metric-value v-neutral">{enb:.1f}</div>
                        <div style="font-size:10px; color:{THEME['text_muted']}">Independent Risk Sources</div>
                    </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("#### Correlation Matrix")
            st.caption("Shows the linear relationship between asset returns. Values range from -1 (perfect negative) to +1 (perfect positive).")
            
            if not result.corr.empty:
                fig_corr = _plot_correlation_heatmap(result.corr)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Correlation insights
                st.markdown("#### Correlation Insights")
                
                corr_matrix = result.corr.values
                n = corr_matrix.shape[0]
                tickers_list = list(result.corr.columns)
                
                # Find highest and lowest correlations
                correlations = []
                for i in range(n):
                    for j in range(i+1, n):
                        correlations.append({
                            'pair': f"{tickers_list[i]} / {tickers_list[j]}",
                            'corr': corr_matrix[i, j]
                        })
                
                if correlations:
                    corr_df = pd.DataFrame(correlations).sort_values('corr', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔴 Highest Correlations (Most Similar)**")
                        top_corr = corr_df.head(3)
                        for _, row in top_corr.iterrows():
                            color = THEME['danger'] if row['corr'] > 0.7 else THEME['warning']
                            st.markdown(f"<span style='color:{color}'>• {row['pair']}: {row['corr']:.3f}</span>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**🟢 Lowest Correlations (Best Diversifiers)**")
                        bottom_corr = corr_df.tail(3).iloc[::-1]
                        for _, row in bottom_corr.iterrows():
                            color = THEME['success'] if row['corr'] < 0.3 else THEME['text_secondary']
                            st.markdown(f"<span style='color:{color}'>• {row['pair']}: {row['corr']:.3f}</span>", unsafe_allow_html=True)

        with tab3:
            st.markdown("####  Complete Performance Metrics")
            
            # Performance
            st.markdown("** Returns & Performance**")
            perf_cols = st.columns(4)
            
            perf_metrics = [
                ("Annual Return", result.metrics.get("Annual Return", 0) * 100, "%"),
                ("Volatility", result.metrics.get("Annual Volatility", 0) * 100, "%"),
                ("Sharpe Ratio", result.metrics.get("Sharpe Ratio", 0), ""),
                ("Sortino Ratio", result.metrics.get("Sortino Ratio", 0), ""),
            ]
            
            for col, (label, val, unit) in zip(perf_cols, perf_metrics):
                with col:
                    val_class = "v-positive" if val > 0 else "v-negative" if val < 0 else "v-neutral"
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value {val_class}">{val:.2f}{unit}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk
            st.markdown("**⚠️ Risk Metrics**")
            risk_cols = st.columns(4)
            
            risk_metrics = [
                ("Max Drawdown", result.metrics.get("Max Drawdown", 0) * 100, "%"),
                ("Calmar Ratio", result.metrics.get("Calmar Ratio", 0), ""),
                ("VaR (95%)", result.metrics.get("VaR 95%", 0) * 100, "%"),
                ("CVaR (95%)", result.metrics.get("CVaR 95%", 0) * 100, "%"),
            ]
            
            for col, (label, val, unit) in zip(risk_cols, risk_metrics):
                with col:
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value v-negative">{val:.2f}{unit}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Weights Table
            st.markdown("**⚖️ Portfolio Weights**")
            weights_df = pd.DataFrame({
                "Asset": result.weights.index,
                "Weight": [f"{w*100:.2f}%" for w in result.weights.values],
                "Weight (Decimal)": [f"{w:.4f}" for w in result.weights.values],
            })
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

        with tab4:
            st.markdown("####  Price Data")
            st.caption(f" {len(prices)} data points | Period: {period} | Interval: {interval}")
            
            st.dataframe(prices.tail(50), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv_prices = prices.to_csv()
                st.download_button(
                    "📥 Download Price Data (CSV)",
                    csv_prices,
                    "portfolio_prices.csv",
                    "text/csv"
                )
            
            with col2:
                # Export portfolio summary
                summary = {
                    "Method": method,
                    "Period": period,
                    "Interval": interval,
                    "Rebalancing": rebalance,
                    **result.metrics,
                }
                summary_df = pd.DataFrame([summary])
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Portfolio Summary (CSV)",
                    csv_summary,
                    "portfolio_summary.csv",
                    "text/csv"
                )


def render() -> None:
    """Entry point for the module."""
    render_quant_b(show_header=False)