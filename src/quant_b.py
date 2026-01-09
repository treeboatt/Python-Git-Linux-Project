# src/quant_b.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data_loader import fetch_data


# -----------------------------
# Helpers
# -----------------------------
def _is_yfinance_combo_supported(period: str, interval: str) -> bool:
    """
    Heuristic rules for yfinance limitations (not perfect, but avoids most 'empty' fetches).
    """
    period = (period or "").lower().strip()
    interval = (interval or "").lower().strip()

    intraday = interval.endswith("m") or interval.endswith("h")
    if not intraday:
        return True  # daily data generally works for long history

    # Conservative limits to avoid empty results:
    if interval in {"1m", "2m", "5m"} and period in {"3mo", "6mo", "1y", "2y", "5y", "max"}:
        return False
    if interval in {"15m", "30m", "60m", "90m", "1h"} and period in {"6mo", "1y", "2y", "5y", "max"}:
        return False

    return True


def _suggest_fixed_combo(period: str, interval: str) -> Tuple[str, str]:
    """
    If combo is not supported, suggest a safer alternative.
    """
    if _is_yfinance_combo_supported(period, interval):
        return period, interval

    # easiest safe fallback:
    # keep period, increase interval to daily
    if period in {"3mo", "6mo", "1y"}:
        return period, "1d"

    # otherwise shrink period
    return "1mo", interval


def _safe_float(x: float, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _annualization_factor_from_interval(interval: str) -> float:
    interval = (interval or "").lower().strip()

    if interval.endswith("d"):
        return np.sqrt(252.0)

    if interval.endswith("m"):
        try:
            mins = int(interval[:-1])
        except Exception:
            mins = 5
        bars_per_day = 390.0 / max(mins, 1)
        return np.sqrt(252.0 * bars_per_day)

    if interval.endswith("h"):
        try:
            hours = int(interval[:-1])
        except Exception:
            hours = 1
        bars_per_day = 6.5 / max(hours, 0.5)
        return np.sqrt(252.0 * bars_per_day)

    return np.sqrt(252.0)


def _to_price_matrix(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        df = fetch_data(t, period=period, interval=interval)
        if df is None or df.empty or "Close" not in df.columns:
            continue
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
    rets = np.log(prices).diff()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return rets


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    dd = (equity_curve / peak) - 1.0
    return float(dd.min())


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.array(w, dtype=float)
    w[~np.isfinite(w)] = 0.0
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _risk_parity_weights(cov: np.ndarray, max_iter: int = 2000, tol: float = 1e-8) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    cov = cov.copy() + np.eye(n) * 1e-10

    target = 1.0 / n
    for _ in range(max_iter):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            break
        mrc = cov @ w
        rc = (w * mrc) / port_var
        err = rc - target
        if np.max(np.abs(err)) < tol:
            break
        w = w * (target / np.clip(rc, 1e-12, None))
        w = _normalize_weights(w)

    return w


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    cov = cov.copy() + np.eye(cov.shape[0]) * 1e-10
    ones = np.ones(cov.shape[0])
    inv = np.linalg.pinv(cov)
    w = inv @ ones
    return _normalize_weights(w)


# -----------------------------
# Portfolio simulation
# -----------------------------
@dataclass
class PortfolioResult:
    weights: pd.Series
    equity: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    corr: pd.DataFrame
    cov: pd.DataFrame
    normalized_prices: pd.DataFrame


class QuantBPortfolio:
    def __init__(self, prices: pd.DataFrame, interval: str):
        self.prices = prices.copy()
        self.interval = interval
        self.returns = _compute_returns(self.prices)
        self.ann_factor = _annualization_factor_from_interval(interval)

    def simulate(
        self,
        method: str,
        custom_weights: Optional[Dict[str, float]] = None,
        rebalance: str = "No Rebalance",
    ) -> PortfolioResult:
        if self.prices.empty or self.returns.empty:
            return PortfolioResult(
                weights=pd.Series(dtype=float),
                equity=pd.Series(dtype=float),
                returns=pd.Series(dtype=float),
                metrics={},
                corr=pd.DataFrame(),
                cov=pd.DataFrame(),
                normalized_prices=pd.DataFrame(),
            )

        tickers = list(self.prices.columns)
        rets = self.returns.copy()

        cov = rets.cov()
        corr = rets.corr()

        w0 = self._get_weights(method, tickers, cov.values, custom_weights)
        port_rets = self._portfolio_returns_with_rebalance(rets, w0, rebalance)

        equity = (1.0 + port_rets).cumprod()
        equity.name = "Portfolio"

        metrics = self._compute_metrics(port_rets, equity, w0, rets)

        norm_prices = self.prices / self.prices.iloc[0]
        norm_prices["Portfolio"] = equity / equity.iloc[0]

        return PortfolioResult(
            weights=pd.Series(w0, index=tickers, name="Weights"),
            equity=equity,
            returns=port_rets,
            metrics=metrics,
            corr=corr,
            cov=cov,
            normalized_prices=norm_prices,
        )

    def _get_weights(
        self,
        method: str,
        tickers: List[str],
        cov: np.ndarray,
        custom_weights: Optional[Dict[str, float]],
    ) -> np.ndarray:
        method = (method or "").strip()
        n = len(tickers)

        if method == "Equal Weight":
            return np.ones(n) / n

        if method == "Custom":
            w = np.zeros(n, dtype=float)
            custom_weights = custom_weights or {}
            for i, t in enumerate(tickers):
                w[i] = _safe_float(custom_weights.get(t, 0.0), 0.0)
            return _normalize_weights(w)

        if method == "Min Variance":
            return _min_variance_weights(cov)

        if method == "Risk Parity":
            return _risk_parity_weights(cov)

        return np.ones(n) / n

    def _portfolio_returns_with_rebalance(self, rets: pd.DataFrame, w0: np.ndarray, rebalance: str) -> pd.Series:
        w0 = np.array(w0, dtype=float)

        if rebalance == "No Rebalance":
            wealth = (1.0 + rets).cumprod()
            alloc = wealth.mul(w0, axis=1)
            port_equity = alloc.sum(axis=1)
            port_rets = port_equity.pct_change().fillna(0.0)
            port_rets.name = "Portfolio Returns"
            return port_rets

        if rebalance == "Daily":
            rule = "1D"
        elif rebalance == "Hourly":
            rule = "1H"
        else:
            rule = None

        if rule is None:
            wealth = (1.0 + rets).cumprod()
            alloc = wealth.mul(w0, axis=1)
            port_equity = alloc.sum(axis=1)
            port_rets = port_equity.pct_change().fillna(0.0)
            port_rets.name = "Portfolio Returns"
            return port_rets

        groups = rets.groupby(pd.Grouper(freq=rule))
        out = []
        for _, g in groups:
            if g.empty:
                continue
            wealth = (1.0 + g).cumprod()
            alloc = wealth.mul(w0, axis=1)
            port_equity = alloc.sum(axis=1)
            pr = port_equity.pct_change().fillna(0.0)
            out.append(pr)

        port_rets = pd.concat(out).sort_index()
        port_rets.name = "Portfolio Returns"
        return port_rets

    def _compute_metrics(self, port_rets: pd.Series, equity: pd.Series, w: np.ndarray, asset_rets: pd.DataFrame) -> Dict[str, float]:
        if port_rets.empty:
            return {}

        mean = float(port_rets.mean())
        std = float(port_rets.std(ddof=0))

        # NOTE: kept as-is (approx), to not change the "fond".
        ann_ret = (1.0 + mean) ** (self.ann_factor**2) - 1.0 if mean > -1 else 0.0
        ann_vol = std * self.ann_factor
        sharpe = 0.0 if ann_vol == 0 else ann_ret / ann_vol

        mdd = _max_drawdown(equity)

        cov = asset_rets.cov().values
        w = _normalize_weights(w)
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 0.0))

        indiv_vol = np.sqrt(np.maximum(np.diag(cov), 0.0))
        weighted_avg_vol = float(np.sum(w * indiv_vol))
        diversification_ratio = 0.0 if port_vol == 0 else weighted_avg_vol / port_vol

        corr = asset_rets.corr().values
        n = corr.shape[0]
        if n >= 2:
            off_diag = corr[~np.eye(n, dtype=bool)]
            avg_corr = float(np.nanmean(off_diag))
        else:
            avg_corr = 0.0

        s2 = float(np.sum(w**2))
        enb = 1.0 / s2 if s2 > 0 else 0.0

        return {
            "Annualized Return (approx)": round(_safe_float(ann_ret), 4),
            "Annualized Volatility": round(_safe_float(ann_vol), 4),
            "Sharpe (rf=0)": round(_safe_float(sharpe), 4),
            "Max Drawdown": round(_safe_float(mdd), 4),
            "Diversification Ratio": round(_safe_float(diversification_ratio), 4),
            "Avg Pairwise Corr": round(_safe_float(avg_corr), 4),
            "Effective N Bets": round(_safe_float(enb), 2),
        }


# -----------------------------
# Plotly visuals
# -----------------------------
def _plot_normalized_prices_plotly(norm_prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in norm_prices.columns:
        fig.add_trace(
            go.Scatter(
                x=norm_prices.index,
                y=norm_prices[col],
                mode="lines",
                name=col,
                connectgaps=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title="Time", showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True)
    fig.update_yaxes(title="Normalized value", showgrid=True)
    return fig


def _plot_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            hovertemplate="Corr(%{y}, %{x}) = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# -----------------------------
# Streamlit UI (Quant B)
# -----------------------------
def render_quant_b(show_header: bool = False) -> None:
    # IMPORTANT: header is now optional to avoid title duplication (app.py handles the hero)
    if show_header:
        st.subheader("Quant B - Multi-asset portfolio")
        st.caption("Portfolio simulation (>=3 assets), interactive weights, rebalancing and risk metrics.")

    with st.expander("Settings", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            tickers_raw = st.text_input(
                "Tickers (comma-separated, >=3)",
                value="AAPL, MSFT, GOOGL",
                help="Example: AAPL, MSFT, GOOGL, BTC-USD, EURUSD=X",
            )
        with c2:
            period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
        with c3:
            interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "1d"], index=2)

        c4, c5 = st.columns([1, 1])
        with c4:
            method = st.selectbox(
                "Weighting method",
                ["Equal Weight", "Custom", "Min Variance", "Risk Parity"],
                index=0,
                help="Custom: choose weights. Min Variance: minimum-variance portfolio. Risk Parity: equal risk contributions.",
            )
        with c5:
            rebalance = st.selectbox(
                "Rebalancing",
                ["No Rebalance", "Daily", "Hourly"],
                index=0,
                help="No Rebalance = buy-and-hold drift. Daily/Hourly = periodic rebalance.",
            )

    # yfinance combo validation (fixed + NOT duplicated)
    if not _is_yfinance_combo_supported(period, interval):
        suggested_period, suggested_interval = _suggest_fixed_combo(period, interval)
        st.warning(
            f"⚠️ yfinance limitation detected: period='{period}' with interval='{interval}' often returns no data. "
            f"Using '{suggested_period}' / '{suggested_interval}' instead."
        )
        period, interval = suggested_period, suggested_interval

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    if len(tickers) < 3:
        st.error("Please provide at least 3 tickers for Quant B.")
        return

    with st.spinner("Fetching data..."):
        prices = _to_price_matrix(tickers, period=period, interval=interval)

    if prices.empty:
        st.error("No data retrieved. Try other tickers or period/interval.")
        return

    st.caption(
        f"Loaded: **{prices.shape[1]} assets** • **{prices.shape[0]} rows** • "
        f"Period **{period}** • Interval **{interval}** • Method **{method}** • Rebalance **{rebalance}**"
    )

    # Custom weights UI
    custom_weights: Optional[Dict[str, float]] = None
    if method == "Custom":
        with st.container(border=True):
            st.markdown("### Custom weights")
            st.caption("Weights are normalized to sum to 1.")
            cols = st.columns(min(4, len(prices.columns)))
            cw: Dict[str, float] = {}
            for i, t in enumerate(prices.columns):
                with cols[i % len(cols)]:
                    cw[t] = st.number_input(t, min_value=0.0, max_value=1.0, value=1.0 / len(prices.columns))
            custom_weights = cw

    analyzer = QuantBPortfolio(prices=prices, interval=interval)
    result = analyzer.simulate(method=method, custom_weights=custom_weights, rebalance=rebalance)

    if result.equity.empty:
        st.error("Portfolio computation failed (empty result).")
        return

    tab_overview, tab_corr, tab_diag, tab_data = st.tabs(["Overview", "Correlation", "Diagnostics", "Data"])

    with tab_overview:
        with st.container(border=True):
            st.markdown("### Main chart")
            st.caption("Assets vs portfolio (normalized) — hover/zoom enabled.")
            st.plotly_chart(_plot_normalized_prices_plotly(result.normalized_prices), use_container_width=True)

        with st.container(border=True):
            st.markdown("### Current values")
            last_prices = prices.iloc[-1].copy()
            current = pd.DataFrame(
                {
                    "Last price": last_prices,
                    "Weight": result.weights.reindex(prices.columns).fillna(0.0),
                }
            )
            st.dataframe(current.style.format({"Last price": "{:.4f}", "Weight": "{:.2%}"}), use_container_width=True)

        with st.container(border=True):
            st.markdown("### Portfolio metrics")
            m = result.metrics or {}
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Ann. return", f"{m.get('Annualized Return (approx)', 0.0):.2%}")
            k2.metric("Ann. vol", f"{m.get('Annualized Volatility', 0.0):.2%}")
            k3.metric("Sharpe (rf=0)", f"{m.get('Sharpe (rf=0)', 0.0):.2f}")
            k4.metric("Max drawdown", f"{m.get('Max Drawdown', 0.0):.2%}")

            k5, k6, k7 = st.columns(3)
            k5.metric("Diversification ratio", f"{m.get('Diversification Ratio', 0.0):.2f}")
            k6.metric("Avg pairwise corr", f"{m.get('Avg Pairwise Corr', 0.0):.2f}")
            k7.metric("Effective N bets", f"{m.get('Effective N Bets', 0.0):.2f}")

    with tab_corr:
        st.markdown("### Correlation matrix")
        if not result.corr.empty:
            st.plotly_chart(_plot_corr_heatmap(result.corr), use_container_width=True)
        else:
            st.info("Correlation matrix not available.")

    with tab_diag:
        st.markdown("### Diagnostics")
        diag = pd.DataFrame({"Portfolio returns": result.returns}).dropna()
        st.dataframe(diag.describe().T.style.format("{:.6f}"), use_container_width=True)

    with tab_data:
        st.markdown("### Data")
        st.markdown("**Prices (tail)**")
        st.dataframe(prices.tail(300), use_container_width=True)

        st.markdown("**Normalized (tail)**")
        st.dataframe(result.normalized_prices.tail(300).style.format("{:.4f}"), use_container_width=True)

        st.markdown("**Weights**")
        st.dataframe(result.weights.to_frame("Weight").style.format("{:.2%}"), use_container_width=True)

        st.markdown("**Comparison (end value)**")
        end_vals = result.normalized_prices.iloc[-1].sort_values(ascending=False)
        st.dataframe(end_vals.to_frame("End normalized value").style.format("{:.4f}"), use_container_width=True)


def render() -> None:
    render_quant_b(show_header=False)
