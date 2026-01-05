import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt


from streamlit_autorefresh import st_autorefresh

from src.data_loader import fetch_data
from src.quant_a import QuantAAnalyzer
from src.quant_b import render as render_quant_b


st.set_page_config(page_title="Quant Dashboard", page_icon="📈", layout="wide")

# --------------------
# GLOBAL HERO CSS
# --------------------
st.markdown("""
<style>
.hero{
  border:1px solid rgba(255,255,255,0.10);
  background: radial-gradient(1200px 500px at 20% -10%, rgba(46,91,255,0.25), transparent 60%),
              radial-gradient(900px 400px at 100% 0%, rgba(0,200,150,0.12), transparent 55%),
              rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  margin: 6px 0 18px 0;
}
.hero-title{
  font-size: 34px;
  font-weight: 900;
  letter-spacing: -0.5px;
  margin-bottom: 4px;
}
.hero-sub{
  font-size: 13px;
  color: rgba(255,255,255,0.70);
  margin-bottom: 12px;
}
.hero-pills{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}
.pill{
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  background: rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
  font-size:12px;
  color: rgba(255,255,255,0.85);
}
</style>
""", unsafe_allow_html=True)

# Auto refresh 5 minutes (requirement)
st_autorefresh(interval=300000, key="auto_refresh_5min")


@st.cache_data(ttl=300)
def cached_fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return fetch_data(ticker, period=period, interval=interval)


def _clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    # garder seulement colonnes utiles si elles existent
    for col in ["Close", "SMA_Short", "SMA_Long"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"])
    return out


def format_signal(last_row: pd.Series):
    if pd.isna(last_row.get("SMA_Short")) or pd.isna(last_row.get("SMA_Long")):
        return ("NEUTRAL", "#9CA3AF", "Not enough data for moving averages.")
    if last_row["SMA_Short"] > last_row["SMA_Long"]:
        return ("BUY", "#22C55E", "SMA short above SMA long.")
    if last_row["SMA_Short"] < last_row["SMA_Long"]:
        return ("SELL", "#EF4444", "SMA short below SMA long.")
    return ("NEUTRAL", "#9CA3AF", "SMA short equals SMA long.")


def plot_quant_a(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", mode="lines",
        connectgaps=False
    ))

    if "SMA_Short" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_Short"],
            name="SMA short", mode="lines",
            connectgaps=False
        ))

    if "SMA_Long" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_Long"],
            name="SMA long", mode="lines",
            connectgaps=False
        ))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title="Time", showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True)
    fig.update_yaxes(title="Price", showgrid=True)
    return fig

def render_hero(title: str, subtitle: str, pills: list[str]):
    pills_html = "".join([f'<span class="pill">{p}</span>' for p in pills])

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
            <div class="hero-pills">{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Go to",
        ["Quant A (single asset)", "Quant B (portfolio)", "About"],
        index=0
    )
    st.divider()
    if st.button("Force refresh now", help="Clears cache and reloads the app"):
        st.cache_data.clear()
        st.rerun()

st.title("Quant Research Dashboard")
st.caption("Live data via public API. Auto refresh every 5 minutes. Two modules integrated.")


# -----------------------
# Quant A
# -----------------------
if page == "Quant A (single asset)":
    render_hero(
    title="Quant A - Single Asset Strategy",
    subtitle="Moving-average crossover strategy with real-time risk metrics",
    pills=[
        "Signal: SMA crossover",
        "Metrics: Sharpe, Drawdown",
        "Data: yfinance",
        "Mode: Real-time"
    ]
)

    # Settings card
    with st.container(border=True):
        st.markdown("**Settings**")
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])

        with c1:
            ticker = st.text_input(
                "Symbol",
                value="AAPL",
                help="Example: AAPL, MSFT, BTC-USD, EURUSD=X"
            ).strip().upper()

        with c2:
            short_window = st.slider(
                "SMA short",
                5, 80, 20,
                help="Short moving average window"
            )

        with c3:
            long_window = st.slider(
                "SMA long",
                10, 200, 83,
                help="Long moving average window"
            )

        with c4:
            period = st.selectbox(
                "Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                index=1,
                help="Historical range to download"
            )

        with c5:
            interval = st.selectbox(
                "Interval",
                ["1m", "2m", "5m", "15m", "30m", "60m", "1d"],
                index=2,
                help="Sampling frequency"
            )

        # Guard rails for SMA windows
        if short_window >= long_window:
            st.warning("SMA short should be < SMA long. Auto-adjusting long window.")
            long_window = max(long_window, short_window + 1)

    # Fetch + spinner
    with st.spinner("Fetching market data…"):
        df = cached_fetch_data(ticker, period=period, interval=interval)

    if df is None or df.empty:
        st.error("No data retrieved. Try another symbol or a different period/interval.")
        st.stop()

    # Strategy
    analyzer = QuantAAnalyzer(df)
    df_strat = analyzer.apply_strategy(short_window=short_window, long_window=long_window)

    if df_strat is None or df_strat.empty:
        st.error("Strategy output is empty. Try smaller windows or another symbol.")
        st.stop()

    last_row = df_strat.iloc[-1]
    last_price = float(last_row["Close"]) if "Close" in df_strat.columns else 0.0
    metrics = analyzer.get_metrics()
    signal_text, signal_color, signal_help = format_signal(last_row)

    # KPIs card
    with st.container(border=True):
        st.markdown("**Key metrics**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Last price", f"{last_price:.2f}")
        k2.markdown(
            f"""
            <div style="padding:12px;border-radius:16px;border:1px solid rgba(255,255,255,0.10);
                        background:rgba(255,255,255,0.04);">
              <div style="font-size:12px;color:rgba(229,231,235,0.70);">Signal</div>
              <div style="font-size:22px;font-weight:900;color:{signal_color};">{signal_text}</div>
              <div style="font-size:12px;color:rgba(229,231,235,0.70);">{signal_help}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        k3.metric("Max drawdown", f"{metrics.get('Max Drawdown', 0.0):.2%}")
        k4.metric("Sharpe (rf=0)", f"{metrics.get('Sharpe Ratio', 0.0):.2f}")

    # Chart card (matplotlib version kept)
    with st.container(border=True):
        st.markdown("**Chart**")
        df_plot = _clean_timeseries(df_strat)
        st.plotly_chart(plot_quant_a(df_plot, ticker), use_container_width=True)


    # Data
    with st.expander("Show data (last 300 rows)"):
        st.dataframe(df_strat.tail(300), use_container_width=True)


# -----------------------
# Quant B
# -----------------------
elif page == "Quant B (portfolio)":
    render_hero(
    title="Quant B - Multi-Asset Portfolio",
    subtitle="Portfolio simulation, diversification & correlation analysis",
    pills=[
        "≥ 3 assets",
        "Metrics: Sharpe, DD, Corr",
        "Weights: configurable",
        "Focus: Risk & Diversification"
    ]
)
    render_quant_b()


# =============================
# Page: About (premium clickable sections) — FIXED + better layout
# =============================
else:
    from streamlit_option_menu import option_menu

    # ---- Premium CSS for clickable cards + center layout fix
    st.markdown(
        """
        <style>
          /* Fix: prevent weird right shift by controlling main container width */
          .block-container {
            max-width: 1200px !important;
            padding-top: 2.2rem !important;
          }

          .about-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 14px;
            margin-bottom: 14px;
          }
          @media (max-width: 1100px) {
            .about-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 700px) {
            .about-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); }
          }

          .about-card {
            position: relative;
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            padding: 16px 16px 14px 16px;
            min-height: 128px;
            transition: transform 120ms ease, border 120ms ease;
          }
          .about-card:hover {
            transform: translateY(-2px);
            border: 1px solid rgba(255,255,255,0.16);
          }

          .bg-vision::before,
          .bg-arch::before,
          .bg-metrics::before,
          .bg-ux::before,
          .bg-ops::before,
          .bg-future::before {
            content: "";
            position: absolute;
            inset: 0;
            opacity: 0.85;
          }

          /* Subtle “image-like” gradients (no external files) */
          .bg-vision::before  { background: radial-gradient(circle at 20% 20%, rgba(46,91,255,0.55), transparent 50%),
                                         radial-gradient(circle at 80% 70%, rgba(0,220,255,0.30), transparent 50%),
                                         linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); }
          .bg-arch::before    { background: radial-gradient(circle at 30% 30%, rgba(0,255,170,0.40), transparent 55%),
                                         radial-gradient(circle at 80% 70%, rgba(46,91,255,0.35), transparent 55%),
                                         linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); }
          .bg-metrics::before { background: radial-gradient(circle at 25% 20%, rgba(255,184,0,0.40), transparent 55%),
                                         radial-gradient(circle at 80% 70%, rgba(46,91,255,0.25), transparent 55%),
                                         linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); }
          .bg-ux::before      { background: radial-gradient(circle at 25% 30%, rgba(255,0,200,0.28), transparent 55%),
                                         radial-gradient(circle at 80% 70%, rgba(0,220,255,0.22), transparent 55%),
                                         linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); }
          .bg-ops::before     { background: radial-gradient(circle at 25% 30%, rgba(180,255,0,0.20), transparent 55%),
                                         radial-gradient(circle at 80% 70%, rgba(46,91,255,0.25), transparent 55%),
                                         linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); }
          .bg-future::before  { background: radial-gradient(circle at 30% 25%, rgba(46,91,255,0.45), transparent 55%),
                                         radial-gradient(circle at 75% 70%, rgba(255,70,70,0.22), transparent 55%),
                                         linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); }

          .about-card * { position: relative; z-index: 2; }
          .about-title { font-size: 15px; font-weight: 900; color: #E5E7EB; margin-bottom: 6px; }
          .about-desc  { font-size: 12px; color: rgba(229,231,235,0.78); line-height: 1.35; }
          .about-chip  { display:inline-block; margin-top:10px; font-size:11px; padding:6px 10px;
                         border-radius:999px; border: 1px solid rgba(255,255,255,0.10);
                         background: rgba(0,0,0,0.18); color: rgba(229,231,235,0.90); }
          .softline { height:1px; width:100%; background: rgba(255,255,255,0.08); margin: 14px 0; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Small helper: custom “badges”
    # -----------------------------
    def soft_badge(text: str) -> None:
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                padding:6px 14px;
                border-radius:999px;
                background:rgba(255,255,255,0.08);
                border:1px solid rgba(255,255,255,0.15);
                font-size:12px;
                color:#E5E7EB;
                margin-right:8px;
                margin-bottom:6px;
            ">
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------
    # Header
    # -----------------------------
    st.header("About")
    st.write("A clean, modular quantitative research dashboard built for real-time exploration and risk-aware decisions.")

    # Badges row (fixed: NOT inside any column by mistake)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        soft_badge("API: yfinance")
    with c2:
        soft_badge("Auto-refresh: 5 min")
    with c3:
        soft_badge("UI: Streamlit + Plotly")
    with c4:
        soft_badge("Modules: Quant A & Quant B")

    st.markdown('<div class="softline"></div>', unsafe_allow_html=True)

    # -----------------------------
    # Clickable sections (menu)
    # -----------------------------
    selected = option_menu(
        menu_title=None,
        options=["Vision", "Architecture", "Metrics", "UX", "Ops & Run", "Future Work"],
        icons=["bullseye", "diagram-3", "bar-chart-line", "magic", "gear", "rocket"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#E5E7EB", "font-size": "16px"},
            "nav-link": {
                "font-size": "13px",
                "text-align": "center",
                "margin": "0px 6px",
                "padding": "8px 10px",
                "border-radius": "999px",
                "color": "#E5E7EB",
                "background-color": "rgba(255,255,255,0.05)",
                "border": "1px solid rgba(255,255,255,0.08)",
            },
            "nav-link-selected": {
                "background-color": "rgba(46,91,255,0.35)",
                "border": "1px solid rgba(46,91,255,0.55)",
                "color": "#FFFFFF",
            },
        }
    )

    # -----------------------------
    # “Overview cards” (visual)
    # -----------------------------
    st.markdown(
        """
        <div class="about-grid">
          <div class="about-card bg-vision">
            <div class="about-title">Vision</div>
            <div class="about-desc">Why the dashboard exists, what problem it solves.</div>
            <div class="about-chip">Research mindset</div>
          </div>
          <div class="about-card bg-arch">
            <div class="about-title">Architecture</div>
            <div class="about-desc">Clean separation: data → analytics → UI.</div>
            <div class="about-chip">Maintainable</div>
          </div>
          <div class="about-card bg-metrics">
            <div class="about-title">Metrics</div>
            <div class="about-desc">Sharpe, drawdown, diversification, correlation explained.</div>
            <div class="about-chip">Interpretability</div>
          </div>
          <div class="about-card bg-ux">
            <div class="about-title">UX</div>
            <div class="about-desc">Interactive charts, tooltips, clean settings.</div>
            <div class="about-chip">Client-friendly</div>
          </div>
          <div class="about-card bg-ops">
            <div class="about-title">Ops & Run</div>
            <div class="about-desc">How to run locally + deployment checklist.</div>
            <div class="about-chip">Production-ish</div>
          </div>
          <div class="about-card bg-future">
            <div class="about-title">Future Work</div>
            <div class="about-desc">Extensions that turn it into a full research platform.</div>
            <div class="about-chip">Roadmap</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="softline"></div>', unsafe_allow_html=True)

    # -----------------------------
    # Section content (clean + useful)
    # -----------------------------
    if selected == "Vision":
        st.subheader("🎯 Vision")
        st.markdown(
            """
**Quant Research Dashboard** is a lightweight research terminal.

It enables you to:
- Explore time series in real-time (public API data)
- Prototype strategies quickly (Quant A)
- Understand portfolio risk & diversification (Quant B)
- Keep the UI simple while providing research-grade metrics

The goal is **clarity + robustness**, not black-box signals.
            """
        )

    elif selected == "Architecture":
        st.subheader("🏗️ Architecture")
        st.markdown(
            """
**Design principle:** separation of concerns.

- **Data layer**: `src/data_loader.py` fetches and cleans market data.
- **Quant A**: single-asset strategy + metrics (Sharpe, drawdown).
- **Quant B**: portfolio simulation (≥3 assets) + risk/diversification metrics.
- **UI**: Streamlit for layout + Plotly for interactive charts.

This structure supports clean Git collaboration (branches + pull requests).
            """
        )
        st.info("Each module is isolated so teammates can work independently with minimal merge conflicts.")

    elif selected == "Metrics":
        st.subheader("📊 Metrics")
        st.markdown("Key metrics are explained below for interpretability:")

        with st.expander("Sharpe Ratio (rf = 0)"):
            st.write("Risk-adjusted performance. Higher is better. Negative means underperformance relative to volatility.")

        with st.expander("Max Drawdown"):
            st.write("Worst peak-to-trough decline of the equity curve. Important downside risk measure.")

        with st.expander("Correlation Matrix (Quant B)"):
            st.write("Shows how assets move together. Lower average correlation generally improves diversification.")

        with st.expander("Diversification Ratio (Quant B)"):
            st.write("Weighted average asset volatility divided by portfolio volatility. Higher is better diversification.")

        with st.expander("Effective Number of Bets (Quant B)"):
            st.write("1 / sum(w²). Higher = more evenly distributed weights (less concentration).")

    elif selected == "UX":
        st.subheader("✨ UX & Product choices")
        st.markdown(
            """
Decisions that improve user experience:

- **Dark theme** to reduce eye fatigue
- **Plotly charts** for hover, zoom and precise inspection
- **Auto-refresh every 5 minutes** + manual override
- **Safe handling of API constraints** (yfinance period/interval limitations)
- **Settings grouped** to keep the main view readable
            """
        )
        st.success("The dashboard behaves like a product: robust, interactive and easy to understand.")

    elif selected == "Ops & Run":
        st.subheader("⚙️ Ops & Run")

        st.markdown("### Run locally")
        st.code(
            """python -m venv .venv
# Windows:
.\\.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
""",
            language="bash",
        )

        st.markdown("### Deployment checklist (Linux)")
        st.markdown(
            """
- Run the app 24/7 using a service manager (systemd recommended)
- Add a cron job for daily report generation (`scripts/daily_report.py`)
- Write a README: setup, usage, screenshots, and troubleshooting
            """
        )
        st.warning("If you want, I can give you the exact systemd + cron files ready to copy/paste.")

    else:  # Future Work
        st.subheader("🚀 Future Work")
        st.markdown(
            """
Possible extensions (research-grade improvements):

- Transaction costs + slippage for realistic backtests
- Volatility targeting / risk parity variants
- Efficient frontier (Markowitz) visualization
- Exportable reports (CSV/HTML/PDF) from the UI
- Multi-source data (AlphaVantage / FRED / ECB) + caching
            """
        )
        st.info("Mentioning these shows maturity and improves the perceived quality of the project.")
