import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from src.data_loader import fetch_data
from src.quant_a import QuantAAnalyzer
from src.quant_b import render as render_quant_b

# --- CONFIGURATION ---
st.set_page_config(page_title="Quant Terminal", page_icon="⚡", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

def set_page(page_name):
    st.session_state.page = page_name

# --- DONNÉES ---
INDEX_HOLDINGS = {
    "SPY": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "JPM", "V"],
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "PEP"],
    "DIA": ["UNH", "GS", "MSFT", "HD", "CAT", "AMGN", "CRM", "V", "MCD", "TRV"],
    "IWM": ["SMCI", "MSTR", "CVNA", "ELF", "SPS", "FCNCA", "CMA", "ONTO", "GKOS", "TGLS"]
}

ASSET_MAP = {
    "Indices": {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "DIA": "Dow Jones"},
    "Tech": {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "TSLA": "Tesla", "GOOGL": "Google", "AMZN": "Amazon"},
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"}
}

# --- CSS (Design KPI Cards & Layout) ---
st.markdown("""
<style>
    [data-testid="stHeader"] { display: none; }
    .stApp { background-color: #F8FAFC; }
    
    .block-container { 
        padding-top: 3.5rem !important; 
        padding-bottom: 2rem !important; 
        max-width: 1600px !important; 
        margin: 0 auto !important;
    }

    /* TICKER */
    .ticker-wrap {
        position: fixed; top: 0; left: 0; width: 100%; height: 45px;
        background-color: #0F172A; color: white; z-index: 99999;
        display: flex; align-items: center; overflow: hidden;
        border-bottom: 3px solid #F97316;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        font-family: 'Roboto Mono', monospace; font-size: 0.85rem;
    }
    .ticker-content { display: flex; animation: marquee 60s linear infinite; }
    .ticker-item { display: flex; align-items: center; padding: 0 25px; white-space: nowrap; }
    .ticker-symbol { font-weight: 700; margin-right: 8px; color: #E2E8F0; }
    .ticker-change-up { color: #4ADE80; font-weight: 600; }
    .ticker-change-down { color: #F87171; font-weight: 600; }
    @keyframes marquee { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }

    /* NAVIGATION */
    div.stButton > button { 
        border-radius: 12px !important; border: 1px solid #E2E8F0 !important; 
        font-weight: 700 !important; font-size: 1.2rem !important; color: #334155 !important; 
        height: 4.5rem !important; background: white !important; transition: all 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important; margin-top: 10px;
    }
    div.stButton > button:hover { 
        border-color: #F97316 !important; color: #C2410C !important; 
        background: #FFF7ED !important; transform: translateY(-2px); 
    }

    /* HEADER TITRE */
    .app-title { font-size: 3rem; font-weight: 900; color: #1E293B; letter-spacing: -1px; line-height: 1; }
    .app-subtitle { font-size: 1.1rem; color: #94A3B8; font-weight: 500; font-style: italic;}
    .title-container { padding-left: 10px; }

    /* --- NOUVEAU : KPI CARDS --- */
    .kpi-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #F97316; /* Bordure orange distinctive */
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-label { font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 800; color: #1E293B; }
    .kpi-sub { font-size: 0.85rem; color: #94A3B8; margin-top: 4px; }

    /* DASHBOARD CARDS */
    .metric-card { 
        background: white; padding: 20px; border-radius: 12px; 
        border: 1px solid #E2E8F0; text-align: center; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 100%; 
    }
    .action-box {
        background: white; border: 1px solid #E2E8F0; border-radius: 16px; 
        padding: 30px; text-align: center; transition: transform 0.2s; height: 100%; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .action-box:hover { border-color: #F97316; transform: translateY(-3px); box-shadow: 0 10px 20px -5px rgba(0,0,0,0.08); }

    hr { margin: 20px 0 !important; border-top: 1px solid #E2E8F0; }
    button[kind="primary"] { background-color: #F97316 !important; border-color: #F97316 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=300000, key="auto_refresh_5min")

@st.cache_data(ttl=300)
def cached_fetch_data(t, p, i): return fetch_data(t, period=p, interval=i)

# --- TICKER ---
def render_ticker():
    symbols = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "AMD", "META"]
    items_html = ""
    for s in symbols:
        try:
            df = cached_fetch_data(s, "2d", "1d")
            if not df.empty:
                pct = df["Close"].pct_change().iloc[-1]
                color = "ticker-change-up" if pct >= 0 else "ticker-change-down"
                arrow = "▲" if pct >= 0 else "▼"
                items_html += f"""<div class="ticker-item"><span class="ticker-symbol">{s.replace("-USD","")}</span><span class="{color}">{arrow} {pct:.2%}</span></div>"""
        except: continue
    st.markdown(f"""<div class="ticker-wrap"><div class="ticker-content">{items_html} {items_html}</div></div>""", unsafe_allow_html=True)

render_ticker()

# --- HEADER NAVIGATION ---
c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1], gap="medium")
with c1: 
    st.markdown("""<div class="title-container"><div class="app-title">⚡ Quant Terminal</div><div class="app-subtitle">Your portfolio, automated.</div></div>""", unsafe_allow_html=True)
with c2: st.button("🏠 Dashboard", use_container_width=True, on_click=set_page, args=("Dashboard",))
with c3: st.button("📈 Single Asset", use_container_width=True, on_click=set_page, args=("Quant A",))
with c4: st.button("💼 Portfolio", use_container_width=True, on_click=set_page, args=("Quant B",))

st.markdown("<hr>", unsafe_allow_html=True)

# --- CHART (Adapté aux nouvelles stratégies) ---
def plot_chart(df, ticker):
    fig = go.Figure()
    # PRIX
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color='#F97316', width=2), fill='tozeroy', fillcolor='rgba(249, 115, 22, 0.05)'))
    
    # MOYENNES MOBILES (Si présentes)
    if "SMA_Short" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Short"], name="SMA Short", line=dict(color='#38BDF8', width=1.5)))
    if "SMA_Long" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Long"], name="SMA Long", line=dict(color='#94A3B8', width=1.5)))
    
    # BANDES DE BOLLINGER (Si présentes)
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color='rgba(148, 163, 184, 0.5)', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color='rgba(148, 163, 184, 0.5)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)'))

    # EQUITY CURVE
    if "Equity_Curve" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Equity_Curve"], name="Strategy", line=dict(color='#10B981', width=2), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Base 100"))

    fig.update_layout(template="plotly_white", height=550, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.05), xaxis=dict(showgrid=True, gridcolor='#F1F5F9'), yaxis=dict(showgrid=True, gridcolor='#F1F5F9'))
    return fig

# --- HELPER HOLDINGS ---
def get_holdings_data(index_ticker):
    if index_ticker not in INDEX_HOLDINGS: return None
    comps = INDEX_HOLDINGS[index_ticker]
    data = []
    for c in comps:
        try:
            d = fetch_data(c, "2d", "1d")
            if not d.empty: data.append({"Ticker": c, "Price": d["Close"].iloc[-1], "Change": d["Close"].pct_change().iloc[-1]})
        except: continue
    return pd.DataFrame(data) if data else None

# --- PAGES ---

if st.session_state.page == "Dashboard":
    st.markdown("<h1 style='text-align: center; margin-bottom: 20px; margin-top:0px;'>Market Overview</h1>", unsafe_allow_html=True)
    
    tickers = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL", "MSFT"]
    data = []
    for s in tickers:
        d = cached_fetch_data(s, "2d", "1d")
        if not d.empty: data.append({"t": s, "p": d["Close"].iloc[-1], "chg": d["Close"].pct_change().iloc[-1]})
    
    if data:
        df_m = pd.DataFrame(data)
        best = df_m.loc[df_m['chg'].idxmax()]
        worst = df_m.loc[df_m['chg'].idxmin()]
        sent = df_m['chg'].mean()
        sent_txt, sent_col = ("BULLISH", "#16A34A") if sent > 0 else ("BEARISH", "#DC2626")

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"""<div class="metric-card"><div class="m-lbl">🚀 Top Mover</div><div class="m-val" style="color:#16A34A">{best['t']}</div><div style="font-size:0.9rem">{best['chg']:+.2%}</div></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="metric-card"><div class="m-lbl">Sentiment</div><div class="m-val" style="color:{sent_col}">{sent_txt}</div><div style="font-size:0.9rem">Avg: {sent:+.2%}</div></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="metric-card"><div class="m-lbl">📉 Laggard</div><div class="m-val" style="color:#DC2626">{worst['t']}</div><div style="font-size:0.9rem">{worst['chg']:+.2%}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Market Pulse")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        for i, tick in enumerate(["SPY", "QQQ", "BTC-USD", "ETH-USD"]):
            with [c1, c2, c3, c4][i]:
                d = cached_fetch_data(tick, "2d", "1d")
                if not d.empty: st.metric(tick.replace("-USD",""), f"${d['Close'].iloc[-1]:,.2f}", f"{d['Close'].pct_change().iloc[-1]:+.2%}")
        st.markdown("<hr style='margin:10px 0 !important'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for i, tick in enumerate(["NVDA", "TSLA", "AAPL", "MSFT"]):
            with [c1, c2, c3, c4][i]:
                d = cached_fetch_data(tick, "2d", "1d")
                if not d.empty: st.metric(tick, f"${d['Close'].iloc[-1]:,.2f}", f"{d['Close'].pct_change().iloc[-1]:+.2%}")

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("""<div class="action-box"><h3 style="margin:0">📈 Single Asset</h3><p style="font-size:0.9rem; color:#64748B; margin-bottom:10px">Deep dive & Backtest.</p></div>""", unsafe_allow_html=True)
        st.button("Open Module A", use_container_width=True, type="primary", on_click=set_page, args=("Quant A",))
    with b2:
        st.markdown("""<div class="action-box"><h3 style="margin:0">💼 Portfolio</h3><p style="font-size:0.9rem; color:#64748B; margin-bottom:10px">Allocation & Diversification.</p></div>""", unsafe_allow_html=True)
        st.button("Open Module B", use_container_width=True, type="primary", on_click=set_page, args=("Quant B",))

elif st.session_state.page == "Quant A":
    c_left, c_right = st.columns([1, 4]) 
    
    # --- CONTROLES LATERAUX ---
    params = {}
    with c_left:
        with st.container(border=True):
            st.markdown("**Settings**")
            cat = st.selectbox("Market", list(ASSET_MAP.keys()), label_visibility="collapsed")
            tick = st.selectbox("Asset", list(ASSET_MAP[cat].keys()), label_visibility="collapsed")
            st.markdown("---")
            
            # SELECTION STRATEGIE
            strat = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion", "Bollinger Bands", "Buy & Hold"])
            
            # SLIDERS DYNAMIQUES SELON STRATEGIE
            if strat == "SMA Crossover":
                params['short_window'] = st.slider("Fast MA", 5, 50, 20)
                params['long_window'] = st.slider("Slow MA", 20, 200, 50)
            elif strat == "RSI Mean Reversion":
                params['rsi_window'] = st.slider("RSI Window", 5, 30, 14)
                params['rsi_lower'] = st.slider("Oversold (<)", 10, 40, 30)
                params['rsi_upper'] = st.slider("Overbought (>)", 60, 90, 70)
            elif strat == "Bollinger Bands":
                params['bb_window'] = st.slider("Window", 10, 50, 20)
                params['bb_std'] = st.slider("Std Dev", 1.0, 3.0, 2.0)
            
            st.markdown("---")
            per = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)

    with c_right:
        df = cached_fetch_data(tick, per, "1d")
        if df is not None and not df.empty:
            analyzer = QuantAAnalyzer(df)
            
            # Application de la stratégie choisie
            df_strat = analyzer.apply_strategy(strategy_type=strat, **params)
            met = analyzer.get_metrics()
            
            last = df["Close"].iloc[-1]
            chg = df["Close"].pct_change().iloc[-1]
            col_chg = "#16A34A" if chg >= 0 else "#DC2626"
            
            # INTERPRETATION SIGNAL
            sig_txt = "NEUTRAL"
            if 'Signal' in df_strat.columns:
                last_sig = df_strat['Signal'].iloc[-1]
                if last_sig == 1: sig_txt = "BUY ZONE"
                elif last_sig == 0 and strat != "Buy & Hold": sig_txt = "SELL / CASH"
                elif strat == "Buy & Hold": sig_txt = "INVESTED"

            # HEADER
            c_head, c_met = st.columns([1.5, 3.5])
            with c_head:
                st.markdown(f"<h2 style='margin:0'>{tick}</h2>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:2rem;font-weight:800'>${last:,.2f}</span> <span style='color:{col_chg};font-weight:bold;font-size:1.2rem'>{chg:+.2%}</span>", unsafe_allow_html=True)
            
            # --- AFFICHAGE METRIQUES AVEC NOUVELLES CARTES ---
            with c_met:
                m1, m2, m3, m4 = st.columns(4)
                
                # Fonction helper pour HTML propre
                def kpi(label, value, sub, color="#1E293B"):
                    return f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value" style="color:{color}">{value}</div>
                        <div class="kpi-sub">{sub}</div>
                    </div>
                    """
                
                c_ret = "#16A34A" if met.get('Total Return',0) >= 0 else "#DC2626"
                
                with m1: st.markdown(kpi("Return", f"{met.get('Total Return',0):+.1%}", "Strategy", c_ret), unsafe_allow_html=True)
                with m2: st.markdown(kpi("Sharpe", f"{met.get('Sharpe Ratio',0):.2f}", "Risk Adj."), unsafe_allow_html=True)
                with m3: st.markdown(kpi("Max DD", f"{met.get('Max Drawdown',0):.1%}", "Drawdown", "#DC2626"), unsafe_allow_html=True)
                with m4: st.markdown(kpi("Signal", sig_txt, "Current", "#F97316"), unsafe_allow_html=True)

            st.markdown("<hr style='margin:15px 0'>", unsafe_allow_html=True)
            st.plotly_chart(plot_chart(df_strat, tick), use_container_width=True)

            if tick in INDEX_HOLDINGS:
                st.markdown("### 🏗️ Top Holdings")
                df_hold = get_holdings_data(tick)
                if df_hold is not None:
                    cols = st.columns(5)
                    for i, row in df_hold.iterrows():
                        col_idx = i % 5
                        c_val = "#16A34A" if row['Change'] >= 0 else "#DC2626"
                        with cols[col_idx]:
                            st.markdown(f"""
                            <div style="background:white;padding:10px;border-radius:8px;border:1px solid #E2E8F0;text-align:center;margin-bottom:8px;">
                                <div style="font-weight:bold;font-size:0.9rem">{row['Ticker']}</div>
                                <div style="color:{c_val};font-size:0.85rem;">{row['Change']:+.2%}</div>
                            </div>
                            """, unsafe_allow_html=True)

        else: st.error(f"Data unavailable for {tick}.")

elif st.session_state.page == "Quant B":
    render_quant_b()