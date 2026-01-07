import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from src.data_loader import fetch_data
from src.quant_a import QuantAAnalyzer
from src.quant_b import render as render_quant_b

# --- CONFIGURATION DE LA PAGE ---
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

# --- CSS BIG UI & COMPACT ---
st.markdown("""
<style>
    /* 1. CACHER LA BARRE STREAMLIT (Deploy, Menu...) */
    [data-testid="stHeader"] { display: none; }
    
    /* 2. FOND & LAYOUT SERRÉ */
    .stApp { background-color: #F8FAFC; }
    
    .block-container { 
        padding-top: 3.5rem !important; /* Juste la place pour le ticker (45px + marge) */
        padding-bottom: 2rem !important; 
        max-width: 1600px !important; 
        margin: 0 auto !important;
    }

    /* 3. TICKER TAPE (Fixe en haut) */
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

    /* 4. BOUTONS NAVIGATION (BIG SIZE) */
    div.stButton > button { 
        border-radius: 12px !important; 
        border: 1px solid #E2E8F0 !important; 
        font-weight: 700 !important; 
        font-size: 1.2rem !important; /* Texte plus gros */
        color: #334155 !important; 
        height: 4.5rem !important; /* Boutons plus hauts */
        background: white !important; 
        transition: all 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
        margin-top: 10px; /* Alignement avec le titre */
    }
    div.stButton > button:hover { 
        border-color: #F97316 !important; color: #C2410C !important; 
        background: #FFF7ED !important; transform: translateY(-2px); 
        box-shadow: 0 5px 12px rgba(249, 115, 22, 0.15) !important;
    }

    /* 5. TITRE SITE (BIG SIZE) */
    .app-title { font-size: 3rem; font-weight: 900; color: #1E293B; letter-spacing: -1px; line-height: 1; margin-bottom: 0px; }
    .app-subtitle { font-size: 1.1rem; color: #94A3B8; font-weight: 500; margin-top: 0px; margin-bottom: 0px; font-style: italic;}
    .title-container { padding-left: 10px; }

    /* 6. ELEMENTS GRAPHIQUES */
    .metric-card { 
        background: white; padding: 20px; border-radius: 12px; 
        border: 1px solid #E2E8F0; text-align: center; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 100%; 
    }
    .m-lbl { font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 2px; }
    .m-val { font-size: 1.8rem; font-weight: 800; color: #0F172A; }
    
    .action-box {
        background: white; border: 1px solid #E2E8F0; border-radius: 16px; 
        padding: 30px; text-align: center; transition: transform 0.2s; height: 100%; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .action-box:hover { border-color: #F97316; transform: translateY(-3px); box-shadow: 0 10px 20px -5px rgba(0,0,0,0.08); }

    hr { margin: 20px 0 !important; border-top: 1px solid #E2E8F0; }
    
    /* Boutons Primary */
    button[kind="primary"] {
        background-color: #F97316 !important; border-color: #F97316 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=300000, key="auto_refresh_5min")

@st.cache_data(ttl=300)
def cached_fetch_data(t, p, i): return fetch_data(t, period=p, interval=i)

# --- TICKER BAR ---
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

# --- NAVIGATION HEADER (OPTIMISÉ & TAGLINE) ---
# c1 plus large pour le titre
c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1], gap="medium")

with c1: 
    st.markdown("""
        <div class="title-container">
            <div class="app-title">⚡ Quant Terminal</div>
            <div class="app-subtitle">Your portfolio, automated.</div>
        </div>
    """, unsafe_allow_html=True)

with c2: st.button("🏠 Dashboard", use_container_width=True, on_click=set_page, args=("Dashboard",))
with c3: st.button("📈 Single Asset", use_container_width=True, on_click=set_page, args=("Quant A",))
with c4: st.button("💼 Portfolio", use_container_width=True, on_click=set_page, args=("Quant B",))

st.markdown("<hr>", unsafe_allow_html=True)

# --- GRAPH FUNCTION ---
def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color='#F97316', width=2), fill='tozeroy', fillcolor='rgba(249, 115, 22, 0.05)'))
    if "SMA_Short" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Short"], name="Fast MA", line=dict(color='#38BDF8', width=1.5)))
    if "SMA_Long" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["SMA_Long"], name="Slow MA", line=dict(color='#94A3B8', width=1.5, dash='dot')))
    if "Equity_Curve" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Equity_Curve"], name="Strategy", line=dict(color='#10B981', width=2), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Base 100"))

    fig.update_layout(template="plotly_white", height=500, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.05), xaxis=dict(showgrid=True, gridcolor='#F1F5F9'), yaxis=dict(showgrid=True, gridcolor='#F1F5F9'))
    return fig

# --- HELPER HOLDINGS ---
def get_holdings_data(index_ticker):
    if index_ticker not in INDEX_HOLDINGS: return None
    comps = INDEX_HOLDINGS[index_ticker]
    data = []
    for c in comps:
        try:
            d = fetch_data(c, "2d", "1d")
            if not d.empty:
                data.append({"Ticker": c, "Price": d["Close"].iloc[-1], "Change": d["Close"].pct_change().iloc[-1]})
        except: continue
    return pd.DataFrame(data) if data else None

# --- PAGES ---

if st.session_state.page == "Dashboard":
    # Titre collé au HR
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
        with c1: st.markdown(f"""<div class="metric-card"><div class="m-lbl">🚀 Top Mover</div><div class="m-val" style="color:#16A34A">{best['t']}</div><div class="m-sub">{best['chg']:+.2%}</div></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="metric-card"><div class="m-lbl">Sentiment</div><div class="m-val" style="color:{sent_col}">{sent_txt}</div><div class="m-sub">Avg: {sent:+.2%}</div></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="metric-card"><div class="m-lbl">📉 Laggard</div><div class="m-val" style="color:#DC2626">{worst['t']}</div><div class="m-sub">{worst['chg']:+.2%}</div></div>""", unsafe_allow_html=True)

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
    
    with c_left:
        with st.container(border=True):
            st.markdown("**Settings**")
            cat = st.selectbox("Market", list(ASSET_MAP.keys()), label_visibility="collapsed")
            tick = st.selectbox("Asset", list(ASSET_MAP[cat].keys()), label_visibility="collapsed")
            st.markdown("---")
            strat = st.selectbox("Strategy", ["SMA Crossover", "Buy & Hold"])
            if strat == "SMA Crossover":
                s_w = st.slider("Fast", 5, 50, 20)
                l_w = st.slider("Slow", 20, 200, 50)
            else: s_w, l_w = 20, 50
            st.markdown("---")
            per = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)

    with c_right:
        df = cached_fetch_data(tick, per, "1d")
        if df is not None and not df.empty:
            analyzer = QuantAAnalyzer(df)
            df_strat = analyzer.apply_strategy(short_window=s_w, long_window=l_w)
            met = analyzer.get_metrics()
            
            last = df["Close"].iloc[-1]
            chg = df["Close"].pct_change().iloc[-1]
            col_chg = "#16A34A" if chg >= 0 else "#DC2626"
            
            sig_txt = "NEUTRAL"
            if strat == "SMA Crossover" and "SMA_Short" in df_strat.columns:
                if df_strat["SMA_Short"].iloc[-1] > df_strat["SMA_Long"].iloc[-1]: sig_txt = "BUY ZONE"
                elif df_strat["SMA_Short"].iloc[-1] < df_strat["SMA_Long"].iloc[-1]: sig_txt = "SELL ZONE"

            c_head, c_met = st.columns([1.5, 3.5])
            with c_head:
                st.markdown(f"<h2 style='margin:0'>{tick}</h2>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:2rem;font-weight:800'>${last:,.2f}</span> <span style='color:{col_chg};font-weight:bold;font-size:1.2rem'>{chg:+.2%}</span>", unsafe_allow_html=True)
            with c_met:
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("Return", f"{met.get('Total Return',0):+.1%}")
                with m2: st.metric("Sharpe", f"{met.get('Sharpe Ratio',0):.2f}")
                with m3: st.metric("Max DD", f"{met.get('Max Drawdown',0):.1%}")
                with m4: st.metric("Signal", sig_txt)

            st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
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