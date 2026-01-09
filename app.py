"""
Quant Terminal Pro - Professional Quantitative Finance Platform
Main Streamlit Application with Virtual Trading
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Page Configuration
st.set_page_config(
    page_title="Quant Terminal Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Internal Module Imports
from src.quant_a import render_quant_a, run_strategy, STRATEGIES, apply_rsi_strategy
from src.quant_b import render_quant_b
from src.data_loader import clear_cache, get_latest_price, fetch_data
from src.utils import COLORS, get_paris_time, is_market_open, POPULAR_TICKERS, format_currency


# ============================================================
# DESIGN SYSTEM - PROFESSIONAL COLOR PALETTE
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
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    defaults = {
        'current_page': "Dashboard",
        'virtual_balance': 100000.0,
        'portfolio': {},
        'trade_history': [],
        'initial_balance': 100000.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# TICKER BAR - LIVE MARKET DATA SCROLL
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_ticker_html_content():
    tickers = ["SPY", "QQQ", "DIA", "BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    items_html = ""
    for ticker in tickers:
        try:
            data = get_latest_price(ticker)
            if data:
                is_up = data['is_positive']
                color = THEME['success'] if is_up else THEME['danger']
                arrow = "▲" if is_up else "▼"
                items_html += f'<div class="ticker-item"><span class="ticker-symbol">{ticker}</span><span class="ticker-price">${data["price"]:,.2f}</span><span class="ticker-change" style="color:{color}">{arrow} {abs(data["change_pct"]):.2f}%</span></div>'
        except:
            continue
    return items_html


# ============================================================
# GLOBAL STYLES
# ============================================================
def render_global_styles():
    ticker_items_html = get_ticker_html_content()
    
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}

.mono, .ticker-price, .metric-value, .card-value,
div[data-testid="stMetricValue"],
div[data-testid="stDataFrame"] {{
    font-family: 'JetBrains Mono', monospace !important;
}}

.stApp {{
    background: linear-gradient(180deg, {THEME['bg_primary']} 0%, {THEME['bg_secondary']} 100%);
    color: {THEME['text_primary']};
}}

.main .block-container {{
    max-width: 1400px;
    padding-top: 4.5rem !important;
    padding-left: 2rem;
    padding-right: 2rem;
    margin: auto;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

@keyframes ticker-scroll {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}

.ticker-wrap {{
    width: 100%;
    overflow: hidden;
    background: linear-gradient(90deg, {THEME['bg_secondary']} 0%, {THEME['bg_tertiary']} 50%, {THEME['bg_secondary']} 100%);
    border-bottom: 1px solid {THEME['border']};
    position: fixed;
    top: 0;
    left: 0;
    z-index: 9999;
    height: 42px;
    display: flex;
    align-items: center;
}}

.ticker-move {{
    display: flex;
    white-space: nowrap;
    animation: ticker-scroll 80s linear infinite;
}}

.ticker-move:hover {{ animation-play-state: paused; }}

.ticker-item {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0 28px;
    font-size: 13px;
    border-right: 1px solid {THEME['border']};
}}

.ticker-symbol {{ font-weight: 600; color: {THEME['text_primary']}; }}
.ticker-price {{ color: {THEME['text_secondary']}; font-size: 12px; }}
.ticker-change {{ font-weight: 600; font-size: 12px; }}

.header-container {{ text-align: center; margin-bottom: 1.5rem; }}

.header-title {{
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, {THEME['accent_primary']} 0%, {THEME['accent_secondary']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}}

.header-subtitle {{ color: {THEME['text_muted']}; font-size: 1rem; font-weight: 400; }}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.status-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}}

.status-open {{ background: {THEME['success']}; box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }}
.status-closed {{ background: {THEME['danger']}; box-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }}

.card {{
    background: {THEME['bg_secondary']};
    border: 1px solid {THEME['border']};
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}}

.card:hover {{
    border-color: rgba(0, 212, 170, 0.25);
    box-shadow: 0 8px 32px rgba(0, 212, 170, 0.08);
    transform: translateY(-2px);
}}

.card-accent {{ border-left: 3px solid {THEME['accent_primary']}; }}

.card-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {THEME['text_muted']};
    margin-bottom: 8px;
}}

.card-value {{ font-size: 28px; font-weight: 700; color: {THEME['text_primary']}; line-height: 1.2; }}
.card-change {{ font-size: 13px; font-weight: 500; margin-top: 4px; }}
.card-change.positive {{ color: {THEME['success']}; }}
.card-change.negative {{ color: {THEME['danger']}; }}
.card-change.neutral {{ color: {THEME['text_muted']}; }}

.metric-card {{
    background: {THEME['bg_tertiary']};
    border: 1px solid {THEME['border']};
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}}

.metric-card:hover {{ border-color: {THEME['border_light']}; background: {THEME['bg_hover']}; }}
.metric-label {{ font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; color: {THEME['text_muted']}; margin-bottom: 6px; }}
.metric-value {{ font-size: 22px; font-weight: 700; }}
.metric-positive {{ color: {THEME['success']}; }}
.metric-negative {{ color: {THEME['danger']}; }}
.metric-neutral {{ color: {THEME['text_primary']}; }}

.table-header {{
    display: grid;
    grid-template-columns: 2fr 1.5fr 1fr 1fr 1fr;
    padding: 12px 16px;
    background: {THEME['bg_tertiary']};
    border-radius: 8px 8px 0 0;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: {THEME['text_muted']};
}}

.table-row {{
    display: grid;
    grid-template-columns: 2fr 1.5fr 1fr 1fr 1fr;
    padding: 14px 16px;
    border-bottom: 1px solid {THEME['border']};
    align-items: center;
    font-size: 14px;
}}

.table-row:hover {{ background: {THEME['bg_tertiary']}; }}

.signal-card {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    background: {THEME['bg_tertiary']};
    border: 1px solid {THEME['border']};
    border-radius: 10px;
    margin-bottom: 10px;
}}

.signal-card:hover {{ transform: translateX(4px); border-color: {THEME['border_light']}; }}
.signal-buy {{ border-left: 3px solid {THEME['success']}; }}
.signal-sell {{ border-left: 3px solid {THEME['danger']}; }}
.signal-ticker {{ font-weight: 700; font-size: 15px; color: {THEME['text_primary']}; }}
.signal-strategy {{ font-size: 11px; color: {THEME['text_muted']}; margin-top: 2px; }}
.signal-type {{ font-weight: 700; font-size: 12px; text-transform: uppercase; }}

.badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; }}
.badge-success {{ background: rgba(16, 185, 129, 0.15); color: {THEME['success']}; }}
.badge-danger {{ background: rgba(239, 68, 68, 0.15); color: {THEME['danger']}; }}

.section-title {{
    font-size: 13px;
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

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {THEME['bg_secondary']};
    border: 1px solid {THEME['border']};
    border-radius: 12px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {THEME['bg_tertiary']};
    padding: 4px;
    border-radius: 10px;
}}

.stTabs [data-baseweb="tab"] {{ border-radius: 8px; padding: 8px 20px; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ background: rgba(0, 212, 170, 0.15) !important; color: {THEME['accent_primary']} !important; }}

.stButton > button {{
    background: linear-gradient(135deg, {THEME['accent_primary']} 0%, {THEME['accent_secondary']} 100%);
    border: none;
    border-radius: 8px;
    font-weight: 600;
}}

.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 212, 170, 0.4);
}}

span[data-baseweb="tag"] {{ background: {THEME['bg_tertiary']} !important; border: 1px solid {THEME['border']} !important; }}
</style>
<div class="ticker-wrap"><div class="ticker-move">{ticker_items_html}{ticker_items_html}</div></div>
""", unsafe_allow_html=True)


# ============================================================
# HEADER & NAVIGATION
# ============================================================
def render_header():
    st.markdown("""
        <div class="header-container">
            <div class="header-title">Quant Terminal Pro</div>
            <div class="header-subtitle">Professional Quantitative Analysis Platform</div>
        </div>
    """, unsafe_allow_html=True)
    
    pages = ["Dashboard", "Single Asset", "Portfolio", "Virtual Trading"]
    current_idx = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
    
    selected = option_menu(
        menu_title=None,
        options=pages,
        icons=["grid-3x3-gap", "graph-up-arrow", "pie-chart", "lightning-charge"],
        default_index=current_idx,
        orientation="horizontal",
        styles={
            "container": {"padding": "8px", "background-color": THEME['bg_secondary'], "border-radius": "12px", "border": f"1px solid {THEME['border']}"},
            "icon": {"color": THEME['accent_primary'], "font-size": "16px"},
            "nav-link": {"font-size": "14px", "font-weight": "500", "text-align": "center", "border-radius": "8px", "margin": "0 4px", "color": THEME['text_secondary'], "padding": "10px 20px"},
            "nav-link-selected": {"background": f"linear-gradient(135deg, {THEME['accent_primary']} 0%, {THEME['accent_secondary']} 100%)", "color": "white", "font-weight": "600"},
        }
    )
    
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def calculate_portfolio_value():
    total = st.session_state.virtual_balance
    for ticker, position in st.session_state.portfolio.items():
        price_data = get_latest_price(ticker)
        if price_data:
            total += position['shares'] * price_data['price']
    return total


@st.cache_data(ttl=600)
def scan_market_signals():
    signals = []
    scan_list = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "GOOGL", "META", "BTC-USD", "ETH-USD"]
    for ticker in scan_list:
        try:
            df = fetch_data(ticker, period="1mo", interval="1d")
            if df.empty:
                continue
            res_rsi = apply_rsi_strategy(df.copy(), window=14)
            last_rsi = res_rsi['RSI'].iloc[-1]
            if last_rsi < 30:
                signals.append({"ticker": ticker, "type": "BUY", "strategy": "RSI Oversold", "value": f"RSI {last_rsi:.0f}"})
            elif last_rsi > 70:
                signals.append({"ticker": ticker, "type": "SELL", "strategy": "RSI Overbought", "value": f"RSI {last_rsi:.0f}"})
            ret_5d = df['Close'].pct_change(5).iloc[-1]
            if ret_5d > 0.10:
                signals.append({"ticker": ticker, "type": "BUY", "strategy": "Strong Momentum", "value": f"+{ret_5d*100:.1f}% (5d)"})
        except:
            continue
    return signals


# ============================================================
# DASHBOARD PAGE
# ============================================================
# ============================================================
# DASHBOARD PAGE
# ============================================================
def render_dashboard():
    paris_time = get_paris_time()
    market_open = is_market_open()
    
    # 3 Colonnes : Status | Heure | Refresh
    # J'ai gardé les proportions
    hud_left, hud_right, hud_refresh = st.columns([3, 1.5, 0.2])
    
    with hud_left:
        status_class = "status-open" if market_open else "status-closed"
        status_text = "MARKET OPEN" if market_open else "MARKET CLOSED"
        status_color = THEME['success'] if market_open else THEME['danger']
        
        next_open = ""
        if not market_open:
            hour = paris_time.hour
            weekday = paris_time.weekday()
            if weekday >= 5:
                next_open = " · Opens Monday 15:30 CET"
            elif hour >= 22:
                next_open = " · Opens Monday 15:30 CET" if weekday == 4 else " · Opens Tomorrow 15:30 CET"
            elif hour < 15 or (hour == 15 and paris_time.minute < 30):
                next_open = " · Opens Today 15:30 CET"
        
        # MODIFICATION ICI : padding-top passé à 15px pour descendre le texte
        st.markdown(f"""
        <div style="display:flex;align-items:center;padding-top:15px;">
            <span class="status-dot {status_class}"></span>
            <span style="font-weight:600;color:{status_color};text-transform:uppercase;font-size:12px;letter-spacing:1px;">{status_text}</span>
            <span style="color:{THEME['text_muted']};font-size:12px;margin-left:8px;">{next_open}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with hud_right:
        # MODIFICATION ICI : padding-top passé à 15px pour descendre l'heure aussi
        st.markdown(f"""
        <div style="text-align:right;padding-top:15px;">
            <span style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;color:{THEME['text_primary']}">{paris_time.strftime('%H:%M')}</span>
            <span style="color:{THEME['border_light']};margin:0 10px;">|</span>
            <span style="font-size:12px;color:{THEME['text_muted']};">{paris_time.strftime('%A, %d %B')}</span>
            <span style="font-size:12px;color:{THEME['accent_primary']};font-weight:600;margin-left:5px;">PARIS</span>
        </div>
        """, unsafe_allow_html=True)

    with hud_refresh:
        # Le bouton reste à sa place, c'est le texte à gauche qui est descendu pour s'aligner
        st.markdown('<div style="margin-top: 2px;"></div>', unsafe_allow_html=True)
        if st.button("↻", key="refresh_hud", help="Refresh Data", use_container_width=True):
            st.rerun()
    
    st.markdown("<div style='margin-bottom:25px;'></div>", unsafe_allow_html=True)
    
    # ... (Le reste de la fonction reste identique) ...
    # Market Overview
    st.markdown('<div class="section-title"> Market Overview</div>', unsafe_allow_html=True)
    
    indices = [("SPY", "S&P 500", "🇺🇸"), ("QQQ", "NASDAQ 100", "💻"), ("BTC-USD", "Bitcoin", "₿"), ("GC=F", "Gold", "🥇")]
    
    cols = st.columns(4)
    for col, (ticker, name, icon) in zip(cols, indices):
        data = get_latest_price(ticker)
        if data:
            change_class = "positive" if data['is_positive'] else "negative"
            arrow = "↑" if data['is_positive'] else "↓"
            with col:
                st.markdown(f"""<div class="card card-accent"><div class="card-label">{icon} {name}</div><div class="card-value">${data['price']:,.2f}</div><div class="card-change {change_class}">{arrow} {abs(data['change_pct']):.2f}%</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Content
    col_left, col_right = st.columns([2.2, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="section-title"> Market Movers</div>', unsafe_allow_html=True)
        
        with st.container(border=True):
            watchlist = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META", "AMZN", "GOOGL"]
            
            st.markdown(f"""<div class="table-header"><div>SYMBOL</div><div>PRICE</div><div>CHANGE</div><div>VOLUME</div><div>STATUS</div></div>""", unsafe_allow_html=True)
            
            for ticker in watchlist:
                data = get_latest_price(ticker)
                if data:
                    change_color = THEME['success'] if data['is_positive'] else THEME['danger']
                    badge_class = "badge-success" if data['is_positive'] else "badge-danger"
                    arrow = "▲" if data['is_positive'] else "▼"
                    vol_str = f"{data['volume']/1e6:.1f}M" if data['volume'] > 1e6 else f"{data['volume']/1e3:.0f}K"
                    
                    st.markdown(f"""<div class="table-row"><div style="font-weight:600;">{ticker}</div><div class="mono">${data['price']:,.2f}</div><div style="color:{change_color};font-weight:500;">{arrow} {abs(data['change_pct']):.2f}%</div><div class="mono" style="color:{THEME['text_muted']}">{vol_str}</div><div><span class="badge {badge_class}">{'BULLISH' if data['is_positive'] else 'BEARISH'}</span></div></div>""", unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="section-title"> Portfolio Summary</div>', unsafe_allow_html=True)
        
        total_val = calculate_portfolio_value()
        pnl = total_val - st.session_state.initial_balance
        pnl_pct = (pnl / st.session_state.initial_balance) * 100
        pnl_class = "positive" if pnl >= 0 else "negative"
        
        st.markdown(f"""<div class="card card-accent" style="margin-bottom:16px;"><div class="card-label">Total Equity</div><div class="card-value">${total_val:,.2f}</div><div class="card-change {pnl_class}">{'+' if pnl >= 0 else ''}{pnl:,.2f} ({pnl_pct:+.2f}%)</div></div>""", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Cash</div><div class="metric-value metric-neutral">${st.session_state.virtual_balance:,.0f}</div></div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Positions</div><div class="metric-value metric-neutral">{len(st.session_state.portfolio)}</div></div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Start Trading", type="primary", use_container_width=True):
            st.session_state.current_page = "Virtual Trading"
            st.rerun()

# ============================================================
# VIRTUAL TRADING PAGE
# ============================================================
def render_virtual_trading():
    total_val = calculate_portfolio_value()
    pnl = total_val - st.session_state.initial_balance
    pnl_pct = (pnl / st.session_state.initial_balance) * 100
    invested = total_val - st.session_state.virtual_balance
    
    st.markdown('<div class="section-title"> Paper Trading Dashboard</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    
    metrics_data = [
        ("Total Portfolio", f"${total_val:,.2f}", f"{pnl:+,.2f} ({pnl_pct:+.2f}%)", "positive" if pnl >= 0 else "negative"),
        ("Cash Balance", f"${st.session_state.virtual_balance:,.2f}", "Available to trade", "neutral"),
        ("Total Invested", f"${invested:,.2f}", f"In {len(st.session_state.portfolio)} positions", "neutral"),
        ("Total Trades", str(len(st.session_state.trade_history)), "Executed orders", "neutral"),
    ]
    
    for col, (label, value, subtext, status) in zip([m1, m2, m3, m4], metrics_data):
        with col:
            st.markdown(f"""<div class="card card-accent"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-change {status}">{subtext}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_main, col_side = st.columns([2, 1], gap="large")
    
    with col_main:
        st.markdown('<div class="section-title"> Active Positions</div>', unsafe_allow_html=True)
        
        if not st.session_state.portfolio:
            st.info("No active positions. Use the trading panel to place your first order.")
        else:
            with st.container(border=True):
                st.markdown(f"""<div class="table-header" style="grid-template-columns:1.5fr 1fr 1fr 1fr 1fr 1fr;"><div>SYMBOL</div><div>QTY</div><div>AVG PRICE</div><div>CURRENT</div><div>P&L</div><div>ACTION</div></div>""", unsafe_allow_html=True)
                
                for ticker, pos in list(st.session_state.portfolio.items()):
                    data = get_latest_price(ticker)
                    if data:
                        current_val = pos['shares'] * data['price']
                        cost_val = pos['shares'] * pos['avg_price']
                        pos_pnl = current_val - cost_val
                        badge_class = "badge-success" if pos_pnl >= 0 else "badge-danger"
                        
                        c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1, 1, 1])
                        c1.markdown(f"**{ticker}**")
                        c2.markdown(f"{pos['shares']}")
                        c3.markdown(f"${pos['avg_price']:.2f}")
                        c4.markdown(f"${data['price']:.2f}")
                        c5.markdown(f"<span class='badge {badge_class}'>{pos_pnl:+.2f}</span>", unsafe_allow_html=True)
                        with c6:
                            if st.button("Sell", key=f"sell_{ticker}", type="secondary"):
                                val = pos['shares'] * data['price']
                                st.session_state.virtual_balance += val
                                del st.session_state.portfolio[ticker]
                                st.session_state.trade_history.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "ticker": ticker, "type": "SELL", "qty": pos['shares'], "price": data['price']})
                                st.toast(f"Sold {pos['shares']} {ticker} @ ${data['price']:.2f}", icon="✅")
                                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title"> Trade History</div>', unsafe_allow_html=True)
        
        if st.session_state.trade_history:
            hist_df = pd.DataFrame(st.session_state.trade_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No trades executed yet.")
    
    with col_side:
        with st.container(border=True):
            st.markdown("#### Quick Trade")
            ticker = st.selectbox("Symbol", options=POPULAR_TICKERS, index=0)
            data = get_latest_price(ticker)
            
            if data:
                price_color = THEME['success'] if data['is_positive'] else THEME['danger']
                st.markdown(f"""<div style="margin-bottom:12px;"><div style="font-size:11px;color:{THEME['text_muted']};text-transform:uppercase;">Current Price</div><div style="font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace;">${data['price']:,.2f}</div><div style="font-size:12px;color:{price_color}">{data['change_pct']:+.2f}%</div></div>""", unsafe_allow_html=True)
                
                action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
                qty = st.number_input("Shares", min_value=1, value=10)
                total_cost = qty * data['price']
                
                st.markdown(f"""<div style="text-align:right;padding:8px 0;border-top:1px solid {THEME['border']};margin-top:8px;"><span style="color:{THEME['text_muted']};font-size:12px;">Est. Total:</span><span style="font-weight:700;font-size:16px;margin-left:8px;">${total_cost:,.2f}</span></div>""", unsafe_allow_html=True)
                
                if st.button("Execute Order", type="primary", use_container_width=True):
                    if action == "BUY":
                        if total_cost <= st.session_state.virtual_balance:
                            st.session_state.virtual_balance -= total_cost
                            pos = st.session_state.portfolio.get(ticker, {'shares': 0, 'avg_price': 0})
                            new_shares = pos['shares'] + qty
                            new_avg = ((pos['shares'] * pos['avg_price']) + total_cost) / new_shares
                            st.session_state.portfolio[ticker] = {'shares': new_shares, 'avg_price': new_avg}
                            st.session_state.trade_history.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "ticker": ticker, "type": "BUY", "qty": qty, "price": data['price']})
                            st.toast(f"Bought {qty} {ticker} @ ${data['price']:.2f}", icon="✅")
                            st.rerun()
                        else:
                            st.toast("Insufficient funds", icon="❌")
                    else:
                        owned = st.session_state.portfolio.get(ticker, {}).get('shares', 0)
                        if qty <= owned:
                            st.session_state.virtual_balance += total_cost
                            st.session_state.portfolio[ticker]['shares'] -= qty
                            if st.session_state.portfolio[ticker]['shares'] == 0:
                                del st.session_state.portfolio[ticker]
                            st.session_state.trade_history.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "ticker": ticker, "type": "SELL", "qty": qty, "price": data['price']})
                            st.toast(f"Sold {qty} {ticker} @ ${data['price']:.2f}", icon="✅")
                            st.rerun()
                        else:
                            st.toast("Insufficient shares", icon="❌")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Live Signals")
        
        signals = scan_market_signals()
        if not signals:
            st.info("No strong signals detected.")
        else:
            for sig in signals[:5]:
                is_buy = sig['type'] == 'BUY'
                signal_class = "signal-buy" if is_buy else "signal-sell"
                signal_color = THEME['success'] if is_buy else THEME['danger']
                st.markdown(f"""<div class="signal-card {signal_class}"><div><div class="signal-ticker">{sig['ticker']}</div><div class="signal-strategy">{sig['strategy']}</div></div><div style="text-align:right;"><div class="signal-type" style="color:{signal_color}">{sig['type']}</div><div style="font-size:11px;color:{THEME['text_muted']}">{sig['value']}</div></div></div>""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
def render_footer():
    paris_time = get_paris_time()
    st.markdown(f"""<div style="text-align:center;padding:40px 0 20px 0;border-top:1px solid {THEME['border']};margin-top:40px;"><div style="color:{THEME['text_muted']};font-size:12px;">Quant Terminal Pro © 2026 · Paris: {paris_time.strftime('%H:%M')} · <span style="color:{THEME['success']}">● Online</span></div></div>""", unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    init_session_state()
    render_global_styles()
    st_autorefresh(interval=300000, key="auto_refresh")
    render_header()
    
    page = st.session_state.current_page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Single Asset":
        render_quant_a(show_header=True)
    elif page == "Portfolio":
        render_quant_b(show_header=True)
    elif page == "Virtual Trading":
        render_virtual_trading()
    
    render_footer()


if __name__ == "__main__":
    main()