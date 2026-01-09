"""
Main Streamlit Application
Quantitative Finance Dashboard - Single Asset & Portfolio Analysis
Auto-refreshes every 5 minutes as required.
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import datetime
import pytz

# page config must be first streamlit command
st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# import our modules (after page config)
from src.quant_b import render_quant_b
from src.data_loader import clear_cache, get_latest_price
from src.utils import COLORS, get_paris_time, is_market_open


# ============================================================
# SESSION STATE INITIALIZATION (FIX FOR NAVIGATION BUG)
# ============================================================

def init_session_state():
    """Initialize session state for persistent navigation."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'menu_key' not in st.session_state:
        st.session_state.menu_key = 0


def get_page_index(page_name: str) -> int:
    """Get the index of a page in the menu."""
    pages = ["Home", "Single Asset", "Portfolio", "Settings"]
    return pages.index(page_name) if page_name in pages else 0


# ============================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================

def inject_custom_css():
    """Inject custom CSS for better styling."""
    st.markdown("""
    <style>
    /* main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* cards/containers */
    [data-testid="stExpander"] {
        background-color: rgba(30, 30, 46, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
    }
    
    /* metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] > div {
        font-size: 0.9rem;
    }
    
    /* tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 30, 46, 0.4);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.3);
    }
    
    /* buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* selectbox */
    [data-testid="stSelectbox"] > div > div {
        background-color: rgba(30, 30, 46, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
    }
    
    /* text input */
    [data-testid="stTextInput"] > div > div > input {
        background-color: rgba(30, 30, 46, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
    }
    
    /* slider */
    [data-testid="stSlider"] > div > div > div {
        background-color: #6366f1;
    }
    
    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* custom header */
    .main-header {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 15px;
        padding: 20px 30px;
        margin-bottom: 20px;
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-open {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-closed {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* info boxes */
    .info-box {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# AUTO-REFRESH (5 MINUTES = 300000 MS)
# ============================================================

def setup_auto_refresh():
    """Setup auto-refresh every 5 minutes."""
    count = st_autorefresh(interval=300000, limit=None, key="auto_refresh")
    return count


# ============================================================
# HEADER COMPONENT
# ============================================================

def render_header():
    """Render the main header with status info."""
    paris_time = get_paris_time()
    market_status = is_market_open()
    
    st.markdown("""
    <div class="main-header">
        <h1>📈 Quantitative Finance Dashboard</h1>
        <p>Real-time market analysis, backtesting strategies, and portfolio optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        status_class = "status-open" if market_status else "status-closed"
        status_text = "● Market Open" if market_status else "● Market Closed"
        st.markdown(f'<span class="status-indicator {status_class}">{status_text}</span>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"🕐 **Paris:** {paris_time.strftime('%H:%M:%S')}")
    
    with col3:
        st.markdown(f"📅 **Date:** {paris_time.strftime('%Y-%m-%d')}")
    
    with col4:
        st.markdown("🔄 **Auto-refresh:** 5 min")


# ============================================================
# SIDEBAR WITH FIXED NAVIGATION
# ============================================================

def render_sidebar():
    """Render sidebar with navigation and info."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="background: linear-gradient(90deg, #6366f1, #8b5cf6); 
                       -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent;
                       font-size: 1.5rem;">
                🎯 Quant Platform
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # get current page index for default_index
        current_index = get_page_index(st.session_state.current_page)
        
        # navigation menu with persistent selection
        selected = option_menu(
            menu_title=None,
            options=["Home", "Single Asset", "Portfolio", "Settings"],
            icons=["house", "graph-up", "pie-chart", "gear"],
            menu_icon="cast",
            default_index=current_index,
            key=f"nav_menu_{st.session_state.menu_key}",
            styles={
                "container": {"padding": "5px", "background-color": "transparent"},
                "icon": {"color": "#8b5cf6", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "5px 0",
                    "padding": "10px 15px",
                    "border-radius": "8px",
                    "--hover-color": "rgba(99, 102, 241, 0.2)",
                },
                "nav-link-selected": {
                    "background-color": "rgba(99, 102, 241, 0.3)",
                    "font-weight": "600",
                },
            }
        )
        
        # update session state when menu selection changes
        if selected != st.session_state.current_page:
            st.session_state.current_page = selected
        
        st.markdown("---")
        
        # quick ticker lookup
        st.markdown("### 🔍 Quick Lookup")
        quick_ticker = st.text_input("Ticker", value="AAPL", key="quick_lookup")
        
        if quick_ticker:
            price_data = get_latest_price(quick_ticker)
            if price_data:
                delta_color = "normal" if price_data['is_positive'] else "inverse"
                st.metric(
                    label=quick_ticker.upper(),
                    value=f"${price_data['price']:.2f}",
                    delta=f"{price_data['change_pct']:.2f}%",
                    delta_color=delta_color
                )
            else:
                st.warning("Ticker not found")
        
        st.markdown("---")
        
        # about section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Quant Dashboard v1.0**
            
            A professional quantitative finance platform for:
            - Single asset analysis & backtesting
            - Multi-asset portfolio optimization
            - ML-based price predictions
            
            Built with Streamlit, yfinance, and love.
            
            ---
            
            **Team:** Quant Research
            
            **Data:** Yahoo Finance API
            
            **Refresh:** Every 5 minutes
            """)
        
        # cache control
        if st.button("🔄 Force Refresh Data"):
            clear_cache()
            st.success("Cache cleared!")
            st.rerun()
        
        return selected


# ============================================================
# HOME PAGE
# ============================================================

def render_home():
    """Render the home page with overview."""
    st.markdown("## 🏠 Welcome to Quant Dashboard")
    
    st.markdown("""
    <div class="info-box">
        <strong>📊 What is this?</strong><br>
        A professional-grade quantitative finance platform for analyzing individual assets 
        and building optimized portfolios. Features include backtesting, ML predictions, 
        and real-time data.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Single Asset Analysis (Quant A)
        
        Analyze individual assets with:
        - **8 Trading Strategies**: SMA, EMA, RSI, Bollinger, MACD, Momentum, Mean Reversion, Buy & Hold
        - **Backtesting**: Test strategies on historical data
        - **Performance Metrics**: Sharpe, Sortino, Max Drawdown, VaR
        - **ML Predictions**: Prophet, Linear Regression, Momentum forecasts
        - **Interactive Charts**: Candlesticks, signals, indicators, drawdown
        
        Perfect for deep-diving into a single stock, crypto, or forex pair.
        """)
        
        if st.button("Go to Single Asset →", key="goto_quant_a"):
            st.session_state.current_page = "Single Asset"
            st.session_state.menu_key += 1
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 📊 Portfolio Analysis (Quant B)
        
        Build and analyze portfolios with:
        - **7 Allocation Methods**: Equal Weight, Min Variance, Risk Parity, Max Sharpe, HRP
        - **Rebalancing Options**: None, Daily, Weekly, Monthly
        - **Risk Analysis**: Correlation matrix, diversification ratio
        - **Efficient Frontier**: Visualize risk-return tradeoff
        - **Risk Contribution**: See how each asset impacts portfolio risk
        
        Perfect for building diversified portfolios with multiple assets.
        """)
        
        if st.button("Go to Portfolio →", key="goto_quant_b"):
            st.session_state.current_page = "Portfolio"
            st.session_state.menu_key += 1
            st.rerun()
    
    st.markdown("---")
    
    # market overview
    st.markdown("### 🌍 Market Overview")
    
    market_tickers = ["^GSPC", "^DJI", "^IXIC", "BTC-USD", "EURUSD=X", "GC=F"]
    market_names = ["S&P 500", "Dow Jones", "NASDAQ", "Bitcoin", "EUR/USD", "Gold"]
    
    cols = st.columns(6)
    
    for i, (ticker, name) in enumerate(zip(market_tickers, market_names)):
        with cols[i]:
            price_data = get_latest_price(ticker)
            if price_data:
                st.metric(
                    label=name,
                    value=f"${price_data['price']:,.2f}" if price_data['price'] > 100 else f"${price_data['price']:.4f}",
                    delta=f"{price_data['change_pct']:.2f}%",
                    delta_color="normal" if price_data['is_positive'] else "inverse"
                )
            else:
                st.metric(label=name, value="N/A", delta="--")
    
    st.markdown("---")
    
    # features grid
    st.markdown("### ✨ Platform Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **🔄 Real-Time Data**
        
        Auto-refreshes every 5 minutes with live market data from Yahoo Finance.
        """)
    
    with col2:
        st.markdown("""
        **📊 8 Strategies**
        
        From simple moving averages to advanced momentum and mean-reversion strategies.
        """)
    
    with col3:
        st.markdown("""
        **🤖 ML Predictions**
        
        Forecast prices using Prophet, Linear Regression, and custom models.
        """)
    
    with col4:
        st.markdown("""
        **📈 Portfolio Optimization**
        
        Risk parity, min variance, max Sharpe, and more allocation methods.
        """)


# ============================================================
# SETTINGS PAGE
# ============================================================

def render_settings():
    """Render settings page."""
    st.markdown("## ⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Display Settings")
        
        st.checkbox("Dark mode", value=True, disabled=True, help="Dark mode is always on")
        st.checkbox("Show signals on charts", value=True, key="show_signals")
        st.checkbox("Show confidence intervals", value=True, key="show_ci")
        
        st.markdown("### 📊 Default Parameters")
        
        st.selectbox(
            "Default Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=3,
            key="default_period"
        )
        
        st.selectbox(
            "Default Interval",
            ["1m", "5m", "15m", "1h", "1d"],
            index=4,
            key="default_interval"
        )
    
    with col2:
        st.markdown("### 🔧 Cache & Data")
        
        st.info("Data is cached for 5 minutes to reduce API calls and improve performance.")
        
        if st.button("🗑️ Clear All Cache", type="primary"):
            clear_cache()
            st.success("✅ All cached data cleared!")
        
        st.markdown("### 📈 Risk Parameters")
        
        st.number_input(
            "Risk-free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1,
            key="risk_free_rate",
            help="Annual risk-free rate for Sharpe ratio calculation"
        )
        
        st.number_input(
            "VaR Confidence Level (%)",
            min_value=90.0,
            max_value=99.9,
            value=95.0,
            step=0.5,
            key="var_confidence",
            help="Confidence level for Value at Risk"
        )
    
    st.markdown("---")
    
    st.markdown("### 📝 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Version:** 1.0.0
        
        **Python:** 3.10+
        
        **Framework:** Streamlit
        """)
    
    with col2:
        st.markdown("""
        **Data Source:** Yahoo Finance
        
        **Refresh Rate:** 5 minutes
        
        **Cache TTL:** 300 seconds
        """)
    
    with col3:
        st.markdown("""
        **Timezone:** Europe/Paris
        
        **Market Hours:** 9:30-16:00 ET
        
        **Trading Days:** 252/year
        """)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application entry point."""
    # initialize session state first
    init_session_state()
    
    # inject custom CSS
    inject_custom_css()
    
    # setup auto-refresh
    refresh_count = setup_auto_refresh()
    
    # render sidebar and get navigation selection
    selected = render_sidebar()
    
    # render header
    render_header()
    
    st.markdown("---")
    
    # render selected page based on session state
    current_page = st.session_state.current_page
    
    if current_page == "Home":
        render_home()
    elif current_page == "Single Asset":
        st.info("📈 Quant A est sur l’autre branche : `quant-a-module`.")
    elif current_page == "Portfolio":
        render_quant_b(show_header=True)
    elif current_page == "Settings":
        render_settings()
    
    # footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 20px;">
            <p>Quant Dashboard © 2024 | Data provided by Yahoo Finance | Auto-refreshes every 5 minutes</p>
            <p>Built with ❤️ using Streamlit, Plotly, and Python</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
