import streamlit as st
import pandas as pd
from src.data_loader import fetch_data
from src.quant_a import QuantAAnalyzer

st.set_page_config(page_title="Projet Finance - Quant A", layout="wide")

st.title("Dashboard Analyse Quantitative (Module A)")

st.sidebar.header("Parametres")

ticker = st.sidebar.text_input("Symbole", value="AAPL")
short_window = st.sidebar.slider("Moyenne Mobile Courte", 5, 50, 20)
long_window = st.sidebar.slider("Moyenne Mobile Longue", 50, 200, 50)

if st.sidebar.button("Rafraichir"):
    st.cache_data.clear()
    st.rerun()

st.write(f"Analyse Technique : {ticker}")

try:
    # On prend 5 jours au lieu de 1 mois pour eviter l'ecran noir
    df = fetch_data(ticker, period="5d", interval="5m")

    if not df.empty:
        analyzer = QuantAAnalyzer(df)
        df_analyzed = analyzer.apply_strategy(short_window, long_window)
        metrics = analyzer.get_metrics()

        col1, col2, col3, col4 = st.columns(4)
        
        last_price = df_analyzed['Close'].iloc[-1]
        signal = "ACHAT" if df_analyzed['Signal'].iloc[-1] == 1 else "NEUTRE"

        col1.metric("Prix", f"{last_price:.2f} $")
        col2.metric("Signal", signal)
        col3.metric("Drawdown", f"{metrics['Max Drawdown']*100:.2f} %")
        col4.metric("Sharpe", f"{metrics['Sharpe Ratio']:.2f}")

        st.line_chart(df_analyzed[['Close', 'SMA_Short', 'SMA_Long']])

        with st.expander("Voir les donnees"):
            st.dataframe(df_analyzed.tail())

    else:
        st.write("Aucune donnee disponible")

except Exception as e:
    st.write(f"Erreur : {e}")