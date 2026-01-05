import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajoute le dossier racine au chemin pour pouvoir importer src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quant_a import QuantAAnalyzer

def test_quant_a_logic():
    # 1. Créer une fausse donnée (Mock data)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    # Prix qui monte de 100 à 199
    prices = np.arange(100, 200) 
    df = pd.DataFrame({"Close": prices}, index=dates)

    # 2. Lancer la stratégie
    analyzer = QuantAAnalyzer(df)
    res = analyzer.apply_strategy(short_window=10, long_window=20)

    # 3. Vérifications (Assertions)
    assert "SMA_Short" in res.columns, "La colonne SMA_Short doit exister"
    assert "Signal" in res.columns, "La colonne Signal doit exister"
    
    # Vérifie que le bonus Equity Curve est bien là (si tu as fait la modif précédente)
    if "Equity_Curve" in res.columns:
        assert res["Equity_Curve"].iloc[-1] > 100, "La stratégie devrait gagner de l'argent sur un prix qui monte"
    
    # Vérifier qu'on ne crash pas sur les métriques
    metrics = analyzer.get_metrics()
    assert isinstance(metrics["Sharpe Ratio"], float)

def test_empty_dataframe():
    # Vérifier que le code résiste à un tableau vide
    df_empty = pd.DataFrame()
    analyzer = QuantAAnalyzer(df_empty)
    res = analyzer.apply_strategy()
    assert res.empty, "Doit renvoyer un DF vide si entrée vide"