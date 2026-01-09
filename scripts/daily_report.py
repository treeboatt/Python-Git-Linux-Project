#!/usr/bin/env python3
"""
Daily Report Generator - Professional Edition
Generates a comprehensive, styled PDF report at 8pm via cron job.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pytz
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from src.data_loader import get_daily_summary, get_period_stats
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# Liste des actifs à surveiller
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "SPY", "QQQ", "DIA",
    "BTC-USD", "ETH-USD",
    "EURUSD=X", "GC=F", "CL=F",
]

REPORTS_DIR = PROJECT_ROOT / "reports"
PARIS_TZ = pytz.timezone("Europe/Paris")

# --- CLASSE PDF PERSONNALISÉE ---
if HAS_FPDF:
    class ProPDF(FPDF):
        def header(self):
            # Fond sombre pour l'en-tête (Style "Dark Mode")
            self.set_fill_color(16, 24, 32) # Gris très foncé/Bleuté
            self.rect(0, 0, 210, 40, 'F')
            
            # Titre Principal
            self.set_font("Helvetica", "B", 24)
            self.set_text_color(255, 255, 255) # Blanc
            self.set_y(10)
            self.cell(0, 10, "QUANT TERMINAL PRO", ln=True, align="C")
            
            # Sous-titre
            self.set_font("Helvetica", "", 10)
            self.set_text_color(0, 212, 170) # Teal (Couleur accent du site)
            self.cell(0, 10, "DAILY MARKET REPORT & ANALYTICS", ln=True, align="C")
            self.ln(20)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Generated automatically via Cron | Page {self.page_no()}', 0, 0, 'C')

def collect_daily_data(tickers: list) -> dict:
    """Collect daily data for all tickers."""
    logger.info(f"Collecting data for {len(tickers)} tickers...")
    results = {
        "timestamp": datetime.now(PARIS_TZ).isoformat(),
        "date": datetime.now(PARIS_TZ).strftime("%Y-%m-%d"),
        "assets": {},
        "errors": [],
    }
    for ticker in tickers:
        try:
            summary = get_daily_summary(ticker)
            if summary:
                stats = get_period_stats(ticker, "1mo")
                results["assets"][ticker] = {"daily": summary, "monthly_stats": stats}
            else:
                results["errors"].append(f"No data: {ticker}")
        except Exception as e:
            results["errors"].append(f"{ticker}: {e}")
    return results

def calculate_metrics(data: dict) -> dict:
    """Calculate aggregate metrics."""
    assets = data.get("assets", {})
    if not assets:
        return {}
    returns = []
    gainers, losers = [], []
    for ticker, info in assets.items():
        daily = info.get("daily", {})
        if daily:
            change = daily.get("change_pct", 0)
            returns.append(change)
            if change > 0:
                gainers.append((ticker, change))
            elif change < 0:
                losers.append((ticker, change))
    gainers.sort(key=lambda x: x[1], reverse=True)
    losers.sort(key=lambda x: x[1])
    return {
        "total_assets": len(assets),
        "avg_return": float(np.mean(returns)) if len(returns) > 0 else 0,
        "positive_count": len(gainers),
        "negative_count": len(losers),
        "top_gainers": gainers[:5],
        "top_losers": losers[:5],
    }

def generate_text_report(data: dict, metrics: dict) -> str:
    """Basic text fallback."""
    return f"Daily Report {data['date']}\nAvg Return: {metrics.get('avg_return', 0):.2f}%"

def generate_json_report(data: dict, metrics: dict) -> str:
    return json.dumps({"data": data, "metrics": metrics}, indent=2, default=str)

def generate_pdf_report(data: dict, metrics: dict, path: Path) -> bool:
    """Generate the Professional Styled PDF."""
    if not HAS_FPDF:
        return False
    
    try:
        pdf = ProPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # --- SECTION 1: MARKET OVERVIEW ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(16, 24, 32)
        pdf.cell(0, 10, f"MARKET OVERVIEW  |  {data['date']}", ln=True)
        
        # Ligne de séparation
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Métriques en gros
        pdf.set_font("Helvetica", "", 12)
        
        # Avg Return avec couleur
        avg_ret = metrics.get('avg_return', 0)
        pdf.cell(40, 10, "Average Return: ")
        
        if avg_ret >= 0:
            pdf.set_text_color(16, 185, 129) # Vert Success
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(30, 10, f"+{avg_ret:.2f}%")
        else:
            pdf.set_text_color(239, 68, 68) # Rouge Danger
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(30, 10, f"{avg_ret:.2f}%")
            
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(10, 10, " | ")
        pdf.cell(30, 10, f"Total Assets: {metrics.get('total_assets', 0)}")
        pdf.cell(10, 10, " | ")
        pdf.cell(50, 10, f"Gainers: {metrics.get('positive_count', 0)} / Losers: {metrics.get('negative_count', 0)}", ln=True)
        pdf.ln(10)

        # --- SECTION 2: TOP MOVERS ---
        y_top = pdf.get_y()
        
        # Colonne Gainers
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(236, 253, 245) # Fond Vert très clair
        pdf.cell(90, 8, " TOP GAINERS", 1, 0, 'L', True)
        
        # Espace
        pdf.cell(10, 8, "")
        
        # Colonne Losers
        pdf.set_fill_color(254, 242, 242) # Fond Rouge très clair
        pdf.cell(90, 8, " TOP LOSERS", 1, 1, 'L', True)
        
        pdf.set_font("Helvetica", "", 10)
        gainers = metrics.get("top_gainers", [])
        losers = metrics.get("top_losers", [])
        
        for i in range(5):
            # Gainers
            if i < len(gainers):
                ticker, change = gainers[i]
                pdf.set_text_color(0, 0, 0)
                pdf.cell(60, 8, f"  {ticker}", 'L')
                pdf.set_text_color(16, 185, 129) # Vert
                pdf.cell(30, 8, f"+{change:.2f}%", 'R')
            else:
                pdf.cell(90, 8, "")
            
            # Espace milieu
            pdf.cell(10, 8, "")
            
            # Losers
            if i < len(losers):
                ticker, change = losers[i]
                pdf.set_text_color(0, 0, 0)
                pdf.cell(60, 8, f"  {ticker}", 'L')
                pdf.set_text_color(239, 68, 68) # Rouge
                pdf.cell(30, 8, f"{change:.2f}%", 'R', ln=True)
            else:
                pdf.cell(90, 8, "", ln=True)

        pdf.ln(15)

        # --- SECTION 3: DETAILED TABLE ---
        pdf.set_text_color(16, 24, 32)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "DETAILED ASSET PERFORMANCE", ln=True)
        pdf.set_draw_color(16, 24, 32)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # En-têtes Tableau
        pdf.set_fill_color(241, 245, 249) # Gris clair style admin
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(40, 10, "Ticker", 0, 0, 'L', True)
        pdf.cell(40, 10, "Close Price", 0, 0, 'R', True)
        pdf.cell(40, 10, "Daily Change", 0, 0, 'R', True)
        pdf.cell(70, 10, "Volume", 0, 1, 'R', True)
        
        # Lignes
        pdf.set_font("Helvetica", "", 10)
        fill = False
        
        # Trier par performance
        sorted_assets = sorted(data.get("assets", {}).items(), key=lambda x: x[1]['daily']['change_pct'], reverse=True)
        
        for ticker, info in sorted_assets:
            daily = info.get("daily", {})
            if daily:
                pdf.set_text_color(0, 0, 0)
                pdf.cell(40, 8, ticker)
                pdf.cell(40, 8, f"${daily.get('close', 0):,.2f}", 0, 0, 'R')
                
                # Couleur change
                change = daily.get('change_pct', 0)
                if change >= 0:
                    pdf.set_text_color(16, 185, 129)
                    txt_change = f"+{change:.2f}%"
                else:
                    pdf.set_text_color(239, 68, 68)
                    txt_change = f"{change:.2f}%"
                pdf.cell(40, 8, txt_change, 0, 0, 'R')
                
                pdf.set_text_color(100, 116, 139) # Volume en gris (muted)
                pdf.cell(70, 8, f"{daily.get('volume', 0):,}", 0, 1, 'R')
                
                # Ligne grise fine entre chaque
                pdf.set_draw_color(240, 240, 240)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())

        pdf.output(str(path))
        return True
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return False

def save_reports(data: dict, metrics: dict) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = data["date"]
    saved = {}
    
    text_path = REPORTS_DIR / f"report_{date_str}.txt"
    text_path.write_text(generate_text_report(data, metrics))
    saved["text"] = str(text_path)
    
    json_path = REPORTS_DIR / f"report_{date_str}.json"
    json_path.write_text(generate_json_report(data, metrics))
    saved["json"] = str(json_path)
    
    pdf_path = REPORTS_DIR / f"report_{date_str}.pdf"
    if generate_pdf_report(data, metrics, pdf_path):
        saved["pdf"] = str(pdf_path)
    
    (REPORTS_DIR / "latest_report.txt").write_text(generate_text_report(data, metrics))
    return saved

def main(watchlist: list = None):
    logger.info("=" * 50)
    logger.info("Starting Pro Report Generation")
    logger.info("=" * 50)
    if watchlist is None: watchlist = DEFAULT_WATCHLIST
    try:
        data = collect_daily_data(watchlist)
        metrics = calculate_metrics(data)
        saved = save_reports(data, metrics)
        logger.info("Report Generation Complete")
        for fmt, path in saved.items(): logger.info(f"  {fmt}: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        custom = sys.argv[1].split(",")
        success = main(watchlist=custom)
    else:
        success = main()
    sys.exit(0 if success else 1)