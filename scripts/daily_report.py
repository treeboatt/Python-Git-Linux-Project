#!/usr/bin/env python3
"""
Daily Report Generator
Generates a comprehensive daily report at 8pm via cron job.
Saves reports locally on the VM as required by the project.
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
    from src.data_loader import fetch_data, get_daily_summary, get_period_stats
    from src.utils import compute_volatility, compute_max_drawdown, safe_float, format_percentage
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "SPY", "QQQ", "DIA",
    "BTC-USD", "ETH-USD",
    "EURUSD=X", "GC=F", "CL=F",
]

REPORTS_DIR = PROJECT_ROOT / "reports"
PARIS_TZ = pytz.timezone("Europe/Paris")


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
                logger.info(f"  OK {ticker}: ${summary['close']:.2f} ({summary['change_pct']:+.2f}%)")
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
    
    returns = np.array(returns)
    
    return {
        "total_assets": len(assets),
        "avg_return": float(np.mean(returns)) if len(returns) > 0 else 0,
        "positive_count": len(gainers),
        "negative_count": len(losers),
        "top_gainers": gainers[:5],
        "top_losers": losers[:5],
    }


def generate_text_report(data: dict, metrics: dict) -> str:
    """Generate plain text report."""
    lines = [
        "=" * 60,
        "DAILY MARKET REPORT - QUANT TERMINAL",
        f"Generated: {data['timestamp']}",
        "=" * 60, "",
        "SUMMARY",
        "-" * 40,
        f"Date: {data['date']}",
        f"Assets: {metrics.get('total_assets', 0)}",
        f"Avg Return: {metrics.get('avg_return', 0):+.2f}%",
        f"Gainers: {metrics.get('positive_count', 0)} | Losers: {metrics.get('negative_count', 0)}",
        "",
        "TOP GAINERS",
        "-" * 40,
    ]
    
    for ticker, change in metrics.get("top_gainers", []):
        lines.append(f"  {ticker:12} {change:+.2f}%")
    
    lines.extend(["", "TOP LOSERS", "-" * 40])
    
    for ticker, change in metrics.get("top_losers", []):
        lines.append(f"  {ticker:12} {change:+.2f}%")
    
    lines.extend(["", "DETAILED DATA", "-" * 40])
    lines.append(f"{'Ticker':<12} {'Close':>10} {'Change':>10} {'Volume':>12}")
    
    for ticker, info in data.get("assets", {}).items():
        daily = info.get("daily", {})
        if daily:
            lines.append(
                f"{ticker:<12} ${daily.get('close', 0):>9.2f} "
                f"{daily.get('change_pct', 0):>+9.2f}% "
                f"{daily.get('volume', 0):>12,}"
            )
    
    lines.extend(["", "=" * 60, "End of Report", "=" * 60])
    return "\n".join(lines)


def generate_json_report(data: dict, metrics: dict) -> str:
    """Generate JSON report."""
    return json.dumps({
        "metadata": {"generated": data["timestamp"], "date": data["date"]},
        "summary": metrics,
        "assets": data["assets"],
        "errors": data["errors"],
    }, indent=2, default=str)


def generate_pdf_report(data: dict, metrics: dict, path: Path) -> bool:
    """Generate PDF report."""
    if not HAS_FPDF:
        return False
    
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 15, "Daily Market Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Date: {data['date']}", ln=True, align="C")
        pdf.ln(10)
        
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Assets: {metrics.get('total_assets', 0)}", ln=True)
        pdf.cell(0, 8, f"Avg Return: {metrics.get('avg_return', 0):+.2f}%", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top Gainers:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for ticker, change in metrics.get("top_gainers", [])[:3]:
            pdf.cell(0, 6, f"  {ticker}: {change:+.2f}%", ln=True)
        
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top Losers:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for ticker, change in metrics.get("top_losers", [])[:3]:
            pdf.cell(0, 6, f"  {ticker}: {change:+.2f}%", ln=True)
        
        pdf.output(str(path))
        return True
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return False


def save_reports(data: dict, metrics: dict) -> dict:
    """Save all reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    date_str = data["date"]
    saved = {}
    
    # Text
    text_path = REPORTS_DIR / f"report_{date_str}.txt"
    text_content = generate_text_report(data, metrics)
    text_path.write_text(text_content)
    saved["text"] = str(text_path)
    
    # JSON
    json_path = REPORTS_DIR / f"report_{date_str}.json"
    json_path.write_text(generate_json_report(data, metrics))
    saved["json"] = str(json_path)
    
    # PDF
    pdf_path = REPORTS_DIR / f"report_{date_str}.pdf"
    if generate_pdf_report(data, metrics, pdf_path):
        saved["pdf"] = str(pdf_path)
    
    # Latest copies
    (REPORTS_DIR / "latest_report.txt").write_text(text_content)
    
    return saved


def main(watchlist: list = None):
    """Main entry point."""
    logger.info("=" * 50)
    logger.info("Starting Daily Report Generation")
    logger.info("=" * 50)
    
    if watchlist is None:
        watchlist = DEFAULT_WATCHLIST
    
    try:
        data = collect_daily_data(watchlist)
        metrics = calculate_metrics(data)
        saved = save_reports(data, metrics)
        
        logger.info("Report Generation Complete")
        for fmt, path in saved.items():
            logger.info(f"  {fmt}: {path}")
        
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