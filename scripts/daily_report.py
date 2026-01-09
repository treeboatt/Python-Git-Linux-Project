#!/usr/bin/env python3
"""
Daily Report Generator
Generates a comprehensive daily report at 8pm via cron job.
Saves reports locally on the VM as required by the project.

Usage:
    python scripts/daily_report.py
    
Cron setup (runs at 8pm Paris time every day):
    0 20 * * * cd /path/to/project && /path/to/python scripts/daily_report.py >> /var/log/quant_report.log 2>&1
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# add project root to path so we can import our modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pytz
import json
import logging

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# try importing our modules
try:
    from src.data_loader import fetch_data, get_daily_summary, get_period_stats
    from src.utils import (
        compute_volatility,
        compute_max_drawdown,
        compute_sharpe_ratio,
        compute_returns,
        format_percentage,
        safe_float,
    )
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.info("Make sure you're running from the project root directory")
    sys.exit(1)

# try importing PDF generation
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
    logger.warning("fpdf2 not installed. PDF reports will be skipped.")


# ============================================================
# CONFIGURATION
# ============================================================

# default watchlist for daily report
DEFAULT_WATCHLIST = [
    # US stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    # indices
    "^GSPC", "^DJI", "^IXIC",
    # crypto
    "BTC-USD", "ETH-USD",
    # forex
    "EURUSD=X", "GBPUSD=X",
    # commodities
    "GC=F", "CL=F",
]

# report output directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# timezone
PARIS_TZ = pytz.timezone("Europe/Paris")


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_daily_data(tickers: list) -> dict:
    """
    Collect daily data for all tickers in the watchlist.
    """
    logger.info(f"Collecting data for {len(tickers)} tickers...")
    
    results = {
        "timestamp": datetime.now(PARIS_TZ).isoformat(),
        "date": datetime.now(PARIS_TZ).strftime("%Y-%m-%d"),
        "assets": {},
        "errors": [],
    }
    
    for ticker in tickers:
        try:
            # get daily summary
            summary = get_daily_summary(ticker)
            
            if summary:
                # get period stats for additional context
                stats_1mo = get_period_stats(ticker, "1mo")
                
                results["assets"][ticker] = {
                    "daily": summary,
                    "monthly_stats": stats_1mo,
                }
                logger.info(f"  ✓ {ticker}: ${summary['close']:.2f} ({summary['change_pct']:+.2f}%)")
            else:
                results["errors"].append(f"No data for {ticker}")
                logger.warning(f"  ✗ {ticker}: No data available")
                
        except Exception as e:
            results["errors"].append(f"{ticker}: {str(e)}")
            logger.error(f"  ✗ {ticker}: {e}")
    
    return results


def calculate_portfolio_metrics(data: dict) -> dict:
    """
    Calculate aggregate portfolio metrics from collected data.
    """
    assets = data.get("assets", {})
    
    if not assets:
        return {}
    
    # collect daily returns
    returns = []
    total_value = 0
    gainers = []
    losers = []
    
    for ticker, info in assets.items():
        daily = info.get("daily", {})
        
        if daily:
            change_pct = daily.get("change_pct", 0)
            returns.append(change_pct)
            
            if change_pct > 0:
                gainers.append((ticker, change_pct))
            elif change_pct < 0:
                losers.append((ticker, change_pct))
    
    # sort gainers and losers
    gainers.sort(key=lambda x: x[1], reverse=True)
    losers.sort(key=lambda x: x[1])
    
    # calculate stats
    returns = np.array(returns)
    
    metrics = {
        "total_assets": len(assets),
        "avg_return": float(np.mean(returns)) if len(returns) > 0 else 0,
        "median_return": float(np.median(returns)) if len(returns) > 0 else 0,
        "std_return": float(np.std(returns)) if len(returns) > 0 else 0,
        "positive_count": len(gainers),
        "negative_count": len(losers),
        "unchanged_count": len(assets) - len(gainers) - len(losers),
        "top_gainers": gainers[:5],
        "top_losers": losers[:5],
        "best_performer": gainers[0] if gainers else None,
        "worst_performer": losers[0] if losers else None,
    }
    
    return metrics


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_text_report(data: dict, metrics: dict) -> str:
    """
    Generate a plain text report.
    """
    lines = []
    
    # header
    lines.append("=" * 70)
    lines.append("DAILY MARKET REPORT")
    lines.append(f"Generated: {data['timestamp']}")
    lines.append("=" * 70)
    lines.append("")
    
    # summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Date: {data['date']}")
    lines.append(f"Assets Tracked: {metrics.get('total_assets', 0)}")
    lines.append(f"Average Return: {metrics.get('avg_return', 0):+.2f}%")
    lines.append(f"Gainers: {metrics.get('positive_count', 0)} | Losers: {metrics.get('negative_count', 0)}")
    lines.append("")
    
    # top movers
    lines.append("TOP GAINERS")
    lines.append("-" * 40)
    for ticker, change in metrics.get("top_gainers", []):
        lines.append(f"  {ticker:12} {change:+.2f}%")
    if not metrics.get("top_gainers"):
        lines.append("  No gainers today")
    lines.append("")
    
    lines.append("TOP LOSERS")
    lines.append("-" * 40)
    for ticker, change in metrics.get("top_losers", []):
        lines.append(f"  {ticker:12} {change:+.2f}%")
    if not metrics.get("top_losers"):
        lines.append("  No losers today")
    lines.append("")
    
    # detailed data
    lines.append("DETAILED DATA")
    lines.append("-" * 40)
    lines.append(f"{'Ticker':<12} {'Close':>10} {'Change':>10} {'High':>10} {'Low':>10} {'Volume':>12}")
    lines.append("-" * 70)
    
    for ticker, info in data.get("assets", {}).items():
        daily = info.get("daily", {})
        if daily:
            lines.append(
                f"{ticker:<12} "
                f"${daily.get('close', 0):>9.2f} "
                f"{daily.get('change_pct', 0):>+9.2f}% "
                f"${daily.get('high', 0):>9.2f} "
                f"${daily.get('low', 0):>9.2f} "
                f"{daily.get('volume', 0):>12,}"
            )
    
    lines.append("")
    
    # monthly context
    lines.append("MONTHLY CONTEXT (30-day stats)")
    lines.append("-" * 40)
    lines.append(f"{'Ticker':<12} {'Return':>10} {'Volatility':>12} {'Best Day':>10} {'Worst Day':>10}")
    lines.append("-" * 70)
    
    for ticker, info in data.get("assets", {}).items():
        stats = info.get("monthly_stats", {})
        if stats:
            lines.append(
                f"{ticker:<12} "
                f"{stats.get('total_return', 0)*100:>+9.2f}% "
                f"{stats.get('volatility', 0)*100:>11.2f}% "
                f"{stats.get('best_day', 0)*100:>+9.2f}% "
                f"{stats.get('worst_day', 0)*100:>+9.2f}%"
            )
    
    lines.append("")
    
    # errors if any
    if data.get("errors"):
        lines.append("ERRORS")
        lines.append("-" * 40)
        for error in data["errors"]:
            lines.append(f"  ! {error}")
        lines.append("")
    
    # footer
    lines.append("=" * 70)
    lines.append("End of Report")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_json_report(data: dict, metrics: dict) -> str:
    """
    Generate a JSON report for programmatic access.
    """
    report = {
        "metadata": {
            "generated_at": data["timestamp"],
            "date": data["date"],
            "version": "1.0",
        },
        "summary": metrics,
        "assets": data["assets"],
        "errors": data["errors"],
    }
    
    return json.dumps(report, indent=2, default=str)


def generate_pdf_report(data: dict, metrics: dict, output_path: Path) -> bool:
    """
    Generate a PDF report.
    Returns True if successful, False otherwise.
    """
    if not HAS_FPDF:
        logger.warning("PDF generation skipped: fpdf2 not installed")
        return False
    
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # title
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 15, "Daily Market Report", ln=True, align="C")
        
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Generated: {data['date']}", ln=True, align="C")
        pdf.ln(10)
        
        # summary section
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        
        pdf.cell(0, 8, f"Assets Tracked: {metrics.get('total_assets', 0)}", ln=True)
        pdf.cell(0, 8, f"Average Return: {metrics.get('avg_return', 0):+.2f}%", ln=True)
        pdf.cell(0, 8, f"Gainers: {metrics.get('positive_count', 0)} | Losers: {metrics.get('negative_count', 0)}", ln=True)
        pdf.ln(5)
        
        # top movers
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Top Movers", ln=True)
        pdf.set_font("Helvetica", "", 10)
        
        # gainers
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Top Gainers:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for ticker, change in metrics.get("top_gainers", [])[:3]:
            pdf.cell(0, 6, f"  {ticker}: {change:+.2f}%", ln=True)
        
        pdf.ln(3)
        
        # losers
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Top Losers:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for ticker, change in metrics.get("top_losers", [])[:3]:
            pdf.cell(0, 6, f"  {ticker}: {change:+.2f}%", ln=True)
        
        pdf.ln(5)
        
        # asset table
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Asset Details", ln=True)
        
        # table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(25, 8, "Ticker", border=1)
        pdf.cell(25, 8, "Close", border=1, align="R")
        pdf.cell(25, 8, "Change", border=1, align="R")
        pdf.cell(25, 8, "High", border=1, align="R")
        pdf.cell(25, 8, "Low", border=1, align="R")
        pdf.cell(30, 8, "Volume", border=1, align="R")
        pdf.ln()
        
        # table rows
        pdf.set_font("Helvetica", "", 8)
        for ticker, info in list(data.get("assets", {}).items())[:20]:  # limit rows
            daily = info.get("daily", {})
            if daily:
                pdf.cell(25, 6, ticker, border=1)
                pdf.cell(25, 6, f"${daily.get('close', 0):.2f}", border=1, align="R")
                pdf.cell(25, 6, f"{daily.get('change_pct', 0):+.2f}%", border=1, align="R")
                pdf.cell(25, 6, f"${daily.get('high', 0):.2f}", border=1, align="R")
                pdf.cell(25, 6, f"${daily.get('low', 0):.2f}", border=1, align="R")
                pdf.cell(30, 6, f"{daily.get('volume', 0):,}", border=1, align="R")
                pdf.ln()
        
        # save
        pdf.output(str(output_path))
        return True
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return False


# ============================================================
# MAIN EXECUTION
# ============================================================

def save_reports(data: dict, metrics: dict) -> dict:
    """
    Save all report formats to disk.
    """
    # ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    date_str = data["date"]
    saved_files = {}
    
    # text report
    text_path = REPORTS_DIR / f"report_{date_str}.txt"
    text_report = generate_text_report(data, metrics)
    text_path.write_text(text_report)
    saved_files["text"] = str(text_path)
    logger.info(f"Saved text report: {text_path}")
    
    # json report
    json_path = REPORTS_DIR / f"report_{date_str}.json"
    json_report = generate_json_report(data, metrics)
    json_path.write_text(json_report)
    saved_files["json"] = str(json_path)
    logger.info(f"Saved JSON report: {json_path}")
    
    # pdf report
    pdf_path = REPORTS_DIR / f"report_{date_str}.pdf"
    if generate_pdf_report(data, metrics, pdf_path):
        saved_files["pdf"] = str(pdf_path)
        logger.info(f"Saved PDF report: {pdf_path}")
    
    # also save a "latest" symlink/copy for easy access
    latest_text = REPORTS_DIR / "latest_report.txt"
    latest_text.write_text(text_report)
    
    latest_json = REPORTS_DIR / "latest_report.json"
    latest_json.write_text(json_report)
    
    return saved_files


def main(watchlist: list = None):
    """
    Main entry point for daily report generation.
    """
    start_time = datetime.now()
    logger.info("=" * 50)
    logger.info("Starting Daily Report Generation")
    logger.info(f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    # use default watchlist if none provided
    if watchlist is None:
        watchlist = DEFAULT_WATCHLIST
    
    try:
        # collect data
        data = collect_daily_data(watchlist)
        
        # calculate metrics
        metrics = calculate_portfolio_metrics(data)
        
        # save reports
        saved_files = save_reports(data, metrics)
        
        # summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("Report Generation Complete")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Assets processed: {metrics.get('total_assets', 0)}")
        logger.info(f"Files saved: {len(saved_files)}")
        for fmt, path in saved_files.items():
            logger.info(f"  - {fmt}: {path}")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # allow custom watchlist via command line
    if len(sys.argv) > 1:
        custom_watchlist = sys.argv[1].split(",")
        success = main(watchlist=custom_watchlist)
    else:
        success = main()
    
    sys.exit(0 if success else 1)