# Quantitative Finance Dashboard

Interactive quantitative analysis platform developed for the Python, Git, Linux for Finance course.

---

## Overview

This project is a real-time financial dashboard that provides single-asset analysis (Quant A) and multi-asset portfolio optimization (Quant B). The application fetches live data from Yahoo Finance, runs trading strategy backtests, and offers ML-based price predictions.

Designed to run continuously on a Linux VM with automated daily reporting via cron.

---

## Project Structure

```
project/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── data_loader.py      # Data fetching with caching (5 min TTL)
│   ├── quant_a.py          # Single asset analysis module
│   ├── quant_b.py          # Portfolio analysis module
│   ├── predictions.py      # ML prediction models
│   └── utils.py            # Shared utilities and metrics
│
├── scripts/
│   ├── daily_report.py     # Automated report generation
│   └── cron_setup.sh       # Cron job configuration
│
├── tests/
│   └── test_strategies.py  # Unit tests (pytest)
│
└── reports/                # Generated reports (TXT, JSON, PDF)
```

---

## Quant A - Single Asset Analysis

Dedicated module for analyzing individual assets (stocks, crypto, forex, indices).

**Trading Strategies (8)**

| Strategy | Description |
|----------|-------------|
| SMA Crossover | Buy when short MA crosses above long MA |
| EMA Crossover | Same logic with exponential moving averages |
| RSI Mean Reversion | Buy oversold (<30), sell overbought (>70) |
| Bollinger Bands | Buy at lower band, sell at upper band |
| MACD | Buy when MACD crosses above signal line |
| Momentum | Buy when recent return exceeds threshold |
| Mean Reversion | Buy when z-score indicates oversold |
| Buy & Hold | Benchmark strategy, always invested |

**Performance Metrics**

Total return, annualized return, volatility, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, VaR (95%), CVaR (95%), win rate, profit factor.

**ML Predictions**

Prophet, Linear Regression, Momentum-based, Exponential Smoothing — all with configurable confidence intervals.

**Charts**

Interactive candlestick charts, buy/sell signals, technical indicators, drawdown visualization, returns distribution.

---

## Quant B - Portfolio Analysis

Dedicated module for building and analyzing multi-asset portfolios.

**Allocation Methods (7)**

| Method | Description |
|--------|-------------|
| Equal Weight | 1/n allocation to each asset |
| Inverse Volatility | Higher weight to less volatile assets |
| Minimum Variance | Closed-form solution for lowest risk |
| Risk Parity | Equal risk contribution from each asset |
| Maximum Sharpe | Monte Carlo optimization (5000 simulations) |
| HRP | Hierarchical Risk Parity using clustering |
| Custom | User-defined weights |

**Rebalancing Options**

None (drift), daily, weekly, monthly.

**Analysis Tools**

Correlation heatmap, risk contribution breakdown, diversification ratio, efficient frontier visualization.

---

## Installation

Requirements: Python 3.10+

```bash
# Clone the repository
git clone <repository-url>
cd project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Start the dashboard
streamlit run app.py
```

Access the dashboard at [http://localhost:8501](http://localhost:8508/)

For external access (VM deployment):

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Cron Setup (Linux VM)

The `cron_setup.sh` script automatically configures scheduled tasks.

```bash
chmod +x scripts/cron_setup.sh
./scripts/cron_setup.sh
```

This installs:
- Daily report generation at 8pm
- Health check every 10 minutes
- Auto-restart on VM reboot

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Key Dependencies

- streamlit - Web interface
- streamlit-autorefresh - Auto-refresh every 5 minutes
- streamlit-option-menu - Navigation menu
- yfinance - Financial data from Yahoo Finance
- plotly - Interactive charts
- pandas / numpy - Data manipulation
- scikit-learn - ML models
- prophet - Time series forecasting
- fpdf2 - PDF report generation
- pytest - Unit testing

---

## Commands Reference

```bash
# Start application
streamlit run app.py

# Run tests
pytest tests/ -v

# Configure cron jobs
./scripts/cron_setup.sh

# Generate report manually
python scripts/daily_report.py

# View application logs
tail -f logs/app.log

# View report logs
tail -f logs/daily_report.log

# Check current crontab
crontab -l

# Clear data cache (also available in UI)
# Click "Force Refresh Data" in sidebar
```

---

## Authors

Master Finance — Python, Git, Linux for Finance Project
