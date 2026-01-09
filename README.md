# QUANTITATIVE FINANCE DASHBOARD

- **Authors:** Matthieu HANNA GERGUIS & Maxime GRUEZ
- **Course:** A4-IF3 | Python, Git, Linux for Finance
- **Institution:** ESILV (Ecole Supérieure d'Ingénieurs Léonard de Vinci)

---

## ACCESS THE PLATFORM

Click the links below to access the dashboard:

- **Live Deployment (AWS):** http://13.60.215.60:8501
- **Private/Local Access:** http://localhost:8501
- **Status:** Online - Hosted on Linux VM (AWS EC2)
- **Address of website used for scrapping / API consumption: ** https://finance.yahoo.com

---

## OVERVIEW

This project is a professional-grade real-time financial dashboard developed to perform advanced quantitative analysis. It combines single-asset technical analysis with multi-asset portfolio optimization strategies.

The application is deployed on a Linux Virtual Machine and features a robust automation architecture using Cron jobs for continuous operation, self-healing (health checks), and automated daily reporting.

### Key Features
- **Quant A:** Single-asset analysis with 8 backtested trading strategies and Machine Learning price predictions.
- **Quant B:** Portfolio optimization using Modern Portfolio Theory (MPT) and Hierarchical Risk Parity (HRP).
- **Automation:** Fully automated infrastructure managing data fetching and system stability.
- **Reporting:** Automatic generation of daily PDF market reports stored locally on the server.

---

## PROJECT STRUCTURE

```text
project/
│
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python dependencies list
│
├── src/                    # Core Logic Modules
│   ├── data_loader.py      # Data fetching with caching (TTL 5 min)
│   ├── quant_a.py          # Module: Single Asset Analysis & Strategies
│   ├── quant_b.py          # Module: Portfolio Optimization & Efficient Frontier
│   ├── predictions.py      # Module: ML Models (Prophet, Linear Reg, Smoothing)
│   └── utils.py            # Shared mathematical utilities and metrics
│
├── scripts/                # Infrastructure & Automation
│   ├── daily_report.py     # Engine for generating PDF reports
│   ├── run_report.sh       # Shell wrapper for cron execution
│   └── cron_setup.sh       # Server configuration script (Cron, Timezone)
│
├── tests/                  # Quality Assurance
│   └── test_strategies.py  # Unit tests using Pytest
│
└── reports/                # Generated Content
    ├── latest_report.txt   # Raw data summary of the last run
    └── report_YYYY-MM-DD.pdf # Professional Daily PDF Reports

```

---

## MODULE DETAILS

### 1. Quant A - Single Asset Analysis

This module analyzes individual assets (Equity, Crypto, Forex, Indices) using real-time data from Yahoo Finance.

**Trading Strategies Implemented:**

| Strategy | Description |
| --- | --- |
| SMA Crossover | Buy signal when Short MA crosses above Long MA |
| EMA Crossover | Exponential variation of the standard MA Crossover |
| RSI Mean Reversion | Contrarian strategy based on overbought (>70) / oversold (<30) levels |
| Bollinger Bands | Volatility-based strategy utilizing standard deviation bands |
| MACD | Trend-following momentum strategy using signal line crossovers |
| Momentum | Position entry based on Rate of Change (ROC) thresholds |
| Mean Reversion | Statistical arbitrage strategy based on Z-Score analysis |
| Buy & Hold | Benchmark strategy for performance comparison |

**Performance Metrics:**
Total Return, Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, VaR (95%), CVaR (95%), Win Rate.

**Machine Learning:**
Integration of Prophet, Linear Regression, and Exponential Smoothing for trend forecasting with configurable confidence intervals.

### 2. Quant B - Portfolio Optimization

This module constructs and analyzes multi-asset portfolios.

**Allocation Algorithms:**

| Method | Description |
| --- | --- |
| Equal Weight | Naive diversification (1/N allocation) |
| Inverse Volatility | Weights inversely proportional to asset risk |
| Minimum Variance | Mathematical optimization for lowest portfolio volatility |
| Risk Parity | Equal risk contribution from each asset |
| Maximum Sharpe | Monte Carlo optimization (5000 simulations) for max risk-adjusted return |
| HRP | Hierarchical Risk Parity using clustering algorithms |

---

## AUTOMATED REPORTING

The project includes a custom reporting engine powered by `fpdf2` and scheduled via Linux Cron.

* **Schedule:** Runs daily at 20:00 (Europe/Paris Time).
* **Output:** Generates a "Dark Mode" styled PDF containing market summaries, top gainers/losers, and volume analysis.
* **Storage:** Reports are archived in the `reports/` directory on the VM.
* **Example:** A sample report is available in this repository as `reports/report_2026-01-09.pdf`.

---

## INSTALLATION & DEPLOYMENT

### Local Development

Requirements: Python 3.10+

```bash
# 1. Clone the repository
git clone <repository-url>
cd project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start application
streamlit run app.py

```

### Server Deployment (Linux/AWS)

The repository includes a setup script to automate the production environment configuration.

```bash
# Give execution permissions
chmod +x scripts/cron_setup.sh

# Run setup (Configures Cron, Timezone, and Logs)
./scripts/cron_setup.sh

```

**Installed Cron Jobs:**

1. **Daily Reporting:** 20:00 Daily - Triggers PDF generation.
2. **Health Check:** Every 10 mins - Checks app status and restarts if down.
3. **Auto-Reboot:** @reboot - Launches application automatically on server start.

---

## TESTING

Unit tests are implemented to validate trading logic and strategy calculations.

```bash
pytest tests/ -v

```

---

## COMMANDS REFERENCE

### Application Management

```bash
# Start Dashboard
streamlit run app.py

# Run in background (Server mode)
nohup streamlit run app.py --server.port 8501 > logs/app.log 2>&1 &

```

### Logs & Monitoring

```bash
# View App Logs
tail -f logs/app.log

# View Cron/System Logs
grep CRON /var/log/syslog

```

### Manual Operations

```bash
# Generate PDF Report manually
python scripts/daily_report.py

# Check Cron configuration
crontab -l
