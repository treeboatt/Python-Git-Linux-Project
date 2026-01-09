#!/usr/bin/env bash
# ============================================================
# CRON SETUP SCRIPT (robust)
# - Daily report at 20:00 (server time) -> set server TZ to Europe/Paris
# - Keep Streamlit app running
# - Writes logs in /var/log/quant_dashboard if possible, else ./logs
# - Compatible with .venv/ or venv/
# ============================================================

set -euo pipefail

# ---------- Pretty logs ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------- Detect project root ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
log_info "Project root: $PROJECT_ROOT"

# ---------- Find python (fallback) ----------
if command -v python3 >/dev/null 2>&1; then
  SYS_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  SYS_PYTHON="$(command -v python)"
else
  log_error "Python not found. Please install Python 3."
  exit 1
fi

# ---------- Detect venv (.venv or venv) ----------
VENV_DIR=""
if [ -d "$PROJECT_ROOT/.venv" ]; then
  VENV_DIR="$PROJECT_ROOT/.venv"
elif [ -d "$PROJECT_ROOT/venv" ]; then
  VENV_DIR="$PROJECT_ROOT/venv"
fi

PYTHON_BIN="$SYS_PYTHON"
STREAMLIT_BIN="streamlit"

if [ -n "$VENV_DIR" ]; then
  log_info "Virtual environment detected: $VENV_DIR"
  if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  fi
  if [ -x "$VENV_DIR/bin/streamlit" ]; then
    STREAMLIT_BIN="$VENV_DIR/bin/streamlit"
  fi
else
  log_warning "No .venv/ or venv/ found. Will use system Python/Streamlit."
fi

log_info "Python bin: $PYTHON_BIN"
log_info "Streamlit bin: $STREAMLIT_BIN"

# ---------- Logs dir (try /var/log first, else local) ----------
LOG_DIR="/var/log/quant_dashboard"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
  LOG_DIR="$PROJECT_ROOT/logs"
  mkdir -p "$LOG_DIR"
fi
log_success "Logs directory: $LOG_DIR"

# ---------- Ensure reports dir ----------
mkdir -p "$PROJECT_ROOT/reports"
log_success "Reports directory: $PROJECT_ROOT/reports"

# ---------- Wrapper scripts ----------
log_info "Creating wrapper scripts..."

REPORT_WRAPPER="$PROJECT_ROOT/scripts/run_daily_report.sh"
APP_WRAPPER="$PROJECT_ROOT/scripts/run_app.sh"
HEALTH_CHECK="$PROJECT_ROOT/scripts/health_check.sh"

cat > "$REPORT_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

# activate venv if present
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
fi

PY_BIN="$PYTHON_BIN"
"\$PY_BIN" "$PROJECT_ROOT/scripts/daily_report.py" >> "$LOG_DIR/daily_report.log" 2>&1
echo "\$(date '+%Y-%m-%d %H:%M:%S') - Daily report completed" >> "$LOG_DIR/cron.log"
EOF

cat > "$APP_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

# activate venv if present
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
fi

# If already running, do nothing
if pgrep -f "streamlit run.*app.py" >/dev/null 2>&1; then
  echo "\$(date '+%Y-%m-%d %H:%M:%S') - Streamlit already running" >> "$LOG_DIR/app.log"
  exit 0
fi

echo "\$(date '+%Y-%m-%d %H:%M:%S') - Starting Streamlit app" >> "$LOG_DIR/app.log"
nohup "$STREAMLIT_BIN" run "$PROJECT_ROOT/app.py" --server.port 8501 --server.address 0.0.0.0 >> "$LOG_DIR/app.log" 2>&1 &
echo "\$(date '+%Y-%m-%d %H:%M:%S') - Streamlit started (PID \$!)" >> "$LOG_DIR/app.log"
EOF

cat > "$HEALTH_CHECK" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

if ! pgrep -f "streamlit run.*app.py" >/dev/null 2>&1; then
  echo "\$(date '+%Y-%m-%d %H:%M:%S') - App not running, restarting..." >> "$LOG_DIR/health.log"
  "$APP_WRAPPER"
else
  echo "\$(date '+%Y-%m-%d %H:%M:%S') - App is healthy" >> "$LOG_DIR/health.log"
fi
EOF

chmod +x "$REPORT_WRAPPER" "$APP_WRAPPER" "$HEALTH_CHECK"
log_success "Wrappers created:
- $REPORT_WRAPPER
- $APP_WRAPPER
- $HEALTH_CHECK"

# ---------- Cron entries ----------
# NOTE: 20:00 is SERVER time -> set VM timezone to Europe/Paris
CRON_DAILY_REPORT="0 20 * * * $REPORT_WRAPPER"
CRON_HEALTH_CHECK="*/10 * * * * $HEALTH_CHECK"
CRON_APP_STARTUP="@reboot $APP_WRAPPER"

# ---------- Backup existing crontab ----------
CRON_BACKUP="$PROJECT_ROOT/scripts/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
crontab -l > "$CRON_BACKUP" 2>/dev/null || echo "# No existing crontab" > "$CRON_BACKUP"
log_info "Backed up existing crontab to: $CRON_BACKUP"

# ---------- Remove old entries (avoid duplicates) ----------
EXISTING_CRON="$(crontab -l 2>/dev/null || true)"
CLEANED_CRON="$(echo "$EXISTING_CRON" \
  | grep -v "run_daily_report.sh" \
  | grep -v "health_check.sh" \
  | grep -v "run_app.sh" \
  | grep -v "# Quant Dashboard Cron Jobs" \
  || true)"

# Install cleaned + our header + our jobs
{
  echo "$CLEANED_CRON"
  echo ""
  echo "# Quant Dashboard Cron Jobs"
  echo "$CRON_DAILY_REPORT"
  echo "$CRON_HEALTH_CHECK"
  echo "$CRON_APP_STARTUP"
} | crontab -

# ---------- Summary ----------
echo ""
echo "============================================================"
echo -e "${GREEN}CRON SETUP COMPLETE${NC}"
echo "============================================================"
echo ""
echo "Installed cron jobs (server time):"
echo ""
echo "  1) Daily Report:   $CRON_DAILY_REPORT"
echo "  2) Health Check:   $CRON_HEALTH_CHECK"
echo "  3) On reboot start:$CRON_APP_STARTUP"
echo ""
echo "Logs directory: $LOG_DIR"
echo "Reports saved to: $PROJECT_ROOT/reports/"
echo ""
log_info "Current crontab:"
crontab -l
echo ""
log_success "Done."
echo ""
log_warning "IMPORTANT: ensure VM timezone is Europe/Paris for '20:00 Paris time':"
echo "  timedatectl"
echo "  sudo timedatectl set-timezone Europe/Paris"
