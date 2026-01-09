#!/usr/bin/env bash
# ============================================================
# CRON SETUP SCRIPT - Quant Terminal
# - Daily report at 20:00 (Paris time)
# - Health check every 10 minutes
# - Auto-restart on VM reboot
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "[INFO] $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
log_info "Project: $PROJECT_ROOT"

# Find Python
if command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    log_error "Python3 not found"
    exit 1
fi

# Detect venv
VENV_DIR=""
[ -d "$PROJECT_ROOT/.venv" ] && VENV_DIR="$PROJECT_ROOT/.venv"
[ -d "$PROJECT_ROOT/venv" ] && VENV_DIR="$PROJECT_ROOT/venv"

if [ -n "$VENV_DIR" ]; then
    log_info "Venv: $VENV_DIR"
    PYTHON="$VENV_DIR/bin/python"
    STREAMLIT="$VENV_DIR/bin/streamlit"
else
    STREAMLIT="streamlit"
fi

# Logs directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_ROOT/reports"

# Create wrapper scripts
REPORT_SCRIPT="$PROJECT_ROOT/scripts/run_report.sh"
APP_SCRIPT="$PROJECT_ROOT/scripts/run_app.sh"
HEALTH_SCRIPT="$PROJECT_ROOT/scripts/health_check.sh"

cat > "$REPORT_SCRIPT" <<EOF
#!/bin/bash
cd "$PROJECT_ROOT"
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"
"$PYTHON" "$PROJECT_ROOT/scripts/daily_report.py" >> "$LOG_DIR/report.log" 2>&1
echo "\$(date) - Report done" >> "$LOG_DIR/cron.log"
EOF

cat > "$APP_SCRIPT" <<EOF
#!/bin/bash
cd "$PROJECT_ROOT"
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"
if pgrep -f "streamlit run.*app.py" >/dev/null; then
    echo "\$(date) - App already running" >> "$LOG_DIR/app.log"
    exit 0
fi
nohup "$STREAMLIT" run "$PROJECT_ROOT/app.py" --server.port 8501 --server.address 0.0.0.0 >> "$LOG_DIR/app.log" 2>&1 &
echo "\$(date) - App started" >> "$LOG_DIR/app.log"
EOF

cat > "$HEALTH_SCRIPT" <<EOF
#!/bin/bash
cd "$PROJECT_ROOT"
if ! pgrep -f "streamlit run.*app.py" >/dev/null; then
    echo "\$(date) - Restarting app" >> "$LOG_DIR/health.log"
    "$APP_SCRIPT"
else
    echo "\$(date) - App OK" >> "$LOG_DIR/health.log"
fi
EOF

chmod +x "$REPORT_SCRIPT" "$APP_SCRIPT" "$HEALTH_SCRIPT"
log_success "Scripts created"

# Cron entries
CRON_REPORT="0 20 * * * $REPORT_SCRIPT"
CRON_HEALTH="*/10 * * * * $HEALTH_SCRIPT"
CRON_REBOOT="@reboot $APP_SCRIPT"

# Backup existing crontab
crontab -l > "$PROJECT_ROOT/scripts/crontab_backup.txt" 2>/dev/null || true

# Install cron jobs
EXISTING=$(crontab -l 2>/dev/null | grep -v "run_report.sh\|health_check.sh\|run_app.sh\|Quant Terminal" || true)

{
    echo "$EXISTING"
    echo ""
    echo "# Quant Terminal Cron Jobs"
    echo "$CRON_REPORT"
    echo "$CRON_HEALTH"
    echo "$CRON_REBOOT"
} | crontab -

echo ""
echo "============================================================"
log_success "CRON SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Jobs installed:"
echo "  1) Daily report:  20:00"
echo "  2) Health check:  every 10 min"
echo "  3) Auto-start:    on reboot"
echo ""
echo "Logs: $LOG_DIR"
echo "Reports: $PROJECT_ROOT/reports"
echo ""
crontab -l
echo ""
log_warning "Set timezone: sudo timedatectl set-timezone Europe/Paris"