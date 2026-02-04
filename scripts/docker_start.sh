#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Chatterbox Docker Startup Script
# =============================================================================

DEBUG="${DEBUG:-0}"

# Colors and symbols
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

ok() { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; }
info() { echo -e "${CYAN}→${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
header() { echo -e "\n${BOLD}$*${NC}"; }
debug() { [[ "$DEBUG" == "1" ]] && echo -e "[debug] $*" || true; }

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
START_API="${START_API:-1}"

# Gradio mode: "turbo" (default), "original", or "none"
# Legacy vars START_GRADIO_TURBO/START_GRADIO_ORIGINAL still work for backwards compat
GRADIO_MODE="${GRADIO_MODE:-turbo}"
if [[ "${START_GRADIO_TURBO:-}" == "1" ]]; then
  GRADIO_MODE="turbo"
elif [[ "${START_GRADIO_ORIGINAL:-}" == "1" ]]; then
  GRADIO_MODE="original"
elif [[ "${START_GRADIO_TURBO:-}" == "0" && "${START_GRADIO_ORIGINAL:-}" == "0" ]]; then
  GRADIO_MODE="none"
fi

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-10050}"

export GRADIO_HOST="${GRADIO_HOST:-0.0.0.0}"
export GRADIO_TURBO_PORT="${GRADIO_TURBO_PORT:-7860}"
export GRADIO_ORIGINAL_PORT="${GRADIO_ORIGINAL_PORT:-7861}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"

# Pass debug to API
export API_DEBUG="$DEBUG"

header "═══════════════════════════════════════════════════"
header "  Chatterbox TTS - Docker Startup"
header "═══════════════════════════════════════════════════"
echo ""
header "Configuration:"
if [[ "$START_API" == "1" ]]; then
  ok "API Server: http://${API_HOST}:${API_PORT}"
else
  info "API Server: disabled"
fi
case "$GRADIO_MODE" in
  turbo)
    ok "Gradio Turbo: http://${GRADIO_HOST}:${GRADIO_TURBO_PORT}"
    [[ "$GRADIO_SHARE" == "1" ]] && info "  Public sharing: enabled (URL will appear below)"
    ;;
  original)
    ok "Gradio Original: http://${GRADIO_HOST}:${GRADIO_ORIGINAL_PORT}"
    [[ "$GRADIO_SHARE" == "1" ]] && info "  Public sharing: enabled (URL will appear below)"
    ;;
  none)
    info "Gradio: disabled"
    ;;
  *)
    fail "Unknown GRADIO_MODE: $GRADIO_MODE (use 'turbo', 'original', or 'none')"
    exit 1
    ;;
esac
echo ""

debug "DEBUG mode enabled"
debug "  START_API=$START_API"
debug "  GRADIO_MODE=$GRADIO_MODE"
debug "  GRADIO_SHARE=$GRADIO_SHARE"

# -----------------------------------------------------------------------------
# API arguments
# -----------------------------------------------------------------------------
API_ARGS=()
if [[ "${NO_PRELOAD_TURBO:-0}" == "1" ]]; then
  API_ARGS+=("--no-preload-turbo")
fi
if [[ "${PRELOAD_ORIGINAL:-0}" == "1" ]]; then
  API_ARGS+=("--preload-original")
fi
if [[ -n "${PRELOAD_VOICE:-}" ]]; then
  API_ARGS+=("--preload-voice" "${PRELOAD_VOICE}")
fi

# -----------------------------------------------------------------------------
# Process management
# -----------------------------------------------------------------------------
declare -a PIDS=()
declare -A PID_NAMES=()

cleanup() {
  echo ""
  warn "Shutting down all services..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      debug "Killing ${PID_NAMES[$pid]:-process} (PID $pid)"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  # Give processes time to exit gracefully
  sleep 1
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      debug "Force killing PID $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  ok "Shutdown complete."
}

trap cleanup SIGINT SIGTERM EXIT

# -----------------------------------------------------------------------------
# Start services
# -----------------------------------------------------------------------------

header "Starting Services..."

if [[ "$START_API" == "1" ]]; then
  info "Starting API server..."
  python /workspace/chatterbox/tts_api.py --host "$API_HOST" --port "$API_PORT" "${API_ARGS[@]}" &
  pid=$!
  PIDS+=("$pid")
  PID_NAMES[$pid]="API"
  debug "API started with PID $pid"
fi

if [[ "$GRADIO_MODE" == "turbo" ]]; then
  info "Starting Gradio Turbo..."
  # Suppress noisy warnings (diffusers deprecation, float32 conversion)
  python -W ignore::FutureWarning -W ignore::UserWarning /workspace/chatterbox/gradio_tts_turbo_app_enhanced.py &
  pid=$!
  PIDS+=("$pid")
  PID_NAMES[$pid]="Gradio-Turbo"
  debug "Gradio Turbo started with PID $pid"
elif [[ "$GRADIO_MODE" == "original" ]]; then
  info "Starting Gradio Original..."
  python -W ignore::FutureWarning -W ignore::UserWarning /workspace/chatterbox/gradio_tts_app_enhanced.py &
  pid=$!
  PIDS+=("$pid")
  PID_NAMES[$pid]="Gradio-Original"
  debug "Gradio Original started with PID $pid"
fi

# -----------------------------------------------------------------------------
# Wait or drop to shell
# -----------------------------------------------------------------------------

if [[ ${#PIDS[@]} -eq 0 ]]; then
  warn "No services selected. Dropping into bash."
  trap - SIGINT SIGTERM EXIT
  exec bash
fi

echo ""
ok "All services started (${#PIDS[@]} processes)"
info "Press Ctrl+C to stop"
echo ""
header "═══════════════════════════════════════════════════"
echo ""

# Wait for any process to exit
wait -n "${PIDS[@]}" 2>/dev/null || true
exit_code=$?
debug "A process exited with code $exit_code"

# If one process dies, shut down the rest
cleanup
