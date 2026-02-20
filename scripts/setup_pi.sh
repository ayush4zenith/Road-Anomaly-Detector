#!/usr/bin/env bash
# setup_pi.sh - Prepare a Raspberry Pi (or Debian-based) host for the Road-Anomaly-Detection project
#
# Features:
#   - Optional apt-get system package installation
#   - Create a Python 3 virtualenv at .venv and install requirements.txt
#   - Create models/pretrained directory and optionally download/extract a model archive
#
# Usage:
#   ./scripts/setup_pi.sh [--non-interactive] [--skip-apt] [--model-url <url>]
#
# Notes:
#   - This script is written for Debian-based systems (Raspberry Pi OS).
#   - If your platform differs, use --skip-apt and manually install system deps.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly PROJECT_ROOT
readonly VENV_DIR="$PROJECT_ROOT/.venv"
readonly REQ_FILE="$PROJECT_ROOT/requirements.txt"
readonly MODELS_DIR="$PROJECT_ROOT/models/pretrained"

# Default option values
NON_INTERACTIVE=0
SKIP_APT=0
MODEL_URL=""
INSTALL_TFLITE=0
SKIP_VENV=0

# =============================================================================
# Cleanup trap
# =============================================================================

TMP_ARCHIVE=""
# shellcheck disable=SC2329
cleanup() {
    if [[ -n "$TMP_ARCHIVE" && -f "$TMP_ARCHIVE" ]]; then
        rm -f "$TMP_ARCHIVE"
    fi
}
trap cleanup EXIT

# =============================================================================
# Helper functions
# =============================================================================

check_command() {
    command -v "$1" >/dev/null 2>&1
}

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

maybe_sudo() {
    if [[ "$(id -u)" -ne 0 ]]; then
        if check_command sudo; then
            sudo "$@"
        else
            log_error "This script requires root privileges. Re-run as root or install sudo."
            exit 1
        fi
    else
        "$@"
    fi
}

show_help() {
    cat <<EOF
Usage: $0 [options]

Options:
    --non-interactive    Skip confirmation prompts
    --skip-apt           Don't attempt apt-get installs (useful on non-Debian systems)
    --skip-venv          Don't create/activate a virtualenv; install into current Python env
    --model-url <url>    Download and extract a model archive into models/pretrained
    --install-tflite     Attempt to pip-install tflite-runtime into the venv
    -h, --help           Show this help

This script focuses on a minimal TFLite-only Pi setup: system packages (optional),
a Python venv (optional), installing Python deps from $REQ_FILE, and an optional
TFLite runtime install.
EOF
}

# =============================================================================
# Argument parsing
# =============================================================================

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --non-interactive) NON_INTERACTIVE=1; shift ;;
        --skip-apt) SKIP_APT=1; shift ;;
        --model-url)
            if [[ -z "${2:-}" ]]; then
                log_error "--model-url requires a URL argument"
                exit 1
            fi
            MODEL_URL="$2"; shift 2 ;;
        --install-tflite) INSTALL_TFLITE=1; shift ;;
        --skip-venv) SKIP_VENV=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        *)
            log_error "Unknown argument: $1"
            show_help
            exit 1 ;;
    esac
done

# =============================================================================
# Main setup
# =============================================================================

log_info "Project root: $PROJECT_ROOT"

# Interactive confirmation
if [[ "$NON_INTERACTIVE" -eq 0 ]]; then
    read -rp "Proceed with setup (this may install system packages)? [Y/n] " yn
    yn="${yn:-Y}"
    if [[ ! "$yn" =~ ^[Yy] ]]; then
        log_info "Aborted by user."
        exit 0
    fi
fi

# System packages (apt)
if [[ "$SKIP_APT" -eq 0 ]]; then
    if check_command apt-get; then
        log_info "Updating apt and installing system packages (may take several minutes)..."
        maybe_sudo apt-get update -y
        maybe_sudo apt-get install -y \
            python3 python3-venv python3-dev python3-pip build-essential \
            libatlas-base-dev libopenblas-dev liblapack-dev libjpeg-dev libpng-dev \
            ffmpeg git cmake pkg-config
    else
        log_warn "apt-get not found; skipping system package installation."
    fi
else
    log_info "--skip-apt set: skipping apt package installation"
fi

# Ensure pip is available
log_info "Ensuring pip is available and up-to-date"
if check_command python3; then
    maybe_sudo python3 -m pip install --upgrade pip setuptools wheel || true
else
    log_error "python3 not found. Please install Python3 before continuing."
    exit 1
fi

# Python virtual environment
if [[ "$SKIP_VENV" -eq 1 ]]; then
    log_info "--skip-venv set: using current Python environment"
else
    log_info "Creating Python virtualenv at: $VENV_DIR"
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
fi

# Install Python dependencies
if [[ -f "$REQ_FILE" ]]; then
    log_info "Installing Python dependencies from $REQ_FILE"
    pip install --upgrade pip setuptools wheel
    pip install -r "$REQ_FILE"
else
    log_warn "$REQ_FILE not found. Create it and run pip install -r manually."
fi

# Optional TFLite runtime install
if [[ "$INSTALL_TFLITE" -eq 1 ]]; then
    log_info "Attempting to install tflite-runtime..."
    if pip install tflite-runtime; then
        log_info "tflite-runtime installed successfully from PyPI"
    else
        log_warn "pip install tflite-runtime failed."
        log_info "On Raspberry Pi you may need a platform-specific wheel."
        log_info "See: https://www.tensorflow.org/lite/guide/python"
    fi
fi

# Create models directory
log_info "Creating models directory: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# Download model archive if URL provided
if [[ -n "$MODEL_URL" ]]; then
    log_info "Downloading model archive from: $MODEL_URL"
    TMP_ARCHIVE="/tmp/models_archive_$$"

    if check_command curl; then
        curl -fSL "$MODEL_URL" -o "$TMP_ARCHIVE"
    elif check_command wget; then
        wget -q "$MODEL_URL" -O "$TMP_ARCHIVE"
    else
        log_error "Neither curl nor wget available to download MODEL_URL"
        exit 1
    fi

    log_info "Extracting into $MODELS_DIR"
    if file "$TMP_ARCHIVE" | grep -qi 'zip archive'; then
        if check_command unzip; then
            unzip -o "$TMP_ARCHIVE" -d "$MODELS_DIR"
        else
            log_error "zip archive found but unzip not installed. Install unzip and re-run."
            exit 1
        fi
    else
        tar -xvf "$TMP_ARCHIVE" -C "$MODELS_DIR"
    fi
fi

# =============================================================================
# Completion message
# =============================================================================

echo
log_info "Setup complete! Quick next steps:"
echo "  1) Activate the venv: source $VENV_DIR/bin/activate"
echo "  2) Run the demo: python3 src/main.py"
echo "  3) Place optimized model files (.tflite) in: $MODELS_DIR"
echo
log_info "For persistent service, create a systemd unit running under the venv."

# Deactivate venv if we activated it
deactivate 2>/dev/null || true

exit 0
