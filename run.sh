#!/bin/bash
echo "==================================================="
echo "  Qwen3-TTS CPU Verber Installer & Launcher (UV)"
echo "==================================================="
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "[System] 'uv' is not detected. Installing 'uv' via pip..."
    python3 -m pip install uv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install 'uv'. Please install it manually."
        exit 1
    fi
else
    echo "[System] 'uv' detected successfully."
fi

# Create Virtual Environment using uv
if [ ! -d ".venv" ]; then
    echo "[System] Creating virtual environment in .venv using uv..."
    uv venv .venv
fi

# Activate Virtual Environment
source .venv/bin/activate

# Install PyTorch
echo "[System] Installing PyTorch via uv..."
uv pip install torch torchaudio

# Install remaining dependencies
echo "[System] Installing other required packages via uv..."
uv pip install fastapi uvicorn qwen-tts soundfile numpy python-multipart psutil gradio

# Run the server
echo
echo "==================================================="
echo "  Verber Server running on http://127.0.0.1:8000"
echo "==================================================="
echo
python3 main.py
