@echo off
echo ===================================================
echo   Qwen3-TTS CPU Verber Installer ^& Launcher (UV)
echo ===================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.12 and try again.
    pause
    exit /b 1
)

:: Check if uv is installed, if not, install it
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [System] 'uv' is not detected. Installing 'uv' via pip...
    python -m pip install uv
    if %errorlevel% neq 0 (
        echo Error: Failed to install 'uv'. Please install it manually.
        pause
        exit /b 1
    )
) else (
    echo [System] 'uv' detected successfully.
)

:: Create Virtual Environment using uv
if not exist .venv (
    echo [System] Creating virtual environment in .venv using uv...
    uv venv .venv
)

:: Activate Virtual Environment
call .venv\Scripts\activate.bat

:: Install CPU-only PyTorch and Torchaudio using uv
echo [System] Installing CPU-only PyTorch (AMD/Intel CPU optimized) via uv...
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Install remaining dependencies using uv
echo [System] Installing other required packages via uv...
uv pip install fastapi uvicorn qwen-tts soundfile numpy python-multipart psutil gradio

:: Run the server
echo.
echo ===================================================
echo   Verber Server running on http://127.0.0.1:8000
echo ===================================================
echo.
python main.py

pause
