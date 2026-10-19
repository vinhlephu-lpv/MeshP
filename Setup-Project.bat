@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===================================================
echo   CAI DAT MOI TRUONG MESH STUDIO (CHAY 1 LAN DAU)
echo ===================================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Dang tao moi truong Python (.venv)...
    python -m venv .venv
)

echo [2/4] Dang cai dat thu vien Python (PyTorch CUDA + Hunyuan3D + FastAPI)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
".venv\Scripts\python.exe" -m pip uninstall -y hf-xet >nul 2>&1

echo [3/4] Dang cai dat thu vien Giao dien (React + Electron)...
cd /d "%~dp0frontend"
call npm install
call npm run build

echo [4/4] Dang kiem tra / tai Model AI ve bo nho dem HuggingFace...
cd /d "%~dp0"
".venv\Scripts\python.exe" backend\download_models.py

echo ===================================================
echo   HOAN TAT! Ban co the chay Start-Mesh-Studio.bat
echo ===================================================
pause
