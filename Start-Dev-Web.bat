@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Starting Mesh Studio Backend on http://127.0.0.1:8765 ...
start "Mesh Backend" ".venv\Scripts\python.exe" backend\server.py
cd /d "%~dp0\frontend"
echo Starting Vite Dev Server on http://localhost:5173 ...
npm run dev -- --open
