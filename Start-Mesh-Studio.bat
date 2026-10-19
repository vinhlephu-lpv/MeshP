@echo off
chcp 65001 >nul
cd /d "%~dp0\frontend"
echo Starting Mesh Studio Desktop App...
start "" "%~dp0frontend\node_modules\electron\dist\electron.exe" "%~dp0frontend"
