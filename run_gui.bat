@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m eda_report.gui
if errorlevel 1 pause
