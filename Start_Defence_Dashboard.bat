@echo off
title Defence Dashboard
cd /d "%~dp0"
echo Opening the defence dashboard... keep this window open. Close it to stop.
python launch_defence_dashboard.py
pause
