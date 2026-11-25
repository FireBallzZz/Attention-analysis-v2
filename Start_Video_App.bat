@echo off
title Video Analysis App
cd /d "%~dp0"
echo Starting the video analysis app on http://127.0.0.1:8080 ... keep this window open.
python launch_video_upload.py --port 8080
pause
