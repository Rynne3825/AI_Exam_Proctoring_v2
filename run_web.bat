@echo off
title AI Exam Proctoring - Samsung SIC Demo Web
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" http://127.0.0.1:8000
python web_app.py
pause
