@echo off
setlocal
cd /d D:\PyChram_project\CognitiveSemCom
title CognitiveSemCom E8 Visible Run

set PYTHON_EXE=D:\anconda\envs\pytorch\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo Starting CognitiveSemCom E8 visible run...
echo Project: %CD%
echo Python:  %PYTHON_EXE%
echo.
"%PYTHON_EXE%" "D:\PyChram_project\CognitiveSemCom\scripts\run_visible_e8.py" --trials 40

echo.
echo Window is staying open for inspection.
pause
