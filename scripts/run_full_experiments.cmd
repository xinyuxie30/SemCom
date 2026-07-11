@echo off
setlocal
cd /d D:\PyChram_project\CognitiveSemCom
title CognitiveSemCom Full Experiments E1-E8

set PYTHON_EXE=D:\anconda\envs\pytorch\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo Starting CognitiveSemCom full experiment suite E1-E8...
echo Project: %CD%
echo Python:  %PYTHON_EXE%
echo.
"%PYTHON_EXE%" "D:\PyChram_project\CognitiveSemCom\scripts\run_full_experiments.py"

echo.
echo Window is staying open for inspection.
pause
