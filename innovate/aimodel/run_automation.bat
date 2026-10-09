@echo off
TITLE Company Predictive Dashboard - Automated Model Runner
echo =================================================================
echo   🏢 COMPANY PREDICTIVE SERVICE - AUTOMATED MODEL RUNNER
echo =================================================================
echo.

:: 1. Run Python Prediction Pipeline
echo [1/2] Running Machine Learning Pipeline on Input CSVs...
.venv\Scripts\python predict.py %*

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Pipeline execution failed. Please check input CSV format.
    pause
    exit /b %ERRORLEVEL%
)

:: 2. Launch / Refresh Power BI Desktop
echo.
echo [2/2] Launching Power BI Desktop...
start "" "C:\Users\mohit\AppData\Local\Microsoft\WindowsApps\PBIDesktopStore.exe"

echo.
echo =================================================================
echo   ✅ AUTOMATION COMPLETE!
echo   • Power BI datasets refreshed in 'output/powerbi/'
echo   • Click 'Refresh' in Power BI to view updated dashboard.
echo =================================================================
echo.
pause
