@echo off
echo ========================================================
echo   Campus Placement Predictor - Interactive Dashboard
echo   Lazyskool (YuvaBot Pvt. Ltd.) ^| Google Antigravity
echo ========================================================
echo.

IF EXIST ..\valorant-performance-intelligence\.venv\Scripts\streamlit.exe (
    ..\valorant-performance-intelligence\.venv\Scripts\streamlit.exe run src/app.py
) ELSE IF EXIST .venv\Scripts\streamlit.exe (
    .venv\Scripts\streamlit.exe run src/app.py
) ELSE (
    streamlit run src/app.py
)

pause
