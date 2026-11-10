#!/usr/bin/env bash
echo "========================================================"
echo "  Campus Placement Predictor - Interactive Dashboard"
echo "  Lazyskool (YuvaBot Pvt. Ltd.) | Google Antigravity"
echo "========================================================"
echo ""

if [ -f "../valorant-performance-intelligence/.venv/bin/streamlit" ]; then
    ../valorant-performance-intelligence/.venv/bin/streamlit run src/app.py
elif [ -f ".venv/bin/streamlit" ]; then
    .venv/bin/streamlit run src/app.py
else
    streamlit run src/app.py
fi
