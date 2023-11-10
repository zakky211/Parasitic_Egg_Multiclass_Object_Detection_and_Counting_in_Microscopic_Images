@echo off
echo Please wait...
echo Do not close this window
call conda activate streamlit
call streamlit run app.py
@pause