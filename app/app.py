import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.config import APP_TITLE, APP_ICON, APP_LAYOUT

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT, initial_sidebar_state="expanded")

from app.pages.dashboard import render_dashboard
from app.pages.predict import render_predict

with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.markdown("---")
    page = st.radio("Navigate", ["🔮 Predict Response Time", "📊 Model Dashboard"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Toronto Fire Services · Open Data")
    st.caption("Model: XGBoost Regressor")

if page == "🔮 Predict Response Time":
    render_predict()
else:
    render_dashboard()