import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import json
import numpy as np
import pandas as pd
import streamlit as st
from src.config import (
    MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH,
    INCIDENT_TYPES, CALL_SOURCES, ALARM_LEVELS,
    TORONTO_WARDS, STATION_AREAS, APP_TITLE, APP_ICON, APP_LAYOUT,
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import joblib
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

def build_input(hour, day_of_week, month, is_weekend, incident_type, call_source,
                alarm_level, ward, station_area, dist_station_m, hydrants_500m, feature_cols):
    row = pd.Series(0.0, index=feature_cols)

    def _set(col, val):
        if col in row.index:
            row[col] = float(val)

    _set("hour_of_day", hour)
    _set("day_of_week", day_of_week)
    _set("month", month)
    _set("is_weekend", is_weekend)
    _set("dist_nearest_station_m", dist_station_m)
    _set("hydrants_within_500m", hydrants_500m)

    try:
        _set("INCIDENT_WARD", float(ward))
        _set("INCIDENT_STATION_AREA", float(station_area))
    except ValueError:
        pass

    def _ohe(prefix, value):
        _set(f"{prefix}_{value.upper().replace(' ', '_').replace('-', '_')}", 1.0)

    _ohe("FINAL_INCIDENT_TYPE", incident_type)
    _ohe("CALL_SOURCE", call_source)
    _ohe("EVENT_ALARM_LEVEL", alarm_level)

    return row.values.reshape(1, -1).astype(np.float32)

model, scaler, feature_cols = load_model()

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown("---")

if model is None:
    st.error("No trained model found. Run: `python main.py --skip-pipeline --train --no-tune`")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⏰ Time")
    hour = st.slider("Hour of day", 0, 23, 14)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_label = st.selectbox("Day of week", day_names, index=2)
    day_of_week = day_names.index(day_label)
    is_weekend = int(day_of_week >= 5)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_label = st.selectbox("Month", month_names, index=5)
    month = month_names.index(month_label) + 1

with col2:
    st.subheader("🚨 Incident")
    incident_type = st.selectbox("Incident type", INCIDENT_TYPES)
    call_source   = st.selectbox("Call source", CALL_SOURCES)
    alarm_level   = st.selectbox("Alarm level", ALARM_LEVELS)
    ward          = st.selectbox("Ward", TORONTO_WARDS, index=0)
    station_area  = st.selectbox("Station area", STATION_AREAS, index=0)

with col3:
    st.subheader("📍 Location")
    dist_station_m = st.number_input("Distance to nearest station (m)", min_value=0, max_value=20000, value=1500, step=100)
    hydrants_500m  = st.number_input("Hydrants within 500 m", min_value=0, max_value=50, value=5)

st.markdown("---")

if st.button("⚡ Predict Response Time", type="primary", use_container_width=True):
    X        = build_input(hour, day_of_week, month, is_weekend, incident_type, call_source,
                           alarm_level, ward, station_area, dist_station_m, hydrants_500m, feature_cols)
    pred_min = float(np.clip(model.predict(scaler.transform(X))[0], 0, None))

    st.markdown("### Result")
    st.metric("Estimated Response Time", f"{pred_min:.1f} min", f"{pred_min * 60:.0f} seconds")

    if pred_min <= 4:
        st.success("🟢 Within the 4-minute target")
    elif pred_min <= 6:
        st.warning("🟡 Slightly above target")
    elif pred_min <= 10:
        st.warning("🟠 Moderately above target")
    else:
        st.error("🔴 Significantly above target")