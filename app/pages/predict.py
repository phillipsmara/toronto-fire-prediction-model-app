from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from src.config import (
    MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH,
    INCIDENT_TYPES, CALL_SOURCES, ALARM_LEVELS,
    TORONTO_WARDS, STATION_AREAS,
)

@st.cache_resource(show_spinner="Loading model…")
def _load_artefacts():
    import joblib
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

def _build_input_row(hour, day_of_week, month, is_weekend, incident_type, call_source,
                     alarm_level, ward, station_area, dist_station_m, hydrants_500m, feature_cols):
    row = {col: 0.0 for col in feature_cols}
    row["hour_of_day"]            = float(hour)
    row["day_of_week"]            = float(day_of_week)
    row["month"]                  = float(month)
    row["is_weekend"]             = float(is_weekend)
    row["dist_nearest_station_m"] = float(dist_station_m)
    row["hydrants_within_500m"]   = float(hydrants_500m)
    try:
        row["INCIDENT_WARD"]         = float(ward)
        row["INCIDENT_STATION_AREA"] = float(station_area)
    except ValueError:
        pass

    def _set_ohe(prefix, value):
        col = f"{prefix}_{value.upper().replace(' ', '_').replace('-', '_')}"
        if col in row:
            row[col] = 1.0

    _set_ohe("FINAL_INCIDENT_TYPE", incident_type)
    _set_ohe("CALL_SOURCE", call_source)
    _set_ohe("EVENT_ALARM_LEVEL", alarm_level)
    return np.array([list(row.values())], dtype=np.float32)

def render_predict():
    st.title("🔮 Predict Response Time")
    st.markdown("Adjust the incident parameters below to get an estimated fire response time from Toronto Fire Services.")

    model, scaler, feature_cols = _load_artefacts()

    if model is None:
        st.error("**No trained model found.**\n\nRun: `python main.py --skip-pipeline --train --no-tune`")
        return

    st.success("✅ Model loaded — ready to predict.", icon="🤖")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⏰ Time & Date")
        hour = st.slider("Hour of day", 0, 23, 14, help="0 = midnight, 12 = noon")
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_label = st.selectbox("Day of week", day_names, index=2)
        day_of_week = day_names.index(day_label)
        is_weekend = int(day_of_week >= 5)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_label = st.selectbox("Month", month_names, index=5)
        month = month_names.index(month_label) + 1

    with col2:
        st.subheader("🚨 Incident Details")
        incident_type = st.selectbox("Incident type", INCIDENT_TYPES)
        call_source   = st.selectbox("Call source", CALL_SOURCES)
        alarm_level   = st.selectbox("Alarm level", ALARM_LEVELS)
        ward          = st.selectbox("Ward", TORONTO_WARDS, index=0)
        station_area  = st.selectbox("Station area", STATION_AREAS, index=0)

    with col3:
        st.subheader("📍 Location Factors")
        dist_station_m = st.number_input("Distance to nearest station (m)", min_value=0, max_value=20000, value=1500, step=100)
        hydrants_500m  = st.number_input("Hydrants within 500 m", min_value=0, max_value=50, value=5)

    st.markdown("---")

    if st.button("⚡ Predict Response Time", type="primary", use_container_width=True):
        X = _build_input_row(
            hour=hour, day_of_week=day_of_week, month=month, is_weekend=is_weekend,
            incident_type=incident_type, call_source=call_source, alarm_level=alarm_level,
            ward=ward, station_area=station_area, dist_station_m=dist_station_m,
            hydrants_500m=hydrants_500m, feature_cols=feature_cols,
        )
        X_scaled = scaler.transform(X)
        pred_min = float(np.clip(model.predict(X_scaled)[0], 0, None))
        pred_sec = pred_min * 60

        st.markdown("### 🏁 Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Estimated Response Time", f"{pred_min:.1f} min")
        r2.metric("In Seconds", f"{pred_sec:.0f} s")

        if pred_min <= 4:
            label = "🟢 Excellent (within target)"
        elif pred_min <= 6:
            label = "🟡 Good"
        elif pred_min <= 10:
            label = "🟠 Moderate"
        else:
            label = "🔴 Slow — review contributing factors"

        r3.metric("Assessment", label)
        st.markdown("---")
        st.info("**Note:** Toronto Fire Services targets a 4-minute response time for Priority 1 calls. This prediction is based on historical incident data and spatial features.")

        with st.expander("📋 Input summary"):
            st.json({
                "Hour of day": hour, "Day of week": day_label, "Month": month_label,
                "Is weekend": bool(is_weekend), "Incident type": incident_type,
                "Call source": call_source, "Alarm level": alarm_level,
                "Ward": ward, "Station area": station_area,
                "Distance to station (m)": dist_station_m, "Hydrants within 500 m": hydrants_500m,
            })