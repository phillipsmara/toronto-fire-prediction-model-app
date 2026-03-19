from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from src.config import MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH, TRAINING_CSV, EXPLAINABILITY_DIR

@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    import joblib
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

@st.cache_data(show_spinner="Loading dataset sample…")
def _load_sample(n: int = 5000) -> pd.DataFrame | None:
    if not os.path.exists(TRAINING_CSV):
        return None
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    return df.sample(n=n, random_state=42) if len(df) > n else df

def _metrics_section(model, scaler, feature_cols):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = _load_sample(n=20000)
    if df is None or "response_time_seconds" not in df.columns:
        st.warning("training_dataset.csv not found — cannot compute metrics.")
        return

    target = (df["response_time_seconds"] / 60.0).clip(0, 120)
    feat   = [c for c in feature_cols if c in df.columns]
    X      = df[feat].fillna(0).values.astype(np.float32)
    y      = target.values.astype(np.float32)
    mask   = ~np.isnan(y)
    X, y   = X[mask], y[mask]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_s = scaler.transform(X_test)
    y_pred   = np.clip(model.predict(X_test_s), 0, None)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    w1   = np.mean(np.abs(y_test - y_pred) <= 1.0) * 100
    w2   = np.mean(np.abs(y_test - y_pred) <= 2.0) * 100

    st.markdown("### 📈 Model Performance (hold-out test set)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{mae:.2f} min")
    c2.metric("RMSE", f"{rmse:.2f} min")
    c3.metric("R²", f"{r2:.3f}")
    c4.metric("Within ±1 min", f"{w1:.1f}%")
    c5.metric("Within ±2 min", f"{w2:.1f}%")

def _feature_importance_section(model, feature_cols):
    import plotly.graph_objects as go

    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
    labels   = feat_imp.index.tolist()[::-1]
    values   = feat_imp.values[::-1]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker=dict(color=values, colorscale="YlOrRd", showscale=False)))
    fig.update_layout(title="Top 20 Features by XGBoost Gain", xaxis_title="Importance (Gain)",
                      margin=dict(l=10, r=10, t=40, b=10), height=520, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def _response_dist_section():
    import plotly.graph_objects as go

    df = _load_sample(n=20000)
    if df is None or "response_time_seconds" not in df.columns:
        return

    mins = (df["response_time_seconds"] / 60.0).clip(0, 30)
    fig  = go.Figure(go.Histogram(x=mins, nbinsx=60, marker_color="#f97316", opacity=0.85))
    fig.add_vline(x=4, line_dash="dash", line_color="#facc15",
                  annotation_text="4 min target", annotation_position="top right")
    fig.update_layout(title="Distribution of Historical Response Times (capped at 30 min)",
                      xaxis_title="Response Time (minutes)", yaxis_title="Incidents",
                      template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

def _shap_images_section():
    shap_summary   = os.path.join(EXPLAINABILITY_DIR, "shap_summary.png")
    shap_waterfall = os.path.join(EXPLAINABILITY_DIR, "shap_waterfall.png")
    xgb_imp        = os.path.join(EXPLAINABILITY_DIR, "xgb_feature_importance.png")

    if not any(os.path.exists(p) for p in [shap_summary, shap_waterfall, xgb_imp]):
        st.info("No SHAP plots found. Generate them with:\n```\npython main.py --skip-pipeline --explain\n```")
        return

    st.markdown("### 🔬 SHAP Explainability")
    tabs  = []
    paths = []
    if os.path.exists(shap_summary):   tabs.append("SHAP Summary");        paths.append(shap_summary)
    if os.path.exists(shap_waterfall): tabs.append("SHAP Waterfall");       paths.append(shap_waterfall)
    if os.path.exists(xgb_imp):        tabs.append("XGBoost Importance");   paths.append(xgb_imp)

    for tab, path in zip(st.tabs(tabs), paths):
        with tab:
            st.image(path, use_container_width=True)

def render_dashboard():
    st.title("📊 Model Dashboard")
    st.markdown("Overview of model performance, feature importance, and historical response time distribution.")

    model, scaler, feature_cols = _load_model()

    if model is None:
        st.error("**No trained model found.**\n\nRun: `python main.py --skip-pipeline --train --no-tune`")
        return

    st.markdown("---")
    _metrics_section(model, scaler, feature_cols)
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("### 🏆 Feature Importance")
        _feature_importance_section(model, feature_cols)
    with right:
        st.markdown("### 🕐 Response Time Distribution")
        _response_dist_section()

    st.markdown("---")
    _shap_images_section()

    with st.expander("ℹ️ Model info"):
        st.json({
            "model_type": type(model).__name__,
            "n_estimators": int(model.n_estimators),
            "max_depth": int(model.max_depth),
            "learning_rate": float(model.learning_rate),
            "n_features": len(feature_cols),
            "model_path": MODEL_PATH,
        })