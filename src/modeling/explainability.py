"""
Feature importance and model explainability for the XGBoost response-time model.

Generates:
  1. XGBoost native feature importance bar chart
  2. SHAP summary plot (beeswarm)
  3. SHAP waterfall plot for a single prediction
  4. Top-N feature summary printed to console

Usage:
    from src.modeling.explainability import explain_model
    explain_model()                       # uses default paths
    explain_model(csv_path="data/processed/training_dataset.csv", n_samples=5000)
"""
from __future__ import annotations

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")


# Paths

MODEL_DIR         = "models"
MODEL_PATH        = os.path.join(MODEL_DIR, "xgb_response_time.joblib")
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.json")

PLOTS_DIR         = "models/explainability"


# Helpers

def _load_artefacts():
    for path in [MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artefact missing: {path}\n"
                "Train the model first with `python -m src.modeling.train`."
            )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def _sample_data(
    csv_path: str,
    feature_cols: list[str],
    scaler,
    n_samples: int,
    random_state: int,
) -> tuple[np.ndarray, list[str]]:
    """Load processed CSV, align columns, scale, and return a sample."""
    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  Filling {len(missing)} missing feature(s) with 0.")
        for col in missing:
            df[col] = 0.0

    df = df[feature_cols].fillna(0)

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state)
        print(f"  Sampled {n_samples:,} rows for SHAP computation.")
    else:
        print(f"  Using all {len(df):,} rows for SHAP computation.")

    X = scaler.transform(df.values.astype(np.float32))
    return X, list(df.columns)


def _clean_feature_name(name: str) -> str:
    """Make feature names more human-readable for plots."""
    mapping = {
        "dist_nearest_station_m": "Distance to Nearest Station (m)",
        "hydrants_within_500m": "Hydrants Within 500 m",
        "hour_of_day": "Hour of Day",
        "day_of_week": "Day of Week",
        "month": "Month",
        "year": "Year",
        "is_weekend": "Is Weekend",
        "response_time_seconds": "Response Time (s)",
        "PERSONS_RESCUED": "Persons Rescued",
        "EVENT_ALARM_LEVEL": "Alarm Level",
        "INCIDENT_STATION_AREA": "Station Area",
        "INCIDENT_WARD": "Ward",
    }
    return mapping.get(name, name.replace("_", " ").title())


# Plot 1: XGBoost native feature importance

def plot_xgb_importance(
    model,
    feature_cols: list[str],
    top_n: int = 25,
    save_path: str | None = None,
):
    """Bar chart of XGBoost's built-in 'gain' feature importance."""

    importances = model.feature_importances_
    feat_imp = (
        pd.Series(importances, index=feature_cols)
        .sort_values(ascending=False)
        .head(top_n)
    )

    labels = [_clean_feature_name(n) for n in feat_imp.index]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.95, len(feat_imp)))[::-1]
    bars = ax.barh(labels[::-1], feat_imp.values[::-1], color=colors[::-1], height=0.7)

    ax.set_xlabel("Feature Importance (Gain)", color="#cccccc", fontsize=11)
    ax.set_title(
        f"Top {top_n} Features — XGBoost Gain",
        color="white", fontsize=14, fontweight="bold", pad=14
    )
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="x", color="#2a2a2a", linewidth=0.8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return feat_imp


# Plot 2: SHAP summary (beeswarm)

def plot_shap_summary(
    model,
    X: np.ndarray,
    feature_cols: list[str],
    top_n: int = 20,
    save_path: str | None = None,
):
    """SHAP beeswarm summary plot showing impact of each feature."""
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP plots. "
              "Run: pip install shap")
        return None

    print("  Computing SHAP values (this may take a moment) …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Rank by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    X_top    = X[:, top_idx]
    sv_top   = shap_values[:, top_idx]
    labels   = [_clean_feature_name(feature_cols[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    shap.summary_plot(
        sv_top, X_top,
        feature_names=labels,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    ax = plt.gca()
    ax.set_facecolor("#0f1117")
    fig = plt.gcf()
    fig.patch.set_facecolor("#0f1117")
    ax.set_title(
        "SHAP Feature Impact on Predicted Response Time",
        color="white", fontsize=13, fontweight="bold", pad=12
    )
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return shap_values


# Plot 3: SHAP waterfall for a single prediction

def plot_shap_waterfall(
    model,
    X: np.ndarray,
    feature_cols: list[str],
    sample_idx: int = 0,
    save_path: str | None = None,
):
    """Waterfall plot showing how each feature pushes one prediction up/down."""
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping waterfall plot.")
        return

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    sv_single  = shap_values[sample_idx]
    base_value = explainer.expected_value

    # Keep only the top-15 features by absolute SHAP value for clarity
    top_idx  = np.argsort(np.abs(sv_single))[::-1][:15]
    sv_top   = sv_single[top_idx]
    labels   = [_clean_feature_name(feature_cols[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    cumulative = base_value
    y_positions = range(len(sv_top) - 1, -1, -1)

    for pos, (val, label) in zip(y_positions, zip(sv_top, labels)):
        color = "#ef4444" if val > 0 else "#22c55e"
        ax.barh(pos, val, left=cumulative, color=color, height=0.6, alpha=0.9)
        cumulative += val

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, color="#cccccc", fontsize=9)
    ax.axvline(base_value, color="#888888", linewidth=1, linestyle="--", alpha=0.7,
               label=f"Base value: {base_value:.2f} min")
    ax.axvline(cumulative, color="#facc15", linewidth=1.5, linestyle="-",
               label=f"Prediction: {cumulative:.2f} min")

    ax.set_xlabel("Predicted Response Time (minutes)", color="#cccccc", fontsize=11)
    ax.set_title(
        f"SHAP Waterfall — Sample #{sample_idx}",
        color="white", fontsize=13, fontweight="bold", pad=12
    )
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="#cccccc", fontsize=9)
    ax.grid(axis="x", color="#2a2a2a", linewidth=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")

    plt.close(fig)


# Console summary

def print_top_features(feat_imp: pd.Series, top_n: int = 15):
    print(f"\n{'─'*55}")
    print(f"  {'Rank':<6} {'Feature':<38} {'Importance':>9}")
    print(f"{'─'*55}")
    for rank, (feat, val) in enumerate(feat_imp.head(top_n).items(), 1):
        label = _clean_feature_name(feat)[:37]
        print(f"  {rank:<6} {label:<38} {val:>9.5f}")
    print(f"{'─'*55}\n")


# Main entry point

def explain_model(
    csv_path: str = "data/processed/training_dataset.csv",
    n_samples: int = 5000,
    top_n: int = 20,
    random_state: int = 42,
    waterfall_idx: int = 0,
):
    """
    Generate all explainability artefacts for the trained model.

    Parameters
    ----------
    csv_path      : Path to processed training CSV
    n_samples     : Max rows to use for SHAP computation (performance)
    top_n         : Number of top features to display in plots
    random_state  : Seed for sampling reproducibility
    waterfall_idx : Row index to use for the SHAP waterfall plot
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n=== Loading model artefacts ===")
    model, scaler, feature_cols = _load_artefacts()

    print("\n=== Preparing data sample ===")
    X, cols = _sample_data(csv_path, feature_cols, scaler, n_samples, random_state)

    # Plot 1: XGBoost native importance
    print("\n=== XGBoost native feature importance ===")
    feat_imp = plot_xgb_importance(
        model, cols, top_n=top_n,
        save_path=os.path.join(PLOTS_DIR, "xgb_feature_importance.png")
    )
    print_top_features(feat_imp)

    # Plot 2: SHAP summary
    print("=== SHAP summary plot ===")
    plot_shap_summary(
        model, X, cols, top_n=top_n,
        save_path=os.path.join(PLOTS_DIR, "shap_summary.png")
    )

    # Plot 3: SHAP waterfall
    print("=== SHAP waterfall plot ===")
    plot_shap_waterfall(
        model, X, cols, sample_idx=waterfall_idx,
        save_path=os.path.join(PLOTS_DIR, "shap_waterfall.png")
    )

    print(f"\n=== Done — plots saved to: {PLOTS_DIR}/ ===\n")


# CLI

if __name__ == "__main__":
    explain_model()