"""
Power BI exporter.

Keeps the existing CSV schemas used by the dashboard while exporting
results from the updated churn and revenue models.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("output/powerbi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def _status(value, target, better="higher"):
    """
    Return a dashboard-friendly status based on a target.
    """
    if better == "higher":
        return "Meets Target" if value >= target else "Below Target"
    else:
        return "Meets Target" if value <= target else "Above Target"


def _safe_float(value):
    if value is None:
        return np.nan

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return np.nan


# ============================================================
# CHURN PREDICTIONS
# ============================================================

def export_churn_predictions(
    predictions_df,
    output_path=None
):
    """
    Export customer churn predictions.

    IMPORTANT:
    Existing Power BI columns are preserved.
    """

    if output_path is None:
        output_path = OUTPUT_DIR / "churn_predictions_enriched.csv"

    df = predictions_df.copy()

    # Make sure probability is numeric
    if "churn_probability" in df.columns:
        df["churn_probability"] = pd.to_numeric(
            df["churn_probability"],
            errors="coerce"
        )

    # Make sure risk tier exists
    if "risk_tier" not in df.columns and "churn_probability" in df.columns:

        def get_risk(probability):

            if pd.isna(probability):
                return "Unknown"

            if probability < 0.40:
                return "Low Risk"

            elif probability < 0.70:
                return "Medium Risk"

            return "High Risk"

        df["risk_tier"] = df["churn_probability"].apply(get_risk)

    # Revenue at risk
    if "RevenueAtRisk" not in df.columns:

        if (
            "MonthlyCharges" in df.columns
            and "churn_probability" in df.columns
        ):

            df["RevenueAtRisk"] = (
                pd.to_numeric(
                    df["MonthlyCharges"],
                    errors="coerce"
                )
                * df["churn_probability"]
            )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def export_feature_importance(
    feature_importance,
    output_path=None
):
    """
    Export feature importance while preserving:
        feature
        importance
    """

    if output_path is None:
        output_path = OUTPUT_DIR / "feature_importance.csv"

    if isinstance(feature_importance, dict):

        df = pd.DataFrame(
            {
                "feature": list(feature_importance.keys()),
                "importance": list(feature_importance.values())
            }
        )

    elif isinstance(feature_importance, pd.Series):

        df = (
            feature_importance
            .rename("importance")
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    else:
        df = feature_importance.copy()

    df["importance"] = pd.to_numeric(
        df["importance"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# MODEL METRICS
# ============================================================

def export_model_metrics(
    metrics,
    output_path=None
):
    """
    Export model metrics using the existing Power BI schema:

        Model
        Metric
        Value
        CI_Lower
        CI_Upper
        Target_or_Baseline
        Status
    """

    if output_path is None:
        output_path = OUTPUT_DIR / "model_metrics.csv"

    rows = []

    # --------------------------------------------------------
    # Churn metrics
    # --------------------------------------------------------

    churn_metrics = {
        "AUC-ROC": {
            "target": 0.70,
            "better": "higher"
        },

        "F1-Score": {
            "target": 0.50,
            "better": "higher"
        },

        "Precision": {
            "target": 0.50,
            "better": "higher"
        },

        "Recall": {
            "target": 0.70,
            "better": "higher"
        },

        "Brier Score": {
            "target": 0.20,
            "better": "lower"
        },
    }

    for metric_name, config in churn_metrics.items():

        value = None

        # Allow several naming styles
        possible_names = [
            metric_name,
            metric_name.lower(),
            metric_name.replace("-", "_"),
            metric_name.lower().replace("-", "_"),
            metric_name.lower().replace(" ", "_"),
        ]

        for name in possible_names:

            if name in metrics:
                value = metrics[name]
                break

        if value is None:
            continue

        value = _safe_float(value)

        if np.isnan(value):
            continue

        rows.append(
            {
                "Model": "XGBoost + Logistic Ensemble",
                "Metric": metric_name,
                "Value": value,
                "CI_Lower": np.nan,
                "CI_Upper": np.nan,
                "Target_or_Baseline": config["target"],
                "Status": _status(
                    value,
                    config["target"],
                    config["better"]
                ),
            }
        )

    # --------------------------------------------------------
    # Revenue metrics
    # --------------------------------------------------------

    revenue_metrics = {
        "MAPE": {
            "target": 0.05,
            "better": "lower"
        },

        "RMSE": {
            "target": None,
            "better": "lower"
        },

        "Interval Coverage": {
            "target": 0.80,
            "better": "higher"
        },
    }

    for metric_name, config in revenue_metrics.items():

        value = None

        possible_names = [
            metric_name,
            metric_name.lower(),
            metric_name.replace(" ", "_"),
            metric_name.lower().replace(" ", "_"),
        ]

        for name in possible_names:

            if name in metrics:
                value = metrics[name]
                break

        if value is None:
            continue

        value = _safe_float(value)

        if np.isnan(value):
            continue

        target = config["target"]

        if target is None:
            status = "Reported"
        else:
            status = _status(
                value,
                target,
                config["better"]
            )

        rows.append(
            {
                "Model": "Ridge Trend + Seasonality",
                "Metric": metric_name,
                "Value": value,
                "CI_Lower": np.nan,
                "CI_Upper": np.nan,
                "Target_or_Baseline": target,
                "Status": status,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Metric",
            "Value",
            "CI_Lower",
            "CI_Upper",
            "Target_or_Baseline",
            "Status",
        ]
    )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# SALES FORECAST
# ============================================================

def export_sales_forecast(
    sales_df,
    output_path=None
):
    """
    Export the revenue forecast.

    Power BI columns remain:

        Month
        ActualRevenue
        PredictedRevenue
        LowerBound_80
        UpperBound_80
        PeriodType
    """

    if output_path is None:
        output_path = OUTPUT_DIR / "sales_forecast.csv"

    df = sales_df.copy()

    required_columns = [
        "Month",
        "ActualRevenue",
        "PredictedRevenue",
        "LowerBound_80",
        "UpperBound_80",
        "PeriodType",
    ]

    # Add missing columns without changing schema
    for column in required_columns:

        if column not in df.columns:
            df[column] = np.nan

    df = df[required_columns]

    # Normalize month
    df["Month"] = pd.to_datetime(
        df["Month"],
        errors="coerce"
    ).dt.strftime("%Y-%m")

    # Numeric fields
    numeric_columns = [
        "ActualRevenue",
        "PredictedRevenue",
        "LowerBound_80",
        "UpperBound_80",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Sort chronologically
    df["_sort_month"] = pd.to_datetime(
        df["Month"],
        format="%Y-%m",
        errors="coerce"
    )

    df = (
        df
        .sort_values("_sort_month")
        .drop(columns="_sort_month")
        .reset_index(drop=True)
    )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# DRIFT MONITORING
# ============================================================

def export_drift_monitoring(
    drift_df,
    output_path=None
):

    if output_path is None:
        output_path = OUTPUT_DIR / "drift_monitoring.csv"

    drift_df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ============================================================
# RETENTION LIFT
# ============================================================

def export_retention_lift(
    retention_df,
    output_path=None
):

    if output_path is None:
        output_path = (
            OUTPUT_DIR /
            "retention_lift_simulation.csv"
        )

    retention_df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ============================================================
# EXPORT EVERYTHING
# ============================================================

def export_powerbi_tables(
    churn_predictions=None,
    feature_importance=None,
    model_metrics=None,
    sales_forecast=None,
    drift_monitoring=None,
    retention_lift=None,
):
    """
    Export all Power BI datasets.

    Existing Power BI CSV schemas are preserved.
    """

    exported = {}

    if churn_predictions is not None:

        exported["churn_predictions"] = (
            export_churn_predictions(
                churn_predictions
            )
        )

    if feature_importance is not None:

        exported["feature_importance"] = (
            export_feature_importance(
                feature_importance
            )
        )

    if model_metrics is not None:

        exported["model_metrics"] = (
            export_model_metrics(
                model_metrics
            )
        )

    if sales_forecast is not None:

        exported["sales_forecast"] = (
            export_sales_forecast(
                sales_forecast
            )
        )

    if drift_monitoring is not None:

        exported["drift_monitoring"] = (
            export_drift_monitoring(
                drift_monitoring
            )
        )

    if retention_lift is not None:

        exported["retention_lift"] = (
            export_retention_lift(
                retention_lift
            )
        )

    return exported


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Power BI exporter module loaded.")
    print(f"Output directory: {OUTPUT_DIR}")