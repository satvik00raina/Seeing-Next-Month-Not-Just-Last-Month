"""
Enterprise Predictive Analytics Pipeline
----------------------------------------

Runs the complete ML pipeline and exports the results
for Power BI.

Existing Power BI filenames and schemas are preserved.
"""
from src.data_generator import generate_customer_data, generate_monthly_revenue
from pathlib import Path
import pandas as pd
import sys



# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"
POWERBI_DIR = OUTPUT_DIR / "powerbi"

# Make sure Python can find src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ML PIPELINE
# ============================================================

from src.ml_pipeline import run_full_pipeline


# ============================================================
# POWER BI EXPORTER
# ============================================================

from src.powerbi_exporter import (
    export_churn_predictions,
    export_feature_importance,
    export_model_metrics,
    export_sales_forecast,
    export_drift_monitoring,
    export_retention_lift,
)


# ============================================================
# HELPERS
# ============================================================

def ensure_directories():
    """Create required output directories."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    POWERBI_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def save_csv(df, path):
    """
    Save a DataFrame safely to CSV.

    This function is local to main.py so that the exporter
    does not need a safe_to_csv dependency.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        df.to_csv(
            path,
            index=False
        )

        return True

    except PermissionError:

        print()
        print("=" * 70)
        print("ERROR: Could not write CSV")
        print("=" * 70)
        print(f"File: {path}")
        print()
        print(
            "The file is probably open in Power BI, Excel, "
            "or another program."
        )
        print(
            "Close the file and run the pipeline again."
        )
        print("=" * 70)

        return False


# ============================================================
# POWER BI EXPORT
# ============================================================

def export_powerbi_results(results):
    """
    Export all ML results into the existing Power BI CSV files.

    Existing filenames and column names are preserved.
    """

    print()
    print("=" * 70)
    print("EXPORTING POWER BI DATASETS")
    print("=" * 70)

    POWERBI_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    exported_files = []

    # --------------------------------------------------------
    # 1. CHURN PREDICTIONS
    # --------------------------------------------------------

    churn_result = results.get("churn", {})

    churn_predictions = churn_result.get(
        "predictions"
    )

    if churn_predictions is not None:

        path = export_churn_predictions(
            churn_predictions,
            POWERBI_DIR / "churn_predictions_enriched.csv"
        )

        exported_files.append(path)

        print(
            "✓ churn_predictions_enriched.csv"
        )

    else:

        print(
            "⚠ Churn predictions were not available."
        )

    # --------------------------------------------------------
    # 2. FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = churn_result.get(
        "feature_importance"
    )

    if feature_importance is not None:

        path = export_feature_importance(
            feature_importance,
            POWERBI_DIR / "feature_importance.csv"
        )

        exported_files.append(path)

        print(
            "✓ feature_importance.csv"
        )

    else:

        print(
            "⚠ Feature importance was not available."
        )

    # --------------------------------------------------------
    # 3. MODEL METRICS
    # --------------------------------------------------------

    churn_metrics = churn_result.get(
        "metrics",
        {}
    )

    sales_result = results.get(
        "sales",
        {}
    )

    sales_metrics = sales_result.get(
        "metrics",
        {}
    )

    # Combine metrics for the existing exporter.
    combined_metrics = {}

    if isinstance(churn_metrics, dict):
        combined_metrics.update(
            churn_metrics
        )

    if isinstance(sales_metrics, dict):
        combined_metrics.update(
            sales_metrics
        )

    path = export_model_metrics(
        combined_metrics,
        POWERBI_DIR / "model_metrics.csv"
    )

    exported_files.append(path)

    print(
        "✓ model_metrics.csv"
    )

    # --------------------------------------------------------
    # 4. SALES FORECAST
    # --------------------------------------------------------

    sales_forecast = build_sales_forecast_dataframe(
        sales_result
    )

    if sales_forecast is not None:

        path = export_sales_forecast(
            sales_forecast,
            POWERBI_DIR / "sales_forecast.csv"
        )

        exported_files.append(path)

        print(
            "✓ sales_forecast.csv"
        )

    else:

        print(
            "⚠ Sales forecast was not available."
        )

    # --------------------------------------------------------
    # 5. DRIFT MONITORING
    # --------------------------------------------------------

    drift_result = results.get(
        "drift",
        {}
    )

    drift_df = None

    if isinstance(drift_result, pd.DataFrame):

        drift_df = drift_result

    elif isinstance(drift_result, dict):

        drift_df = drift_result.get(
            "data"
        )

        if drift_df is None:
            drift_df = drift_result.get(
                "df"
            )

    if drift_df is not None:

        path = export_drift_monitoring(
            drift_df,
            POWERBI_DIR / "drift_monitoring.csv"
        )

        exported_files.append(path)

        print(
            "✓ drift_monitoring.csv"
        )

    else:

        print(
            "⚠ Drift monitoring data was not available."
        )

    # --------------------------------------------------------
    # 6. RETENTION LIFT
    # --------------------------------------------------------

    retention_result = results.get(
        "retention_lift"
    )

    if retention_result is None:

        retention_result = results.get(
            "retention"
        )

    retention_df = None

    if isinstance(retention_result, pd.DataFrame):

        retention_df = retention_result

    elif isinstance(retention_result, dict):

        retention_df = retention_result.get(
            "data"
        )

        if retention_df is None:
            retention_df = retention_result.get(
                "df"
            )

    if retention_df is not None:

        path = export_retention_lift(
            retention_df,
            POWERBI_DIR / "retention_lift_simulation.csv"
        )

        exported_files.append(path)

        print(
            "✓ retention_lift_simulation.csv"
        )

    else:

        print(
            "⚠ Retention lift data was not available."
        )

    print()
    print(
        f"Power BI: {len(exported_files)} tables generated"
    )

    return exported_files


# ============================================================
# SALES FORECAST CONVERSION
# ============================================================

def build_sales_forecast_dataframe(sales_result):
    """
    Convert the new Ridge revenue model output into the
    existing Power BI sales_forecast schema.

    Existing Power BI columns:

        Month
        ActualRevenue
        PredictedRevenue
        LowerBound_80
        UpperBound_80
        PeriodType
    """

    if not isinstance(
        sales_result,
        dict
    ):
        return None

    rows = []

    # --------------------------------------------------------
    # Actual / backtest predictions
    # --------------------------------------------------------

    actuals = sales_result.get(
        "actuals",
        {}
    )

    test_predictions = sales_result.get(
        "test_predictions",
        {}
    )

    actual_months = actuals.get(
        "months",
        []
    )

    actual_revenue = actuals.get(
        "revenue",
        []
    )

    test_months = test_predictions.get(
        "months",
        []
    )

    predicted_revenue = test_predictions.get(
        "revenue",
        []
    )

    lower_bound = test_predictions.get(
        "lower_bound",
        []
    )

    upper_bound = test_predictions.get(
        "upper_bound",
        []
    )

    # Use the test prediction months as the primary
    # chronological backtest period.

    if len(test_months) > 0:

        for i, month in enumerate(
            test_months
        ):

            actual_value = None

            # Find matching actual month.
            for j, actual_month in enumerate(
                actual_months
            ):

                if str(actual_month)[:7] == str(month)[:7]:

                    if j < len(actual_revenue):
                        actual_value = actual_revenue[j]

                    break

            predicted_value = (
                predicted_revenue[i]
                if i < len(predicted_revenue)
                else None
            )

            lower_value = (
                lower_bound[i]
                if i < len(lower_bound)
                else None
            )

            upper_value = (
                upper_bound[i]
                if i < len(upper_bound)
                else None
            )

            rows.append(
                {
                    "Month": month,
                    "ActualRevenue": actual_value,
                    "PredictedRevenue": predicted_value,
                    "LowerBound_80": lower_value,
                    "UpperBound_80": upper_value,
                    "PeriodType": "Backtest Holdout",
                }
            )

    # --------------------------------------------------------
    # Next month forecast
    # --------------------------------------------------------

    next_forecast = sales_result.get(
        "next_month_forecast"
    )

    if isinstance(
        next_forecast,
        dict
    ):

        forecast_month = next_forecast.get(
            "month"
        )

        forecast_revenue = next_forecast.get(
            "revenue"
        )

        forecast_lower = next_forecast.get(
            "lower_bound"
        )

        forecast_upper = next_forecast.get(
            "upper_bound"
        )

        if forecast_month is not None:

            rows.append(
                {
                    "Month": forecast_month,
                    "ActualRevenue": None,
                    "PredictedRevenue": forecast_revenue,
                    "LowerBound_80": forecast_lower,
                    "UpperBound_80": forecast_upper,
                    "PeriodType": "Next Month Forecast",
                }
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if len(rows) == 0:

        return None

    df = pd.DataFrame(
        rows,
        columns=[
            "Month",
            "ActualRevenue",
            "PredictedRevenue",
            "LowerBound_80",
            "UpperBound_80",
            "PeriodType",
        ]
    )

    return df


# ============================================================
# PRINT MODEL RESULTS
# ============================================================

def print_model_summary(results):
    """
    Print the most important model metrics.
    """

    print()
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    # --------------------------------------------------------
    # Churn
    # --------------------------------------------------------

    churn = results.get(
        "churn",
        {}
    )

    metrics = churn.get(
        "metrics",
        {}
    )

    if metrics:

        print()
        print("CHURN MODEL")
        print("-" * 40)

        print(
            "Model: XGBoost + Logistic Ensemble"
        )

        if "accuracy" in metrics:
            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

        if "auc" in metrics:
            print(
                f"AUC: "
                f"{metrics['auc']:.4f}"
            )

        if "precision" in metrics:
            print(
                f"Precision: "
                f"{metrics['precision']:.4f}"
            )

        if "recall" in metrics:
            print(
                f"Recall: "
                f"{metrics['recall']:.4f}"
            )

        if "f1" in metrics:
            print(
                f"F1: "
                f"{metrics['f1']:.4f}"
            )

        if "brier_score" in metrics:
            print(
                f"Brier Score: "
                f"{metrics['brier_score']:.4f}"
            )

        if "decision_threshold" in metrics:
            print(
                f"Threshold: "
                f"{metrics['decision_threshold']:.4f}"
            )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    sales = results.get(
        "sales",
        {}
    )

    sales_metrics = sales.get(
        "metrics",
        {}
    )

    if sales_metrics:

        print()
        print("REVENUE MODEL")
        print("-" * 40)

        print(
            "Model: Ridge Trend + Seasonality"
        )

        if "rmse" in sales_metrics:
            print(
                f"RMSE: "
                f"${sales_metrics['rmse']:,.2f}"
            )

        if "mape" in sales_metrics:
            print(
                f"MAPE: "
                f"{sales_metrics['mape']:.2f}%"
            )

        if "r2" in sales_metrics:
            print(
                f"R²: "
                f"{sales_metrics['r2']:.4f}"
            )

        if "interval_coverage" in sales_metrics:
            print(
                f"Prediction interval coverage: "
                f"{sales_metrics['interval_coverage']:.1%}"
            )

    # --------------------------------------------------------
    # Next month
    # --------------------------------------------------------

    next_forecast = sales.get(
        "next_month_forecast"
    )

    if isinstance(
        next_forecast,
        dict
    ):

        month = next_forecast.get(
            "month"
        )

        revenue = next_forecast.get(
            "revenue"
        )

        if month is not None and revenue is not None:

            print()
            print(
                f"Next month ({month}): "
                f"${revenue:,.2f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PREDICTIVE DASHBOARD — ENTERPRISE EDITION")
    print("=" * 70)

    ensure_directories()

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Power BI output: {POWERBI_DIR}"
    )

    # --------------------------------------------------------
    # RUN ML PIPELINE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING ML PIPELINE")
    print("=" * 70)

    customer_df = generate_customer_data()
    revenue_df = generate_monthly_revenue()

    results = run_full_pipeline(
        customer_df,
        revenue_df
    )

    if results is None:

        print()
        print(
            "ERROR: run_full_pipeline() returned None."
        )

        return 1

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    print_model_summary(
        results
    )

    # --------------------------------------------------------
    # POWER BI EXPORT
    # --------------------------------------------------------

    try:

        exported_files = export_powerbi_results(
            results
        )

    except PermissionError as exc:

        print()
        print(
            "ERROR: A Power BI/CSV file is locked."
        )
        print(exc)

        return 1

    except Exception as exc:

        print()
        print("=" * 70)
        print("POWER BI EXPORT ERROR")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Power BI tables generated: "
        f"{len(exported_files)}"
    )

    print(
        f"Output directory: "
        f"{POWERBI_DIR}"
    )

    print()

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )