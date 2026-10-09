"""
Flask Web Frontend — Enterprise Churn & Sales Prediction Platform
=================================================================
Run:  .venv\\Scripts\\python app.py
Then: open http://127.0.0.1:5000 in your browser.
"""

import io
import json
import os
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import validate_customer_df, validate_sales_df
from src.ml_pipeline import run_full_pipeline
from src.dashboard import generate_dashboard
from src.powerbi_exporter import export_powerbi_tables

# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "churn-platform-secret-key-2025"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "output", "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Store last run results in memory (single-user desktop tool)
_last_results = {}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _safe_float(v):
    """Convert numpy/pandas scalar to plain Python float."""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def _build_confusion_matrix_data(results):
    """Return [[TN, FP], [FN, TP]] as plain list from results dict."""
    cm = results["churn"].get("confusion_matrix", None)
    if cm is None:
        return None
    if hasattr(cm, "tolist"):
        return cm.tolist()
    return list(cm)


def _build_roc_data(results):
    """Return {fpr: [...], tpr: [...], auc: float} for Plotly."""
    roc = results["churn"].get("roc_curve", {})
    if not roc:
        return None
    return {
        "fpr": [_safe_float(x) for x in roc.get("fpr", [])],
        "tpr": [_safe_float(x) for x in roc.get("tpr", [])],
        "auc": _safe_float(results["churn"]["metrics"].get("auc")),
    }


def _build_calibration_data(results):
    """Return {prob_pred: [...], prob_true: [...]} for Plotly."""
    cal = results["churn"].get("calibration", {})
    if not cal:
        return None
    return {
        # ml_pipeline.py uses prob_true / prob_pred keys
        "prob_pred": [_safe_float(x) for x in cal.get("prob_pred", [])],
        "prob_true": [_safe_float(x) for x in cal.get("prob_true", [])],
    }


def _build_forecast_data(results):
    """Return chart-ready forecast data from ml_pipeline sales results."""
    actuals = results["sales"].get("actuals", {})         # all months + revenue
    train_pred = results["sales"].get("train_predictions", {})
    test_pred = results["sales"].get("test_predictions", {})

    # Historical = full actuals series for the line
    hist_months = actuals.get("months", [])
    hist_revenue = [_safe_float(x) for x in actuals.get("revenue", [])]

    # Forecast = test period with interval bands
    fc_months = test_pred.get("months", [])
    fc_revenue = [_safe_float(x) for x in test_pred.get("revenue", [])]
    lower = [_safe_float(x) for x in test_pred.get("lower_bound", [])]
    upper = [_safe_float(x) for x in test_pred.get("upper_bound", [])]

    # Backtest: actuals for the test months (from the actuals dict)
    n_test = len(fc_months)
    test_actual_vals = hist_revenue[-n_test:] if n_test > 0 else []

    return {
        "hist_months": hist_months[:-n_test] if n_test > 0 else hist_months,
        "hist_revenue": hist_revenue[:-n_test] if n_test > 0 else hist_revenue,
        "forecast_months": fc_months,
        "forecast_revenue": fc_revenue,
        "lower": lower,
        "upper": upper,
        # Holdout backtest: actual vs predicted on test months
        "test_months": fc_months,
        "test_actual": test_actual_vals,
        "test_predicted": fc_revenue,
    }


def _build_drift_data(results):
    """Return list of {feature, psi, ks, status} for the drift table."""
    rows = []
    for feat, info in results["drift"].get("feature_details", {}).items():
        psi = _safe_float(info.get("psi"))
        ks = _safe_float(info.get("ks_stat"))
        if psi is None:
            continue
        if psi >= 0.2:
            status = "HIGH"
        elif psi >= 0.1:
            status = "MEDIUM"
        else:
            status = "LOW"
        rows.append({"feature": feat, "psi": round(psi, 4),
                     "ks": round(ks, 4) if ks else "-", "status": status})
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return rows


def _build_leakage_data(results):
    return results["leakage"].get("checks", [])


def _serialize_results(results):
    """Build a serializable dict for the template / JSON endpoint."""
    m = results["churn"]["metrics"]
    ci = results["churn"].get("bootstrap_ci", {})
    sm = results["sales"]["metrics"]
    meta = results["metadata"]

    def _ci(key):
        block = ci.get(key, {})
        lo = _safe_float(block.get("ci_lower"))
        hi = _safe_float(block.get("ci_upper"))
        if lo is None or hi is None:
            return "N/A"
        return f"[{lo:.4f} – {hi:.4f}]"

    churn_metrics = {
        "AUC":           {"value": _safe_float(m.get("auc")),         "ci": _ci("auc"),       "fmt": ".4f"},
        "F1 Score":      {"value": _safe_float(m.get("f1")),          "ci": _ci("f1"),        "fmt": ".4f"},
        "Precision":     {"value": _safe_float(m.get("precision")),   "ci": _ci("precision"), "fmt": ".4f"},
        "Recall":        {"value": _safe_float(m.get("recall")),      "ci": _ci("recall"),    "fmt": ".4f"},
        "Accuracy":      {"value": _safe_float(m.get("accuracy")),    "ci": "N/A",            "fmt": ".4f"},
        "Brier Score":   {"value": _safe_float(m.get("brier_score")), "ci": "N/A",            "fmt": ".4f"},
        "Baseline AUC":  {"value": _safe_float(m.get("baseline_auc")), "ci": "N/A",           "fmt": ".4f"},
    }

    # Compute MAE for sales
    actuals = results["sales"].get("actuals", {})
    test_pred = results["sales"].get("test_predictions", {})
    try:
        n_test = sm.get("rmse") and len(test_pred.get("revenue", []))
        n = len(test_pred.get("revenue", []))
        if n > 0:
            actual_vals = np.array(actuals.get("revenue", []))[-n:]
            pred_vals = np.array(test_pred.get("revenue", []))
            mae_val = round(float(np.mean(np.abs(actual_vals - pred_vals))), 2)
        else:
            mae_val = None
    except Exception:
        mae_val = None

    sales_metrics = {
        "RMSE":               {"value": _safe_float(sm.get("rmse")),              "fmt": "dollar"},
        "MAE":                {"value": mae_val,                                   "fmt": "dollar"},
        "MAPE":               {"value": _safe_float(sm.get("mape")),              "fmt": "pct"},
        "R²":                 {"value": _safe_float(sm.get("r2")),                "fmt": ".4f"},
        "Interval Coverage":  {"value": _safe_float(sm.get("interval_coverage")), "fmt": "pct"},
    }


    high_risk = results["churn"]["predictions"]
    high_risk_df = high_risk[high_risk["churn_probability"] >= 0.5].sort_values(
        "churn_probability", ascending=False
    ).head(20)

    high_risk_records = []
    for _, row in high_risk_df.iterrows():
        rec = {}
        for col in high_risk_df.columns:
            v = row[col]
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = round(float(v), 4)
            rec[col] = v
        high_risk_records.append(rec)

    next_month = results["sales"]["test_predictions"]
    if next_month and next_month.get("revenue"):
        next_rev = _safe_float(next_month["revenue"][-1])
        next_lower = _safe_float(next_month.get("lower_bound", [None])[-1])
        next_upper = _safe_float(next_month.get("upper_bound", [None])[-1])
        next_month_label = next_month["months"][-1] if next_month.get("months") else "N/A"
    else:
        next_rev = next_lower = next_upper = None
        next_month_label = "N/A"

    return {
        "meta": meta,
        "churn_metrics": churn_metrics,
        "sales_metrics": sales_metrics,
        "confusion_matrix": _build_confusion_matrix_data(results),
        "roc_data": _build_roc_data(results),
        "calibration_data": _build_calibration_data(results),
        "forecast_data": _build_forecast_data(results),
        "drift_rows": _build_drift_data(results),
        "leakage_checks": _build_leakage_data(results),
        "high_risk_table": high_risk_records,
        "high_risk_cols": list(high_risk_df.columns),
        "next_rev": next_rev,
        "next_lower": next_lower,
        "next_upper": next_upper,
        "next_month_label": next_month_label,
        "drift_summary": results["drift"].get("drift_summary", ""),
        "drifted_count": results["drift"].get("drifted_features_count", 0),
        "total_features": results["drift"].get("total_features_monitored", 0),
        "leakage_passed": sum(
            1 for c in results["leakage"].get("checks", [])
            if "PASS" in c.get("status", "")
        ),
        "leakage_total": len(results["leakage"].get("checks", [])),
    }


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle file upload, run pipeline, redirect to results."""
    global _last_results

    errors = []

    # -- Validate uploads --
    if "customers_file" not in request.files or "sales_file" not in request.files:
        return render_template("index.html", error="Both CSV files are required.")

    cust_file = request.files["customers_file"]
    sales_file = request.files["sales_file"]

    if cust_file.filename == "" or sales_file.filename == "":
        return render_template("index.html", error="Please select both CSV files.")

    # -- Save uploads --
    run_id = uuid.uuid4().hex[:8]
    cust_path = os.path.join(UPLOAD_FOLDER, f"{run_id}_customers.csv")
    sales_path = os.path.join(UPLOAD_FOLDER, f"{run_id}_sales.csv")
    cust_file.save(cust_path)
    sales_file.save(sales_path)

    try:
        customer_df = pd.read_csv(cust_path)
        sales_df = pd.read_csv(sales_path)
        customer_df = validate_customer_df(customer_df)
        sales_df = validate_sales_df(sales_df)
    except Exception as e:
        return render_template("index.html", error=f"CSV validation error: {e}")

    try:
        t0 = time.time()
        raw = run_full_pipeline(customer_df, sales_df)

        # Export Power BI & dashboard
        pbi_dir = os.path.join(OUTPUT_FOLDER, "powerbi")
        export_powerbi_tables(raw, customer_df, sales_df, output_dir=pbi_dir)
        dashboard_path = os.path.join(OUTPUT_FOLDER, "dashboard.html")
        generate_dashboard(raw, output_path=dashboard_path)

        elapsed = round(time.time() - t0, 1)
        _last_results["data"] = raw
        _last_results["elapsed"] = elapsed

        serialized = _serialize_results(raw)
        serialized["elapsed"] = elapsed
        serialized["pbi_dir"] = pbi_dir
        serialized["dashboard_path"] = dashboard_path
        _last_results["serialized"] = serialized

    except Exception as e:
        tb = traceback.format_exc()
        return render_template("index.html",
                               error=f"Pipeline error: {e}\n\n{tb}")

    return redirect(url_for("results"))


@app.route("/results")
def results():
    if not _last_results.get("serialized"):
        return redirect(url_for("index"))
    s = _last_results["serialized"]
    # Pass JSON blobs for Plotly
    return render_template(
        "results.html",
        s=s,
        roc_json=json.dumps(s["roc_data"]),
        cal_json=json.dumps(s["calibration_data"]),
        forecast_json=json.dumps(s["forecast_data"]),
        cm_json=json.dumps(s["confusion_matrix"]),
    )


@app.route("/download/<filename>")
def download(filename):
    """Download a Power BI CSV output."""
    pbi_dir = os.path.join(OUTPUT_FOLDER, "powerbi")
    file_path = os.path.join(pbi_dir, filename)
    if not os.path.exists(file_path):
        return f"File not found: {filename}", 404
    return send_file(file_path, as_attachment=True)


@app.route("/download_dashboard")
def download_dashboard():
    dashboard_path = os.path.join(OUTPUT_FOLDER, "dashboard.html")
    if not os.path.exists(dashboard_path):
        return "Dashboard not yet generated.", 404
    return send_file(dashboard_path, as_attachment=True)


@app.route("/api/results")
def api_results():
    if not _last_results.get("serialized"):
        return jsonify({"error": "No results available yet."}), 404
    return jsonify(_last_results["serialized"])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Enterprise Churn & Sales Prediction Platform")
    print("  Open your browser: http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
