"""
Interactive HTML Dashboard Generator
=====================================
Generates a self-contained HTML dashboard with Plotly.js charts.
No server required — just open the HTML file in any browser.
"""

import json
import os
from datetime import datetime


def generate_dashboard(results, output_path="output/dashboard.html"):
    """
    Generate a comprehensive, enterprise-grade HTML dashboard.

    Parameters
    ----------
    results : dict
        Output from ml_pipeline.run_full_pipeline()
    output_path : str
        Where to save the HTML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    churn = results["churn"]
    sales = results["sales"]
    drift = results["drift"]
    leakage = results["leakage"]
    meta = results["metadata"]

    # Prepare data for JSON embedding
    feature_imp = churn["feature_importance"].head(10).to_dict("records")
    predictions = churn["predictions"].head(50).to_dict("records")
    high_risk = churn["high_risk_customers"].to_dict("records")

    # Build the dashboard data object
    dashboard_data = {
        "meta": meta,
        "churn_metrics": churn["metrics"],
        "churn_ci": churn["bootstrap_ci"],
        "roc": churn["roc_curve"],
        "roc_baseline": churn["roc_curve_baseline"],
        "calibration": churn["calibration"],
        "confusion_matrix": churn["confusion_matrix"],
        "feature_importance": feature_imp,
        "predictions": predictions,
        "high_risk": high_risk,
        "churn_rate_train": churn["churn_rate_train"],
        "churn_rate_test": churn["churn_rate_test"],
        "train_size": churn["train_size"],
        "test_size": churn["test_size"],
        "sales_metrics": sales["metrics"],
        "sales_actuals": sales["actuals"],
        "sales_train_pred": sales["train_predictions"],
        "sales_test_pred": sales["test_predictions"],
        "sales_train_size": sales["train_size"],
        "sales_test_size": sales["test_size"],
        "drift_psi": drift["psi"],
        "drift_ks": drift["ks_tests"],
        "drift_summary": drift["drift_summary"],
        "drift_overall": drift["overall_drift_score"],
        "drift_count": drift["drifted_features_count"],
        "drift_total": drift["total_features_monitored"],
        "leakage_checks": leakage["checks"],
        "leakage_passed": leakage["passed"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Convert risk_tier to string for JSON serialization
    for p in dashboard_data["predictions"]:
        p["risk_tier"] = str(p.get("risk_tier", "Unknown"))
    for p in dashboard_data["high_risk"]:
        p["risk_tier"] = str(p.get("risk_tier", "Unknown"))

    html = _build_html(dashboard_data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDashboard saved to: {os.path.abspath(output_path)}")
    return output_path


def _build_html(data):
    """Build the complete HTML string."""
    data_json = json.dumps(data, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Dashboard — Churn & Sales Forecast</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0f1117;
            --bg-card: #1a1d29;
            --bg-card-hover: #222639;
            --text-primary: #e6e9f0;
            --text-secondary: #8b92a5;
            --accent-blue: #4f8df5;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --accent-amber: #fbbf24;
            --accent-purple: #a78bfa;
            --border: #2a2d3a;
            --shadow: 0 4px 24px rgba(0,0,0,0.3);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
        }}
        .header {{
            text-align: center;
            padding: 32px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 32px;
        }}
        .header h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }}
        .kpi-value {{
            font-size: 2rem;
            font-weight: 700;
            margin: 8px 0 4px;
        }}
        .kpi-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .kpi-detail {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            margin-top: 4px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--accent-blue);
            display: inline-block;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
        }}
        .chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }}
        tr:hover td {{
            background: var(--bg-card-hover);
        }}
        .badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .badge-high {{ background: rgba(248,113,113,0.15); color: var(--accent-red); }}
        .badge-medium {{ background: rgba(251,191,36,0.15); color: var(--accent-amber); }}
        .badge-low {{ background: rgba(52,211,153,0.15); color: var(--accent-green); }}
        .badge-pass {{ background: rgba(52,211,153,0.15); color: var(--accent-green); }}
        .badge-warn {{ background: rgba(251,191,36,0.15); color: var(--accent-amber); }}
        .badge-info {{ background: rgba(79,141,245,0.15); color: var(--accent-blue); }}
        .honesty-note {{
            background: linear-gradient(135deg, rgba(79,141,245,0.08), rgba(167,139,250,0.08));
            border: 1px solid rgba(79,141,245,0.2);
            border-radius: 12px;
            padding: 20px;
            margin: 24px 0;
        }}
        .honesty-note h3 {{
            color: var(--accent-blue);
            margin-bottom: 8px;
        }}
        .honesty-note p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        .tab-container {{
            display: flex;
            gap: 4px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0;
            overflow-x: auto;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            color: var(--text-secondary);
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .tab:hover {{ color: var(--text-primary); }}
        .tab.active {{
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .ci-note {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-style: italic;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }}
        .metric-row:last-child {{ border-bottom: none; }}
        .footer {{
            text-align: center;
            padding: 32px 0;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }}
        @media (max-width: 768px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
            body {{ padding: 12px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Predictive Dashboard</h1>
        <p class="subtitle">Churn Risk Analysis & Revenue Forecasting — Enterprise Edition</p>
        <p class="subtitle" style="margin-top:4px;">Generated: {data['generated_at']}</p>
    </div>

    <!-- KPI Row -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Model AUC-ROC</div>
            <div class="kpi-value" style="color: var(--accent-blue);">{data['churn_metrics']['auc']}</div>
            <div class="kpi-detail">95% CI: [{data['churn_ci']['auc']['ci_lower']}, {data['churn_ci']['auc']['ci_upper']}]</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value" style="color: var(--accent-red);">{data['meta']['churn_rate']:.1%}</div>
            <div class="kpi-detail">{data['meta']['n_customers']:,} total customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Forecast MAPE</div>
            <div class="kpi-value" style="color: var(--accent-green);">{data['sales_metrics']['mape']}%</div>
            <div class="kpi-detail">RMSE: ${data['sales_metrics']['rmse']:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">High-Risk Customers</div>
            <div class="kpi-value" style="color: var(--accent-amber);">{len(data['high_risk'])}</div>
            <div class="kpi-detail">Churn probability ≥ 60%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Drift Status</div>
            <div class="kpi-value" style="font-size:1.2rem; color: {'var(--accent-green)' if data['drift_count'] == 0 else 'var(--accent-red)'};">{'Stable' if data['drift_count'] == 0 else f"{data['drift_count']} Drifted"}</div>
            <div class="kpi-detail">PSI max: {data['drift_overall']}</div>
        </div>
    </div>

    <!-- Honesty Note -->
    <div class="honesty-note">
        <h3>🔍 Model Accuracy — Honest Assessment</h3>
        <p>
            <strong>Churn Model:</strong> AUC of {data['churn_metrics']['auc']} (95% CI: {data['churn_ci']['auc']['ci_lower']}–{data['churn_ci']['auc']['ci_upper']}) on a holdout of {data['test_size']:,} customers.
            Baseline logistic regression achieves AUC {data['churn_metrics']['baseline_auc']} — the gradient boosting model provides a
            {((data['churn_metrics']['auc'] - data['churn_metrics']['baseline_auc']) / data['churn_metrics']['baseline_auc'] * 100):.1f}% improvement.
            Brier score: {data['churn_metrics']['brier_score']} (lower is better, 0 = perfect calibration).
        </p>
        <p style="margin-top:8px;">
            <strong>Sales Forecast:</strong> MAPE of {data['sales_metrics']['mape']}% over {data['sales_test_size']}-month horizon.
            80% prediction intervals achieve {data['sales_metrics']['interval_coverage']}% coverage (target: 80%).
            {'✅ Intervals are well-calibrated.' if abs(data['sales_metrics']['interval_coverage'] - 80) <= 10 else '⚠️ Interval calibration needs attention.'}
        </p>
        <p style="margin-top:8px;">
            <strong>Leakage Audit:</strong> {'✅ All checks passed — no leakage detected.' if data['leakage_passed'] else '⚠️ Potential leakage flagged — review the audit below.'}
        </p>
    </div>

    <!-- Navigation Tabs -->
    <div class="tab-container">
        <div class="tab active" onclick="switchTab('churn')">🎯 Churn Analysis</div>
        <div class="tab" onclick="switchTab('sales')">📈 Sales Forecast</div>
        <div class="tab" onclick="switchTab('model')">🏆 Model Performance</div>
        <div class="tab" onclick="switchTab('drift')">📡 Drift Monitor</div>
        <div class="tab" onclick="switchTab('audit')">🔒 Leakage Audit</div>
    </div>

    <!-- Tab: Churn Analysis -->
    <div id="tab-churn" class="tab-content active">
        <div class="section">
            <h2 class="section-title">Churn Risk Distribution</h2>
            <div class="chart-grid">
                <div class="chart-container" id="chart-churn-dist"></div>
                <div class="chart-container" id="chart-feature-imp"></div>
            </div>
        </div>
        <div class="section">
            <h2 class="section-title">High-Risk Customers (Top 20)</h2>
            <div class="card" style="overflow-x:auto;">
                <table id="high-risk-table">
                    <thead>
                        <tr>
                            <th>Customer ID</th>
                            <th>Churn Probability</th>
                            <th>Risk Tier</th>
                            <th>Predicted</th>
                            <th>Actual</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Tab: Sales Forecast -->
    <div id="tab-sales" class="tab-content">
        <div class="section">
            <h2 class="section-title">Revenue Forecast with Prediction Intervals</h2>
            <div class="chart-container" id="chart-sales-forecast"></div>
            <div class="card" style="margin-top:20px;">
                <h3 style="margin-bottom:12px;">Forecast Metrics</h3>
                <div class="metric-row">
                    <span>RMSE (Test)</span>
                    <span style="font-weight:600;">${data['sales_metrics']['rmse']:,.0f}</span>
                </div>
                <div class="metric-row">
                    <span>MAPE (Test)</span>
                    <span style="font-weight:600;">{data['sales_metrics']['mape']}%</span>
                </div>
                <div class="metric-row">
                    <span>R² (Test)</span>
                    <span style="font-weight:600;">{data['sales_metrics']['r2']}</span>
                </div>
                <div class="metric-row">
                    <span>RMSE (Train)</span>
                    <span style="font-weight:600;">${data['sales_metrics']['rmse_train']:,.0f}</span>
                </div>
                <div class="metric-row">
                    <span>80% Interval Coverage</span>
                    <span style="font-weight:600;">{data['sales_metrics']['interval_coverage']}%</span>
                </div>
                <p class="ci-note" style="margin-top:12px;">Prediction intervals via quantile regression (α=0.1, 0.9). Coverage should be near 80%.</p>
            </div>
        </div>
    </div>

    <!-- Tab: Model Performance -->
    <div id="tab-model" class="tab-content">
        <div class="section">
            <h2 class="section-title">Classification Performance</h2>
            <div class="chart-grid">
                <div class="chart-container" id="chart-roc"></div>
                <div class="chart-container" id="chart-calibration"></div>
            </div>
            <div class="chart-grid" style="margin-top:20px;">
                <div class="chart-container" id="chart-confusion"></div>
                <div class="card">
                    <h3 style="margin-bottom:16px;">Metrics with 95% Confidence Intervals</h3>
                    <table>
                        <thead>
                            <tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>AUC-ROC</td>
                                <td style="font-weight:600;">{data['churn_metrics']['auc']}</td>
                                <td class="ci-note">[{data['churn_ci']['auc']['ci_lower']}, {data['churn_ci']['auc']['ci_upper']}]</td>
                            </tr>
                            <tr>
                                <td>F1 Score</td>
                                <td style="font-weight:600;">{data['churn_metrics']['f1']}</td>
                                <td class="ci-note">[{data['churn_ci']['f1']['ci_lower']}, {data['churn_ci']['f1']['ci_upper']}]</td>
                            </tr>
                            <tr>
                                <td>Precision</td>
                                <td style="font-weight:600;">{data['churn_metrics']['precision']}</td>
                                <td class="ci-note">[{data['churn_ci']['precision']['ci_lower']}, {data['churn_ci']['precision']['ci_upper']}]</td>
                            </tr>
                            <tr>
                                <td>Recall</td>
                                <td style="font-weight:600;">{data['churn_metrics']['recall']}</td>
                                <td class="ci-note">[{data['churn_ci']['recall']['ci_lower']}, {data['churn_ci']['recall']['ci_upper']}]</td>
                            </tr>
                            <tr>
                                <td>Accuracy</td>
                                <td style="font-weight:600;">{data['churn_metrics']['accuracy']}</td>
                                <td class="ci-note">—</td>
                            </tr>
                            <tr>
                                <td>Brier Score</td>
                                <td style="font-weight:600;">{data['churn_metrics']['brier_score']}</td>
                                <td class="ci-note">Lower is better</td>
                            </tr>
                        </tbody>
                    </table>
                    <p class="ci-note" style="margin-top:12px;">CIs computed via 500-iteration bootstrap on the holdout set (n={data['test_size']:,}).</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Tab: Drift Monitor -->
    <div id="tab-drift" class="tab-content">
        <div class="section">
            <h2 class="section-title">Feature Distribution Drift</h2>
            <div class="card">
                <p style="font-size:1.1rem; margin-bottom:16px;">{data['drift_summary']}</p>
                <p class="ci-note">
                    Monitoring {data['drift_total']} numeric features.
                    PSI &lt; 0.1 = no drift, 0.1–0.2 = moderate, ≥ 0.2 = significant.
                    Comparing early-tenure (training proxy) vs. late-tenure (production proxy) distributions.
                </p>
            </div>
            <div class="chart-grid">
                <div class="chart-container" id="chart-psi"></div>
                <div class="chart-container" id="chart-ks"></div>
            </div>
        </div>
    </div>

    <!-- Tab: Leakage Audit -->
    <div id="tab-audit" class="tab-content">
        <div class="section">
            <h2 class="section-title">Data Leakage Audit Report</h2>
            <div class="card">
                <table>
                    <thead>
                        <tr><th>Check</th><th>Status</th><th>Details</th></tr>
                    </thead>
                    <tbody id="leakage-table-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Predictive Dashboard v1.0 — Enterprise Edition</p>
        <p>Built with scikit-learn + Plotly.js | {data['meta']['n_customers']:,} customers | {data['meta']['n_months']} months of data</p>
    </div>

    <script>
    // Dashboard data
    const D = {data_json};

    // ========== Tab Switching ==========
    function switchTab(name) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + name).classList.add('active');
        event.target.classList.add('active');
        window.dispatchEvent(new Event('resize'));  // Trigger Plotly relayout
    }}

    const plotlyLayout = {{
        paper_bgcolor: '#1a1d29',
        plot_bgcolor: '#1a1d29',
        font: {{ color: '#e6e9f0', family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' }},
        margin: {{ t: 40, r: 20, b: 40, l: 60 }},
        xaxis: {{ gridcolor: '#2a2d3a', zerolinecolor: '#2a2d3a' }},
        yaxis: {{ gridcolor: '#2a2d3a', zerolinecolor: '#2a2d3a' }},
    }};
    const plotlyConfig = {{ responsive: true, displayModeBar: false }};

    // ========== Churn Distribution ==========
    (function() {{
        const probs = D.predictions.map(p => p.churn_probability);
        Plotly.newPlot('chart-churn-dist', [{{
            x: probs,
            type: 'histogram',
            nbinsx: 30,
            marker: {{ color: '#4f8df5', line: {{ color: '#1a1d29', width: 1 }} }},
            name: 'Customers',
        }}], {{
            ...plotlyLayout,
            title: {{ text: 'Churn Probability Distribution', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'Predicted Churn Probability' }},
            yaxis: {{ ...plotlyLayout.yaxis, title: 'Number of Customers' }},
            shapes: [{{
                type: 'line', x0: 0.6, x1: 0.6, y0: 0, y1: 1,
                yref: 'paper', line: {{ color: '#f87171', width: 2, dash: 'dash' }},
            }}],
            annotations: [{{
                x: 0.62, y: 0.95, xref: 'x', yref: 'paper',
                text: 'High Risk →', showarrow: false,
                font: {{ color: '#f87171', size: 11 }},
            }}],
        }}, plotlyConfig);
    }})();

    // ========== Feature Importance ==========
    (function() {{
        const fi = D.feature_importance.slice().reverse();
        Plotly.newPlot('chart-feature-imp', [{{
            y: fi.map(f => f.feature),
            x: fi.map(f => f.importance),
            type: 'bar',
            orientation: 'h',
            marker: {{ color: fi.map((_, i) => `hsl(${{210 + i * 15}}, 70%, 60%)`) }},
        }}], {{
            ...plotlyLayout,
            title: {{ text: 'Top 10 Feature Importance', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'Importance' }},
            yaxis: {{ ...plotlyLayout.yaxis, automargin: true }},
            margin: {{ ...plotlyLayout.margin, l: 140 }},
        }}, plotlyConfig);
    }})();

    // ========== High Risk Table ==========
    (function() {{
        const tbody = document.querySelector('#high-risk-table tbody');
        D.high_risk.forEach(row => {{
            const tr = document.createElement('tr');
            const badgeClass = row.risk_tier === 'High' ? 'badge-high' : row.risk_tier === 'Medium' ? 'badge-medium' : 'badge-low';
            tr.innerHTML = `
                <td>${{row.customerID}}</td>
                <td style="font-weight:600;">${{(row.churn_probability * 100).toFixed(1)}}%</td>
                <td><span class="badge ${{badgeClass}}">${{row.risk_tier}}</span></td>
                <td>${{row.predicted_churn ? '⚠️ Yes' : '✅ No'}}</td>
                <td>${{row.actual_churn ? '❌ Churned' : '✅ Active'}}</td>
            `;
            tbody.appendChild(tr);
        }});
    }})();

    // ========== Sales Forecast ==========
    (function() {{
        const traces = [
            {{
                x: D.sales_actuals.months,
                y: D.sales_actuals.revenue,
                type: 'scatter', mode: 'lines+markers',
                name: 'Actual Revenue',
                line: {{ color: '#e6e9f0', width: 2 }},
                marker: {{ size: 5 }},
            }},
            {{
                x: D.sales_train_pred.months,
                y: D.sales_train_pred.revenue,
                type: 'scatter', mode: 'lines',
                name: 'Train Fit',
                line: {{ color: '#4f8df5', width: 2, dash: 'dot' }},
            }},
            {{
                x: D.sales_test_pred.months,
                y: D.sales_test_pred.revenue,
                type: 'scatter', mode: 'lines+markers',
                name: 'Forecast',
                line: {{ color: '#34d399', width: 3 }},
                marker: {{ size: 7 }},
            }},
            {{
                x: [...D.sales_test_pred.months, ...D.sales_test_pred.months.slice().reverse()],
                y: [...D.sales_test_pred.upper_bound, ...D.sales_test_pred.lower_bound.slice().reverse()],
                type: 'scatter', fill: 'toself',
                fillcolor: 'rgba(52,211,153,0.12)',
                line: {{ color: 'transparent' }},
                name: '80% Prediction Interval',
                showlegend: true,
            }},
        ];
        Plotly.newPlot('chart-sales-forecast', traces, {{
            ...plotlyLayout,
            title: {{ text: 'Monthly Revenue: Actuals vs. Forecast', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'Month' }},
            yaxis: {{ ...plotlyLayout.yaxis, title: 'Revenue ($)', tickformat: '$,.0f' }},
            legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.3)', font: {{ size: 11 }} }},
            shapes: [{{
                type: 'line',
                x0: D.sales_test_pred.months[0], x1: D.sales_test_pred.months[0],
                y0: 0, y1: 1, yref: 'paper',
                line: {{ color: '#fbbf24', width: 1, dash: 'dash' }},
            }}],
            annotations: [{{
                x: D.sales_test_pred.months[0], y: 1.02, xref: 'x', yref: 'paper',
                text: '← Train | Test →', showarrow: false,
                font: {{ color: '#fbbf24', size: 11 }},
            }}],
        }}, plotlyConfig);
    }})();

    // ========== ROC Curve ==========
    (function() {{
        Plotly.newPlot('chart-roc', [
            {{
                x: D.roc.fpr, y: D.roc.tpr,
                type: 'scatter', mode: 'lines',
                name: `GBM (AUC=${{D.churn_metrics.auc}})`,
                line: {{ color: '#4f8df5', width: 2.5 }},
            }},
            {{
                x: D.roc_baseline.fpr, y: D.roc_baseline.tpr,
                type: 'scatter', mode: 'lines',
                name: `LR Baseline (AUC=${{D.churn_metrics.baseline_auc}})`,
                line: {{ color: '#8b92a5', width: 1.5, dash: 'dash' }},
            }},
            {{
                x: [0, 1], y: [0, 1],
                type: 'scatter', mode: 'lines',
                name: 'Random',
                line: {{ color: '#2a2d3a', width: 1, dash: 'dot' }},
                showlegend: false,
            }},
        ], {{
            ...plotlyLayout,
            title: {{ text: 'ROC Curve', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'False Positive Rate', range: [0, 1] }},
            yaxis: {{ ...plotlyLayout.yaxis, title: 'True Positive Rate', range: [0, 1] }},
            legend: {{ x: 0.5, y: 0.05, bgcolor: 'rgba(0,0,0,0.3)', font: {{ size: 11 }} }},
        }}, plotlyConfig);
    }})();

    // ========== Calibration Plot ==========
    (function() {{
        Plotly.newPlot('chart-calibration', [
            {{
                x: D.calibration.prob_pred, y: D.calibration.prob_true,
                type: 'scatter', mode: 'lines+markers',
                name: 'Model',
                line: {{ color: '#a78bfa', width: 2 }},
                marker: {{ size: 7 }},
            }},
            {{
                x: [0, 1], y: [0, 1],
                type: 'scatter', mode: 'lines',
                name: 'Perfect',
                line: {{ color: '#2a2d3a', width: 1, dash: 'dot' }},
            }},
        ], {{
            ...plotlyLayout,
            title: {{ text: 'Calibration Plot (Reliability Diagram)', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'Mean Predicted Probability', range: [0, 1] }},
            yaxis: {{ ...plotlyLayout.yaxis, title: 'Fraction of Positives', range: [0, 1] }},
            legend: {{ x: 0.6, y: 0.1, bgcolor: 'rgba(0,0,0,0.3)', font: {{ size: 11 }} }},
        }}, plotlyConfig);
    }})();

    // ========== Confusion Matrix ==========
    (function() {{
        const cm = D.confusion_matrix;
        const labels = ['No Churn', 'Churn'];
        const text = cm.map((row, i) => row.map((val, j) => {{
            const total = cm.flat().reduce((a, b) => a + b, 0);
            return `${{val}}<br>(${{(val/total*100).toFixed(1)}}%)`;
        }}));
        Plotly.newPlot('chart-confusion', [{{
            z: cm, x: labels, y: labels,
            type: 'heatmap',
            colorscale: [[0, '#1a1d29'], [1, '#4f8df5']],
            text: text,
            texttemplate: '%{{text}}',
            textfont: {{ color: '#e6e9f0', size: 14 }},
            hoverinfo: 'skip',
            showscale: false,
        }}], {{
            ...plotlyLayout,
            title: {{ text: 'Confusion Matrix', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'Predicted', side: 'bottom' }},
            yaxis: {{ ...plotlyLayout.yaxis, title: 'Actual', autorange: 'reversed' }},
        }}, plotlyConfig);
    }})();

    // ========== PSI Chart ==========
    (function() {{
        const psi = D.drift_psi.sort((a, b) => b.psi - a.psi);
        const colors = psi.map(p => p.psi >= 0.2 ? '#f87171' : p.psi >= 0.1 ? '#fbbf24' : '#34d399');
        Plotly.newPlot('chart-psi', [{{
            x: psi.map(p => p.feature),
            y: psi.map(p => p.psi),
            type: 'bar',
            marker: {{ color: colors }},
        }}], {{
            ...plotlyLayout,
            title: {{ text: 'Population Stability Index (PSI)', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'Feature', tickangle: -45 }},
            yaxis: {{ ...plotlyLayout.yaxis, title: 'PSI' }},
            shapes: [
                {{ type: 'line', x0: -0.5, x1: psi.length - 0.5, y0: 0.1, y1: 0.1, line: {{ color: '#fbbf24', width: 1, dash: 'dash' }} }},
                {{ type: 'line', x0: -0.5, x1: psi.length - 0.5, y0: 0.2, y1: 0.2, line: {{ color: '#f87171', width: 1, dash: 'dash' }} }},
            ],
            margin: {{ ...plotlyLayout.margin, b: 100 }},
        }}, plotlyConfig);
    }})();

    // ========== KS Test Chart ==========
    (function() {{
        const ks = D.drift_ks.sort((a, b) => b.ks_statistic - a.ks_statistic);
        const colors = ks.map(k => k.drift_detected ? '#f87171' : '#34d399');
        Plotly.newPlot('chart-ks', [{{
            x: ks.map(k => k.feature),
            y: ks.map(k => k.ks_statistic),
            type: 'bar',
            marker: {{ color: colors }},
            text: ks.map(k => `p=${{k.p_value.toFixed(4)}}`),
            textposition: 'outside',
            textfont: {{ size: 9, color: '#8b92a5' }},
        }}], {{
            ...plotlyLayout,
            title: {{ text: 'Kolmogorov-Smirnov Test Statistics', font: {{ size: 14 }} }},
            xaxis: {{ ...plotlyLayout.xaxis, title: 'Feature', tickangle: -45 }},
            yaxis: {{ ...plotlyLayout.yaxis, title: 'KS Statistic' }},
            shapes: [
                {{ type: 'line', x0: -0.5, x1: ks.length - 0.5, y0: 0.05, y1: 0.05, line: {{ color: '#fbbf24', width: 1, dash: 'dash' }} }},
            ],
            margin: {{ ...plotlyLayout.margin, b: 100 }},
        }}, plotlyConfig);
    }})();

    // ========== Leakage Audit Table ==========
    (function() {{
        const tbody = document.getElementById('leakage-table-body');
        D.leakage_checks.forEach(check => {{
            const tr = document.createElement('tr');
            let badgeClass = 'badge-pass';
            if (check.status.includes('WARNING')) badgeClass = 'badge-warn';
            else if (check.status.includes('INFO')) badgeClass = 'badge-info';
            tr.innerHTML = `
                <td style="font-weight:500;">${{check.name}}</td>
                <td><span class="badge ${{badgeClass}}">${{check.status}}</span></td>
                <td style="font-size:0.85rem;color:var(--text-secondary);">${{check.detail}}</td>
            `;
            tbody.appendChild(tr);
        }});
    }})();

    </script>
</body>
</html>"""
