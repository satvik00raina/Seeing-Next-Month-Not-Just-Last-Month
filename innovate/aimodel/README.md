# 📊 Predictive Dashboard — Enterprise Edition

> **Seeing Next Month, Not Just Last Month**
>
> A churn risk + revenue forecasting dashboard that trains ML models and shows
> predictions alongside actuals, with honest accuracy notes and enterprise-grade
> monitoring.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python main.py

# 3. Open the dashboard
# → output/dashboard.html (open in any browser)
```

## 🏗️ Architecture

```
aimodel/
├── main.py                    # Entry point — orchestrates everything
├── requirements.txt           # Python dependencies
├── src/
│   ├── data_generator.py      # Synthetic Telco data generator
│   ├── ml_pipeline.py         # Full ML pipeline (leakage + churn + sales + drift)
│   └── dashboard.py           # Interactive HTML dashboard generator
├── data/                      # Generated datasets (auto-created)
│   ├── telco_customers.csv
│   └── monthly_revenue.csv
└── output/                    # Dashboard output (auto-created)
    └── dashboard.html
```

## 📋 What the Dashboard Shows

### 1. Executive KPIs
- **Model AUC-ROC** with 95% confidence intervals
- **Churn rate** across the customer base
- **Forecast MAPE** for revenue predictions
- **High-risk customer count**
- **Drift status** (stable / drifted)

### 2. Churn Analysis (Classification)
- Churn probability distribution with risk threshold
- Top 10 feature importance (Gradient Boosting)
- High-risk customer table with actionable risk tiers
- Model: `GradientBoostingClassifier` vs. `LogisticRegression` baseline

### 3. Sales Forecast (Regression)
- Monthly revenue forecast with **80% prediction intervals** (quantile regression)
- Actuals vs. predictions with train/test split visualization
- RMSE, MAPE, R², and interval coverage metrics

### 4. Model Performance
- ROC curve (GBM vs. baseline)
- Calibration plot (reliability diagram)
- Confusion matrix
- All metrics with **bootstrapped 95% CIs** (500 iterations)

### 5. Drift Monitoring
- **PSI (Population Stability Index)** per feature
- **Kolmogorov-Smirnov tests** with p-values
- Traffic light system: green < 0.1 < amber < 0.2 < red

### 6. Data Leakage Audit
- Target correlation check (|r| > 0.8)
- Derived feature detection (TotalCharges ≈ Monthly × tenure)
- Class imbalance assessment
- Missing value audit
- Temporal integrity verification

## 🛡️ Enterprise-Grade Features

| Feature | Implementation |
|---------|---------------|
| **Backtesting** | Stratified holdout (churn), chronological split (sales) |
| **Error Metrics** | AUC, F1, Precision, Recall, Brier Score, RMSE, MAPE, R² |
| **Confidence Intervals** | 500-iteration bootstrap for all classification metrics |
| **Prediction Intervals** | Quantile regression (α=0.1, 0.9) for revenue forecast |
| **Drift Monitoring** | PSI + KS test across all numeric features |
| **Leakage Prevention** | 6-point automated audit before training |
| **Honest Accuracy Note** | Prominently displayed with limitations and caveats |

## 🔧 Customization

### Using Your Own Data
Replace the data generation in `main.py` with your own CSV loading:

```python
customer_df = pd.read_csv("your_customers.csv")
revenue_df = pd.read_csv("your_revenue.csv")
```

Required columns for customer data: `customerID`, `Churn` (0/1), plus feature columns.
Required columns for revenue data: `month`, `revenue`, `new_customers`, `active_customers`.

### Adjusting the Model
Edit `src/ml_pipeline.py` → `train_churn_model()` to swap algorithms, tune
hyperparameters, or add cross-validation.

## 📊 Sample Output

The dashboard is a **self-contained HTML file** — no server required.
Just open `output/dashboard.html` in any modern browser (Chrome, Edge, Firefox).

## ⚖️ License

MIT — Built for the Student Edition Predictive Dashboard challenge.
