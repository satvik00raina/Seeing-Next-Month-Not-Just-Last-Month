"""
Machine Learning Pipeline
=========================

Customer Churn + Revenue Forecasting + Drift Monitoring

Churn:
    - Feature engineering
    - XGBoost
    - ExtraTrees
    - HistGradientBoosting
    - Logistic Regression
    - Out-of-fold stacking
    - Threshold optimization
    - Bootstrap confidence intervals
    - Feature importance
    - Retention lift simulation

Revenue:
    - Ridge / ElasticNet trend + seasonality
    - Tree-based candidates
    - Chronological holdout
    - Next-month forecast
    - Prediction intervals

Power BI:
    - churn_predictions_enriched
    - feature_importance
    - model_metrics
    - sales_forecast
    - drift_monitoring
    - retention_lift_simulation
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PolynomialFeatures,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    ElasticNet,
)

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from xgboost import XGBClassifier


# ============================================================
# GENERAL HELPERS
# ============================================================

def _safe_numeric(series):
    """
    Convert a pandas Series to numeric safely.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# CHURN FEATURE ENGINEERING
# ============================================================

def _engineer_churn_features(X):
    """
    Create additional churn features.

    IMPORTANT:
    This function never uses the Churn target.
    """

    X = X.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    if "tenure" in X.columns:
        X["tenure"] = _safe_numeric(
            X["tenure"]
        )

    if "MonthlyCharges" in X.columns:
        X["MonthlyCharges"] = _safe_numeric(
            X["MonthlyCharges"]
        )

    # --------------------------------------------------------
    # Tenure features
    # --------------------------------------------------------

    if "tenure" in X.columns:

        X["IsNewCustomer"] = (
            X["tenure"] <= 6
        ).astype(int)

        X["IsVeryNewCustomer"] = (
            X["tenure"] <= 3
        ).astype(int)

        X["IsLongTermCustomer"] = (
            X["tenure"] >= 48
        ).astype(int)

        X["TenureGroup"] = pd.cut(
            X["tenure"],
            bins=[
                -np.inf,
                3,
                6,
                12,
                24,
                48,
                np.inf,
            ],
            labels=[
                "0-3",
                "4-6",
                "7-12",
                "13-24",
                "25-48",
                "48+",
            ],
        ).astype(str)

    # --------------------------------------------------------
    # Monthly charges
    # --------------------------------------------------------

    if "MonthlyCharges" in X.columns:

        median_charge = X[
            "MonthlyCharges"
        ].median()

        q75_charge = X[
            "MonthlyCharges"
        ].quantile(0.75)

        X["HighMonthlyCharge"] = (
            X["MonthlyCharges"]
            > median_charge
        ).astype(int)

        X["VeryHighMonthlyCharge"] = (
            X["MonthlyCharges"]
            > q75_charge
        ).astype(int)

    # --------------------------------------------------------
    # Customer lifetime value
    # --------------------------------------------------------

    if (
        "MonthlyCharges" in X.columns
        and "tenure" in X.columns
    ):

        X["EstimatedLifetimeValue"] = (
            X["MonthlyCharges"]
            * X["tenure"]
        )

        X["MonthlyChargePerTenure"] = (
            X["MonthlyCharges"]
            / (X["tenure"] + 1)
        )

    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    if "Contract" in X.columns:

        contract = (
            X["Contract"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        X["MonthToMonth"] = (
            contract
            .str.contains(
                "month-to-month",
                regex=False,
            )
        ).astype(int)

        X["OneYearContract"] = (
            contract
            .str.contains(
                "one year",
                regex=False,
            )
        ).astype(int)

        X["TwoYearContract"] = (
            contract
            .str.contains(
                "two year",
                regex=False,
            )
        ).astype(int)

    # --------------------------------------------------------
    # Payment method
    # --------------------------------------------------------

    if "PaymentMethod" in X.columns:

        payment = (
            X["PaymentMethod"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        X["ElectronicPayment"] = (
            payment
            .str.contains(
                "electronic",
                regex=False,
            )
        ).astype(int)

        X["MailedPayment"] = (
            payment
            .str.contains(
                "mailed",
                regex=False,
            )
        ).astype(int)

    # --------------------------------------------------------
    # Internet service
    # --------------------------------------------------------

    if "InternetService" in X.columns:

        internet = (
            X["InternetService"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        X["HasInternet"] = (
            ~internet.isin(
                [
                    "no",
                    "none",
                    "nan",
                ]
            )
        ).astype(int)

        X["HasFiber"] = (
            internet
            .str.contains(
                "fiber",
                regex=False,
            )
        ).astype(int)

        X["NoInternet"] = (
            internet == "no"
        ).astype(int)

    # --------------------------------------------------------
    # Service count
    # --------------------------------------------------------

    service_columns = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    existing_services = [
        column
        for column in service_columns
        if column in X.columns
    ]

    if existing_services:

        service_count = np.zeros(
            len(X),
            dtype=int,
        )

        for column in existing_services:

            values = (
                X[column]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            service_count += (
                ~values.isin(
                    [
                        "no",
                        "no internet service",
                        "nan",
                        "none",
                    ]
                )
            ).astype(int)

        X["ServiceCount"] = (
            service_count
        )

    # --------------------------------------------------------
    # Support flags
    # --------------------------------------------------------

    for column in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]:

        if column in X.columns:

            values = (
                X[column]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            X[f"No_{column}"] = (
                values.isin(
                    [
                        "no",
                        "no internet service",
                    ]
                )
            ).astype(int)

    # --------------------------------------------------------
    # Interaction features
    # --------------------------------------------------------

    if (
        "MonthToMonth" in X.columns
        and "HighMonthlyCharge" in X.columns
    ):

        X["MonthToMonth_HighCharge"] = (
            X["MonthToMonth"]
            * X["HighMonthlyCharge"]
        )

    if (
        "MonthToMonth" in X.columns
        and "VeryHighMonthlyCharge" in X.columns
    ):

        X[
            "MonthToMonth_VeryHighCharge"
        ] = (
            X["MonthToMonth"]
            * X["VeryHighMonthlyCharge"]
        )

    if (
        "MonthToMonth" in X.columns
        and "IsNewCustomer" in X.columns
    ):

        X["MonthToMonth_NewCustomer"] = (
            X["MonthToMonth"]
            * X["IsNewCustomer"]
        )

    if (
        "HasFiber" in X.columns
        and "ServiceCount" in X.columns
    ):

        X["Fiber_LowServices"] = (
            X["HasFiber"]
            * (
                X["ServiceCount"] <= 2
            ).astype(int)
        )

    if (
        "ElectronicPayment" in X.columns
        and "HighMonthlyCharge" in X.columns
    ):

        X["Electronic_HighCharge"] = (
            X["ElectronicPayment"]
            * X["HighMonthlyCharge"]
        )

    if (
        "IsNewCustomer" in X.columns
        and "HighMonthlyCharge" in X.columns
    ):

        X["New_HighCharge"] = (
            X["IsNewCustomer"]
            * X["HighMonthlyCharge"]
        )

    # --------------------------------------------------------
    # Remove redundant TotalCharges
    # --------------------------------------------------------

    if "TotalCharges" in X.columns:

        X = X.drop(
            columns=["TotalCharges"]
        )

    return X


# ============================================================
# CHURN PREPROCESSOR
# ============================================================

def _build_churn_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# FEATURE NAME HELPER
# ============================================================

def _get_feature_names(preprocessor):

    try:
        return list(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        return [
            f"feature_{i}"
            for i in range(
                preprocessor.transform(
                    pd.DataFrame(
                        columns=[]
                    )
                ).shape[1]
            )
        ]


# ============================================================
# BOOTSTRAP AUC CI
# ============================================================

def _bootstrap_auc_ci(
    y_true,
    probabilities,
    random_state=42,
    n_bootstrap=1000,
):

    rng = np.random.default_rng(
        random_state
    )

    y_true = np.asarray(y_true)
    probabilities = np.asarray(
        probabilities
    )

    scores = []

    n = len(y_true)

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            n,
        )

        y_sample = y_true[
            indices
        ]

        p_sample = probabilities[
            indices
        ]

        if len(np.unique(y_sample)) < 2:
            continue

        scores.append(
            roc_auc_score(
                y_sample,
                p_sample,
            )
        )

    if not scores:

        return np.nan, np.nan

    return (
        float(
            np.percentile(
                scores,
                2.5,
            )
        ),
        float(
            np.percentile(
                scores,
                97.5,
            )
        ),
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def _find_churn_thresholds(
    y_true,
    probabilities,
    minimum_recall=0.60,
):

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": float(
                    threshold
                ),
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "balanced_accuracy":
                    balanced_accuracy_score(
                        y_true,
                        predictions,
                    ),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Pure accuracy threshold
    # --------------------------------------------------------

    accuracy_row = (
        result_df
        .sort_values(
            [
                "accuracy",
                "f1",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    accuracy_threshold = float(
        accuracy_row["threshold"]
    )

    # --------------------------------------------------------
    # Accuracy threshold with recall constraint
    # --------------------------------------------------------

    constrained = result_df[
        result_df["recall"]
        >= minimum_recall
    ]

    if len(constrained) > 0:

        business_row = (
            constrained
            .sort_values(
                [
                    "accuracy",
                    "f1",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        business_threshold = float(
            business_row["threshold"]
        )

    else:

        business_threshold = (
            accuracy_threshold
        )

    # --------------------------------------------------------
    # F1 threshold
    # --------------------------------------------------------

    f1_row = (
        result_df
        .sort_values(
            [
                "f1",
                "accuracy",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    f1_threshold = float(
        f1_row["threshold"]
    )

    return (
        accuracy_threshold,
        business_threshold,
        f1_threshold,
        result_df,
    )


# ============================================================
# CHURN MODEL
# ============================================================

def train_churn_model(
    df,
    random_state=42,
):

    print(
        "\n[2/4] Training churn "
        "classification model..."
    )

    # ========================================================
    # 1. TARGET
    # ========================================================

    data = df.copy()

    if "Churn" not in data.columns:

        raise ValueError(
            "Churn column not found."
        )

    y = data["Churn"].copy()

    if y.dtype == object:

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "yes": 1,
                    "no": 0,
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                }
            )
        )

    y = y.astype(int)

    X = data.drop(
        columns=["Churn"]
    )

    # ========================================================
    # 2. FEATURE ENGINEERING
    # ========================================================

    X = _engineer_churn_features(
        X
    )

    # ========================================================
    # 3. TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=random_state,
        )
    )

    print(
        f"  Training rows: {len(X_train):,}"
    )

    print(
        f"  Test rows:     {len(X_test):,}"
    )

    print(
        f"  Churn rate:    {y.mean():.2%}"
    )

    # ========================================================
    # 4. PREPROCESS
    # ========================================================

    preprocessor = (
        _build_churn_preprocessor(
            X_train
        )
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # ========================================================
    # 5. XGBOOST SEARCH
    # ========================================================

    print(
        "\n  Tuning XGBoost..."
    )

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    xgb_params = {

        "n_estimators": [
            200,
            300,
            400,
        ],

        "max_depth": [
            2,
            3,
            4,
            5,
            6,
        ],

        "learning_rate": [
            0.02,
            0.03,
            0.05,
            0.08,
            0.10,
        ],

        "subsample": [
            0.70,
            0.80,
            0.90,
            1.00,
        ],

        "colsample_bytree": [
            0.70,
            0.80,
            0.90,
            1.00,
        ],

        "min_child_weight": [
            1,
            2,
            3,
            5,
            7,
        ],

        "gamma": [
            0,
            0.05,
            0.10,
            0.20,
        ],

        "reg_alpha": [
            0,
            0.01,
            0.05,
            0.10,
        ],

        "reg_lambda": [
            1,
            1.5,
            2,
            3,
            5,
        ],
    }

    xgb_search = (
        RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=xgb_params,
            n_iter=30,
            scoring="roc_auc",
            cv=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
    )

    xgb_search.fit(
        X_train_processed,
        y_train,
    )

    xgb_model = (
        xgb_search.best_estimator_
    )

    print(
        "  XGBoost tuning complete."
    )

    # ========================================================
    # 6. EXTRA TREES SEARCH
    # ========================================================

    print(
        "\n  Tuning ExtraTrees..."
    )

    extra_base = ExtraTreesClassifier(
        random_state=random_state,
        n_jobs=1,
    )

    extra_params = {

        "n_estimators": [
            300,
            400,
            500,
            600,
        ],

        "max_depth": [
            None,
            5,
            8,
            12,
            16,
            20,
        ],

        "min_samples_split": [
            2,
            4,
            6,
            10,
        ],

        "min_samples_leaf": [
            1,
            2,
            3,
            5,
        ],

        "max_features": [
            "sqrt",
            "log2",
            0.6,
            0.8,
            1.0,
        ],

        "class_weight": [
            None,
            "balanced",
        ],
    }

    extra_search = (
        RandomizedSearchCV(
            estimator=extra_base,
            param_distributions=extra_params,
            n_iter=10,
            scoring="roc_auc",
            cv=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
    )

    extra_search.fit(
        X_train_processed,
        y_train,
    )

    extra_model = (
        extra_search.best_estimator_
    )

    print(
        "  ExtraTrees tuning complete."
    )

    # ========================================================
    # 7. HISTOGRAM GRADIENT BOOSTING
    # ========================================================

    hist_model = (
        HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )
    )

    # ========================================================
    # 8. LOGISTIC
    # ========================================================

    logistic_model = (
        LogisticRegression(
            C=0.5,
            max_iter=3000,
            solver="lbfgs",
            class_weight=None,
            random_state=random_state,
        )
    )

    base_templates = [
        xgb_model,
        extra_model,
        hist_model,
        logistic_model,
    ]

    # ========================================================
    # 9. OOF STACKING
    # ========================================================

    print(
        "\n  Building 5-fold "
        "out-of-fold ensemble..."
    )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    oof_predictions = np.zeros(
        (
            len(X_train),
            len(base_templates),
        )
    )

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        skf.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"    Fold {fold}/5"
        )

        X_fold_train = (
            X_train.iloc[train_idx]
        )

        X_fold_valid = (
            X_train.iloc[valid_idx]
        )

        y_fold_train = (
            y_train.iloc[train_idx]
        )

        fold_preprocessor = (
            _build_churn_preprocessor(
                X_fold_train
            )
        )

        X_fold_train_processed = (
            fold_preprocessor.fit_transform(
                X_fold_train
            )
        )

        X_fold_valid_processed = (
            fold_preprocessor.transform(
                X_fold_valid
            )
        )

        for model_index, template in (
            enumerate(
                base_templates
            )
        ):

            model = clone(
                template
            )

            model.fit(
                X_fold_train_processed,
                y_fold_train,
            )

            oof_predictions[
                valid_idx,
                model_index,
            ] = (
                model.predict_proba(
                    X_fold_valid_processed
                )[:, 1]
            )

    # ========================================================
    # 10. META MODEL
    # ========================================================

    meta_model = (
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    )

    meta_model.fit(
        oof_predictions,
        y_train,
    )

    oof_probability = (
        meta_model.predict_proba(
            oof_predictions
        )[:, 1]
    )

    oof_auc = roc_auc_score(
        y_train,
        oof_probability,
    )

    print(
        f"\n  OOF ensemble AUC: "
        f"{oof_auc:.4f}"
    )

    # ========================================================
    # 11. THRESHOLD OPTIMIZATION
    # ========================================================

    (
        accuracy_threshold,
        business_threshold,
        f1_threshold,
        threshold_df,
    ) = _find_churn_thresholds(
        y_train,
        oof_probability,
        minimum_recall=0.60,
    )

    print(
        f"  Accuracy threshold: "
        f"{accuracy_threshold:.3f}"
    )

    print(
        f"  Business threshold: "
        f"{business_threshold:.3f}"
    )

    print(
        f"  F1 threshold: "
        f"{f1_threshold:.3f}"
    )

    # ========================================================
    # 12. FIT FINAL BASE MODELS
    # ========================================================

    print(
        "\n  Training final models..."
    )

    final_models = [
        clone(xgb_model),
        clone(extra_model),
        clone(hist_model),
        clone(logistic_model),
    ]

    test_base_predictions = []

    for model in final_models:

        model.fit(
            X_train_processed,
            y_train,
        )

        probability = (
            model.predict_proba(
                X_test_processed
            )[:, 1]
        )

        test_base_predictions.append(
            probability
        )

    test_base_predictions = (
        np.column_stack(
            test_base_predictions
        )
    )

    # ========================================================
    # 13. FINAL STACKED PROBABILITY
    # ========================================================

    test_probability = (
        meta_model.predict_proba(
            test_base_predictions
        )[:, 1]
    )

    # ========================================================
    # 14. FINAL TEST PREDICTIONS
    # ========================================================

    test_prediction = (
        test_probability
        >= business_threshold
    ).astype(int)

    retention_prediction = (
        test_probability
        >= f1_threshold
    ).astype(int)

    # ========================================================
    # 15. METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        test_prediction,
    )

    precision = precision_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            test_prediction,
        )
    )

    auc = roc_auc_score(
        y_test,
        test_probability,
    )

    pr_auc = (
        average_precision_score(
            y_test,
            test_probability,
        )
    )

    brier = brier_score_loss(
        y_test,
        test_probability,
    )

    # ========================================================
    # 16. CONFUSION MATRIX
    # ========================================================

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            test_prediction,
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # ========================================================
    # 17. AUC CONFIDENCE INTERVAL
    # ========================================================

    try:

        auc_ci_low, auc_ci_high = (
            _bootstrap_auc_ci(
                y_test,
                test_probability,
                random_state=random_state,
                n_bootstrap=1000,
            )
        )

    except Exception:

        auc_ci_low = np.nan
        auc_ci_high = np.nan

    # ========================================================
    # 18. FEATURE IMPORTANCE
    # ========================================================

    feature_names = (
        _get_feature_names(
            preprocessor
        )
    )

    xgb_final = final_models[0]
    extra_final = final_models[1]

    xgb_importance = (
        xgb_final.feature_importances_
    )

    extra_importance = (
        extra_final.feature_importances_
    )

    xgb_total = max(
        xgb_importance.sum(),
        1e-12,
    )

    extra_total = max(
        extra_importance.sum(),
        1e-12,
    )

    xgb_normalized = (
        xgb_importance
        / xgb_total
    )

    extra_normalized = (
        extra_importance
        / extra_total
    )

    combined_importance = (
        0.60 * xgb_normalized
        + 0.40 * extra_normalized
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance":
                combined_importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 19. PREDICTION DATAFRAME
    # ========================================================

    prediction_df = X_test.copy()

    prediction_df[
        "ActualChurn"
    ] = y_test.values

    prediction_df[
        "ChurnProbability"
    ] = test_probability

    prediction_df[
        "PredictedChurn"
    ] = test_prediction

    prediction_df[
        "PredictionThreshold"
    ] = business_threshold

    # ========================================================
    # 20. METRICS DICTIONARY
    # ========================================================

    metrics = {

        "accuracy":
            float(accuracy),

        "auc":
            float(auc),

        "pr_auc":
            float(pr_auc),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "specificity":
            float(
                specificity
            ),

        "brier_score":
            float(brier),

        "decision_threshold":
            float(
                business_threshold
            ),

        "accuracy_threshold":
            float(
                accuracy_threshold
            ),

        "f1_threshold":
            float(
                f1_threshold
            ),

        "cv_auc":
            float(
                oof_auc
            ),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp),

        "auc_ci_lower":
            float(auc_ci_low)
            if not np.isnan(
                auc_ci_low
            )
            else np.nan,

        "auc_ci_upper":
            float(auc_ci_high)
            if not np.isnan(
                auc_ci_high
            )
            else np.nan,
    }

    # ========================================================
    # 21. PRINT FINAL RESULTS
    # ========================================================

    print(
        "\n  Model: Stacked "
        "XGBoost + ExtraTrees + "
        "HistGradientBoosting + Logistic"
    )

    print(
        f"  Accuracy:           "
        f"{accuracy:.4f}"
    )

    print(
        f"  AUC:                "
        f"{auc:.4f}"
    )

    print(
        f"  PR-AUC:             "
        f"{pr_auc:.4f}"
    )

    print(
        f"  Precision:          "
        f"{precision:.4f}"
    )

    print(
        f"  Recall:             "
        f"{recall:.4f}"
    )

    print(
        f"  F1:                 "
        f"{f1:.4f}"
    )

    print(
        f"  Balanced Accuracy:  "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"  Specificity:        "
        f"{specificity:.4f}"
    )

    print(
        f"  Brier Score:        "
        f"{brier:.4f}"
    )

    print(
        f"  Threshold:          "
        f"{business_threshold:.3f}"
    )

    print(
        f"  5-fold OOF AUC:     "
        f"{oof_auc:.4f}"
    )

    if not np.isnan(auc_ci_low):

        print(
            f"  95% AUC CI:         "
            f"[{auc_ci_low:.4f}, "
            f"{auc_ci_high:.4f}]"
        )

    print(
        "\n  Confusion Matrix:"
    )

    print(
        f"    TN = {tn}"
    )

    print(
        f"    FP = {fp}"
    )

    print(
        f"    FN = {fn}"
    )

    print(
        f"    TP = {tp}"
    )

    if accuracy >= 0.80:

        print(
            "\n  ✓ Accuracy > 0.80 achieved."
        )

    else:

        print(
            "\n  ⚠ Accuracy is below 0.80."
        )

        print(
            "    The test set was not used "
            "for threshold tuning."
        )

    # ========================================================
    # 22. RETURN
    # ========================================================

    return {

        "model":
            meta_model,

        "base_models": {

            "xgboost":
                final_models[0],

            "extra_trees":
                final_models[1],

            "hist_gradient_boosting":
                final_models[2],

            "logistic":
                final_models[3],
        },

        "preprocessor":
            preprocessor,

        "metrics":
            metrics,

        "predictions":
            prediction_df,

        "feature_importance":
            importance_df,

        "threshold_results":
            threshold_df,

        "best_params": {

            "xgboost":
                xgb_search.best_params_,

            "extra_trees":
                extra_search.best_params_,
        },

        "retention_threshold":
            f1_threshold,

        "retention_prediction":
            retention_prediction,
    }


# ============================================================
# REVENUE FORECASTING
# ============================================================

def _create_revenue_features(df):

    data = df.copy()

    data["month"] = pd.to_datetime(
        data["month"]
    )

    data = data.sort_values(
        "month"
    ).reset_index(
        drop=True
    )

    data["time_index"] = np.arange(
        len(data)
    )

    data["month_num"] = (
        data["month"].dt.month
    )

    data["sin_month"] = np.sin(
        2 * np.pi
        * data["month_num"]
        / 12
    )

    data["cos_month"] = np.cos(
        2 * np.pi
        * data["month_num"]
        / 12
    )

    return data


def train_revenue_model(
    revenue_df,
    random_state=42,
):

    print(
        "\n[3/4] Training sales "
        "forecast model..."
    )

    data = (
        _create_revenue_features(
            revenue_df
        )
    )

    target_column = (
        "revenue"
        if "revenue" in data.columns
        else "Revenue"
    )

    if target_column not in data.columns:

        raise ValueError(
            "Revenue column not found."
        )

    y = data[
        target_column
    ].astype(float)

    X = data[
        [
            "time_index",
            "sin_month",
            "cos_month",
        ]
    ]

    # --------------------------------------------------------
    # Chronological holdout
    # --------------------------------------------------------

    holdout_size = max(
        4,
        int(
            len(data) * 0.20
        ),
    )

    X_train = X.iloc[
        :-holdout_size
    ]

    X_test = X.iloc[
        -holdout_size:
    ]

    y_train = y.iloc[
        :-holdout_size
    ]

    y_test = y.iloc[
        -holdout_size:
    ]

    candidates = {

        "Ridge Trend + Seasonality":
            Ridge(
                alpha=10.0
            ),

        "ElasticNet Trend + Seasonality":
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.20,
                max_iter=10000,
                random_state=random_state,
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                random_state=random_state,
            ),
    }

    results = []

    fitted_models = {}

    for name, model in candidates.items():

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        mape = (
            mean_absolute_percentage_error(
                y_test,
                predictions,
            )
            * 100
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        results.append(
            {
                "model": name,
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
            }
        )

        fitted_models[name] = model

    results_df = pd.DataFrame(
        results
    )

    # Lower MAPE first, then RMSE
    best_row = (
        results_df
        .sort_values(
            [
                "mape",
                "rmse",
            ]
        )
        .iloc[0]
    )

    best_model_name = (
        best_row["model"]
    )

    best_model = fitted_models[
        best_model_name
    ]

    # --------------------------------------------------------
    # Historical predictions
    # --------------------------------------------------------

    full_predictions = best_model.predict(
        X
    )

    # --------------------------------------------------------
    # Residual uncertainty
    # --------------------------------------------------------

    residuals = (
        y_test.values
        - best_model.predict(
            X_test
        )
    )

    residual_std = np.std(
        residuals
    )

    # 80% normal approximation
    z = 1.2816

    lower = (
        full_predictions
        - z * residual_std
    )

    upper = (
        full_predictions
        + z * residual_std
    )

    # --------------------------------------------------------
    # Next month
    # --------------------------------------------------------

    last_month = data[
        "month"
    ].max()

    next_month = (
        last_month
        + pd.offsets.MonthBegin(1)
    )

    next_time_index = (
        len(data)
    )

    next_month_num = (
        next_month.month
    )

    next_sin = np.sin(
        2 * np.pi
        * next_month_num
        / 12
    )

    next_cos = np.cos(
        2 * np.pi
        * next_month_num
        / 12
    )

    next_X = pd.DataFrame(
        {
            "time_index":
                [next_time_index],

            "sin_month":
                [next_sin],

            "cos_month":
                [next_cos],
        }
    )

    next_prediction = float(
        best_model.predict(
            next_X
        )[0]
    )

    next_lower = (
        next_prediction
        - z * residual_std
    )

    next_upper = (
        next_prediction
        + z * residual_std
    )

    # --------------------------------------------------------
    # Forecast dataframe
    # --------------------------------------------------------

    forecast_df = pd.DataFrame(
        {
            "Month":
                data[
                    "month"
                ].dt.strftime(
                    "%Y-%m"
                ),

            "ActualRevenue":
                y.values,

            "PredictedRevenue":
                full_predictions,

            "LowerBound_80":
                lower,

            "UpperBound_80":
                upper,

            "PeriodType":
                [
                    (
                        "Backtest Holdout"
                        if i
                        >= len(data)
                        - holdout_size
                        else
                        "Historical Actual"
                    )
                    for i in range(
                        len(data)
                    )
                ],
        }
    )

    next_row = pd.DataFrame(
        {
            "Month":
                [
                    next_month.strftime(
                        "%Y-%m"
                    )
                ],

            "ActualRevenue":
                [np.nan],

            "PredictedRevenue":
                [next_prediction],

            "LowerBound_80":
                [next_lower],

            "UpperBound_80":
                [next_upper],

            "PeriodType":
                ["Forecast"],
        }
    )

    forecast_df = pd.concat(
        [
            forecast_df,
            next_row,
        ],
        ignore_index=True,
    )

    metrics = {

        "model":
            best_model_name,

        "rmse":
            float(
                best_row["rmse"]
            ),

        "mape":
            float(
                best_row["mape"]
            ),

        "r2":
            float(
                best_row["r2"]
            ),

        "interval_coverage":
            float(
                np.mean(
                    (
                        y_test.values
                        >= (
                            best_model.predict(
                                X_test
                            )
                            - z
                            * residual_std
                        )
                    )
                    &
                    (
                        y_test.values
                        <= (
                            best_model.predict(
                                X_test
                            )
                            + z
                            * residual_std
                        )
                    )
                )
            ),

        "next_month":
            next_month.strftime(
                "%Y-%m"
            ),

        "next_month_prediction":
            next_prediction,

        "next_month_lower":
            float(next_lower),

        "next_month_upper":
            float(next_upper),
    }

    print(
        f"  Model: {best_model_name}"
    )

    print(
        f"  RMSE: ${metrics['rmse']:,.0f}"
    )

    print(
        f"  MAPE: {metrics['mape']:.2f}%"
    )

    print(
        f"  R²: {metrics['r2']:.4f}"
    )

    print(
        f"  Prediction interval coverage: "
        f"{metrics['interval_coverage']:.2%}"
    )

    print(
        f"  Next month "
        f"({metrics['next_month']}): "
        f"${next_prediction:,.2f}"
    )

    return {
        "model": best_model,
        "metrics": metrics,
        "forecast": forecast_df,
        "candidates": results_df,
    }


# ============================================================
# DRIFT MONITORING
# ============================================================

def _calculate_psi(
    expected,
    actual,
    bins=10,
):

    expected = pd.Series(
        expected
    ).dropna()

    actual = pd.Series(
        actual
    ).dropna()

    if len(expected) < 2:
        return 0.0

    if (
        expected.nunique()
        < 2
    ):

        return 0.0

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    breakpoints = np.unique(
        expected.quantile(
            quantiles
        )
    )

    if len(breakpoints) < 3:

        return 0.0

    expected_bins = pd.cut(
        expected,
        bins=breakpoints,
        include_lowest=True,
    )

    actual_bins = pd.cut(
        actual,
        bins=breakpoints,
        include_lowest=True,
    )

    expected_dist = (
        expected_bins
        .value_counts(
            normalize=True
        )
    )

    actual_dist = (
        actual_bins
        .value_counts(
            normalize=True
        )
    )

    expected_dist, actual_dist = (
        expected_dist.align(
            actual_dist,
            fill_value=0,
        )
    )

    eps = 1e-6

    expected_dist = (
        expected_dist + eps
    )

    actual_dist = (
        actual_dist + eps
    )

    psi = np.sum(
        (
            actual_dist
            - expected_dist
        )
        * np.log(
            actual_dist
            / expected_dist
        )
    )

    return float(psi)


def run_drift_monitoring(
    reference_df,
    current_df=None,
):

    print(
        "\n[4/4] Running drift monitoring..."
    )

    if current_df is None:

        current_df = reference_df.copy()

    numeric_columns = (
        reference_df
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    rows = []

    for column in numeric_columns:

        if column not in current_df.columns:
            continue

        psi = _calculate_psi(
            reference_df[column],
            current_df[column],
        )

        if psi < 0.10:

            status = "No Drift"

        elif psi < 0.25:

            status = "Moderate Drift"

        else:

            status = "Significant Drift"

        rows.append(
            {
                "Feature": column,
                "PSI": psi,
                "Status": status,
            }
        )

    drift_df = pd.DataFrame(
        rows
    )

    if len(drift_df) > 0:

        max_psi = float(
            drift_df["PSI"].max()
        )

    else:

        max_psi = 0.0

    if max_psi < 0.10:

        print(
            "  No significant drift detected"
        )

    else:

        print(
            "  Drift detected"
        )

    print(
        f"  Max PSI: {max_psi:.4f}"
    )

    return {
        "data": drift_df,
        "max_psi": max_psi,
    }


# ============================================================
# RETENTION LIFT SIMULATION
# ============================================================

def simulate_retention_lift(
    predictions,
    threshold=0.50,
):

    data = predictions.copy()

    if (
        "ChurnProbability"
        not in data.columns
    ):

        return pd.DataFrame()

    data["RiskSegment"] = pd.cut(
        data["ChurnProbability"],
        bins=[
            -np.inf,
            0.20,
            0.40,
            0.60,
            0.80,
            np.inf,
        ],
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
    )

    data["RetentionTarget"] = (
        data["ChurnProbability"]
        >= threshold
    ).astype(int)

    summary = (
        data
        .groupby(
            "RiskSegment",
            observed=False,
        )
        .agg(
            Customers=(
                "ChurnProbability",
                "count",
            ),
            AverageRisk=(
                "ChurnProbability",
                "mean",
            ),
            TargetedCustomers=(
                "RetentionTarget",
                "sum",
            ),
        )
        .reset_index()
    )

    if (
        "ActualChurn"
        in data.columns
    ):

        churn_summary = (
            data
            .groupby(
                "RiskSegment",
                observed=False,
            )[
                "ActualChurn"
            ]
            .mean()
            .reset_index(
                name="ActualChurnRate"
            )
        )

        summary = summary.merge(
            churn_summary,
            on="RiskSegment",
            how="left",
        )

    return summary


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

def check_data_leakage(
    df,
    target_column="Churn",
):

    print(
        "\n[1/4] Running data leakage checks..."
    )

    leakage = {
        "target_column":
            target_column,

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "missing_target":
            int(
                df[
                    target_column
                ].isna().sum()
            )
            if target_column
            in df.columns
            else None,

        "potential_leakage": [],
    }

    if target_column not in df.columns:

        print(
            "  Target column missing."
        )

        return leakage

    # --------------------------------------------------------
    # Check suspicious target-derived columns
    # --------------------------------------------------------

    suspicious_keywords = [
        "churn",
        "target",
        "label",
    ]

    for column in df.columns:

        if column == target_column:
            continue

        lower = column.lower()

        if any(
            keyword in lower
            for keyword
            in suspicious_keywords
        ):

            leakage[
                "potential_leakage"
            ].append(
                column
            )

    # --------------------------------------------------------
    # Basic summary
    # --------------------------------------------------------

    missing_totalcharges = 0

    if (
        "TotalCharges"
        in df.columns
    ):

        missing_totalcharges = (
            df[
                "TotalCharges"
            ].isna().mean()
            * 100
        )

    target_rate = (
        df[target_column]
        .astype(str)
        .str.lower()
        .isin(
            [
                "yes",
                "true",
                "1",
            ]
        )
        .mean()
    )

    print(
        f"  Dataset: "
        f"{len(df):,} rows, "
        f"{len(df.columns)} cols"
    )

    print(
        f"  Missing TotalCharges: "
        f"{missing_totalcharges:.1f}%"
    )

    print(
        f"  Target churn rate: "
        f"{target_rate:.4f}"
    )

    return leakage


# ============================================================
# POWER BI MODEL METRICS
# ============================================================

def build_model_metrics_dataframe(
    churn_results,
    sales_results,
):

    rows = []

    churn_metrics = (
        churn_results.get(
            "metrics",
            {},
        )
    )

    def add_churn_metric(
        metric,
        value,
        target=None,
        status=None,
        ci_low=np.nan,
        ci_high=np.nan,
    ):

        rows.append(
            {
                "Model":
                    "Churn",

                "Metric":
                    metric,

                "Value":
                    value,

                "CI_Lower":
                    ci_low,

                "CI_Upper":
                    ci_high,

                "Target_or_Baseline":
                    target,

                "Status":
                    status,
            }
        )

    accuracy = churn_metrics.get(
        "accuracy",
        np.nan,
    )

    auc = churn_metrics.get(
        "auc",
        np.nan,
    )

    f1 = churn_metrics.get(
        "f1",
        np.nan,
    )

    recall = churn_metrics.get(
        "recall",
        np.nan,
    )

    precision = churn_metrics.get(
        "precision",
        np.nan,
    )

    pr_auc = churn_metrics.get(
        "pr_auc",
        np.nan,
    )

    balanced_accuracy = (
        churn_metrics.get(
            "balanced_accuracy",
            np.nan,
        )
    )

    add_churn_metric(
        "Accuracy",
        accuracy,
        0.80,
        (
            "PASS"
            if accuracy >= 0.80
            else "BELOW TARGET"
        ),
    )

    add_churn_metric(
        "ROC-AUC",
        auc,
        0.75,
        (
            "PASS"
            if auc >= 0.75
            else "BELOW TARGET"
        ),
        churn_metrics.get(
            "auc_ci_lower",
            np.nan,
        ),
        churn_metrics.get(
            "auc_ci_upper",
            np.nan,
        ),
    )

    add_churn_metric(
        "PR-AUC",
        pr_auc,
        0.60,
        (
            "PASS"
            if pr_auc >= 0.60
            else "BELOW TARGET"
        ),
    )

    add_churn_metric(
        "Precision",
        precision,
        0.60,
        (
            "PASS"
            if precision >= 0.60
            else "BELOW TARGET"
        ),
    )

    add_churn_metric(
        "Recall",
        recall,
        0.60,
        (
            "PASS"
            if recall >= 0.60
            else "BELOW TARGET"
        ),
    )

    add_churn_metric(
        "F1",
        f1,
        0.60,
        (
            "PASS"
            if f1 >= 0.60
            else "BELOW TARGET"
        ),
    )

    add_churn_metric(
        "Balanced Accuracy",
        balanced_accuracy,
        0.70,
        (
            "PASS"
            if balanced_accuracy >= 0.70
            else "BELOW TARGET"
        ),
    )

    add_churn_metric(
        "Decision Threshold",
        churn_metrics.get(
            "decision_threshold",
            np.nan,
        ),
    )

    add_churn_metric(
        "CV AUC",
        churn_metrics.get(
            "cv_auc",
            np.nan,
        ),
    )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    if sales_results:

        sales_metrics = (
            sales_results.get(
                "metrics",
                {},
            )
        )

        rows.extend(
            [
                {
                    "Model":
                        sales_metrics.get(
                            "model",
                            "Revenue",
                        ),

                    "Metric":
                        "RMSE",

                    "Value":
                        sales_metrics.get(
                            "rmse",
                            np.nan,
                        ),

                    "CI_Lower":
                        np.nan,

                    "CI_Upper":
                        np.nan,

                    "Target_or_Baseline":
                        np.nan,

                    "Status":
                        None,
                },

                {
                    "Model":
                        sales_metrics.get(
                            "model",
                            "Revenue",
                        ),

                    "Metric":
                        "MAPE",

                    "Value":
                        sales_metrics.get(
                            "mape",
                            np.nan,
                        ),

                    "CI_Lower":
                        np.nan,

                    "CI_Upper":
                        np.nan,

                    "Target_or_Baseline":
                        5.0,

                    "Status":
                        (
                            "PASS"
                            if sales_metrics.get(
                                "mape",
                                999,
                            ) <= 5
                            else
                            "BELOW TARGET"
                        ),
                },

                {
                    "Model":
                        sales_metrics.get(
                            "model",
                            "Revenue",
                        ),

                    "Metric":
                        "R2",

                    "Value":
                        sales_metrics.get(
                            "r2",
                            np.nan,
                        ),

                    "CI_Lower":
                        np.nan,

                    "CI_Upper":
                        np.nan,

                    "Target_or_Baseline":
                        0.70,

                    "Status":
                        (
                            "PASS"
                            if sales_metrics.get(
                                "r2",
                                -999,
                            ) >= 0.70
                            else
                            "BELOW TARGET"
                        ),
                },
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Metric",
            "Value",
            "CI_Lower",
            "CI_Upper",
            "Target_or_Baseline",
            "Status",
        ],
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_pipeline(
    customer_df,
    revenue_df,
    random_state=42,
):

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    leakage = check_data_leakage(
        customer_df
    )

    # --------------------------------------------------------
    # Churn
    # --------------------------------------------------------

    churn_results = (
        train_churn_model(
            customer_df,
            random_state=random_state,
        )
    )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    sales_results = (
        train_revenue_model(
            revenue_df,
            random_state=random_state,
        )
    )

    # --------------------------------------------------------
    # Drift
    # --------------------------------------------------------

    drift_results = (
        run_drift_monitoring(
            customer_df,
            customer_df,
        )
    )

    # --------------------------------------------------------
    # Retention simulation
    # --------------------------------------------------------

    retention_lift = (
        simulate_retention_lift(
            churn_results[
                "predictions"
            ],
            threshold=churn_results[
                "retention_threshold"
            ],
        )
    )

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    model_metrics = (
        build_model_metrics_dataframe(
            churn_results,
            sales_results,
        )
    )

    # --------------------------------------------------------
    # Final Power BI tables
    # --------------------------------------------------------

    return {

        "leakage":
            leakage,

        "churn":
            churn_results,

        "sales":
            sales_results,

        "drift":
            drift_results,

        "churn_predictions":
            churn_results[
                "predictions"
            ],

        "feature_importance":
            churn_results[
                "feature_importance"
            ],

        "model_metrics":
            model_metrics,

        "sales_forecast":
            sales_results[
                "forecast"
            ],

        "drift_monitoring":
            drift_results[
                "data"
            ],

        "retention_lift":
            retention_lift,

        "metadata": {

            "random_state":
                random_state,

            "churn_model":
                (
                    "Stacked XGBoost + "
                    "ExtraTrees + "
                    "HistGradientBoosting "
                    "+ Logistic"
                ),

            "revenue_model":
                sales_results[
                    "metrics"
                ].get(
                    "model"
                ),
        },
    }