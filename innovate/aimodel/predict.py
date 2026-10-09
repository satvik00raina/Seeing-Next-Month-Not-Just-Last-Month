import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "churn_preprocessor.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Churn model not found:\n{MODEL_PATH}\n"
            "Train the model first."
        )

    model = joblib.load(MODEL_PATH)

    preprocessor = None

    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # Remove target column if it exists.
    # Prediction data should not use the actual Churn value.
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Remove ID-like columns if they exist.
    id_columns = [
        "CustomerID",
        "customerID",
        "customer_id",
        "ID",
        "id"
    ]

    for col in id_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


# ============================================================
# PREDICT CHURN
# ============================================================

def predict_churn(input_csv, output_csv=None):

    print("\nLoading customer data...")

    df = pd.read_csv(input_csv)

    print(f"Input rows: {len(df)}")

    original_df = df.copy()

    X = prepare_data(df)

    model, preprocessor = load_model()

    # --------------------------------------------------------
    # Apply preprocessing if a separate preprocessor exists
    # --------------------------------------------------------

    if preprocessor is not None:
        X_processed = preprocessor.transform(X)
    else:
        X_processed = X

    # --------------------------------------------------------
    # Probability of churn
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        churn_probability = model.predict_proba(
            X_processed
        )[:, 1]

    else:

        predictions = model.predict(X_processed)

        churn_probability = predictions.astype(float)

    # --------------------------------------------------------
    # Churn prediction
    # --------------------------------------------------------

    churn_prediction = (
        churn_probability >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    risk_category = np.select(
        [
            churn_probability >= 0.70,
            churn_probability >= 0.40
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    # --------------------------------------------------------
    # Revenue at risk
    # --------------------------------------------------------

    if "MonthlyCharges" in original_df.columns:

        revenue_at_risk = (
            original_df["MonthlyCharges"]
            * churn_probability
        )

    else:

        revenue_at_risk = np.zeros(
            len(original_df)
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = original_df.copy()

    result["ChurnProbability"] = np.round(
        churn_probability,
        4
    )

    result["PredictedChurn"] = churn_prediction

    result["ChurnRisk"] = risk_category

    result["RevenueAtRisk"] = np.round(
        revenue_at_risk,
        2
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if output_csv is None:

        output_csv = os.path.join(
            MODEL_DIR,
            "churn_predictions_enriched.csv"
        )

    result.to_csv(
        output_csv,
        index=False
    )

    print("\nPrediction completed.")

    print(
        f"High risk customers: "
        f"{(risk_category == 'High').sum()}"
    )

    print(
        f"Medium risk customers: "
        f"{(risk_category == 'Medium').sum()}"
    )

    print(
        f"Low risk customers: "
        f"{(risk_category == 'Low').sum()}"
    )

    print(
        f"\nSaved predictions to:\n{output_csv}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_file = os.path.join(
        BASE_DIR,
        "data",
        "customer_data.csv"
    )

    output_file = os.path.join(
        MODEL_DIR,
        "churn_predictions_enriched.csv"
    )

    predict_churn(
        input_file,
        output_file
    )