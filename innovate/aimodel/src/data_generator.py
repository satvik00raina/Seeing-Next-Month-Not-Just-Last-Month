"""
Synthetic Telco Customer Churn Dataset Generator
=================================================

Generates:
1. A realistic synthetic telco customer churn dataset.
2. Monthly revenue time-series data for sales/revenue forecasting.

The synthetic data contains known relationships so that the ML models
can learn meaningful churn and revenue patterns.
"""

import numpy as np
import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


# ============================================================
# CUSTOMER DATA
# ============================================================

def generate_customer_data(
    n_customers=7043,
    seed=42
):
    """
    Generate synthetic customer data with realistic churn drivers.

    Churn probability is influenced by:
        - Contract type
        - Tenure
        - Monthly charges
        - Online security
        - Tech support
        - Senior citizen status
        - Internet service
        - Paperless billing
        - Payment method

    Returns
    -------
    pd.DataFrame
        Customer-level data containing customer features and Churn.
    """

    rng = np.random.RandomState(seed)

    # --------------------------------------------------------
    # Demographics
    # --------------------------------------------------------

    customer_ids = [
        f"CUST-{i:05d}"
        for i in range(n_customers)
    ]

    gender = rng.choice(
        ["Male", "Female"],
        n_customers
    )

    senior_citizen = rng.binomial(
        1,
        0.16,
        n_customers
    )

    partner = rng.choice(
        ["Yes", "No"],
        n_customers,
        p=[0.48, 0.52]
    )

    dependents = rng.choice(
        ["Yes", "No"],
        n_customers,
        p=[0.30, 0.70]
    )

    # --------------------------------------------------------
    # Account information
    # --------------------------------------------------------

    tenure = np.clip(
        rng.exponential(
            scale=32,
            size=n_customers
        ),
        1,
        72
    ).astype(int)

    contract = rng.choice(
        [
            "Month-to-month",
            "One year",
            "Two year"
        ],
        n_customers,
        p=[0.55, 0.21, 0.24]
    )

    paperless_billing = rng.choice(
        ["Yes", "No"],
        n_customers,
        p=[0.59, 0.41]
    )

    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        n_customers,
        p=[0.34, 0.23, 0.22, 0.21]
    )

    # --------------------------------------------------------
    # Services
    # --------------------------------------------------------

    phone_service = rng.choice(
        ["Yes", "No"],
        n_customers,
        p=[0.90, 0.10]
    )

    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(
            ["Yes", "No"],
            n_customers,
            p=[0.42, 0.58]
        )
    )

    internet_service = rng.choice(
        [
            "DSL",
            "Fiber optic",
            "No"
        ],
        n_customers,
        p=[0.34, 0.44, 0.22]
    )

    def _internet_addon(
        internet_service,
        rng,
        n,
        p_yes=0.35
    ):
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(
                ["Yes", "No"],
                n,
                p=[p_yes, 1 - p_yes]
            )
        )

    online_security = _internet_addon(
        internet_service,
        rng,
        n_customers,
        0.29
    )

    online_backup = _internet_addon(
        internet_service,
        rng,
        n_customers,
        0.34
    )

    device_protection = _internet_addon(
        internet_service,
        rng,
        n_customers,
        0.34
    )

    tech_support = _internet_addon(
        internet_service,
        rng,
        n_customers,
        0.29
    )

    streaming_tv = _internet_addon(
        internet_service,
        rng,
        n_customers,
        0.38
    )

    streaming_movies = _internet_addon(
        internet_service,
        rng,
        n_customers,
        0.39
    )

    # --------------------------------------------------------
    # Monthly charges
    # --------------------------------------------------------

    base_charge = 20.0

    monthly_charges = (
        base_charge
        + rng.normal(
            0,
            2,
            n_customers
        )
    )

    monthly_charges += np.where(
        phone_service == "Yes",
        20,
        0
    )

    monthly_charges += np.where(
        multiple_lines == "Yes",
        10,
        0
    )

    monthly_charges += np.where(
        internet_service == "DSL",
        15,
        0
    )

    monthly_charges += np.where(
        internet_service == "Fiber optic",
        30,
        0
    )

    monthly_charges += np.where(
        online_security == "Yes",
        5,
        0
    )

    monthly_charges += np.where(
        online_backup == "Yes",
        5,
        0
    )

    monthly_charges += np.where(
        device_protection == "Yes",
        5,
        0
    )

    monthly_charges += np.where(
        tech_support == "Yes",
        5,
        0
    )

    monthly_charges += np.where(
        streaming_tv == "Yes",
        10,
        0
    )

    monthly_charges += np.where(
        streaming_movies == "Yes",
        10,
        0
    )

    monthly_charges = np.round(
        np.clip(
            monthly_charges,
            18.0,
            120.0
        ),
        2
    )

    # --------------------------------------------------------
    # Total charges
    # --------------------------------------------------------

    total_charges = np.round(
        monthly_charges * tenure
        + rng.normal(
            0,
            50,
            n_customers
        ),
        2
    )

    total_charges = np.clip(
        total_charges,
        monthly_charges,
        None
    )

    # --------------------------------------------------------
    # Churn probability
    # --------------------------------------------------------

    logit = (
        -1.5

        + 0.8
        * (contract == "Month-to-month").astype(float)

        - 0.6
        * (contract == "Two year").astype(float)

        - 0.025 * tenure

        + 0.012 * monthly_charges

        + 0.4 * senior_citizen

        - 0.3
        * (online_security == "Yes").astype(float)

        - 0.3
        * (tech_support == "Yes").astype(float)

        + 0.3
        * (internet_service == "Fiber optic").astype(float)

        + 0.2
        * (paperless_billing == "Yes").astype(float)

        + 0.3
        * (payment_method == "Electronic check").astype(float)

        + rng.normal(
            0,
            0.3,
            n_customers
        )
    )

    churn_prob = _sigmoid(
        logit
    )

    churn = (
        rng.random(n_customers)
        < churn_prob
    ).astype(int)

    # --------------------------------------------------------
    # Assemble customer DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    # --------------------------------------------------------
    # Small amount of realistic missing data
    # --------------------------------------------------------

    missing_count = int(
        n_customers * 0.002
    )

    missing_idx = rng.choice(
        df.index,
        size=missing_count,
        replace=False
    )

    df.loc[
        missing_idx,
        "TotalCharges"
    ] = np.nan

    return df


# ============================================================
# MONTHLY REVENUE DATA
# ============================================================

def generate_monthly_revenue(
    n_months=60,
    seed=42
):
    """
    Generate monthly revenue time-series data.

    Generates:
        - month
        - revenue
        - new_customers
        - churned_customers
        - active_customers

    The series contains:
        - long-term upward trend
        - annual seasonality
        - quarterly seasonality
        - random noise
        - customer growth
        - churn variation

    Returns
    -------
    pd.DataFrame
    """

    rng = np.random.RandomState(
        seed
    )

    # --------------------------------------------------------
    # Correct monthly dates
    # --------------------------------------------------------
    #
    # Do NOT use timedelta(days=30.4375 * i).
    #
    # That can create duplicate months because calendar months
    # do not all contain the same number of days.
    #

    months = pd.date_range(
        start="2021-01-01",
        periods=n_months,
        freq="MS"
    )

    # --------------------------------------------------------
    # Revenue trend
    # --------------------------------------------------------

    base_revenue = 350000

    trend = np.linspace(
        0,
        250000,
        n_months
    )

    # --------------------------------------------------------
    # Annual seasonality
    # --------------------------------------------------------

    time_index = np.arange(
        n_months
    )

    yearly_season = (
        25000
        * np.sin(
            2 * np.pi * time_index / 12
            - np.pi / 3
        )
    )

    # --------------------------------------------------------
    # Quarterly seasonality
    # --------------------------------------------------------

    quarterly_season = (
        8000
        * np.cos(
            2 * np.pi * time_index / 3
        )
    )

    # --------------------------------------------------------
    # Random noise
    # --------------------------------------------------------

    noise = rng.normal(
        0,
        6000,
        n_months
    )

    revenue = (
        base_revenue
        + trend
        + yearly_season
        + quarterly_season
        + noise
    )

    revenue = np.round(
        np.clip(
            revenue,
            300000,
            None
        ),
        2
    )

    # --------------------------------------------------------
    # Customer counts
    # --------------------------------------------------------

    base_active = 4200

    new_customers = np.clip(
        rng.poisson(
            190,
            n_months
        )
        + (trend / 1200).astype(int),
        100,
        450
    )

    churn_rate_series = (
        0.03
        + 0.008
        * np.sin(
            2 * np.pi * time_index / 12
        )
    )

    active_customers = np.zeros(
        n_months
    )

    churned_customers = np.zeros(
        n_months
    )

    # --------------------------------------------------------
    # Customer evolution
    # --------------------------------------------------------

    for i in range(n_months):

        if i == 0:

            active_customers[i] = (
                base_active
                + new_customers[i]
            )

        else:

            active_customers[i] = (
                active_customers[i - 1]
                + new_customers[i]
            )

        churned = int(
            active_customers[i]
            * churn_rate_series[i]
        )

        churned_customers[i] = churned

        active_customers[i] -= churned

    # --------------------------------------------------------
    # Assemble revenue DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        {
            "month": months.strftime("%Y-%m"),
            "revenue": revenue,
            "new_customers": new_customers.astype(int),
            "churned_customers": churned_customers.astype(int),
            "active_customers": active_customers.astype(int),
        }
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import os

    os.makedirs(
        "data",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate customer data
    # --------------------------------------------------------

    customers = generate_customer_data()

    print(
        f"Generated {len(customers)} customers"
    )

    print(
        f"Churn rate: "
        f"{customers['Churn'].mean():.1%}"
    )

    customers.to_csv(
        "data/telco_customers.csv",
        index=False
    )

    # --------------------------------------------------------
    # Generate revenue data
    # --------------------------------------------------------

    revenue = generate_monthly_revenue()

    print(
        f"Generated {len(revenue)} months of revenue data"
    )

    print(
        f"Revenue period: "
        f"{revenue['month'].iloc[0]} "
        f"to "
        f"{revenue['month'].iloc[-1]}"
    )

    revenue.to_csv(
        "data/monthly_revenue.csv",
        index=False
    )

    print(
        "Data saved to data/ directory"
    )