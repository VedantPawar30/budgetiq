"""
OLIVE — ML Models Engine
Time-series forecasting (Holt-Winters), XGBoost prediction, and SHAP explainability.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap

warnings.filterwarnings("ignore")

# ─── Feature columns used for XGBoost ───────────────────────────────────────
ALLOCATION_COLS = [
    "health_alloc_cr", "education_alloc_cr", "agriculture_alloc_cr",
    "infrastructure_alloc_cr", "water_alloc_cr", "energy_alloc_cr",
]

SOCIO_ECONOMIC_FEATURES = [
    "population", "population_density", "urban_population_pct",
    "sc_st_population_pct", "working_age_population_pct", "female_male_ratio",
    "gdp_per_capita", "gsdp_growth_rate", "poverty_rate", "unemployment_rate",
    "average_household_income", "tax_revenue_per_capita", "credit_deposit_ratio",
    "imr", "mmr", "life_expectancy", "hospital_beds_per_1000", "doctors_per_1000",
    "immunization_coverage_pct", "malnutrition_rate",
    "literacy_rate", "ger_secondary", "ger_higher",
    "dropout_rate_secondary", "student_teacher_ratio",
    "irrigation_coverage_pct", "crop_yield_index", "farmer_income_inr",
    "electrification_rate", "internet_penetration_pct", "road_density",
    "safe_water_access_pct", "sanitation_coverage_pct",
    "renewable_energy_pct",
    "absorption_capacity", "uplift_multiplier",
]

ALL_FEATURES = ALLOCATION_COLS + SOCIO_ECONOMIC_FEATURES

NEED_INDEX_COLS = [
    "health_need_index", "education_need_index",
    "infrastructure_need_index", "overall_need_index",
]

SECTOR_NAMES = ["Healthcare", "Education", "Agriculture", "Infrastructure", "Water", "Energy"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. TIME-SERIES FORECASTING (Holt-Winters / Linear Trend)
# ═══════════════════════════════════════════════════════════════════════════

def _forecast_series(values):
    """Forecast the next value given a short time series.

    Uses Holt's linear trend (exponential smoothing) when possible,
    falls back to simple linear regression for very short series.

    Args:
        values: array-like of yearly values (7 data points for 2018-2024).

    Returns:
        float: Forecasted next-period value.
    """
    y = np.array(values, dtype=float)
    n = len(y)

    if n < 3:
        # Too short — just return last value
        return float(y[-1])

    try:
        # Holt's linear trend model (no seasonality for yearly data)
        model = ExponentialSmoothing(
            y, trend="add", seasonal=None, initialization_method="estimated"
        )
        fit = model.fit(optimized=True)
        forecast = fit.forecast(1)
        return max(float(forecast[0]), 5.0)
    except Exception:
        # Fallback to simple linear regression
        X = np.arange(n).reshape(-1, 1)
        lr = LinearRegression().fit(X, y)
        pred = lr.predict([[n]])[0]
        return max(float(pred), 5.0)


def train_prophet_forecasts(df):
    """Train time-series models on state-level need indices and forecast 2025.

    Uses Holt-Winters exponential smoothing (additive trend, no seasonality).
    We aggregate to state level (5 states x 4 indices = 20 models) for speed,
    then distribute forecasts back to districts proportionally.

    Args:
        df: The OLIVE dataset (2018-2024).

    Returns:
        dict: {state: {need_col: forecast_2025_value}} at state level
        pd.DataFrame: District-level 2025 forecasts
    """
    state_forecasts = {}
    district_forecasts = []

    for state in df["state"].unique():
        state_df = df[df["state"] == state]
        state_forecasts[state] = {}

        for col in NEED_INDEX_COLS:
            # Aggregate to state-year level (mean across districts)
            ts = state_df.groupby("year")[col].mean().sort_index()
            predicted_val = _forecast_series(ts.values)
            state_forecasts[state][col] = round(predicted_val, 2)

        # Distribute to districts proportionally based on last known year
        last_year = state_df[state_df["year"] == 2024]
        for _, row in last_year.iterrows():
            d_forecast = {
                "district": row["district"],
                "state": state,
                "state_code": row["state_code"],
                "state_tier": row["state_tier"],
                "year": 2025,
            }
            for col in NEED_INDEX_COLS:
                # Scale district's value proportionally
                state_2024_mean = last_year[col].mean()
                if state_2024_mean > 0:
                    ratio = row[col] / state_2024_mean
                    d_forecast[col] = round(state_forecasts[state][col] * ratio, 2)
                else:
                    d_forecast[col] = state_forecasts[state][col]
            district_forecasts.append(d_forecast)

    forecast_df = pd.DataFrame(district_forecasts)
    print(f"[OK] Forecasting: Trained {len(NEED_INDEX_COLS) * len(df['state'].unique())} models, "
          f"forecasted {len(forecast_df)} district-level values for 2025")
    return state_forecasts, forecast_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. XGBOOST PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def train_xgboost_model(df, target_col="overall_welfare_score"):
    """Train an XGBoost regressor to predict welfare/satisfaction scores.

    Args:
        df: The OLIVE dataset.
        target_col: Target variable to predict.

    Returns:
        model: Trained XGBRegressor
        metrics: dict with MAE, R2 on test set
        feature_names: list of feature names used
    """
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any NaN
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 3),
        "r2": round(r2_score(y_test, y_pred), 3),
    }
    
    print(f"[OK] XGBoost ({target_col}): MAE={metrics['mae']}, R2={metrics['r2']}")
    return model, metrics, feature_cols


def predict_outcomes(model, X_new, feature_names):
    """Predict outcomes for new/proposed allocations.

    Args:
        model: Trained XGBRegressor.
        X_new: DataFrame or dict with allocation + socio-economic features.
        feature_names: Feature column names the model was trained on.

    Returns:
        np.ndarray: Predicted values.
    """
    if isinstance(X_new, dict):
        X_new = pd.DataFrame([X_new])

    # Ensure columns match
    for col in feature_names:
        if col not in X_new.columns:
            X_new[col] = 0

    X_new = X_new[feature_names].fillna(0)
    return model.predict(X_new)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════

def get_shap_values(model, X, feature_names):
    """Compute SHAP values for the XGBoost model.

    Args:
        model: Trained XGBRegressor.
        X: Feature DataFrame.
        feature_names: Feature column names.

    Returns:
        shap.Explanation: SHAP explanation object.
    """
    if isinstance(X, dict):
        X = pd.DataFrame([X])
    X = X[feature_names].fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def generate_natural_language_explanation(model, district_row, feature_names, df_latest):
    """Generate human-readable SHAP explanations for a district.

    Args:
        model: Trained XGBRegressor.
        district_row: Series/dict for the selected district.
        feature_names: Feature names used in training.
        df_latest: Latest year's full DataFrame for context.

    Returns:
        list[dict]: List of explanation dicts with sector, direction, reason.
    """
    X = pd.DataFrame([district_row])[feature_names].fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X)

    feature_shap = dict(zip(feature_names, shap_vals.values[0]))

    # Map features to sectors
    sector_features = {
        "Healthcare": ["health_alloc_cr", "imr", "mmr", "life_expectancy",
                        "hospital_beds_per_1000", "doctors_per_1000",
                        "immunization_coverage_pct", "malnutrition_rate"],
        "Education": ["education_alloc_cr", "literacy_rate", "ger_secondary",
                       "ger_higher", "dropout_rate_secondary", "student_teacher_ratio"],
        "Agriculture": ["agriculture_alloc_cr", "irrigation_coverage_pct",
                          "crop_yield_index", "farmer_income_inr"],
        "Infrastructure": ["infrastructure_alloc_cr", "electrification_rate",
                            "internet_penetration_pct", "road_density"],
        "Water & Energy": ["water_alloc_cr", "energy_alloc_cr",
                            "safe_water_access_pct", "sanitation_coverage_pct",
                            "renewable_energy_pct"],
    }

    # Human-readable feature names
    feature_labels = {
        "health_alloc_cr": "Healthcare Allocation",
        "imr": "Infant Mortality Rate (IMR)",
        "mmr": "Maternal Mortality Ratio (MMR)",
        "life_expectancy": "Life Expectancy",
        "hospital_beds_per_1000": "Hospital Beds per 1000",
        "doctors_per_1000": "Doctors per 1000",
        "immunization_coverage_pct": "Immunization Coverage",
        "malnutrition_rate": "Malnutrition Rate",
        "education_alloc_cr": "Education Allocation",
        "literacy_rate": "Literacy Rate",
        "ger_secondary": "Secondary GER",
        "ger_higher": "Higher Education GER",
        "dropout_rate_secondary": "Secondary Dropout Rate",
        "student_teacher_ratio": "Student-Teacher Ratio",
        "agriculture_alloc_cr": "Agriculture Allocation",
        "irrigation_coverage_pct": "Irrigation Coverage",
        "crop_yield_index": "Crop Yield Index",
        "farmer_income_inr": "Farmer Income",
        "infrastructure_alloc_cr": "Infrastructure Allocation",
        "electrification_rate": "Electrification Rate",
        "internet_penetration_pct": "Internet Penetration",
        "road_density": "Road Density",
        "water_alloc_cr": "Water Allocation",
        "energy_alloc_cr": "Energy Allocation",
        "safe_water_access_pct": "Safe Water Access",
        "sanitation_coverage_pct": "Sanitation Coverage",
        "renewable_energy_pct": "Renewable Energy Share",
        "absorption_capacity": "Absorption Capacity",
        "poverty_rate": "Poverty Rate",
        "unemployment_rate": "Unemployment Rate",
        "gdp_per_capita": "GDP per Capita",
        "population": "Population",
        "population_density": "Population Density",
        "urban_population_pct": "Urban Population %",
        "uplift_multiplier": "State Tier Uplift",
    }

    explanations = []

    for sector, feats in sector_features.items():
        relevant = [(f, feature_shap.get(f, 0)) for f in feats if f in feature_shap]
        relevant.sort(key=lambda x: abs(x[1]), reverse=True)

        if not relevant:
            continue

        top_feat, top_shap = relevant[0]
        feat_label = feature_labels.get(top_feat, top_feat)
        feat_val = district_row.get(top_feat, 0)
        abs_cap = district_row.get("absorption_capacity", 0)

        # Determine direction
        direction = "increased" if top_shap > 0 else "decreased"
        impact = "positively" if top_shap > 0 else "negatively"

        # Compute percentile relative to dataset
        if top_feat in df_latest.columns:
            percentile = (df_latest[top_feat] < feat_val).mean() * 100
            pct_label = f"({percentile:.0f}th percentile)"
        else:
            pct_label = ""

        # Build natural language explanation
        if abs(top_shap) > 0.5:
            strength = "significantly"
        elif abs(top_shap) > 0.2:
            strength = "moderately"
        else:
            strength = "slightly"

        explanation = (
            f"{sector} allocation was {strength} {direction} because "
            f"**{feat_label}** {pct_label} {impact} influences welfare outcomes. "
            f"Current value: **{feat_val:,.1f}**"
        )
        if abs_cap > 0:
            explanation += f", and this district's absorption capacity is **{abs_cap*100:.0f}%**."
        else:
            explanation += "."

        explanations.append({
            "sector": sector,
            "direction": direction,
            "strength": strength,
            "top_feature": feat_label,
            "shap_value": round(top_shap, 3),
            "feature_value": feat_val,
            "explanation": explanation,
        })

    return explanations
