"""
OLIVE — Data Generator Engine
Generates a realistic synthetic dataset for Indian public budget allocation.
Timeframe: FY 2018–2024 | Scale: 5 states, 50 districts
"""

import numpy as np
import pandas as pd
import os

# ─── Seed for reproducibility ───────────────────────────────────────────────
np.random.seed(42)

# ─── Constants ───────────────────────────────────────────────────────────────
YEARS = list(range(2018, 2025))  # 2018-2024
NUM_YEARS = len(YEARS)

# State definitions with tiers and realistic base characteristics
STATES = {
    "Maharashtra": {
        "state_code": "MH", "tier": "A",
        "districts": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad",
                       "Thane", "Kolhapur", "Solapur", "Amravati", "Ratnagiri"],
        "base_gdp": 180000, "base_pop": 1200000, "urban_bias": 0.55,
        "base_literacy": 82, "base_health": 70,
    },
    "Karnataka": {
        "state_code": "KA", "tier": "A",
        "districts": ["Bengaluru", "Mysuru", "Hubli-Dharwad", "Mangaluru",
                       "Belagavi", "Kalaburagi", "Davangere", "Ballari",
                       "Vijayapura", "Shivamogga"],
        "base_gdp": 165000, "base_pop": 900000, "urban_bias": 0.48,
        "base_literacy": 76, "base_health": 65,
    },
    "Uttar Pradesh": {
        "state_code": "UP", "tier": "C",
        "districts": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Prayagraj",
                       "Meerut", "Gorakhpur", "Jhansi", "Bareilly", "Moradabad"],
        "base_gdp": 55000, "base_pop": 2500000, "urban_bias": 0.25,
        "base_literacy": 68, "base_health": 45,
    },
    "Rajasthan": {
        "state_code": "RJ", "tier": "B",
        "districts": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer",
                       "Bikaner", "Alwar", "Bharatpur", "Sikar", "Pali"],
        "base_gdp": 85000, "base_pop": 1100000, "urban_bias": 0.30,
        "base_literacy": 67, "base_health": 50,
    },
    "Odisha": {
        "state_code": "OD", "tier": "C",
        "districts": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur",
                       "Sambalpur", "Balasore", "Puri", "Koraput",
                       "Kalahandi", "Mayurbhanj"],
        "base_gdp": 62000, "base_pop": 700000, "urban_bias": 0.22,
        "base_literacy": 73, "base_health": 48,
    },
}

UPLIFT_MAP = {"A": 1.4, "B": 1.1, "C": 0.9}


def _clamp(val, lo, hi):
    """Clamp a value or array between lo and hi."""
    return np.clip(val, lo, hi)


def _trend(base, year, rate=0.01, noise=0.02):
    """Apply a yearly growth trend with noise."""
    years_elapsed = year - 2018
    growth = base * (1 + rate) ** years_elapsed
    return growth * (1 + np.random.uniform(-noise, noise))


def _gen_district_rows(state_name, state_info):
    """Generate all rows for a single state across all years."""
    rows = []
    tier = state_info["tier"]
    uplift = UPLIFT_MAP[tier]
    code = state_info["state_code"]

    for di, district in enumerate(state_info["districts"]):
        # District-level random offsets (persistent across years)
        d_offset = np.random.uniform(-0.15, 0.15)
        is_urban_hub = di < 3  # First 3 districts are urban hubs

        for year in YEARS:
            row = {}
            yr_idx = year - 2018

            # ── Identifiers ──────────────────────────────────
            row["district"] = district
            row["state"] = state_name
            row["state_code"] = code
            row["year"] = year
            row["state_tier"] = tier

            # ── Demographics ─────────────────────────────────
            base_pop = state_info["base_pop"] * (1 + d_offset * 0.5)
            row["population"] = int(_trend(base_pop, year, 0.012, 0.01))
            row["population_density"] = int(
                row["population"] / np.random.uniform(800, 5000)
            ) if not is_urban_hub else int(
                row["population"] / np.random.uniform(100, 400)
            )
            urban_pct = _clamp(
                (state_info["urban_bias"] + (0.2 if is_urban_hub else -0.1)
                 + yr_idx * 0.008 + np.random.uniform(-0.03, 0.03)),
                0.08, 0.95
            )
            row["urban_population_pct"] = round(urban_pct * 100, 1)
            row["rural_population_pct"] = round(100 - row["urban_population_pct"], 1)
            row["sc_st_population_pct"] = round(
                _clamp(np.random.uniform(12, 35) + (-5 if tier == "A" else 5), 8, 45), 1
            )
            row["working_age_population_pct"] = round(
                _clamp(60 + np.random.uniform(-5, 8) + yr_idx * 0.3, 52, 72), 1
            )
            row["female_male_ratio"] = round(
                _clamp(0.92 + np.random.uniform(-0.05, 0.08) + yr_idx * 0.003, 0.82, 1.05), 3
            )

            # ── Economic ─────────────────────────────────────
            gdp_base = state_info["base_gdp"] * (1 + d_offset * 0.3)
            if is_urban_hub:
                gdp_base *= 1.4
            row["gdp_per_capita"] = round(_trend(gdp_base, year, 0.06, 0.03), 0)
            row["gsdp_growth_rate"] = round(
                _clamp(np.random.uniform(4, 12) + (2 if tier == "A" else -1), 1, 16), 1
            )
            row["poverty_rate"] = round(
                _clamp(
                    (40 - 15 * (tier == "A") - 5 * (tier == "B"))
                    * (1 - 0.02 * yr_idx) + np.random.uniform(-5, 5),
                    3, 55
                ), 1
            )
            row["unemployment_rate"] = round(
                _clamp(np.random.uniform(3, 12) + (2 if tier == "C" else -1), 1.5, 18), 1
            )
            row["average_household_income"] = round(
                _trend(
                    120000 * (1.5 if tier == "A" else 1.0 if tier == "B" else 0.7),
                    year, 0.05, 0.04
                ), 0
            )
            row["tax_revenue_per_capita"] = round(
                _trend(
                    8000 * (1.8 if tier == "A" else 1.1 if tier == "B" else 0.6),
                    year, 0.04, 0.03
                ), 0
            )
            row["credit_deposit_ratio"] = round(
                _clamp(np.random.uniform(50, 110) + (15 if is_urban_hub else -10), 35, 130), 1
            )

            # ── Health Indicators ────────────────────────────
            imr_base = 55 - 20 * (tier == "A") - 8 * (tier == "B")
            row["imr"] = round(_clamp(
                imr_base * (1 - 0.03 * yr_idx) + np.random.uniform(-5, 5), 5, 70
            ), 1)
            row["mmr"] = round(_clamp(
                (200 - 80 * (tier == "A") - 30 * (tier == "B"))
                * (1 - 0.04 * yr_idx) + np.random.uniform(-15, 15), 30, 300
            ), 0)
            row["u5mr"] = round(row["imr"] * np.random.uniform(1.2, 1.6), 1)
            row["life_expectancy"] = round(_clamp(
                65 + 5 * (tier == "A") + 2 * (tier == "B")
                + yr_idx * 0.3 + np.random.uniform(-2, 2), 58, 78
            ), 1)
            row["hospital_beds_per_1000"] = round(_clamp(
                (2.5 if tier == "A" else 1.5 if tier == "B" else 0.8)
                * (1.3 if is_urban_hub else 0.9)
                + yr_idx * 0.05 + np.random.uniform(-0.3, 0.3),
                0.3, 5.0
            ), 2)
            row["doctors_per_1000"] = round(_clamp(
                (1.2 if tier == "A" else 0.7 if tier == "B" else 0.4)
                * (1.5 if is_urban_hub else 0.8)
                + yr_idx * 0.03 + np.random.uniform(-0.15, 0.15),
                0.1, 3.0
            ), 2)
            row["health_centres_count"] = int(_clamp(
                np.random.uniform(30, 120) * (1.3 if tier == "A" else 0.8)
                + yr_idx * 2, 15, 200
            ))
            row["immunization_coverage_pct"] = round(_clamp(
                70 + 10 * (tier == "A") + yr_idx * 1.5 + np.random.uniform(-5, 5),
                45, 99
            ), 1)
            row["malnutrition_rate"] = round(_clamp(
                (35 - 12 * (tier == "A") - 5 * (tier == "B"))
                * (1 - 0.02 * yr_idx) + np.random.uniform(-4, 4),
                8, 50
            ), 1)

            # Health Budget
            health_alloc = _trend(
                (500 if tier == "A" else 300 if tier == "B" else 200)
                * (1 + d_offset * 0.3),
                year, 0.07, 0.05
            )
            row["health_alloc_cr"] = round(health_alloc, 2)
            eff = _clamp(np.random.uniform(0.55, 0.95) + (0.1 if tier == "A" else 0), 0.4, 1.0)
            row["health_spent_cr"] = round(health_alloc * eff, 2)
            row["health_efficiency"] = round(eff, 3)

            # ── Education Indicators ─────────────────────────
            lit_base = state_info["base_literacy"] + (5 if is_urban_hub else -3)
            row["literacy_rate"] = round(_clamp(
                lit_base + yr_idx * 0.8 + np.random.uniform(-2, 2), 50, 99
            ), 1)
            row["ger_primary"] = round(_clamp(
                95 + yr_idx * 0.5 + np.random.uniform(-5, 3), 80, 110
            ), 1)
            row["ger_secondary"] = round(_clamp(
                70 + 8 * (tier == "A") + yr_idx * 1.0 + np.random.uniform(-5, 5), 45, 105
            ), 1)
            row["ger_higher"] = round(_clamp(
                25 + 10 * (tier == "A") + 3 * (tier == "B")
                + yr_idx * 1.2 + np.random.uniform(-5, 5),
                10, 60
            ), 1)
            row["dropout_rate_primary"] = round(_clamp(
                (8 - 3 * (tier == "A")) * (1 - 0.04 * yr_idx)
                + np.random.uniform(-2, 2), 0.5, 18
            ), 1)
            row["dropout_rate_secondary"] = round(_clamp(
                (18 - 6 * (tier == "A")) * (1 - 0.03 * yr_idx)
                + np.random.uniform(-4, 4), 2, 35
            ), 1)
            row["student_teacher_ratio"] = round(_clamp(
                (25 if tier == "A" else 32 if tier == "B" else 40)
                * (1 - 0.01 * yr_idx) + np.random.uniform(-3, 3),
                15, 55
            ), 0)
            row["schools_per_1000"] = round(_clamp(
                (5 if tier == "A" else 4 if tier == "B" else 3)
                + yr_idx * 0.1 + np.random.uniform(-0.5, 0.5),
                1.5, 8
            ), 2)
            row["gender_parity_index"] = round(_clamp(
                0.90 + 0.05 * (tier == "A") + yr_idx * 0.008
                + np.random.uniform(-0.03, 0.03),
                0.75, 1.05
            ), 3)

            # Education Budget
            edu_alloc = _trend(
                (600 if tier == "A" else 400 if tier == "B" else 250)
                * (1 + d_offset * 0.3),
                year, 0.06, 0.04
            )
            row["education_alloc_cr"] = round(edu_alloc, 2)
            eff_e = _clamp(np.random.uniform(0.60, 0.92) + (0.08 if tier == "A" else 0), 0.45, 1.0)
            row["education_spent_cr"] = round(edu_alloc * eff_e, 2)
            row["education_efficiency"] = round(eff_e, 3)

            # ── Agriculture Indicators ───────────────────────
            row["agricultural_land_pct"] = round(_clamp(
                (55 if tier == "C" else 45 if tier == "B" else 35)
                + np.random.uniform(-10, 10) - yr_idx * 0.3,
                15, 75
            ), 1)
            row["irrigation_coverage_pct"] = round(_clamp(
                (50 + 15 * (tier == "A") + yr_idx * 1.5
                 + np.random.uniform(-8, 8)),
                20, 95
            ), 1)
            row["crop_yield_index"] = round(_clamp(
                (75 + 10 * (tier == "A") + yr_idx * 1.0
                 + np.random.uniform(-8, 8)),
                40, 110
            ), 1)
            row["farmer_income_inr"] = round(_trend(
                (95000 if tier == "A" else 72000 if tier == "B" else 55000),
                year, 0.04, 0.05
            ), 0)
            row["msp_benefit_pct"] = round(_clamp(
                np.random.uniform(30, 70) + 10 * (tier == "A"), 20, 85
            ), 1)
            row["kisan_credit_coverage"] = round(_clamp(
                np.random.uniform(25, 65) + 10 * (tier == "A") + yr_idx * 1.0,
                15, 80
            ), 1)
            row["drought_frequency_index"] = round(_clamp(
                np.random.uniform(0.1, 0.8) + 0.15 * (tier == "C"), 0.05, 1.0
            ), 2)

            # Agriculture Budget
            agri_alloc = _trend(
                (400 if tier == "A" else 350 if tier == "B" else 250)
                * (1 + d_offset * 0.25),
                year, 0.05, 0.04
            )
            row["agriculture_alloc_cr"] = round(agri_alloc, 2)
            eff_a = _clamp(np.random.uniform(0.50, 0.88) + (0.08 if tier == "A" else 0), 0.35, 1.0)
            row["agriculture_spent_cr"] = round(agri_alloc * eff_a, 2)
            row["agriculture_efficiency"] = round(eff_a, 3)

            # ── Infrastructure Indicators ────────────────────
            row["road_density"] = round(_clamp(
                (1.5 if tier == "A" else 1.0 if tier == "B" else 0.6)
                * (1.4 if is_urban_hub else 0.9)
                + yr_idx * 0.04 + np.random.uniform(-0.15, 0.15),
                0.2, 3.0
            ), 2)
            row["nh_connectivity_index"] = round(_clamp(
                np.random.uniform(40, 85) + 10 * (tier == "A")
                + yr_idx * 1.5, 25, 100
            ), 1)
            row["railway_density"] = round(_clamp(
                (0.05 if tier == "A" else 0.03 if tier == "B" else 0.02)
                * (1.5 if is_urban_hub else 0.8)
                + yr_idx * 0.002 + np.random.uniform(-0.01, 0.01),
                0.005, 0.12
            ), 4)
            row["electrification_rate"] = round(_clamp(
                (90 if tier == "A" else 80 if tier == "B" else 65)
                + yr_idx * 2.0 + np.random.uniform(-3, 3),
                50, 100
            ), 1)
            row["internet_penetration_pct"] = round(_clamp(
                (30 if tier == "A" else 18 if tier == "B" else 10)
                + yr_idx * 6 + (15 if is_urban_hub else 0)
                + np.random.uniform(-5, 5),
                5, 90
            ), 1)
            row["mobile_connectivity_pct"] = round(_clamp(
                (70 if tier == "A" else 55 if tier == "B" else 40)
                + yr_idx * 3.5 + np.random.uniform(-5, 5),
                30, 99
            ), 1)

            # Infrastructure Budget
            infra_alloc = _trend(
                (700 if tier == "A" else 500 if tier == "B" else 350)
                * (1 + d_offset * 0.3),
                year, 0.08, 0.05
            )
            row["infrastructure_alloc_cr"] = round(infra_alloc, 2)
            eff_i = _clamp(np.random.uniform(0.50, 0.90) + (0.08 if tier == "A" else 0), 0.35, 1.0)
            row["infrastructure_spent_cr"] = round(infra_alloc * eff_i, 2)
            row["infrastructure_efficiency"] = round(eff_i, 3)

            # ── Water & Energy ───────────────────────────────
            row["safe_water_access_pct"] = round(_clamp(
                (75 if tier == "A" else 60 if tier == "B" else 45)
                + yr_idx * 2.0 + np.random.uniform(-5, 5),
                30, 99
            ), 1)
            row["sanitation_coverage_pct"] = round(_clamp(
                (70 if tier == "A" else 55 if tier == "B" else 35)
                + yr_idx * 3.0 + np.random.uniform(-5, 5),
                20, 99
            ), 1)
            row["groundwater_stress_index"] = round(_clamp(
                np.random.uniform(0.2, 0.8) + 0.1 * (tier == "C"), 0.05, 1.0
            ), 2)
            row["renewable_energy_pct"] = round(_clamp(
                (20 if tier == "A" else 12 if tier == "B" else 8)
                + yr_idx * 2.5 + np.random.uniform(-4, 4),
                3, 50
            ), 1)
            row["energy_consumption_per_capita"] = round(_trend(
                (1200 if tier == "A" else 800 if tier == "B" else 500),
                year, 0.04, 0.03
            ), 0)

            # Water & Energy Budget
            row["water_alloc_cr"] = round(_trend(
                (250 if tier == "A" else 180 if tier == "B" else 120)
                * (1 + d_offset * 0.2),
                year, 0.06, 0.04
            ), 2)
            row["energy_alloc_cr"] = round(_trend(
                (300 if tier == "A" else 200 if tier == "B" else 130)
                * (1 + d_offset * 0.2),
                year, 0.07, 0.05
            ), 2)

            rows.append(row)
    return rows


def _compute_outcome_scores(df):
    """Compute sector outcome scores and overall welfare + satisfaction."""

    # Health outcome score (0-100): lower IMR/MMR is better, higher coverage is better
    df["health_outcome_score"] = _clamp(
        (100 - df["imr"]) * 0.25
        + (100 - df["malnutrition_rate"]) * 0.20
        + df["immunization_coverage_pct"] * 0.20
        + df["life_expectancy"] * 0.8  # scaled from ~60-78 → ~48-62
        + df["hospital_beds_per_1000"] * 5  # scaled
        - df["mmr"] * 0.05,
        10, 100
    )
    # Normalize to 0-100
    hmin, hmax = df["health_outcome_score"].min(), df["health_outcome_score"].max()
    df["health_outcome_score"] = ((df["health_outcome_score"] - hmin) / (hmax - hmin) * 80 + 15).round(1)

    # Education outcome score
    df["education_outcome_score"] = _clamp(
        df["literacy_rate"] * 0.30
        + df["ger_secondary"] * 0.20
        + df["ger_higher"] * 0.15
        + (100 - df["dropout_rate_secondary"]) * 0.15
        + df["gender_parity_index"] * 20,
        10, 100
    )
    emin, emax = df["education_outcome_score"].min(), df["education_outcome_score"].max()
    df["education_outcome_score"] = ((df["education_outcome_score"] - emin) / (emax - emin) * 80 + 15).round(1)

    # Agriculture outcome score
    df["agriculture_outcome_score"] = _clamp(
        df["crop_yield_index"] * 0.30
        + df["irrigation_coverage_pct"] * 0.25
        + df["kisan_credit_coverage"] * 0.20
        + df["msp_benefit_pct"] * 0.15
        + (1 - df["drought_frequency_index"]) * 10,
        10, 100
    )
    amin, amax = df["agriculture_outcome_score"].min(), df["agriculture_outcome_score"].max()
    df["agriculture_outcome_score"] = ((df["agriculture_outcome_score"] - amin) / (amax - amin) * 80 + 15).round(1)

    # Infrastructure outcome score
    df["infrastructure_outcome_score"] = _clamp(
        df["electrification_rate"] * 0.25
        + df["internet_penetration_pct"] * 0.20
        + df["road_density"] * 15
        + df["nh_connectivity_index"] * 0.15
        + df["mobile_connectivity_pct"] * 0.15,
        10, 100
    )
    imin, imax = df["infrastructure_outcome_score"].min(), df["infrastructure_outcome_score"].max()
    df["infrastructure_outcome_score"] = ((df["infrastructure_outcome_score"] - imin) / (imax - imin) * 80 + 15).round(1)

    # Overall welfare score (weighted average of sector scores)
    df["overall_welfare_score"] = (
        df["health_outcome_score"] * 0.30
        + df["education_outcome_score"] * 0.25
        + df["agriculture_outcome_score"] * 0.20
        + df["infrastructure_outcome_score"] * 0.25
    ).round(1)

    # Citizen satisfaction score (1-5): correlated with welfare but noisy
    df["citizen_satisfaction_score"] = _clamp(
        (df["overall_welfare_score"] / 100 * 4 + 0.5)
        + np.random.uniform(-0.4, 0.4, len(df)),
        1.0, 5.0
    ).round(2)

    return df


def _compute_need_indices(df):
    """Compute need indices and engineered features."""

    # Health need index (higher = more need)
    df["health_need_index"] = _clamp(
        (df["imr"] / 70 * 30)
        + (df["malnutrition_rate"] / 50 * 25)
        + ((100 - df["immunization_coverage_pct"]) / 55 * 20)
        + ((80 - df["life_expectancy"]).clip(lower=0) / 22 * 15)
        + ((3 - df["hospital_beds_per_1000"]).clip(lower=0) / 2.7 * 10),
        5, 100
    ).round(1)

    # Education need index
    df["education_need_index"] = _clamp(
        ((100 - df["literacy_rate"]) / 50 * 30)
        + (df["dropout_rate_secondary"] / 35 * 25)
        + ((100 - df["ger_higher"]) / 90 * 20)
        + ((55 - df["student_teacher_ratio"]).clip(lower=0) / 40 * 15).clip(upper=15)
        + ((1.0 - df["gender_parity_index"]).clip(lower=0) / 0.25 * 10),
        5, 100
    ).round(1)

    # Infrastructure need index
    df["infrastructure_need_index"] = _clamp(
        ((100 - df["electrification_rate"]) / 50 * 25)
        + ((100 - df["internet_penetration_pct"]) / 95 * 25)
        + ((2.0 - df["road_density"]).clip(lower=0) / 1.8 * 20)
        + ((100 - df["nh_connectivity_index"]) / 75 * 15)
        + ((100 - df["mobile_connectivity_pct"]) / 70 * 15),
        5, 100
    ).round(1)

    # Overall need index (average)
    df["overall_need_index"] = (
        (df["health_need_index"] + df["education_need_index"]
         + df["infrastructure_need_index"]) / 3
    ).round(1)

    # Uplift multiplier based on state tier
    df["uplift_multiplier"] = df["state_tier"].map(UPLIFT_MAP)

    # Absorption capacity: average of all efficiency columns per district
    eff_cols = ["health_efficiency", "education_efficiency",
                "agriculture_efficiency", "infrastructure_efficiency"]
    df["absorption_capacity"] = df[eff_cols].mean(axis=1).round(3)

    return df


def generate_dataset(save_path=None):
    """Generate the complete OLIVE synthetic dataset.

    Args:
        save_path: Optional path to save the CSV. If None, saves to current directory.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    all_rows = []
    for state_name, state_info in STATES.items():
        all_rows.extend(_gen_district_rows(state_name, state_info))

    df = pd.DataFrame(all_rows)

    # Add auto-increment ID
    df.insert(0, "id", range(1, len(df) + 1))

    # Compute derived columns
    df = _compute_outcome_scores(df)
    df = _compute_need_indices(df)

    # Sort
    df = df.sort_values(["state", "district", "year"]).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "olive_dataset.csv")
    df.to_csv(save_path, index=False)
    print(f"[OK] Dataset generated: {len(df)} rows, {len(df.columns)} columns -> {save_path}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(f"\nColumns ({len(df.columns)}):\n{list(df.columns)}")
