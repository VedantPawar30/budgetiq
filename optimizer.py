"""
OLIVE — Optimization Engine
Layer 1: Linear Programming (PuLP) for hard constraints.
Layer 2: NSGA-II (pymoo) for multi-objective optimization.
"""

import numpy as np
import pandas as pd
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
)
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

# Sector keys matching allocation columns
SECTORS = ["health", "education", "agriculture", "infrastructure", "water", "energy"]
ALLOC_COLS = [f"{s}_alloc_cr" for s in SECTORS]
SECTOR_LABELS = {
    "health": "Healthcare",
    "education": "Education",
    "agriculture": "Agriculture",
    "infrastructure": "Infrastructure",
    "water": "Water",
    "energy": "Energy",
}


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1: LINEAR PROGRAMMING (PuLP)
# ═══════════════════════════════════════════════════════════════════════════

def run_pulp_optimization(total_budget, historical_alloc, min_pct=0.08):
    """Find feasible allocation bounds using linear programming.

    Constraints:
        - Total allocation <= total_budget
        - Each sector >= min_pct * historical average allocation
        - Each sector <= 40% of total budget (no single sector dominates)

    Args:
        total_budget: Total budget in Cr.
        historical_alloc: dict {sector: historical_avg_alloc_cr}
        min_pct: Minimum fraction of historical allocation guaranteed.

    Returns:
        dict: {sector: (lower_bound, upper_bound)} feasible ranges for each sector.
        dict: {sector: optimal_alloc} LP-optimal solution.
    """
    prob = LpProblem("BudgetAllocation_LP", LpMaximize)

    # Decision variables
    alloc_vars = {}
    for s in SECTORS:
        lb = max(historical_alloc.get(s, 0) * min_pct, total_budget * 0.03)
        ub = total_budget * 0.40
        alloc_vars[s] = LpVariable(f"alloc_{s}", lowBound=lb, upBound=ub)

    # Constraint: total <= budget
    prob += lpSum(alloc_vars.values()) <= total_budget, "TotalBudgetCap"

    # Constraint: total >= 90% of budget (don't under-spend dramatically)
    prob += lpSum(alloc_vars.values()) >= total_budget * 0.90, "MinSpend"

    # Objective: maximize total allocation (push towards full utilization)
    # weighted by sector historical efficiency (prefer sectors that spend well)
    prob += lpSum(alloc_vars.values()), "MaximizeUtilization"

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(f"[WARN] PuLP status: {LpStatus[prob.status]}, using equal split fallback")
        equal = total_budget / len(SECTORS)
        bounds = {s: (equal * 0.5, equal * 2.0) for s in SECTORS}
        solution = {s: equal for s in SECTORS}
        return bounds, solution

    solution = {s: value(alloc_vars[s]) for s in SECTORS}

    # Derive feasible bounds (±30% around LP solution, capped)
    bounds = {}
    for s in SECTORS:
        sol = solution[s]
        lb = max(sol * 0.5, total_budget * 0.03)
        ub = min(sol * 1.8, total_budget * 0.40)
        bounds[s] = (lb, ub)

    print(f"[OK] PuLP: Optimal solution found. Total allocated = Rs.{sum(solution.values()):,.0f} Cr")
    return bounds, solution


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2: NSGA-II MULTI-OBJECTIVE OPTIMIZATION (pymoo)
# ═══════════════════════════════════════════════════════════════════════════

class BudgetOptProblem(ElementwiseProblem):
    """Multi-objective budget allocation problem for NSGA-II.

    Objectives (all minimized by pymoo, so we negate maximization targets):
        1. Minimize -predicted_welfare (maximize welfare)
        2. Minimize -absorption_weighted_alloc (maximize efficiency)
        3. Minimize regional_disparity (std dev of district welfare scores)

    Constraints:
        - Sum of allocations <= total_budget
        - Sum of allocations >= 0.90 * total_budget
    """

    def __init__(self, total_budget, sector_bounds, welfare_model, base_features,
                 feature_names, district_absorption, **kwargs):
        """
        Args:
            total_budget: Total budget cap.
            sector_bounds: dict {sector: (lb, ub)} from PuLP.
            welfare_model: Trained XGBoost model.
            base_features: DataFrame of base socio-economic features for districts.
            feature_names: Feature names for the model.
            district_absorption: Series with absorption_capacity per district.
        """
        self.total_budget = total_budget
        self.sector_bounds = sector_bounds
        self.welfare_model = welfare_model
        self.base_features = base_features
        self.feature_names = feature_names
        self.district_absorption = district_absorption
        self.n_districts = len(base_features)

        n_var = len(SECTORS)
        xl = np.array([sector_bounds[s][0] for s in SECTORS])
        xu = np.array([sector_bounds[s][1] for s in SECTORS])

        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_ieq_constr=2,
            xl=xl,
            xu=xu,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        alloc_dict = dict(zip(ALLOC_COLS, x))

        # Create feature set: base features + new allocations
        X_pred = self.base_features.copy()
        for col, val in alloc_dict.items():
            if col in X_pred.columns:
                X_pred[col] = val

        X_pred = X_pred[self.feature_names].fillna(0)

        # Predict welfare scores for all districts
        welfare_scores = self.welfare_model.predict(X_pred)

        # Objective 1: Maximize mean welfare → minimize negative
        f1 = -np.mean(welfare_scores)

        # Objective 2: Maximize absorption-weighted allocation
        abs_cap = self.district_absorption.values
        weighted_util = np.mean(abs_cap) * np.sum(x) / self.total_budget
        f2 = -weighted_util

        # Objective 3: Minimize disparity (std dev of welfare scores)
        f3 = np.std(welfare_scores)

        out["F"] = [f1, f2, f3]

        # Constraints: g <= 0
        total = np.sum(x)
        out["G"] = [
            total - self.total_budget,           # total <= budget
            0.90 * self.total_budget - total,     # total >= 90% budget
        ]


def run_nsga2_optimization(total_budget, bounds, welfare_model, base_features,
                            feature_names, district_absorption,
                            pop_size=50, n_gen=80):
    """Run NSGA-II multi-objective optimization.

    Args:
        total_budget: Total budget.
        bounds: From PuLP layer.
        welfare_model: Trained XGBoost.
        base_features: District base features DataFrame.
        feature_names: Model feature names.
        district_absorption: Absorption capacity per district.
        pop_size: GA population size.
        n_gen: Number of generations.

    Returns:
        dict: Best compromise allocation {sector: amount}.
        np.ndarray: Pareto front objectives.
    """
    problem = BudgetOptProblem(
        total_budget=total_budget,
        sector_bounds=bounds,
        welfare_model=welfare_model,
        base_features=base_features,
        feature_names=feature_names,
        district_absorption=district_absorption,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    result = pymoo_minimize(
        problem, algorithm, termination, seed=42, verbose=False
    )

    if result.X is None or len(result.X) == 0:
        print("[WARN] NSGA-II: No feasible solution found, using PuLP defaults")
        return None, None

    # Pick the best compromise solution (closest to ideal point)
    F = result.F
    # Normalize objectives
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-8)
    # Ideal point is origin after normalization
    distances = np.sqrt(np.sum(F_norm ** 2, axis=1))
    best_idx = np.argmin(distances)

    best_x = result.X[best_idx] if result.X.ndim > 1 else result.X
    optimal_alloc = {s: round(float(best_x[i]), 2) for i, s in enumerate(SECTORS)}

    print(f"[OK] NSGA-II: Found {len(result.F)} Pareto solutions. "
          f"Best total = Rs.{sum(optimal_alloc.values()):,.0f} Cr")

    return optimal_alloc, result.F


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

def get_optimal_allocation(total_budget, df_latest, welfare_model, feature_names,
                            pop_size=50, n_gen=80):
    """Full optimization pipeline: PuLP bounds → NSGA-II optimization.

    Args:
        total_budget: Total budget in Cr.
        df_latest: DataFrame for the latest year (filtered to selected state/districts).
        welfare_model: Trained XGBoost model.
        feature_names: Feature columns used by the model.
        pop_size: NSGA-II population size.
        n_gen: Number of generations.

    Returns:
        dict: Optimal allocation per sector.
        dict: Historical average allocation per sector.
        np.ndarray: Pareto front (if NSGA-II ran).
    """
    # Compute historical averages
    historical_alloc = {}
    for s in SECTORS:
        col = f"{s}_alloc_cr"
        if col in df_latest.columns:
            historical_alloc[s] = df_latest[col].mean()
        else:
            historical_alloc[s] = total_budget / len(SECTORS)

    # Layer 1: PuLP
    bounds, lp_solution = run_pulp_optimization(total_budget, historical_alloc)

    # Prepare base features for NSGA-II
    base_features = df_latest[[c for c in feature_names if c in df_latest.columns]].copy()
    for col in feature_names:
        if col not in base_features.columns:
            base_features[col] = 0

    district_absorption = df_latest["absorption_capacity"] if "absorption_capacity" in df_latest.columns \
        else pd.Series([0.7] * len(df_latest))

    # Layer 2: NSGA-II
    nsga_alloc, pareto_front = run_nsga2_optimization(
        total_budget=total_budget,
        bounds=bounds,
        welfare_model=welfare_model,
        base_features=base_features,
        feature_names=feature_names,
        district_absorption=district_absorption,
        pop_size=pop_size,
        n_gen=n_gen,
    )

    if nsga_alloc is None:
        nsga_alloc = lp_solution

    return nsga_alloc, historical_alloc, pareto_front
