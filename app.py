"""
OLIVE — Citizen Transparency Dashboard
Streamlit app with 4 tabs: Budget Allocation, Spending vs Allocation,
Before/After Comparisons, and Explainability (SHAP).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BudgetIQ — Budget Optimization Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for premium dark theme ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2634 100%);
        border-right: 1px solid rgba(99, 179, 148, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #63b394 !important;
    }

    /* ── Header gradient ────────────────────── */
    .olive-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        border: 1px solid rgba(99, 179, 148, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .olive-header h1 {
        background: linear-gradient(90deg, #63b394, #a8e6cf, #63b394);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .olive-header p {
        color: #8faabe;
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
    }

    /* ── Metric Cards ───────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #1a2634, #0f1923);
        border: 1px solid rgba(99, 179, 148, 0.15);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(99, 179, 148, 0.15);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #63b394;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: #8faabe;
        margin-top: 6px;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .delta {
        font-size: 0.78rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .delta-positive { color: #4ade80; }
    .delta-negative { color: #f87171; }

    /* ── Section headers ────────────────────── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #c4dce8;
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 179, 148, 0.2);
    }

    /* ── Explanation cards ───────────────────── */
    .explanation-card {
        background: linear-gradient(145deg, #1a2634, #16202c);
        border-left: 4px solid #63b394;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
    }
    .explanation-card.decrease {
        border-left-color: #f59e0b;
    }
    .explanation-card .sector-badge {
        display: inline-block;
        background: rgba(99, 179, 148, 0.15);
        color: #63b394;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    .explanation-card .explanation-text {
        color: #c4dce8;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ── Tabs styling ───────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* ── Hide default streamlit elements ────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,25,35,0.6)",
    font=dict(family="Inter", color="#c4dce8", size=12),
    title_font=dict(size=16, color="#e2eef5"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=40, r=20, t=50, b=40),
)

SECTOR_COLORS = {
    "Healthcare": "#4ade80",
    "Education": "#60a5fa",
    "Agriculture": "#fbbf24",
    "Infrastructure": "#f87171",
    "Water": "#22d3ee",
    "Energy": "#c084fc",
}

SECTOR_COLORS_LIST = list(SECTOR_COLORS.values())


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING AND MODEL TRAINING (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Load or generate the OLIVE dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "olive_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from data_generator import generate_dataset
        df = generate_dataset(csv_path)
    return df


@st.cache_resource(show_spinner=False)
def train_models(_df):
    """Train XGBoost models and return them."""
    from models import train_xgboost_model
    welfare_model, welfare_metrics, feature_names = train_xgboost_model(_df, "overall_welfare_score")
    satisfaction_model, sat_metrics, _ = train_xgboost_model(_df, "citizen_satisfaction_score")
    return {
        "welfare_model": welfare_model,
        "welfare_metrics": welfare_metrics,
        "satisfaction_model": satisfaction_model,
        "satisfaction_metrics": sat_metrics,
        "feature_names": feature_names,
    }


@st.cache_data(show_spinner=False)
def run_prophet(_df):
    """Run Prophet forecasting."""
    from models import train_prophet_forecasts
    state_forecasts, forecast_df = train_prophet_forecasts(_df)
    return state_forecasts, forecast_df


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def render_metric_card(value, label, delta=None, delta_pct=None):
    """Render a premium metric card."""
    delta_html = ""
    if delta is not None:
        cls = "delta-positive" if delta >= 0 else "delta-negative"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="delta {cls}">{sign}{delta:.1f}'
        if delta_pct is not None:
            delta_html += f' ({sign}{delta_pct:.1f}%)'
        delta_html += '</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def create_allocation_comparison_chart(optimal, historical, title="Budget Allocation Comparison"):
    """Create grouped bar chart comparing optimal vs historical allocation."""
    sectors = list(optimal.keys())
    labels = [s.title() for s in sectors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Historical Avg",
        x=labels,
        y=[historical.get(s, 0) for s in sectors],
        marker_color="rgba(148, 163, 184, 0.6)",
        marker_line=dict(color="rgba(148, 163, 184, 0.8)", width=1),
        text=[f"₹{historical.get(s, 0):,.0f}" for s in sectors],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.add_trace(go.Bar(
        name="Optimized",
        x=labels,
        y=[optimal[s] for s in sectors],
        marker_color=SECTOR_COLORS_LIST[:len(sectors)],
        marker_line=dict(color="rgba(255,255,255,0.2)", width=1),
        text=[f"₹{optimal[s]:,.0f}" for s in sectors],
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        barmode="group",
        yaxis_title="Allocation (₹ Cr)",
        xaxis_title="",
        height=420,
    )
    return fig


def create_allocation_pie(optimal):
    """Create pie chart for optimal allocation."""
    sectors = list(optimal.keys())
    labels = [s.title() for s in sectors]
    values = [optimal[s] for s in sectors]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=SECTOR_COLORS_LIST[:len(sectors)],
                    line=dict(color="rgba(15,25,35,0.8)", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f} Cr<br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Optimized Allocation Distribution",
        height=400,
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div class="olive-header">
        <h1>BudgetIQ</h1>
        <p>Optimized Budget Allocation</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ────────────────────────────────────────────
    with st.spinner("🔄 Loading dataset..."):
        df = load_data()

    # ── Sidebar controls ────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        states = sorted(df["state"].unique())
        selected_state = st.selectbox("🗺️ Select State", states, index=0)

        state_districts = sorted(df[df["state"] == selected_state]["district"].unique())
        selected_district = st.selectbox("📍 Select District", ["All Districts"] + list(state_districts))

        st.markdown("---")
        st.markdown("### 💰 Budget Settings")

        # Compute a sensible default from historical data
        state_df = df[df["state"] == selected_state]
        latest_year_df = state_df[state_df["year"] == state_df["year"].max()]

        alloc_cols = ["health_alloc_cr", "education_alloc_cr", "agriculture_alloc_cr",
                      "infrastructure_alloc_cr", "water_alloc_cr", "energy_alloc_cr"]
        hist_total = latest_year_df[alloc_cols].sum(axis=1).mean()
        default_budget = int(round(hist_total * 1.1 / 100) * 100)  # round to 100s

        total_budget = st.slider(
            "Total Budget (₹ Cr)",
            min_value=500,
            max_value=20000,
            value=max(500, min(default_budget, 20000)),
            step=100,
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        with st.spinner("Training models..."):
            model_bundle = train_models(df)

        st.success(f"XGBoost R² = {model_bundle['welfare_metrics']['r2']}")
        st.caption(f"MAE = {model_bundle['welfare_metrics']['mae']}")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#5a7a8a; font-size:0.75rem;'>"
            "OLIVE v1.0 • Prototype<br>Built for Public Transparency"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Filter data based on selection ──────────────────────
    if selected_district == "All Districts":
        filtered_df = state_df
    else:
        filtered_df = state_df[state_df["district"] == selected_district]

    latest_filtered = filtered_df[filtered_df["year"] == filtered_df["year"].max()]

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Budget Allocation",
        "📊 Spending vs Allocation",
        "🔄 Before / After",
        "🧠 Why This Allocation?",
        "🧪 What-If Simulator",
    ])

    # ═════════════════════════════════════════════════════════
    # TAB 1: BUDGET ALLOCATION (Simulation Hub)
    # ═════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">🎯 Optimization Simulation Hub</div>', unsafe_allow_html=True)

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_optim = st.button("🚀 Run Optimization", type="primary", use_container_width=True)
        with col_info:
            st.caption(
                f"Will optimize ₹{total_budget:,} Cr across 6 sectors for "
                f"**{selected_state}** — {selected_district}"
            )

        if run_optim or st.session_state.get("optim_result") is not None:
            if run_optim:
                with st.spinner("⚡ Running PuLP + NSGA-II optimization..."):
                    from optimizer import get_optimal_allocation
                    optimal, historical, pareto = get_optimal_allocation(
                        total_budget=total_budget,
                        df_latest=latest_filtered,
                        welfare_model=model_bundle["welfare_model"],
                        feature_names=model_bundle["feature_names"],
                        pop_size=40,
                        n_gen=60,
                    )
                    st.session_state["optim_result"] = (optimal, historical, pareto)
            else:
                optimal, historical, pareto = st.session_state["optim_result"]

            # Summary metrics
            st.markdown('<div class="section-header">📈 Optimization Results</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            total_opt = sum(optimal.values())
            total_hist = sum(historical.values())
            with m1:
                render_metric_card(f"₹{total_opt:,.0f} Cr", "Optimized Total",
                                    delta=total_opt - total_hist)
            with m2:
                render_metric_card(f"₹{total_hist:,.0f} Cr", "Historical Avg Total")
            with m3:
                utilization = total_opt / total_budget * 100
                render_metric_card(f"{utilization:.1f}%", "Budget Utilization")
            with m4:
                change = (total_opt - total_hist) / total_hist * 100 if total_hist > 0 else 0
                render_metric_card(f"{change:+.1f}%", "Overall Change")

            st.markdown("")

            # Charts
            c1, c2 = st.columns([3, 2])
            with c1:
                fig_bar = create_allocation_comparison_chart(optimal, historical)
                st.plotly_chart(fig_bar, use_container_width=True)
            with c2:
                fig_pie = create_allocation_pie(optimal)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Sector detail table
            st.markdown('<div class="section-header">📋 Sector Breakdown</div>', unsafe_allow_html=True)
            sector_data = []
            for s in optimal:
                opt_val = optimal[s]
                hist_val = historical.get(s, 0)
                chg = ((opt_val - hist_val) / hist_val * 100) if hist_val > 0 else 0
                sector_data.append({
                    "Sector": s.title(),
                    "Optimized (₹ Cr)": f"₹{opt_val:,.1f}",
                    "Historical Avg (₹ Cr)": f"₹{hist_val:,.1f}",
                    "Change": f"{chg:+.1f}%",
                    "Share": f"{opt_val / total_opt * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(sector_data), use_container_width=True, hide_index=True)

        else:
            st.info("👈 Configure budget and click **Run Optimization** to see results.")

    # ═════════════════════════════════════════════════════════
    # TAB 2: SPENDING VS ALLOCATION
    # ═════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">📊 Actual Spending vs Budget Allocation</div>',
                    unsafe_allow_html=True)

        sector_pairs = [
            ("Healthcare", "health_alloc_cr", "health_spent_cr", "health_efficiency"),
            ("Education", "education_alloc_cr", "education_spent_cr", "education_efficiency"),
            ("Agriculture", "agriculture_alloc_cr", "agriculture_spent_cr", "agriculture_efficiency"),
            ("Infrastructure", "infrastructure_alloc_cr", "infrastructure_spent_cr", "infrastructure_efficiency"),
        ]

        # Grouped bar chart: allocated vs spent by sector (latest year)
        alloc_vals = []
        spent_vals = []
        eff_vals = []
        labels = []

        for name, alloc_col, spent_col, eff_col in sector_pairs:
            labels.append(name)
            alloc_vals.append(latest_filtered[alloc_col].mean())
            spent_vals.append(latest_filtered[spent_col].mean())
            eff_vals.append(latest_filtered[eff_col].mean() * 100)

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Bar(
            name="Allocated",
            x=labels, y=alloc_vals,
            marker_color="rgba(99, 179, 148, 0.7)",
            text=[f"₹{v:,.0f}" for v in alloc_vals],
            textposition="outside",
        ))
        fig_vs.add_trace(go.Bar(
            name="Spent",
            x=labels, y=spent_vals,
            marker_color="rgba(96, 165, 250, 0.7)",
            text=[f"₹{v:,.0f}" for v in spent_vals],
            textposition="outside",
        ))
        fig_vs.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Budget Allocated vs Actually Spent — {selected_state} ({latest_filtered['year'].max()})",
            barmode="group",
            yaxis_title="Amount (₹ Cr)",
            height=420,
        )
        st.plotly_chart(fig_vs, use_container_width=True)

        # Efficiency cards
        st.markdown('<div class="section-header">⚡ Spending Efficiency by Sector</div>', unsafe_allow_html=True)
        eff_cols = st.columns(4)
        for i, (name, _, _, _) in enumerate(sector_pairs):
            with eff_cols[i]:
                eff = eff_vals[i]
                gap = alloc_vals[i] - spent_vals[i]
                render_metric_card(
                    f"{eff:.1f}%", f"{name} Efficiency",
                    delta=-gap, delta_pct=None
                )

        # Trend over years
        st.markdown('<div class="section-header">📈 Efficiency Trends Over Time</div>', unsafe_allow_html=True)
        trend_data = filtered_df.groupby("year").agg({
            eff_col: "mean" for _, _, _, eff_col in sector_pairs
        }).reset_index()

        fig_trend = go.Figure()
        for i, (name, _, _, eff_col) in enumerate(sector_pairs):
            fig_trend.add_trace(go.Scatter(
                x=trend_data["year"],
                y=trend_data[eff_col] * 100,
                name=name,
                mode="lines+markers",
                line=dict(width=2.5),
                marker=dict(size=8),
            ))
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            title="Spending Efficiency Trends (2018-2024)",
            yaxis_title="Efficiency (%)",
            xaxis_title="Year",
            height=380,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Heatmap of efficiency across districts
        if selected_district == "All Districts" and len(state_districts) > 1:
            st.markdown('<div class="section-header">🗺️ District Efficiency Heatmap</div>',
                        unsafe_allow_html=True)
            heatmap_data = latest_filtered.set_index("district")[
                [eff_col for _, _, _, eff_col in sector_pairs]
            ] * 100
            heatmap_data.columns = [name for name, _, _, _ in sector_pairs]

            fig_heat = px.imshow(
                heatmap_data,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                labels=dict(color="Efficiency (%)"),
            )
            fig_heat.update_layout(
                **PLOTLY_LAYOUT,
                title="Spending Efficiency by District & Sector",
                height=400,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ═════════════════════════════════════════════════════════
    # TAB 3: BEFORE / AFTER COMPARISONS
    # ═════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">🔄 Before/After Impact Prediction</div>',
                    unsafe_allow_html=True)

        # Current scores
        current_welfare = latest_filtered["overall_welfare_score"].mean()
        current_satisfaction = latest_filtered["citizen_satisfaction_score"].mean()
        current_health = latest_filtered["health_outcome_score"].mean()
        current_edu = latest_filtered["education_outcome_score"].mean()
        current_agri = latest_filtered["agriculture_outcome_score"].mean()
        current_infra = latest_filtered["infrastructure_outcome_score"].mean()

        # Predict new scores if optimization was run
        if st.session_state.get("optim_result") is not None:
            optimal, _, _ = st.session_state["optim_result"]
            from models import predict_outcomes

            # Create feature set with new allocations
            X_new = latest_filtered[model_bundle["feature_names"]].copy()
            alloc_map = {
                "health_alloc_cr": optimal.get("health", 0),
                "education_alloc_cr": optimal.get("education", 0),
                "agriculture_alloc_cr": optimal.get("agriculture", 0),
                "infrastructure_alloc_cr": optimal.get("infrastructure", 0),
                "water_alloc_cr": optimal.get("water", 0),
                "energy_alloc_cr": optimal.get("energy", 0),
            }
            for col, val in alloc_map.items():
                if col in X_new.columns:
                    X_new[col] = val

            predicted_welfare = predict_outcomes(
                model_bundle["welfare_model"], X_new, model_bundle["feature_names"]
            ).mean()
            predicted_satisfaction = predict_outcomes(
                model_bundle["satisfaction_model"], X_new, model_bundle["feature_names"]
            ).mean()

            # Estimate sector score changes (proportional to allocation changes)
            hist = st.session_state["optim_result"][1]
            def sector_delta(sector_key, current_score):
                hist_val = hist.get(sector_key, 1)
                opt_val = optimal.get(sector_key, hist_val)
                pct_change = (opt_val - hist_val) / hist_val if hist_val > 0 else 0
                return current_score * (1 + pct_change * 0.3)  # Dampened effect

            pred_health = min(sector_delta("health", current_health), 95)
            pred_edu = min(sector_delta("education", current_edu), 95)
            pred_agri = min(sector_delta("agriculture", current_agri), 95)
            pred_infra = min(sector_delta("infrastructure", current_infra), 95)

            # ── Comparison metrics ──
            st.markdown("#### 🎯 Overall Impact")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                render_metric_card(
                    f"{predicted_welfare:.1f}", "Predicted Welfare",
                    delta=predicted_welfare - current_welfare,
                    delta_pct=(predicted_welfare - current_welfare) / current_welfare * 100
                )
            with mc2:
                render_metric_card(
                    f"{current_welfare:.1f}", "Current Welfare"
                )
            with mc3:
                render_metric_card(
                    f"{predicted_satisfaction:.2f}", "Predicted Satisfaction",
                    delta=predicted_satisfaction - current_satisfaction,
                )
            with mc4:
                render_metric_card(
                    f"{current_satisfaction:.2f}", "Current Satisfaction"
                )

            st.markdown("")

            # Radar chart
            st.markdown("#### 📡 Sector Score Radar")
            categories = ["Healthcare", "Education", "Agriculture", "Infrastructure"]
            current_scores = [current_health, current_edu, current_agri, current_infra]
            predicted_scores = [pred_health, pred_edu, pred_agri, pred_infra]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=current_scores + [current_scores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Current",
                fillcolor="rgba(148, 163, 184, 0.15)",
                line=dict(color="rgba(148, 163, 184, 0.8)", width=2),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=predicted_scores + [predicted_scores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="After Optimization",
                fillcolor="rgba(99, 179, 148, 0.15)",
                line=dict(color="#63b394", width=2.5),
            ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    bgcolor="rgba(15,25,35,0.6)",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(99,179,148,0.1)"),
                    angularaxis=dict(gridcolor="rgba(99,179,148,0.1)"),
                ),
                title="Sector Outcome Scores: Current vs Projected",
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Sector deltas
            st.markdown("#### 📊 Sector-by-Sector Deltas")
            delta_cols = st.columns(4)
            for i, (name, curr, pred) in enumerate(zip(
                categories, current_scores, predicted_scores
            )):
                with delta_cols[i]:
                    render_metric_card(
                        f"{pred:.1f}", f"{name} (Projected)",
                        delta=pred - curr,
                        delta_pct=(pred - curr) / curr * 100 if curr > 0 else 0,
                    )

            # Prophet forecasts
            st.markdown("#### 🔮 Need-Index Forecasts (2025)")
            with st.spinner("Running forecasts..."):
                state_forecasts, forecast_df = run_prophet(df)

            if selected_state in state_forecasts:
                fc = state_forecasts[selected_state]
                fc_cols = st.columns(len(fc))
                for i, (col, val) in enumerate(fc.items()):
                    with fc_cols[i]:
                        label = col.replace("_need_index", "").replace("_", " ").title()
                        render_metric_card(f"{val:.1f}", f"{label} Need (2025)")
        else:
            st.info("Run optimization in the **Budget Allocation** tab first to see before/after comparisons.")

    # ═════════════════════════════════════════════════════════
    # TAB 4: EXPLAINABILITY (SHAP)
    # ═════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">🧠 Why This Allocation? — Explainable AI</div>',
                    unsafe_allow_html=True)

        # District selector for explanations
        explain_district = st.selectbox(
            "Select district to explain:",
            state_districts,
            key="explain_district",
        )

        district_data = latest_filtered[latest_filtered["district"] == explain_district]
        if len(district_data) == 0:
            st.warning("No data available for this district.")
        else:
            district_row = district_data.iloc[0].to_dict()

            with st.spinner("🔍 Computing SHAP explanations..."):
                from models import generate_natural_language_explanation, get_shap_values

                explanations = generate_natural_language_explanation(
                    model_bundle["welfare_model"],
                    district_row,
                    model_bundle["feature_names"],
                    latest_filtered,
                )

            # Natural Language Explanations
            st.markdown(f"#### 💬 Explanation for **{explain_district}**, {selected_state}")
            st.markdown("")

            for exp in explanations:
                card_class = "explanation-card" if exp["direction"] == "increased" else "explanation-card decrease"
                icon = "📈" if exp["direction"] == "increased" else "📉"
                st.markdown(f"""
                <div class="{card_class}">
                    <div class="sector-badge">{icon} {exp['sector']}</div>
                    <div class="explanation-text">{exp['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)

            # SHAP Feature Importance
            st.markdown("#### 📊 Feature Importance (SHAP Values)")

            X_district = pd.DataFrame([district_row])[model_bundle["feature_names"]].fillna(0)
            shap_vals = get_shap_values(
                model_bundle["welfare_model"], X_district, model_bundle["feature_names"]
            )

            # Top 15 features by absolute SHAP value
            feat_importance = list(zip(model_bundle["feature_names"], shap_vals.values[0]))
            feat_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_feats = feat_importance[:15]

            feat_names = [f[0].replace("_", " ").title() for f in top_feats]
            feat_shap = [f[1] for f in top_feats]
            colors = ["#4ade80" if v >= 0 else "#f87171" for v in feat_shap]

            fig_shap = go.Figure(go.Bar(
                x=feat_shap,
                y=feat_names,
                orientation="h",
                marker_color=colors,
                marker_line=dict(color="rgba(255,255,255,0.1)", width=1),
            ))
            fig_shap.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Top 15 Features Driving Welfare Score — {explain_district}",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis=dict(autorange="reversed"),
                height=500,
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # District profile card
            st.markdown("#### 📋 District Profile Snapshot")
            prof_cols = st.columns(4)
            profile_items = [
                ("Population", f"{district_row.get('population', 0):,.0f}"),
                ("GDP/Capita", f"₹{district_row.get('gdp_per_capita', 0):,.0f}"),
                ("Poverty Rate", f"{district_row.get('poverty_rate', 0):.1f}%"),
                ("Absorption Cap.", f"{district_row.get('absorption_capacity', 0)*100:.0f}%"),
                ("IMR", f"{district_row.get('imr', 0):.1f}"),
                ("Literacy Rate", f"{district_row.get('literacy_rate', 0):.1f}%"),
                ("Electrification", f"{district_row.get('electrification_rate', 0):.1f}%"),
                ("State Tier", f"{district_row.get('state_tier', 'N/A')}"),
            ]
            for i, (label, val) in enumerate(profile_items):
                with prof_cols[i % 4]:
                    render_metric_card(val, label)
                if i == 3:
                    prof_cols = st.columns(4)

    # ═════════════════════════════════════════════════════════
    # TAB 5: WHAT-IF SIMULATOR
    # ═════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">🧪 What-If Budget Simulator</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Manually adjust sector budgets and instantly see how changes affect "
            "predicted welfare, satisfaction, and the remaining budget for other sectors."
        )

        # Compute historical averages for default slider values
        hist_avg = {}
        for s_key, s_col in [("Healthcare", "health_alloc_cr"),
                              ("Education", "education_alloc_cr"),
                              ("Agriculture", "agriculture_alloc_cr"),
                              ("Infrastructure", "infrastructure_alloc_cr"),
                              ("Water", "water_alloc_cr"),
                              ("Energy", "energy_alloc_cr")]:
            hist_avg[s_key] = round(latest_filtered[s_col].mean(), 1) if s_col in latest_filtered.columns else 100.0

        st.markdown("#### 🎚️ Set Your Custom Budget")
        st.markdown(f"**Total budget envelope: ₹{total_budget:,} Cr**")
        st.markdown("")

        # Sliders for each sector in 2-column layout
        sim_alloc = {}
        slider_cols_row1 = st.columns(3)
        slider_cols_row2 = st.columns(3)
        sector_list = ["Healthcare", "Education", "Agriculture", "Infrastructure", "Water", "Energy"]
        sector_colors_emoji = ["🏥", "📚", "🌾", "🏗️", "💧", "⚡"]

        for i, (sector, emoji) in enumerate(zip(sector_list, sector_colors_emoji)):
            col = slider_cols_row1[i] if i < 3 else slider_cols_row2[i - 3]
            with col:
                default_val = min(int(hist_avg.get(sector, 100)), total_budget)
                sim_alloc[sector] = st.slider(
                    f"{emoji} {sector} (₹ Cr)",
                    min_value=0,
                    max_value=total_budget,
                    value=default_val,
                    step=10,
                    key=f"whatif_{sector}",
                )

        # Budget math
        total_allocated = sum(sim_alloc.values())
        remaining = total_budget - total_allocated
        utilization_pct = (total_allocated / total_budget * 100) if total_budget > 0 else 0
        over_budget = total_allocated > total_budget

        # Budget status bar
        st.markdown("---")
        st.markdown("#### 💰 Budget Status")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            render_metric_card(f"₹{total_allocated:,.0f} Cr", "Total Allocated")
        with b2:
            color_class = "delta-negative" if over_budget else "delta-positive"
            render_metric_card(
                f"₹{remaining:,.0f} Cr", "Remaining",
                delta=remaining,
            )
        with b3:
            render_metric_card(f"{utilization_pct:.1f}%", "Utilization")
        with b4:
            render_metric_card(
                "⚠️ OVER" if over_budget else "✅ OK",
                "Budget Status"
            )

        if over_budget:
            st.error(
                f"🚨 You are **₹{-remaining:,.0f} Cr over budget!** "
                f"Reduce allocations by ₹{-remaining:,.0f} Cr to fit within ₹{total_budget:,} Cr."
            )

        # Allocation distribution chart
        st.markdown("---")
        st.markdown("#### 📊 Your Allocation vs Historical")

        sim_labels = list(sim_alloc.keys())
        sim_values = list(sim_alloc.values())
        hist_values = [hist_avg.get(s, 0) for s in sim_labels]

        fig_whatif = go.Figure()
        fig_whatif.add_trace(go.Bar(
            name="Historical Avg",
            x=sim_labels, y=hist_values,
            marker_color="rgba(148, 163, 184, 0.5)",
            text=[f"₹{v:,.0f}" for v in hist_values],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_whatif.add_trace(go.Bar(
            name="Your Scenario",
            x=sim_labels, y=sim_values,
            marker_color=SECTOR_COLORS_LIST[:6],
            text=[f"₹{v:,.0f}" for v in sim_values],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_whatif.update_layout(
            **PLOTLY_LAYOUT,
            title="Your Custom Allocation vs Historical Average",
            barmode="group",
            yaxis_title="Allocation (₹ Cr)",
            height=420,
        )
        st.plotly_chart(fig_whatif, use_container_width=True)

        # Pie chart of custom allocation
        if total_allocated > 0:
            fig_whatif_pie = go.Figure(data=[go.Pie(
                labels=sim_labels,
                values=sim_values,
                hole=0.45,
                marker=dict(colors=SECTOR_COLORS_LIST[:6],
                            line=dict(color="rgba(15,25,35,0.8)", width=2)),
                textinfo="label+percent",
                textfont=dict(size=11),
            )])
            fig_whatif_pie.update_layout(
                **PLOTLY_LAYOUT,
                title="Your Scenario — Budget Split",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_whatif_pie, use_container_width=True)

        # Predict impact with XGBoost
        st.markdown("---")
        st.markdown("#### 🧠 Predicted Impact of Your Scenario")

        from models import predict_outcomes

        X_whatif = latest_filtered[model_bundle["feature_names"]].copy()
        whatif_alloc_map = {
            "health_alloc_cr": sim_alloc["Healthcare"],
            "education_alloc_cr": sim_alloc["Education"],
            "agriculture_alloc_cr": sim_alloc["Agriculture"],
            "infrastructure_alloc_cr": sim_alloc["Infrastructure"],
            "water_alloc_cr": sim_alloc["Water"],
            "energy_alloc_cr": sim_alloc["Energy"],
        }
        for col, val in whatif_alloc_map.items():
            if col in X_whatif.columns:
                X_whatif[col] = val

        whatif_welfare = predict_outcomes(
            model_bundle["welfare_model"], X_whatif, model_bundle["feature_names"]
        ).mean()
        whatif_satisfaction = predict_outcomes(
            model_bundle["satisfaction_model"], X_whatif, model_bundle["feature_names"]
        ).mean()

        current_welfare_val = latest_filtered["overall_welfare_score"].mean()
        current_sat_val = latest_filtered["citizen_satisfaction_score"].mean()

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_metric_card(
                f"{whatif_welfare:.1f}", "Predicted Welfare",
                delta=whatif_welfare - current_welfare_val,
                delta_pct=(whatif_welfare - current_welfare_val) / current_welfare_val * 100,
            )
        with p2:
            render_metric_card(f"{current_welfare_val:.1f}", "Current Welfare")
        with p3:
            render_metric_card(
                f"{whatif_satisfaction:.2f}", "Predicted Satisfaction",
                delta=whatif_satisfaction - current_sat_val,
            )
        with p4:
            render_metric_card(f"{current_sat_val:.2f}", "Current Satisfaction")

        # Show comparison with optimized allocation if available
        if st.session_state.get("optim_result") is not None:
            st.markdown("---")
            st.markdown("#### ⚖️ Your Scenario vs AI-Optimized")

            opt_alloc, _, _ = st.session_state["optim_result"]
            opt_welfare_X = latest_filtered[model_bundle["feature_names"]].copy()
            for col, s_key in [("health_alloc_cr", "health"), ("education_alloc_cr", "education"),
                              ("agriculture_alloc_cr", "agriculture"), ("infrastructure_alloc_cr", "infrastructure"),
                              ("water_alloc_cr", "water"), ("energy_alloc_cr", "energy")]:
                if col in opt_welfare_X.columns:
                    opt_welfare_X[col] = opt_alloc.get(s_key, 0)

            opt_welfare_pred = predict_outcomes(
                model_bundle["welfare_model"], opt_welfare_X, model_bundle["feature_names"]
            ).mean()

            diff = whatif_welfare - opt_welfare_pred
            if diff > 0.5:
                verdict = "🏆 Your scenario **outperforms** the AI-optimized allocation!"
            elif diff < -0.5:
                verdict = f"📉 Your scenario is **{abs(diff):.1f} points below** the AI-optimized allocation."
            else:
                verdict = "🤝 Your scenario is **on par** with the AI-optimized allocation."

            comp_cols = st.columns(3)
            with comp_cols[0]:
                render_metric_card(f"{whatif_welfare:.1f}", "Your Scenario")
            with comp_cols[1]:
                render_metric_card(f"{opt_welfare_pred:.1f}", "AI-Optimized")
            with comp_cols[2]:
                render_metric_card(
                    f"{diff:+.1f}", "Difference",
                    delta=diff,
                )

            st.markdown(verdict)


if __name__ == "__main__":
    main()
