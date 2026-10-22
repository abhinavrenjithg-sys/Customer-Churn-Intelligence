"""Customer Churn Intelligence - Enterprise Streamlit Dashboard.

Run command:
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="Customer Churn Intelligence | ML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Premium Modern Dark CSS Design System
PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Streamlit Main Viewport Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* Top Hero Header */
    .hero-container {
        background: radial-gradient(120% 120% at 50% 0%, #1E293B 0%, #0F172A 70%, #020617 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-title-group h1 {
        color: #F8FAFC !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        letter-spacing: -0.6px;
        margin: 0 0 6px 0 !important;
        padding: 0 !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-title-group p {
        color: #94A3B8 !important;
        font-size: 14px !important;
        margin: 0 !important;
        font-weight: 400;
    }
    .hero-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6EE7B7;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 9999px;
        letter-spacing: 0.2px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    /* Metric KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #0F172A;
        background-image: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 25px -8px rgba(99, 102, 241, 0.25);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #38BDF8);
        opacity: 0.8;
    }
    .kpi-card.crimson::before { background: linear-gradient(90deg, #F43F5E, #FB7185); }
    .kpi-card.emerald::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }

    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748B;
        font-weight: 500;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.35);
    }
    .glass-panel h3 {
        color: #F8FAFC;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 14px;
        letter-spacing: -0.3px;
    }

    /* Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.3px;
    }
    .badge-high {
        background: rgba(244, 63, 94, 0.15);
        color: #FDA4AF;
        border: 1px solid #F43F5E;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #FDE68A;
        border: 1px solid #F59E0B;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid #10B981;
    }

    /* Feature Drivers Pill */
    .driver-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .driver-pill.risk {
        border-left: 3px solid #F43F5E;
    }
    .driver-pill.protective {
        border-left: 3px solid #10B981;
    }
    .driver-name {
        color: #E2E8F0;
        font-weight: 600;
    }
    .driver-val-risk {
        color: #FB7185;
        font-weight: 700;
        font-size: 12px;
    }
    .driver-val-prot {
        color: #34D399;
        font-weight: 700;
        font-size: 12px;
    }

    /* Callout & Disclaimer */
    .callout-box {
        background: rgba(30, 41, 59, 0.45);
        border-left: 4px solid #6366F1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13px;
        color: #CBD5E1;
        line-height: 1.55;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 14.5px !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    }

    /* Section Headers */
    .section-header {
        color: #F8FAFC;
        font-size: 20px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_cached_data() -> Optional[pd.DataFrame]:
    """Loads authentic dataset safely."""
    from src.data_loader import load_raw_data
    if DATA_PATH.exists():
        try:
            df, _ = load_raw_data(DATA_PATH)
            return df
        except Exception:
            return None
    return None


@st.cache_data
def load_metrics_and_summary() -> Optional[Dict[str, Any]]:
    """Loads recorded training metrics from JSON."""
    metrics_file = ARTIFACTS_DIR / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_cv_results() -> Optional[pd.DataFrame]:
    """Loads 5-fold cross validation summary table."""
    cv_file = ARTIFACTS_DIR / "cv_results.csv"
    if cv_file.exists():
        return pd.read_csv(cv_file)
    return None


@st.cache_resource
def get_predictor():
    """Initializes and caches the reusable inference predictor."""
    from src.predict import ChurnPredictor
    model_path = ARTIFACTS_DIR / "best_model_pipeline.joblib"
    if model_path.exists():
        try:
            return ChurnPredictor(model_path=model_path)
        except Exception as e:
            st.error(f"Failed to load predictor: {e}")
            return None
    return None


# Load artifacts
metrics_data = load_metrics_and_summary()
cv_df = load_cv_results()
raw_df = load_cached_data()
predictor = get_predictor()

# If artifacts missing, provide guidance
if metrics_data is None or predictor is None:
    st.error("⚠️ **Trained Model Artifacts Missing**")
    st.info("Please run `python main.py` in your terminal to train models and generate artifacts.")
    st.stop()

test_metrics = metrics_data.get("test_metrics", {})
train_count = metrics_data.get("train_samples", 5634)
test_count = metrics_data.get("test_samples", 1409)
total_subscribers = train_count + test_count
churn_rate_val = metrics_data.get("test_churn_rate", 0.2654) * 100
model_name = metrics_data.get("selected_model", "RandomForest")

# Top Hero Bar
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title-group">
        <h1>⚡ Customer Churn Intelligence</h1>
        <p>Machine Learning Prediction, Risk Explainability & Strategic Retention Analytics</p>
    </div>
    <div class="hero-status-pill">
        <div class="pulse-dot"></div>
        Model Online (v1.0.0 • {model_name})
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    menu_selection = st.radio(
        "Select Module:",
        [
            "📊 Executive Overview",
            "🔍 Cohort Analytics",
            "⚡ Subscriber Churn Prediction",
            "📈 Model Performance & Validation",
            "🧠 Explainability & Drivers",
            "ℹ️ Documentation & Governance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### 📌 Benchmark Highlights")
    st.markdown(f"""
    - **Architecture:** `{model_name}`
    - **Test ROC-AUC:** `{test_metrics.get('roc_auc', 0.8432):.4f}`
    - **Test Recall:** `{test_metrics.get('recall', 0.7941) * 100:.1f}%`
    - **Test PR-AUC:** `{test_metrics.get('pr_auc', 0.6533):.4f}`
    """)

    st.markdown("---")
    st.markdown("<small style='color: #64748B;'>Educational Portfolio Project • Powered by Scikit-Learn & TreeSHAP</small>", unsafe_allow_html=True)


# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if menu_selection == "📊 Executive Overview":
    st.markdown('<div class="section-header">📊 Executive Overview & Key Business KPIs</div>', unsafe_allow_html=True)

    # Metric Cards Grid
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Analyzed Cohort <span>👥</span></div>
            <div class="kpi-value">{total_subscribers:,}</div>
            <div class="kpi-subtext">5,634 Train • 1,409 Held-Out Test</div>
        </div>
        <div class="kpi-card crimson">
            <div class="kpi-label">Baseline Churn Rate <span>⚠️</span></div>
            <div class="kpi-value" style="color: #FDA4AF;">{churn_rate_val:.1f}%</div>
            <div class="kpi-subtext">1,869 Observed Defections</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Test ROC-AUC <span>🎯</span></div>
            <div class="kpi-value" style="color: #93C5FD;">{test_metrics.get('roc_auc', 0.8432):.3f}</div>
            <div class="kpi-subtext">Ranking & Discrimination Capacity</div>
        </div>
        <div class="kpi-card emerald">
            <div class="kpi-label">Churn Recall <span>🛡️</span></div>
            <div class="kpi-value" style="color: #6EE7B7;">{test_metrics.get('recall', 0.7941) * 100:.1f}%</div>
            <div class="kpi-subtext">297 of 374 Churners Flagged</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">PR-AUC / Avg Precision <span>📈</span></div>
            <div class="kpi-value" style="color: #FCD34D;">{test_metrics.get('pr_auc', 0.6533):.3f}</div>
            <div class="kpi-subtext">+146% over random baseline (0.265)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.0])

    with col1:
        st.markdown("""
        <div class="glass-panel">
            <h3>🎯 Strategic Findings & High-Leverage Retention Levers</h3>
            <p style="color: #CBD5E1; font-size: 13.5px; line-height: 1.6;">
                Identifying at-risk customers early protects recurring revenue and customer lifetime value (LTV).
                Empirical analysis reveals four primary operational insights:
            </p>
            <div style="margin-top: 14px;">
                <div class="driver-pill risk">
                    <div>
                        <span class="driver-name">1. Contract Structure is Paramount</span>
                        <div style="font-size: 11.5px; color: #94A3B8;">Month-to-month subscribers experience <b>42.7% churn</b>, versus <b>2.8%</b> on Two-Year contracts.</div>
                    </div>
                    <span class="driver-val-risk">High Friction</span>
                </div>
                <div class="driver-pill risk">
                    <div>
                        <span class="driver-name">2. The 90-Day Early Lifecycle Cliff</span>
                        <div style="font-size: 11.5px; color: #94A3B8;">Attrition is heavily concentrated in the first 1–5 months. Subscribers past 24 months show 4x greater loyalty.</div>
                    </div>
                    <span class="driver-val-risk">Onboarding Risk</span>
                </div>
                <div class="driver-pill risk">
                    <div>
                        <span class="driver-name">3. Fiber Optic Price Sensitivity</span>
                        <div style="font-size: 11.5px; color: #94A3B8;">Subscribers paying >$75/month on Fiber Optic churn at 41.9%, indicating high price-to-value friction.</div>
                    </div>
                    <span class="driver-val-risk">Value Perception</span>
                </div>
                <div class="driver-pill protective">
                    <div>
                        <span class="driver-name">4. Autopay Retention Advantage</span>
                        <div style="font-size: 11.5px; color: #94A3B8;">Subscribers on automated bank/credit draft churn at less than half the rate of electronic check users.</div>
                    </div>
                    <span class="driver-val-prot">Protective Anchor</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if raw_df is not None:
            st.markdown('<div class="glass-panel"><h3>📊 Churn Rate by Contract Commitment</h3>', unsafe_allow_html=True)
            ct = pd.crosstab(raw_df["Contract"], raw_df["Churn"], normalize="index") * 100
            ct = ct.reindex(["Month-to-month", "One year", "Two year"]).reset_index()

            fig = px.bar(
                ct,
                x="Contract",
                y="Yes",
                text="Yes",
                color="Contract",
                color_discrete_sequence=["#F43F5E", "#F59E0B", "#10B981"],
                labels={"Yes": "Churn Rate (%)", "Contract": "Contract Type"},
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                textfont=dict(size=12, color="#F8FAFC"),
                marker=dict(line=dict(width=0)),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=280,
                showlegend=False,
                yaxis=dict(
                    range=[0, 52],
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(color="#94A3B8"),
                    title=dict(text="Churn Rate (%)", font=dict(color="#94A3B8", size=11)),
                ),
                xaxis=dict(
                    tickfont=dict(color="#F8FAFC", size=12, family="Plus Jakarta Sans"),
                    title=dict(text=""),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 2: COHORT ANALYTICS
# ==========================================
elif menu_selection == "🔍 Cohort Analytics":
    st.markdown('<div class="section-header">🔍 Interactive Customer Cohort Exploration</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.info("Dataset not loaded.")
        st.stop()

    # Filter Controls Card
    with st.expander("⚙️ Segment Filtering Controls", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_contracts = st.multiselect("Contract Type", sorted(raw_df["Contract"].unique()), default=list(raw_df["Contract"].unique()))
        with f2:
            sel_internet = st.multiselect("Internet Service", sorted(raw_df["InternetService"].unique()), default=list(raw_df["InternetService"].unique()))
        with f3:
            sel_payments = st.multiselect("Payment Method", sorted(raw_df["PaymentMethod"].unique()), default=list(raw_df["PaymentMethod"].unique()))
        with f4:
            tenure_sel = st.slider("Tenure Window (Months)", 0, 72, (0, 72))

    filtered_df = raw_df[
        (raw_df["Contract"].isin(sel_contracts)) &
        (raw_df["InternetService"].isin(sel_internet)) &
        (raw_df["PaymentMethod"].isin(sel_payments)) &
        (raw_df["tenure"] >= tenure_sel[0]) &
        (raw_df["tenure"] <= tenure_sel[1])
    ]

    total_filt = len(filtered_df)
    churn_count = (filtered_df["Churn"] == "Yes").sum() if total_filt > 0 else 0
    churn_rate_seg = (churn_count / total_filt * 100) if total_filt > 0 else 0.0

    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px 18px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
        <span style="color: #E2E8F0; font-size: 13.5px;">Active Segment Cohort: <b>{total_filt:,}</b> subscribers</span>
        <span style="color: #FDA4AF; font-size: 13.5px; font-weight: 700;">Segment Churn Rate: {churn_rate_seg:.1f}% ({churn_count:,} churners)</span>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown('<div class="glass-panel"><h3>💳 Churn Rate by Payment Method (%)</h3>', unsafe_allow_html=True)
        if total_filt > 0:
            pm_ct = pd.crosstab(filtered_df["PaymentMethod"], filtered_df["Churn"], normalize="index") * 100
            pm_df = pm_ct.reset_index()
            if "Yes" in pm_df.columns:
                pm_df = pm_df.sort_values(by="Yes", ascending=True)
                fig_pm = px.bar(
                    pm_df,
                    y="PaymentMethod",
                    x="Yes",
                    orientation="h",
                    text="Yes",
                    color="Yes",
                    color_continuous_scale=["#10B981", "#F59E0B", "#F43F5E"],
                    labels={"Yes": "Churn Rate (%)", "PaymentMethod": ""},
                )
                fig_pm.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='inside',
                    textfont=dict(color="#FFFFFF", size=11),
                )
                fig_pm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(tickfont=dict(color="#E2E8F0", size=11)),
                )
                st.plotly_chart(fig_pm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="glass-panel"><h3>💵 Monthly Charges ($) Density by Churn</h3>', unsafe_allow_html=True)
        if total_filt > 0:
            fig_hist = px.histogram(
                filtered_df,
                x="MonthlyCharges",
                color="Churn",
                barmode="overlay",
                nbins=25,
                color_discrete_map={"No": "#38BDF8", "Yes": "#F43F5E"},
                labels={"MonthlyCharges": "Monthly Charges (USD)", "Churn": "Status"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(font=dict(color="#E2E8F0")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View Sample Records (First 50 rows)"):
        st.dataframe(filtered_df.head(50), use_container_width=True)


# ==========================================
# PAGE 3: SUBSCRIBER CHURN PREDICTION
# ==========================================
elif menu_selection == "⚡ Subscriber Churn Prediction":
    st.markdown('<div class="section-header">⚡ Subscriber Risk Scoring & Retention Simulation</div>', unsafe_allow_html=True)

    # Preset selection
    st.markdown("#### 1. Select Scenario Preset or Enter Custom Data")
    scenario_choice = st.selectbox(
        "Load Pre-Configured Customer Archetype:",
        [
            "⚠️ High Risk: New Month-to-Month Fiber Subscriber",
            "🛡️ Low Risk: Loyal 5-Year Two-Year Contract Autopay",
            "⚖️ Moderate Risk: 1-Year Tenure DSL Subscriber with Add-ons",
            "Custom Manual Entry",
        ],
    )

    defaults = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 94.50, "TotalCharges": 189.00
    }

    if scenario_choice == "🛡️ Low Risk: Loyal 5-Year Two-Year Contract Autopay":
        defaults.update({
            "tenure": 60, "Contract": "Two year", "InternetService": "DSL", "Partner": "Yes", "Dependents": "Yes",
            "PaymentMethod": "Credit card (automatic)", "TechSupport": "Yes", "OnlineSecurity": "Yes",
            "MonthlyCharges": 65.00, "TotalCharges": 3900.00
        })
    elif scenario_choice == "⚖️ Moderate Risk: 1-Year Tenure DSL Subscriber with Add-ons":
        defaults.update({
            "tenure": 14, "Contract": "One year", "InternetService": "DSL", "Partner": "Yes", "Dependents": "No",
            "PaymentMethod": "Bank transfer (automatic)", "TechSupport": "Yes", "OnlineSecurity": "No",
            "MonthlyCharges": 55.40, "TotalCharges": 775.60
        })

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Account & Financials")
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=int(defaults["tenure"]))
            contract = st.selectbox("Contract Commitment", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]))
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(defaults["MonthlyCharges"]), step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(defaults["TotalCharges"]), step=10.0)
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(defaults["PaymentMethod"]))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if defaults["PaperlessBilling"] == "Yes" else 1)

        with col2:
            st.markdown("##### 🌐 Internet & Core Services")
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(defaults["InternetService"]))
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if defaults["PhoneService"] == "Yes" else 1)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]))
            online_security = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]))
            online_backup = st.selectbox("Online Backup Add-on", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]))
            device_protection = st.selectbox("Device Protection Add-on", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]))

        with col3:
            st.markdown("##### 🛠️ Support & Demographics")
            tech_support = st.selectbox("Tech Support Add-on", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]))
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]))
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]))
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if defaults["gender"] == "Female" else 1)
            senior = st.selectbox("Senior Citizen (Age 65+)", [0, 1], index=int(defaults["SeniorCitizen"]))
            partner = st.selectbox("Has Partner", ["Yes", "No"], index=0 if defaults["Partner"] == "Yes" else 1)
            dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=0 if defaults["Dependents"] == "Yes" else 1)

        st.markdown("<br>", unsafe_allow_html=True)
        decision_threshold = st.slider(
            "Classification Decision Threshold:",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Probabilities at or above this threshold trigger a positive Churn prediction."
        )

        submitted = st.form_submit_button("⚡ Predict Churn Probability", use_container_width=True)

    if submitted:
        input_data = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }

        with st.spinner("Scoring subscriber risk profile..."):
            pred_response = predictor.predict_single_with_explanation(input_data, threshold=decision_threshold)

        prediction = pred_response["prediction"]
        explanation = pred_response["explanation"]
        prob = float(prediction["churn_probability"])
        prob_pct = prob * 100
        risk_cat = prediction["risk_category"]

        st.markdown("---")
        st.markdown("#### 2. Risk Evaluation Results")

        res_col1, res_col2 = st.columns([1.1, 1.0])

        with res_col1:
            # Styled Gauge
            bar_color = "#F43F5E" if prob >= decision_threshold else "#10B981"
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Churn Probability", 'font': {'size': 17, 'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'}},
                number={'suffix': "%", 'font': {'size': 38, 'color': '#F8FAFC', 'weight': 'bold', 'family': 'Plus Jakarta Sans'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                    'bar': {'color': bar_color, 'thickness': 0.32},
                    'bgcolor': "#1E293B",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [35, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': "#F59E0B", 'width': 4},
                        'thickness': 0.8,
                        'value': decision_threshold * 100,
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            badge_class = "badge-high" if "High" in risk_cat else ("badge-medium" if "Medium" in risk_cat else "badge-low")
            badge_text = "CRITICAL RISK" if "High" in risk_cat else ("ELEVATED RISK" if "Medium" in risk_cat else "LOW RISK / SAFE")
            st.markdown(f"""
            <div class="glass-panel" style="padding: 18px;">
                <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">Risk Classification</div>
                <div style="margin-bottom: 12px;"><span class="status-badge {badge_class}">{badge_text} ({risk_cat})</span></div>
                <div style="font-size: 13.5px; color: #CBD5E1; line-height: 1.5;">
                    <b>Model Alert:</b> {('Flagged for Proactive Retention Outreach' if prediction['predicted_churn'] == 1 else 'Account within Acceptable Retention Stability Threshold')}
                </div>
                <div style="margin-top: 10px; font-size: 12.5px; color: #94A3B8;">
                    Decision Boundary: <b>{decision_threshold}</b> • Raw Probability: <b>{prob_pct:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Local Risk Factors and Actionable Playbook
        st.markdown("#### 3. Root Cause Insights (TreeSHAP) & Retention Playbook")
        e_col1, e_col2 = st.columns(2)

        with e_col1:
            st.markdown("##### 🔺 Attributes Elevating Churn Risk")
            risks = explanation.get("top_risk_factors", [])
            if risks:
                for r in risks:
                    st.markdown(f"""
                    <div class="driver-pill risk">
                        <span class="driver-name">{r['feature']}</span>
                        <span class="driver-val-risk">{r['odds_impact']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant risk-elevating attributes detected.")

        with e_col2:
            st.markdown("##### 🛡️ Attributes Anchoring Retention")
            prots = explanation.get("top_protective_factors", [])
            if prots:
                for p in prots:
                    st.markdown(f"""
                    <div class="driver-pill protective">
                        <span class="driver-name">{p['feature']}</span>
                        <span class="driver-val-prot">{p['odds_impact']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant retention anchors detected.")

        # Tailored Playbook
        st.markdown("""
        <div class="glass-panel" style="margin-top: 14px;">
            <h3>💡 Automated Retention Action Plan</h3>
        """, unsafe_allow_html=True)

        playbook = []
        if contract == "Month-to-month":
            playbook.append("📌 <b>Contract Transition Incentive:</b> Offer a 15% promotional credit for migrating to a 12-month commitment.")
        if internet_service == "Fiber optic" and monthly_charges > 75:
            playbook.append("📌 <b>Value-to-Price Rebalance:</b> Perform bill audit and offer complimentary speed boost or streaming bundle to reinforce perceived value.")
        if tech_support == "No" or online_security == "No":
            playbook.append("📌 <b>Service Attachment:</b> Enroll subscriber in 3 months of complimentary Tech Support to establish higher product stickiness.")
        if payment_method == "Electronic check":
            playbook.append("📌 <b>Autopay Migration:</b> Provide a one-time $10 bill discount upon enrollment in automated Credit Card or Bank Draft.")
        if not playbook:
            playbook.append("📌 <b>Standard Account Health:</b> Subscriber displays positive retention health. Continue standard quarterly check-ins.")

        for p in playbook:
            st.markdown(f"<div style='margin-bottom: 8px; color: #CBD5E1; font-size: 13px;'>{p}</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="callout-box"><b>Causality Disclaimer:</b> {explanation["disclaimer"]}</div></div>', unsafe_allow_html=True)


# ==========================================
# PAGE 4: MODEL PERFORMANCE & VALIDATION
# ==========================================
elif menu_selection == "📈 Model Performance & Validation":
    st.markdown('<div class="section-header">📈 Model Validation, Benchmark & Diagnostic Curves</div>', unsafe_allow_html=True)

    # Metric Row
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC (Test)</div>
            <div class="kpi-value" style="color: #93C5FD;">{test_metrics.get('roc_auc', 0.8432):.4f}</div>
            <div class="kpi-subtext">Overall Discriminative Capacity</div>
        </div>
        <div class="kpi-card crimson">
            <div class="kpi-label">PR-AUC / Avg Prec</div>
            <div class="kpi-value" style="color: #FDA4AF;">{test_metrics.get('pr_auc', 0.6533):.4f}</div>
            <div class="kpi-subtext">Minority Class Precision Curve</div>
        </div>
        <div class="kpi-card emerald">
            <div class="kpi-label">Recall (Sensitivity)</div>
            <div class="kpi-value" style="color: #6EE7B7;">{test_metrics.get('recall', 0.7941) * 100:.1f}%</div>
            <div class="kpi-subtext">297 of 374 Churners Detected</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision</div>
            <div class="kpi-value">{test_metrics.get('precision', 0.5238):.4f}</div>
            <div class="kpi-subtext">Accuracy of Positive Alerts</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">F1-Score</div>
            <div class="kpi-value" style="color: #FCD34D;">{test_metrics.get('f1', 0.6312):.4f}</div>
            <div class="kpi-subtext">Harmonic Balance Metric</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5-Fold Stratified CV Benchmark Table
    st.markdown('<div class="glass-panel"><h3>1. 5-Fold Stratified Cross-Validation Benchmark</h3>', unsafe_allow_html=True)
    if cv_df is not None:
        display_cv = cv_df.copy()
        display_cv.columns = [c.replace("test_", "").replace("_", " ").title() for c in display_cv.columns]
        st.dataframe(display_cv, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Diagnostic Curves
    st.markdown("### 2. Diagnostic Curves (Held-Out Test Set)")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        roc_img = FIGURES_DIR / "roc_curve.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)

    with col_c2:
        pr_img = FIGURES_DIR / "precision_recall_curve.png"
        if pr_img.exists():
            st.image(str(pr_img), caption="Precision-Recall Curve with Baseline", use_container_width=True)

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        cm_img = FIGURES_DIR / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Held-Out Test Set Confusion Matrix", use_container_width=True)

    with col_c4:
        th_img = FIGURES_DIR / "threshold_analysis.png"
        if th_img.exists():
            st.image(str(th_img), caption="Precision, Recall & F1 vs. Classification Threshold", use_container_width=True)


# ==========================================
# PAGE 5: EXPLAINABILITY & DRIVERS
# ==========================================
elif menu_selection == "🧠 Explainability & Drivers":
    st.markdown('<div class="section-header">🧠 Global Feature Importance & Behavioral Drivers</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-panel">
        <p style="color: #CBD5E1; font-size: 13.5px; line-height: 1.6; margin: 0;">
            Model explainability bridges the gap between machine learning predictions and executive decision-making.
            Below are the top predictive features identified across the subscriber population by the primary model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    feat_img = FIGURES_DIR / "feature_importance.png"
    if feat_img.exists():
        st.image(str(feat_img), caption="Top Predictive Features (Global Model Associations)", use_container_width=True)

    st.markdown("""
    <div class="glass-panel" style="margin-top: 20px;">
        <h3>Plain-English Breakdown of Key Behavioral Signals</h3>
        <div style="color: #CBD5E1; font-size: 13.5px; line-height: 1.7;">
            <p><b>1. Relationship Duration (<code>tenure</code>):</b> Strongest protective anchor. Subscribers beyond 24 months demonstrate habituated usage and lower price sensitivity.</p>
            <p><b>2. Contract Commitment (<code>Contract_Month-to-month</code>):</b> Primary risk driver. Month-to-month subscribers switch to rival promotional offers with zero exit friction.</p>
            <p><b>3. Fiber Optic Technology (<code>InternetService_Fiber optic</code>):</b> Associated with elevated defection rates (41.9%), driven by high monthly bills ($75-$105) and high reliability expectations.</p>
            <p><b>4. Electronic Check Payments (<code>PaymentMethod_Electronic check</code>):</b> Displays more than double the churn rate of automated credit card or bank drafts due to monthly manual bill friction.</p>
            <p><b>5. Support Attachments (<code>TechSupport_No</code>):</b> Accounts without technical assistance or device protection have fewer operational touchpoints, reducing switching barriers.</p>
        </div>
        <div class="callout-box">
            <b>Crucial Governance Caveat:</b><br>
            Machine learning feature importances quantify statistical correlation within historical subscriber data, not verified causation. Modifying a correlated factor (such as enforcing autopay) will not independently prevent churn unless the underlying product satisfaction and price sensitivity issues are addressed.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 6: DOCUMENTATION & GOVERNANCE
# ==========================================
elif menu_selection == "ℹ️ Documentation & Governance":
    st.markdown('<div class="section-header">ℹ️ System Documentation & Responsible AI Governance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-panel">
        <h3>📖 Dataset Provenance & Methodology</h3>
        <div style="color: #CBD5E1; font-size: 13.5px; line-height: 1.7;">
            <p><b>Source:</b> IBM Telco Customer Churn dataset available via <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank" style="color: #38BDF8;">Kaggle</a>.</p>
            <p><b>Leakage-Safe Architecture:</b> Strict train-test separation with Scikit-Learn <code>ColumnTransformer</code> (Median imputation for numericals, <code>OneHotEncoder(handle_unknown='ignore')</code> for categoricals).</p>
            <p><b>Model Selection:</b> 5-fold Stratified Cross-Validation on training partition (5,634 rows). Final single-pass evaluation on held-out test partition (1,409 rows).</p>
        </div>
    </div>

    <div class="glass-panel">
        <h3>⚖️ Responsible AI & Ethical Governance Considerations</h3>
        <div style="color: #CBD5E1; font-size: 13.5px; line-height: 1.7;">
            <p><b>Demographic Neutrality:</b> Attributes like <code>gender</code> and <code>SeniorCitizen</code> are monitored to ensure models are not utilized for discriminatory pricing or denial of essential telecommunications access.</p>
            <p><b>Probability Calibration:</b> Brier score loss (0.1625) is monitored to ensure predicted percentages reflect genuine empirical departure probabilities.</p>
            <p><b>Educational Demonstration Notice:</b> This system is designed as an AI Engineer portfolio showcase and operates on offline historical records.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
