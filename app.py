"""
Streamlit App: AI Dynamic Traffic Signal Control Analytics
Run with: streamlit run app.py
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page configuration
st.set_page_config(
    page_title="AI Traffic Signal Analytics",
    page_icon="🚥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.title("🚥 AI Dynamic Traffic Signal Control Analytics")
st.markdown("### A/B Testing Case Study: RL PPO Dynamic Control vs Static Fixed Timer")
st.markdown("---")

# Load Datasets
@st.cache_data
def load_data():
    with open("dashboard_summary.json", "r") as f:
        summary = json.load(f)
    return summary

try:
    summary = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}. Please make sure dashboard_summary.json exists in the working directory.")
    st.stop()

# Sidebar options
st.sidebar.header("⚙️ Dashboard Controls")
controller_choice = st.sidebar.multiselect(
    "Filter Controller Types",
    options=["Static Fixed", "RL Dynamic"],
    default=["Static Fixed", "RL Dynamic"]
)

# Top KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Wait Time Reduction",
        value=f"{summary['impact_metrics']['wait_reduction_pct']}%",
        delta="-82.0% delay",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Queue Reduction",
        value=f"{summary['impact_metrics']['queue_reduction_pct']}%",
        delta="-69.6% cars",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Throughput Gain",
        value=f"{summary['impact_metrics']['throughput_gain_pct']}%",
        delta="+76.9% flow",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="Annual CO2 Saved",
        value=f"{summary['impact_metrics']['co2_saved_tons_per_yr']} Tons",
        delta="Saved / Intersection",
        delta_color="normal"
    )

st.markdown("---")

# Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive A/B Summary", 
    "🧪 Statistical Hypothesis Test", 
    "👁️ YOLO Vision Telemetry", 
    "🌱 Sustainability & ROI"
])

# TAB 1: EXECUTIVE SUMMARY
with tab1:
    st.subheader("Performance Comparison (Static vs RL Controller)")
    
    col_a, col_b = st.columns(2)
    
    static_s = summary["static_stats"]
    rl_s = summary["rl_stats"]
    
    with col_a:
        st.info("🔴 **Static 120s Fixed Controller**")
        st.write(f"• **Avg Wait Time:** {static_s['avg_wait_time_sec_mean']} seconds")
        st.write(f"• **Avg Queue Length:** {static_s['avg_queue_mean']} cars")
        st.write(f"• **Throughput:** {static_s['avg_throughput_mean']} cars/sec")
        st.write(f"• **1-Hr CO2 Emissions:** {static_s['total_co2_kg']} kg")

    with col_b:
        st.success("🟢 **RL PPO Dynamic Controller**")
        st.write(f"• **Avg Wait Time:** {rl_s['avg_wait_time_sec_mean']} seconds")
        st.write(f"• **Avg Queue Length:** {rl_s['avg_queue_mean']} cars")
        st.write(f"• **Throughput:** {rl_s['avg_throughput_mean']} cars/sec")
        st.write(f"• **1-Hr CO2 Emissions:** {rl_s['total_co2_kg']} kg")

    st.markdown("---")
    st.subheader("Wait Time & Queue Length Trend Over Time (30s Buckets)")
    
    ts_df = pd.DataFrame(summary["time_series"])
    
    if HAS_PLOTLY:
        fig_wait = px.line(
            ts_df, 
            x="bucket", 
            y=["avg_wait_time_sec_static", "avg_wait_time_sec_rl"],
            labels={"value": "Wait Time (sec)", "bucket": "Simulation Time (seconds)", "variable": "Controller"},
            title="Average Vehicle Wait Time Trajectory",
            color_discrete_map={"avg_wait_time_sec_static": "#f43f5e", "avg_wait_time_sec_rl": "#10b981"}
        )
        st.plotly_chart(fig_wait, use_container_width=True)
        
        fig_queue = px.line(
            ts_df, 
            x="bucket", 
            y=["total_queue_cars_static", "total_queue_cars_rl"],
            labels={"value": "Queue Length (cars)", "bucket": "Simulation Time (seconds)", "variable": "Controller"},
            title="Intersection Queue Length Comparison",
            color_discrete_map={"total_queue_cars_static": "#f59e0b", "total_queue_cars_rl": "#00f2fe"}
        )
        st.plotly_chart(fig_queue, use_container_width=True)
    else:
        st.line_chart(ts_df.set_index("bucket")[["avg_wait_time_sec_static", "avg_wait_time_sec_rl"]])

# TAB 2: STATISTICAL HYPOTHESIS TEST
with tab2:
    st.subheader("Welch's Two-Sample t-Test & Cohen's d Effect Size")
    
    ttest = summary["ttest"]
    
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("t-Statistic", ttest["t_statistic"])
    col_t2.metric("p-Value", ttest["p_value_formatted"])
    col_t3.metric("Cohen's d Effect Size", ttest["cohens_d"])
    
    st.markdown("""
    > **Statistical Interpretation:**
    > - **Null Hypothesis ($H_0$):** Static and RL controllers have equal mean wait times.
    > - **Result:** With $p < 0.0001$, we reject $H_0$. The RL dynamic controller produces a **statistically significant** reduction in vehicle delay.
    > - **Effect Size:** Cohen's $d = 2.23$ confirms an **extremely large practical improvement** ($d > 0.8$).
    """)
    
    # Confidence Intervals Table
    ci_data = pd.DataFrame([
        {
            "Controller": "Static 120s Fixed", 
            "Mean Wait Time (s)": static_s["avg_wait_time_sec_mean"], 
            "95% CI Lower": static_s["avg_wait_ci95_low"], 
            "95% CI Upper": static_s["avg_wait_ci95_high"]
        },
        {
            "Controller": "RL Dynamic Controller", 
            "Mean Wait Time (s)": rl_s["avg_wait_time_sec_mean"], 
            "95% CI Lower": rl_s["avg_wait_ci95_low"], 
            "95% CI Upper": rl_s["avg_wait_ci95_high"]
        }
    ])
    st.table(ci_data)

# TAB 3: YOLO VISION TELEMETRY
with tab3:
    st.subheader("Computer Vision & Sensor Detection Analytics")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.write("#### Vehicle Class Distribution")
        v_mix = pd.DataFrame(list(summary["vehicle_mix"].items()), columns=["Vehicle Class", "Count"])
        if HAS_PLOTLY:
            fig_pie = px.pie(v_mix, names="Vehicle Class", values="Count", title="Detected Vehicle Composition", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.bar_chart(v_mix.set_index("Vehicle Class"))

    with col_v2:
        st.write("#### Average Estimated Speed by Vehicle Type")
        v_speed = pd.DataFrame(list(summary["avg_speed_by_class"].items()), columns=["Vehicle Class", "Avg Speed (km/h)"])
        if HAS_PLOTLY:
            fig_bar = px.bar(v_speed, x="Vehicle Class", y="Avg Speed (km/h)", color="Vehicle Class", title="Speed Distribution")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.bar_chart(v_speed.set_index("Vehicle Class"))

    st.write(f"• **Total Object Detections:** {summary['total_detections']:,}")
    st.write(f"• **Average Model Confidence Score:** {summary['avg_confidence'] * 100:.1f}%")
    st.write(f"• **Emergency Preemption Overrides:** {summary['emergency_count']} events detected")

# TAB 4: SUSTAINABILITY & ROI
with tab4:
    st.subheader("🌱 Sustainability & Environmental ROI Calculator")
    
    st.write("### Interactive City-Scale Calculator")
    
    intersections = st.slider("Select Number of City Intersections", min_value=1, max_value=500, value=25, step=5)
    fuel_price = st.number_input("Average Fuel Price ($ / Liter)", min_value=0.5, max_value=5.0, value=1.20, step=0.05)
    
    annual_co2_saved = summary["impact_metrics"]["co2_saved_tons_per_yr"] * intersections
    annual_fuel_saved = summary["impact_metrics"]["est_annual_fuel_saved_liters"] * intersections
    annual_money_saved = annual_fuel_saved * fuel_price
    
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("City-Wide Annual CO2 Saved", f"{annual_co2_saved:,.1f} Metric Tons")
    col_e2.metric("City-Wide Annual Fuel Saved", f"{annual_fuel_saved:,.0f} Liters")
    col_e3.metric("City-Wide Financial Savings", f"${annual_money_saved:,.2f}")

st.markdown("---")
st.caption("AI Dynamic Traffic Signal Control Project | Built with Python, SciPy, Pandas & Streamlit")
