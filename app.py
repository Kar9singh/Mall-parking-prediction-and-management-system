import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ParkSmart | Mall Parking",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

# ---------------- LOAD PROJECT FILES ----------------
@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "random_forest_model.pkl")
    features = joblib.load(BASE_DIR / "model_features.pkl")
    return model, features

@st.cache_data
def load_data():
    data = pd.read_csv(BASE_DIR / "parking_cleaned.csv")
    data["LastUpdated"] = pd.to_datetime(
        data["LastUpdated"], dayfirst=True, errors="coerce"
    )
    return data.dropna(subset=["LastUpdated"])

try:
    model, model_features = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.info("Check that all three downloaded files are in the same folder as app.py.")
    st.stop()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
}
[data-testid="stSidebar"] {
    background: #10243a;
}
[data-testid="stSidebar"] * {
    color: #f1f5f9;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(115deg, #10243a, #1c5274);
    padding: 28px 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 30px;
    margin: 0;
}
.hero p {
    color: #d7e8f5;
    margin-bottom: 0;
}
.kpi {
    background: white;
    border: 1px solid #e5eaf1;
    border-radius: 15px;
    padding: 19px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(16,36,58,0.04);
}
.kpi-label {
    color: #65758b;
    font-size: 13px;
}
.kpi-value {
    color: #10243a;
    font-size: 27px;
    font-weight: 700;
    margin-top: 9px;
}
.kpi-note {
    color: #738196;
    font-size: 12px;
    margin-top: 5px;
}
div.stButton > button {
    background: #087e8b;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 18px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #06636d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🅿️ ParkSmart")
    st.caption("PARKING INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Occupancy Prediction", "Parking Analytics", "About Project"]
    )

    st.divider()
    st.caption("DATA SOURCE")
    st.write("Historical Birmingham parking data")
    st.caption("Predictions are estimates, not live occupancy.")

# ---------------- PREPARE DATA ----------------
df["Occupancy_Percentage"] = (
    df["Occupancy"] / df["Capacity"].replace(0, np.nan)
) * 100

df = df.dropna(subset=["Occupancy_Percentage"])
locations = sorted(df["SystemCodeNumber"].dropna().unique())

total_locations = len(locations)
total_capacity = int(df.groupby("SystemCodeNumber")["Capacity"].max().sum())
avg_occupancy = df["Occupancy_Percentage"].mean()
latest_time = df["LastUpdated"].max()

# ---------------- REUSABLE KPI ----------------
def show_kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>Mall Parking Management</h1>
        <p>Monitor historical parking patterns and estimate future occupancy.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Parking at a glance")
    st.caption("Summary metrics from the historical dataset, not live sensor readings.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_kpi("Parking Locations", f"{total_locations}", "Unique facilities in dataset")
    with c2:
        show_kpi("Combined Capacity", f"{total_capacity:,}", "Sum of each facility's maximum recorded capacity")
    with c3:
        show_kpi("Average Occupancy", f"{avg_occupancy:.1f}%", "Across valid historical records")
    with c4:
        show_kpi("Latest Record", latest_time.strftime("%d %b %Y") if pd.notna(latest_time) else "N/A", "Timestamp in dataset")

    st.write("")
    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Average occupancy by location")
        loc_summary = (
            df.groupby("SystemCodeNumber")["Occupancy_Percentage"]
            .mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(
            loc_summary,
            x="SystemCodeNumber",
            y="Occupancy_Percentage",
            color="Occupancy_Percentage",
            color_continuous_scale=["#b9e5e8", "#087e8b", "#10243a"],
            labels={
                "SystemCodeNumber": "Parking location",
                "Occupancy_Percentage": "Average occupancy (%)"
            }
        )
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Historical demand distribution")
        demand_df = df.copy()
        demand_df["Demand"] = pd.cut(
            demand_df["Occupancy_Percentage"],
            bins=[-0.01, 40, 75, 100.01],
            labels=["Low", "Medium", "High"]
        )
        demand_counts = (
            demand_df["Demand"].value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Demand").reset_index(name="Records")
        )
        fig = px.pie(
            demand_counts,
            names="Demand",
            values="Records",
            hole=0.62,
            color="Demand",
            color_discrete_map={
                "Low": "#8ac6a5",
                "Medium": "#f0c36d",
                "High": "#df7777"
            }
        )
        fig.update_layout(template="plotly_white", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------- OCCUPANCY PREDICTION ----------------
elif page == "Occupancy Prediction":
    st.markdown("""
    <div class="hero">
        <h1>Occupancy Prediction</h1>
        <p>Estimate parking demand for a selected location and time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("This model uses historical patterns. Results are estimates, not live parking availability.")

    with st.form("prediction_form"):
        location = st.selectbox("Parking location", locations)
        location_records = df[df["SystemCodeNumber"] == location]
        default_capacity = int(location_records["Capacity"].max())

        col1, col2, col3 = st.columns(3)
        with col1:
            capacity = st.number_input(
                "Parking capacity",
                min_value=1,
                value=max(default_capacity, 1),
                step=1
            )
        with col2:
            hour = st.slider("Hour of day", 0, 23, 14)
        with col3:
            day = st.selectbox(
                "Day of week",
                list(range(7)),
                index=0,
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ][x]
            )

        month = st.selectbox("Month", list(range(1, 13)), index=10,
                             format_func=lambda x: pd.Timestamp(2026, x, 1).strftime("%B"))
        submitted = st.form_submit_button("Predict Occupancy", use_container_width=True)

    if submitted:
        input_row = pd.DataFrame(0, index=[0], columns=model_features)

        required_values = {
            "Capacity": capacity,
            "Hour": hour,
            "DayOfWeek": day,
            "Month": month,
            "IsWeekend": int(day >= 5)
        }

        for col, value in required_values.items():
            if col in input_row.columns:
                input_row.loc[0, col] = value

        location_column = "SystemCodeNumber_" + location
        if location_column in input_row.columns:
            input_row.loc[0, location_column] = 1

        prediction = float(model.predict(input_row)[0])
        occupied = int(round(np.clip(prediction, 0, capacity)))
        available = int(capacity - occupied)
        percentage = occupied / capacity * 100

        if percentage <= 40:
            demand = "Low"
        elif percentage <= 75:
            demand = "Medium"
        else:
            demand = "High"

        st.divider()
        st.subheader("Prediction results")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            show_kpi("Estimated Occupied", f"{occupied:,}", "Spaces")
        with r2:
            show_kpi("Estimated Available", f"{available:,}", "Spaces")
        with r3:
            show_kpi("Occupancy Rate", f"{percentage:.1f}%", "Of selected capacity")
        with r4:
            show_kpi("Demand Level", demand, "Project-defined threshold")

        st.progress(percentage / 100)
        st.caption("Low: up to 40% · Medium: above 40% to 75% · High: above 75%")

# ---------------- ANALYTICS ----------------
elif page == "Parking Analytics":
    st.markdown("""
    <div class="hero">
        <h1>Parking Analytics</h1>
        <p>Explore historical occupancy by location and time.</p>
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox("Filter by parking location", locations)
    subset = df[df["SystemCodeNumber"] == selected].copy()

    hourly = (
        subset.groupby("Hour")["Occupancy_Percentage"].mean()
        if "Hour" in subset.columns
        else subset.assign(Hour=subset["LastUpdated"].dt.hour)
             .groupby("Hour")["Occupancy_Percentage"].mean()
    )

    fig = px.line(
        hourly.reset_index(),
        x="Hour",
        y="Occupancy_Percentage",
        markers=True,
        labels={"Occupancy_Percentage": "Average occupancy (%)"}
    )
    fig.update_layout(template="plotly_white", xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Historical records")
    st.dataframe(
        subset[["LastUpdated", "SystemCodeNumber", "Capacity", "Occupancy",
                "Occupancy_Percentage"]].sort_values("LastUpdated", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ---------------- ABOUT ----------------
else:
    st.markdown("""
    <div class="hero">
        <h1>About ParkSmart</h1>
        <p>Mall Parking Management and Occupancy Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project objectives")
    st.markdown("""
    - Analyze historical parking occupancy.
    - Compare parking demand across locations and hours.
    - Predict occupancy using a trained Random Forest model.
    - Estimate available spaces and categorize demand.
    """)

    st.subheader("Technology stack")
    st.markdown("""
    **Python · Pandas · Scikit-learn · Streamlit · Plotly**
    """)

    st.warning(
        "The current dataset is historical Birmingham parking data. "
        "Live monitoring requires a live sensor or API data source."
    )
