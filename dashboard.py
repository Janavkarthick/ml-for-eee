"""
dashboard.py - Smart Grid Control-Room Dashboard
============================================================
PURPOSE:
    Interactive web dashboard styled like a premium utility control-room UI.
    Features a one-way communication architecture:
    - Solar Owner: Only sees solar generation and predictions with detailed weather.
    - Factory: Submits 5-day load requirements + purpose of demand.
    - EB (Admin): Master control room calculating grid deficit, with Tomorrow/5-Day views.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import timedelta

# Set page config FIRST, before any other Streamlit calls
st.set_page_config(
    page_title="Smart Grid Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.forecast import predict_next_n_days, CONFIDENCE_LEVELS
from src.grid_deficit import calculate_extra_power
from src.weather_api import get_5day_weather, predict_solar_from_api
from supabase import create_client, Client

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ============================================================================
#  SUPABASE CLOUD DATABASE
# ============================================================================
SUPABASE_URL = "https://mhuzfghtszonzniasnrv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1odXpmZ2h0c3pvbnpuaWFzbnJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODQyMTk4OTUsImV4cCI6MjA5OTc5NTg5NX0.HT0aBMa1GF0EYrVPyqiXjFc7XQpAVT7yT3YVwyJ21ww"
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_factory_requirements():
    """Load the factory's 5-day submitted load and purpose from Supabase."""
    default_reqs = {
        f"day{i}": {"load": 1500.0, "purpose": "Normal operations"} for i in range(1, 6)
    }
    try:
        # Fetch id=1 from the factory_loads table
        response = supabase_client.table("factory_loads").select("*").eq("id", 1).execute()
        if response.data and len(response.data) > 0:
            cloud_data = response.data[0].get("payload", {})
            # Validate format
            if cloud_data and "day1" in cloud_data and isinstance(cloud_data["day1"], dict):
                return cloud_data
    except Exception as e:
        print(f"Supabase Read Error: {e}")
        
    return default_reqs

def save_factory_requirements(data):
    """Save the factory's submitted load and purpose to Supabase."""
    try:
        # Upsert payload into id=1
        supabase_client.table("factory_loads").upsert({"id": 1, "payload": data}).execute()
    except Exception as e:
        print(f"Supabase Write Error: {e}")
        st.error(f"Cloud Save Failed: {e}")


# ============================================================================
#  CSS INJECTION (Premium Glassmorphism Theme)
# ============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* Base Dark Theme Overrides */
        .stApp { background-color: #0f172a; color: #f1f5f9; }
        h1, h2, h3, h4, h5 { color: #f8fafc !important; font-weight: 600; }
        p, span, div, label { color: #f1f5f9; } /* Force text visibility */
        .stMarkdown p { color: #e2e8f0 !important; }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid rgba(51,65,85,0.5); }
        [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
        
        /* Form inputs visibility */
        div[data-baseweb="input"] { background-color: #334155; }
        div[data-baseweb="input"] input { color: #ffffff !important; }
        
        /* Metric Cards */
        [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700 !important; font-size: 2.2rem !important; }
        [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 1rem !important; font-weight: 500 !important; }
        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, rgba(30,41,59,0.7), rgba(15,23,42,0.9));
            border: 1px solid rgba(51,65,85,0.6);
            border-radius: 16px; padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(56,189,248,0.15);
            border-color: rgba(56,189,248,0.4);
        }

        /* Headers & Boxes */
        .control-room-header {
            background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
            padding: 24px 32px; border-radius: 16px; margin-bottom: 32px;
            box-shadow: 0 10px 25px rgba(59,130,246,0.2);
        }
        .control-room-header h1 { color: white !important; margin: 0 0 8px 0; font-size: 2.2rem; letter-spacing: -0.5px; }
        .control-room-header p { color: #bfdbfe; margin: 0; font-size: 1.1rem; font-weight: 400; }

        /* AI Recommendation Box */
        .ai-box {
            background: rgba(15,23,42,0.8); border-left: 5px solid #8b5cf6;
            border-radius: 8px; padding: 20px 24px; margin-bottom: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .ai-box h3 { color: #c4b5fd !important; margin: 0 0 8px 0; font-size: 1.2rem; }
        
        /* Sub-cards for weather/solar stats */
        .stat-card {
            background: rgba(30,41,59,0.5); border: 1px solid rgba(51,65,85,0.5);
            border-radius: 12px; padding: 16px; text-align: center; margin-bottom: 16px;
        }
        .stat-label { color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px;}
        .stat-value { color: #f1f5f9; font-size: 1.5rem; font-weight: 600;}

        /* Hide Streamlit branding but keep the sidebar toggle visible */
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;} 
        header {background-color: transparent !important;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
#  DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    deficit_path = os.path.join(DATA_DIR, "deficit_results.csv")
    daily_path = os.path.join(DATA_DIR, "daily_features.csv")

    if os.path.exists(deficit_path):
        deficit_df = pd.read_csv(deficit_path)
        deficit_df["date"] = pd.to_datetime(deficit_df["date"])
    else:
        st.warning("Run `python run_pipeline.py` first to generate historical predictions!")
        deficit_df = pd.DataFrame()

    if os.path.exists(daily_path):
        daily_df = pd.read_csv(daily_path)
        daily_df["date"] = pd.to_datetime(daily_df["date"])
    else:
        daily_df = deficit_df.copy()

    return daily_df, deficit_df


@st.cache_data(ttl=300)
def load_forecast():
    try:
        weather_df = get_5day_weather()
        predictions_df = predict_solar_from_api(weather_df)
        
        results = []
        for i, row in predictions_df.iterrows():
            step = i + 1
            results.append({
                "day_offset": step,
                "date": pd.to_datetime(row["Date"]),
                "predicted_temp": row["Temperature"],
                "predicted_humidity": row["Humidity"],
                "predicted_cloud_cover": row["Cloud Cover"],
                "predicted_irradiance": row["Solar Radiation"],
                "predicted_wind_speed": row["Wind Speed"],
                "predicted_solar_kwh": row["Predicted Solar Generation"],
            })
        return results
    except Exception as e:
        print(f"API Forecast failed: {e}. Falling back to synthetic.")
        return predict_next_n_days(n=5)


# ============================================================================
#  REUSABLE COMPONENTS
# ============================================================================
def show_sidebar(role_display_name: str):
    with st.sidebar:
        st.markdown(f"### Currently logged in as")
        st.markdown(f"## {role_display_name}")
        st.markdown("---")

        if st.button("Logout", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================================
#  VIEW 1: SOLAR OWNER
# ============================================================================
def show_solar_owner_view(daily_df, deficit_df, forecasts):
    show_sidebar("☀️ Solar Owner")

    st.markdown("""
    <div class="control-room-header">
        <h1>☀️ Solar Farm Dashboard</h1>
        <p>Real-time generation monitoring & Weather-Driven Forecast</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Latest historical day metric ---
    latest = deficit_df.iloc[-1] if not deficit_df.empty else None
    if latest is not None:
        solar_gen = latest.get("predicted_solar_kwh", latest.get("daily_solar_kwh", 0))
        st.metric("☀️ Today's Current Solar Generation", f"{solar_gen:,.1f} kWh")

    st.markdown("---")
    st.markdown("### 📅 Day-by-Day Forecast Details")
    
    # NEW FEATURE: Day Selector for Weather context
    day_options = {f"Day +{f['day_offset']} ({f['date'].strftime('%b %d')})": f for f in forecasts}
    selected_day_label = st.selectbox("Select Forecast Day to view weather report:", list(day_options.keys()))
    selected_forecast = day_options[selected_day_label]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="border-color:#f59e0b; background:rgba(245,158,11,0.1);">
            <div class="stat-label">Predicted Solar Generation</div>
            <div class="stat-value" style="color:#f59e0b; font-size:2rem;">{selected_forecast['predicted_solar_kwh']:,.1f} kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        with wc1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Temp</div><div class="stat-value">{selected_forecast["predicted_temp"]}°C</div></div>', unsafe_allow_html=True)
        with wc2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Cloud</div><div class="stat-value">{selected_forecast["predicted_cloud_cover"]}%</div></div>', unsafe_allow_html=True)
        with wc3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Humidity</div><div class="stat-value">{selected_forecast["predicted_humidity"]}%</div></div>', unsafe_allow_html=True)
        with wc4:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Irradiance</div><div class="stat-value">{selected_forecast["predicted_irradiance"]}</div></div>', unsafe_allow_html=True)
        with wc5:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Wind</div><div class="stat-value">{selected_forecast["predicted_wind_speed"]}m/s</div></div>', unsafe_allow_html=True)

    st.markdown("#### 5-Day Trend Overview")
    chart_data = pd.DataFrame({
        "Day": [f"+{f['day_offset']} ({f['date'].strftime('%b %d')})" for f in forecasts],
        "Predicted Solar (kWh)": [f["predicted_solar_kwh"] for f in forecasts]
    }).set_index("Day")
    st.bar_chart(chart_data, color="#f59e0b")


# ============================================================================
#  VIEW 2: FACTORY
# ============================================================================
def show_factory_view(daily_df, deficit_df, forecasts):
    show_sidebar("🏭 Factory")

    st.markdown("""
    <div class="control-room-header" style="background: linear-gradient(90deg, #b91c1c 0%, #ef4444 100%);">
        <h1>🏭 Factory Load Planner</h1>
        <p>Submit your power requirements & purpose directly to the EB</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 5-Day Power Requirement Submission")
    st.markdown("Please input your anticipated factory power demand (in kWh) and the primary purpose of this demand for the next 5 days. This will be securely transmitted to the EB (TNPDCL) for grid planning.")

    reqs = load_factory_requirements()

    with st.form("load_submission_form"):
        new_reqs = {}
        for i, f in enumerate(forecasts):
            day_key = f"day{i+1}"
            date_str = f['date'].strftime('%b %d')
            old_data = reqs.get(day_key, {"load": 1500.0, "purpose": ""})
            
            st.markdown(f"**Day +{i+1} ({date_str})**")
            c1, c2 = st.columns([1, 2])
            with c1:
                load_val = st.number_input(f"Demand (kWh)", min_value=0.0, value=float(old_data["load"]), step=10.0, key=f"load_{i}")
            with c2:
                purpose_val = st.text_input(f"Purpose of Demand", value=old_data["purpose"], key=f"purp_{i}", placeholder="e.g. Heavy machining, Weekend maintenance")
            
            new_reqs[day_key] = {"load": load_val, "purpose": purpose_val}
            st.markdown("---")
            
        submitted = st.form_submit_button("Transmit Requirements to EB", type="primary")
        if submitted:
            save_factory_requirements(new_reqs)
            st.success("✅ Power requirements and purposes successfully transmitted to the EB Control Room!")


# ============================================================================
#  VIEW 3: EB / TNPDCL ADMIN
# ============================================================================
def show_admin_view(daily_df, deficit_df, forecasts):
    show_sidebar("🏛️ EB (TNPDCL)")

    st.markdown("""
    <div class="control-room-header" style="background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);">
        <h1>🏛️ EB Master Control Room</h1>
        <p>Predictive Grid Deficit Analysis & Allocation</p>
    </div>
    """, unsafe_allow_html=True)

    reqs = load_factory_requirements()
    
    # NEW FEATURE: Timeframe Toggle
    view_mode = st.radio("Select View Timeframe:", ["Tomorrow (1 Day)", "5-Day Outlook"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if view_mode == "Tomorrow (1 Day)":
        f1 = forecasts[0]
        r1 = reqs["day1"]
        load1 = r1["load"]
        purp1 = r1["purpose"]
        solar1 = f1["predicted_solar_kwh"]
        deficit1 = calculate_extra_power(load1, solar1)
        
        st.markdown(f"### Grid Plan for Tomorrow ({f1['date'].strftime('%b %d')})")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("☀️ Predicted Solar Supply", f"{solar1:,.0f} kWh")
        with c2:
            st.metric("🏭 Factory Demand", f"{load1:,.0f} kWh")
        with c3:
            st.metric("⚡ Grid Deficit to Supply", f"{deficit1:,.0f} kWh", delta="Requires TNPDCL Action", delta_color="inverse")
            
        st.markdown(f"**🏭 Stated Purpose for Demand:** `{purp1}`")
        
        st.markdown("#### ☁️ Tomorrow's Weather Impact")
        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        with wc1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Temp</div><div class="stat-value">{f1["predicted_temp"]}°C</div></div>', unsafe_allow_html=True)
        with wc2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Cloud</div><div class="stat-value">{f1["predicted_cloud_cover"]}%</div></div>', unsafe_allow_html=True)
        with wc3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Humidity</div><div class="stat-value">{f1["predicted_humidity"]}%</div></div>', unsafe_allow_html=True)
        with wc4:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Irradiance</div><div class="stat-value">{f1["predicted_irradiance"]}</div></div>', unsafe_allow_html=True)
        with wc5:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Wind</div><div class="stat-value">{f1["predicted_wind_speed"]}m/s</div></div>', unsafe_allow_html=True)

        if solar1 >= load1:
            st.success(f"✅ The factory will be fully powered by solar tomorrow. Zero grid intervention required.")
        else:
            st.error(f"⚠️ TNPDCL must schedule an extra {deficit1:,.0f} kWh of grid power to prevent blackouts tomorrow due to factory operations ({purp1}).")

    else:
        # 5-Day Outlook
        solar_preds = [f["predicted_solar_kwh"] for f in forecasts]
        factory_loads = [reqs[f"day{i+1}"]["load"] for i in range(len(forecasts))]
        deficits = [calculate_extra_power(load, solar) for load, solar in zip(factory_loads, solar_preds)]
        
        total_5day_solar = sum(solar_preds)
        total_5day_load = sum(factory_loads)
        total_5day_deficit = sum(deficits)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("☀️ 5-Day Solar Supply", f"{total_5day_solar:,.0f} kWh")
        with col2:
            st.metric("🏭 5-Day Factory Demand", f"{total_5day_load:,.0f} kWh")
        with col3:
            st.metric("⚡ 5-Day Grid Deficit", f"{total_5day_deficit:,.0f} kWh", delta="Required from TNPDCL", delta_color="inverse")

        st.markdown("---")

        if total_5day_solar >= total_5day_load:
            st.success("✅ Predicted solar generation completely covers the factory's submitted load requirements for the next 5 days. Zero extra coal/hydro power needs to be allocated.")
        else:
            st.error(f"⚠️ ALERT: TNPDCL must schedule and allocate an extra {total_5day_deficit:,.0f} kWh of conventional grid power to prevent factory blackouts over this 5-day period.")

        st.markdown("### 📊 5-Day Grid Allocation Plan")
        chart_data = pd.DataFrame({
            "Day": [f"+{f['day_offset']} ({f['date'].strftime('%b %d')})" for f in forecasts],
            "Solar Supply": solar_preds,
            "Factory Demand": factory_loads,
            "Grid Deficit (EB)": deficits
        }).set_index("Day")
        st.bar_chart(chart_data, color=["#f59e0b", "#64748b", "#ef4444"])
        
        st.markdown("### 📋 Factory Stated Purposes")
        purpose_df = pd.DataFrame({
            "Day": [f"+{f['day_offset']} ({f['date'].strftime('%b %d')})" for f in forecasts],
            "Requested Load (kWh)": factory_loads,
            "Stated Purpose": [reqs[f"day{i+1}"]["purpose"] for i in range(len(forecasts))]
        })
        st.dataframe(purpose_df, use_container_width=True, hide_index=True)


# ============================================================================
#  LOGIN PAGE
# ============================================================================
def show_landing_page():
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1 style="font-size: 3.5rem; color: #f1f5f9;">Smart Grid Control Center</h1>
        <p style="color: #94a3b8; font-size: 1.2rem; max-width: 600px; margin: 0 auto 40px auto;">
            Choose your portal to access the system.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:rgba(30,41,59,0.5); padding:30px; border-radius:16px; text-align:center; border:1px solid rgba(245,158,11,0.3);">
            <div style="font-size: 3rem; margin-bottom: 16px;">☀️</div>
            <h3 style="color:#f1f5f9; margin-bottom: 8px;">Solar Farm</h3>
            <p style="color:#94a3b8; font-size:0.9rem;">Submit predicted power generation.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Solar Portal", key="btn_solar", use_container_width=True):
            st.session_state["role"] = "solar_owner"
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background:rgba(30,41,59,0.5); padding:30px; border-radius:16px; text-align:center; border:1px solid rgba(239,68,68,0.3);">
            <div style="font-size: 3rem; margin-bottom: 16px;">🏭</div>
            <h3 style="color:#f1f5f9; margin-bottom: 8px;">Factory</h3>
            <p style="color:#94a3b8; font-size:0.9rem;">Submit 5-day load requirements.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Factory Portal", key="btn_factory", use_container_width=True):
            st.session_state["role"] = "factory"
            st.rerun()

    with col3:
        st.markdown("""
        <div style="background:rgba(30,41,59,0.5); padding:30px; border-radius:16px; text-align:center; border:1px solid rgba(59,130,246,0.3);">
            <div style="font-size: 3rem; margin-bottom: 16px;">🏛️</div>
            <h3 style="color:#f1f5f9; margin-bottom: 8px;">EB Control Room</h3>
            <p style="color:#94a3b8; font-size:0.9rem;">Calculate Grid Deficit & Allocate.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter EB Portal", key="btn_admin", use_container_width=True):
            st.session_state["role"] = "admin"
            st.rerun()


# ============================================================================
#  MAIN APP
# ============================================================================
def main():
    inject_custom_css()

    daily_df, deficit_df = load_data()
    forecasts = load_forecast()

    if "role" not in st.session_state:
        show_landing_page()
    elif st.session_state["role"] == "solar_owner":
        show_solar_owner_view(daily_df, deficit_df, forecasts)
    elif st.session_state["role"] == "factory":
        show_factory_view(daily_df, deficit_df, forecasts)
    elif st.session_state["role"] == "admin":
        show_admin_view(daily_df, deficit_df, forecasts)
    else:
        del st.session_state["role"]
        st.rerun()


if __name__ == "__main__":
    main()
