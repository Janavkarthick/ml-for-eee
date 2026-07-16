"""
feature_engineering.py — Step 2: Create daily features for ML models
=====================================================================
PURPOSE:
    - Aggregate the 5-minute cleaned data into DAILY summaries
    - Create SYNTHETIC columns (humidity, cloud cover) since dataset lacks them
    - Create LAG features (yesterday, 2 days ago, 3 days ago)
    - Create ROLLING MEAN features (7-day averages)
    - Generate a SYNTHETIC LOAD column (factory electricity demand)
    - Save the feature-engineered daily dataset

WHY DAILY AGGREGATION?
    Our models predict "tomorrow's weather" and "tomorrow's solar power."
    We need one row per day, not one row per 5-minute reading. So we
    collapse 144 readings/day into daily averages and totals.

VIVA TIP:
    "Feature engineering is the most important step in ML. Good features
    make even simple models perform well. We created lag features so the
    model can see patterns like 'if yesterday was sunny, today probably is too.'"
"""

# ─── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 5-minute readings into one row per day.

    Aggregations:
    - Temperature, wind, irradiance → daily MEAN
    - AC Power → daily SUM converted to kWh
      Formula: sum(Watts) × (5 min / 60 min) / 1000 = kWh

    VIVA TIP: "We multiply watts by 5/60 to convert the 5-minute power
    reading into energy (watt-hours), then divide by 1000 for kilowatt-hours."
    """
    print("[FEAT] Aggregating 5-minute data into daily summaries...")

    # Extract the date from the timestamp column
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    # Define how to aggregate each column
    daily = df.groupby("date").agg(
        avg_module_temp=("MODULE_TEMP", "mean"),       # Average panel temperature
        avg_amb_temp=("Amb_Temp", "mean"),              # Average ambient temperature
        avg_wind_speed=("WIND_Speed", "mean"),          # Average wind speed
        avg_irradiance=("IRR (W/m2)", "mean"),          # Average solar irradiance
        total_ac_power_w=("AC Power in Watts", "sum"),  # Total AC power in Watts
        readings_count=("AC Power in Watts", "count"),  # How many readings that day
    ).reset_index()

    # Convert total watts to kWh
    # Each reading spans 5 minutes = 5/60 hours = 1/12 hour
    # Energy (Wh) = Power (W) × Time (hours)
    # kWh = Wh / 1000
    daily["daily_solar_kwh"] = daily["total_ac_power_w"] * (5 / 60) / 1000

    # Convert date back to datetime for easier manipulation
    daily["date"] = pd.to_datetime(daily["date"])

    print(f"[FEAT] Created {len(daily)} daily rows from {len(df)} raw readings")
    return daily


def add_synthetic_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add SYNTHETIC humidity and cloud cover columns.

    WHY SYNTHETIC?
    - Our dataset only has temp, wind, and irradiance — no humidity or cloud data.
    - Model 1 needs to predict weather including humidity and cloud cover.
    - We create realistic-looking synthetic values based on physical relationships.

    FORMULAS:
    - Humidity ≈ 80 − 0.5 × temperature + random noise
      (Hot days tend to have lower relative humidity)
    - Cloud cover ≈ 100 − (irradiance / max_irradiance × 100) + noise
      (High irradiance = clear sky = low cloud cover)

    VIVA TIP: "These are synthetic — in production you'd use real weather API data.
    But the formulas are physics-inspired: clouds block sunlight, heat reduces
    relative humidity."
    """
    print("[FEAT] Adding synthetic humidity and cloud cover columns...")

    np.random.seed(42)  # For reproducibility (same random numbers every run)

    # Synthetic humidity: inversely related to temperature
    # Base: 80%, decreases by 0.5% per degree C, plus random noise ±5%
    daily["synth_humidity"] = (
        80
        - 0.5 * daily["avg_amb_temp"]
        + np.random.normal(0, 5, len(daily))  # Gaussian noise, std=5
    ).clip(10, 100)  # Keep between 10% and 100%

    # Synthetic cloud cover: inversely related to irradiance
    # If irradiance is at its max, cloud cover ≈ 0 (clear sky)
    max_irr = daily["avg_irradiance"].max()
    daily["synth_cloud_cover"] = (
        100
        - (daily["avg_irradiance"] / max_irr * 100)
        + np.random.normal(0, 8, len(daily))  # Gaussian noise, std=8
    ).clip(0, 100)  # Keep between 0% and 100%

    print("[FEAT] Synthetic weather columns added ✓")
    return daily


def add_synthetic_load(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a synthetic daily factory load (electricity demand) column.

    WHY SYNTHETIC?
    - The user doesn't have real EB (Electricity Board) data.
    - We simulate realistic industrial demand for the deficit calculation.

    FORMULA:
    - Base load = 1600 kWh per day (comparable to avg solar ~1700 kWh)
    - Seasonal variation: +/-300 kWh (higher in summer = more AC usage)
    - Day-of-week effect: weekdays are higher than weekends
    - Random noise: +/-15%

    VIVA TIP: "A real factory uses more power on weekdays (machines running)
    and in summer (air conditioning). We model both patterns."
    """
    print("[FEAT] Generating synthetic factory load column...")

    np.random.seed(123)  # Different seed for load (independent randomness)
    n = len(daily)

    BASE_LOAD_KW = 1600  # Average daily factory demand in kWh

    # Seasonal component: sinusoidal, peaks in summer (around day 180)
    day_of_year = daily["date"].dt.dayofyear
    seasonal = 300 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak ~June

    # Weekday effect: weekdays get +5%, weekends get -15%
    is_weekday = daily["date"].dt.dayofweek < 5  # Mon=0 ... Fri=4 are weekdays
    weekday_factor = np.where(is_weekday, 1.05, 0.85)

    # Random noise: +/-15% (uniform distribution)
    noise = np.random.uniform(0.85, 1.15, n)

    # Combine all components
    daily["load_kwh"] = (BASE_LOAD_KW + seasonal) * weekday_factor * noise


    # Make sure load is always positive
    daily["load_kwh"] = daily["load_kwh"].clip(lower=100)

    print(f"[FEAT] Synthetic load: mean={daily['load_kwh'].mean():.1f} kWh, "
          f"min={daily['load_kwh'].min():.1f}, max={daily['load_kwh'].max():.1f}")
    return daily


def add_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Create LAG and ROLLING features for time-series prediction.

    LAG FEATURES:
    - "What was yesterday's value?" (lag_1)
    - "What was 2 days ago?" (lag_2)
    - "What was 3 days ago?" (lag_3)

    ROLLING FEATURES:
    - "What was the average over the past 7 days?" (rolling_7d)

    WHY LAGS?
    - Weather and solar generation have temporal patterns.
    - Yesterday's weather is the best predictor of today's weather.
    - A 7-day rolling average smooths out day-to-day noise.

    VIVA TIP: "Lag features let the model see 'memory' of past days.
    shift(1) moves the column down by 1 row, so each row can see yesterday's value."
    """
    print("[FEAT] Creating lag and rolling features...")

    # Columns to create lags for
    weather_cols = ["avg_amb_temp", "synth_humidity", "synth_cloud_cover",
                    "avg_wind_speed", "avg_irradiance"]
    solar_cols = ["daily_solar_kwh"]
    all_lag_cols = weather_cols + solar_cols

    for col in all_lag_cols:
        # Lag features: shift the column by N days
        # shift(1) means "use yesterday's value for today's row"
        daily[f"{col}_lag1"] = daily[col].shift(1)  # Yesterday
        daily[f"{col}_lag2"] = daily[col].shift(2)  # 2 days ago
        daily[f"{col}_lag3"] = daily[col].shift(3)  # 3 days ago

        # Rolling 7-day mean: average of the past 7 days (including today)
        # min_periods=1 means "calculate even if <7 days available"
        daily[f"{col}_roll7"] = daily[col].rolling(window=7, min_periods=1).mean()

    # Drop rows where lag features are NaN (first 3 days have no history)
    n_before = len(daily)
    daily = daily.dropna().reset_index(drop=True)
    n_dropped = n_before - len(daily)
    print(f"[FEAT] Created 4 lag/rolling features per column. Dropped {n_dropped} "
          f"initial rows (no lag history).")
    return daily


def engineer_features(cleaned_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Master feature engineering function — runs all steps in order.

    Steps:
    1. Load cleaned data
    2. Aggregate to daily
    3. Add synthetic weather (humidity, cloud cover)
    4. Add synthetic load (factory demand)
    5. Add lag and rolling features
    6. Save to CSV

    Parameters
    ----------
    cleaned_csv : str
        Path to cleaned CSV from Step 1
    output_csv : str
        Where to save the daily feature-engineered CSV
    """
    print("=" * 60)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    # Load the cleaned 5-minute data
    df = pd.read_csv(cleaned_csv)

    # Run each feature engineering step
    daily = aggregate_daily(df)
    daily = add_synthetic_weather(daily)
    daily = add_synthetic_load(daily)
    daily = add_lag_features(daily)

    # Save the feature-engineered daily data
    daily.to_csv(output_csv, index=False)
    print(f"[FEAT] Saved daily features to: {output_csv}")
    print(f"[FEAT] Final shape: {daily.shape[0]} rows × {daily.shape[1]} columns")
    print(f"[FEAT] Columns: {list(daily.columns)}")
    print()

    return daily


# ─── Run as standalone script ─────────────────────────────────────────────────
if __name__ == "__main__":
    engineer_features("data/cleaned_data.csv", "data/daily_features.csv")
