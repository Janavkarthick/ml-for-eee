"""
forecast.py - 6-Day Recursive Multi-Day Forecast Engine
==========================================================
PURPOSE:
    Predict the next N days (default 6) of weather AND solar power
    using a RECURSIVE (autoregressive) approach:

    Day+1: Use actual recent history -> predict weather -> predict solar
    Day+2: Use Day+1's PREDICTIONS as new history -> predict weather -> predict solar
    Day+3: Use Day+2's PREDICTIONS as new history -> predict weather -> predict solar
    ... and so on up to Day+6

    *** IMPORTANT: COMPOUNDING ERROR ***
    Each forecast step depends on the PREVIOUS step's PREDICTION (not actual data).
    This means errors COMPOUND (accumulate) with each step forward:
    - Day+1 is the most accurate (based on real data)
    - Day+2 has error from Day+1's prediction + its own error
    - Day+3 has errors from Day+1 + Day+2 + its own
    - ...
    - Day+6 has the MOST accumulated error (least reliable)

    This is a fundamental limitation of recursive/autoregressive forecasting.
    In production, ensemble methods or direct multi-step models can help reduce this.

HOW IT WORKS:
    We maintain a "sliding buffer" of recent values for each variable
    (temperature, humidity, cloud cover, wind speed, irradiance, solar kWh).
    At each step, we:
    1. Construct feature vectors from the buffer (lag1/2/3 + rolling 7-day mean)
    2. Feed features into Model 1 (weather) and Model 2 (solar)
    3. Append predicted values to the buffer
    4. Repeat for the next day

VIVA TIP:
    "Recursive forecasting is like a game of telephone - each person
    (day) passes a slightly distorted version of the message (prediction)
    to the next. By Day+6, the message may have drifted significantly
    from reality. That's why we show confidence levels that decrease
    with each forecast day."
"""

# --- Imports ----------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)


# --- Confidence levels for each forecast day --------------------------------
# These are displayed in the dashboard to warn users about compounding error.
# Day+1 is based on real data; each subsequent day loses accuracy.
CONFIDENCE_LEVELS = {
    1: {"label": "High confidence",      "color": "#22c55e", "icon": "🟢", "pct": "~90%"},
    2: {"label": "Good confidence",      "color": "#84cc16", "icon": "🟢", "pct": "~80%"},
    3: {"label": "Moderate confidence",  "color": "#eab308", "icon": "🟡", "pct": "~70%"},
    4: {"label": "Fair confidence",      "color": "#f97316", "icon": "🟠", "pct": "~55%"},
    5: {"label": "Low confidence",       "color": "#ef4444", "icon": "🔴", "pct": "~40%"},
    6: {"label": "Lowest confidence",    "color": "#dc2626", "icon": "🔴", "pct": "~30%"},
}


def _build_weather_features(buffers):
    """
    Construct the 20-feature vector for Model 1 (weather predictor)
    from the current sliding buffers.

    Feature order (MUST match training order):
        For each of [amb_temp, humidity, cloud_cover, wind_speed, irradiance]:
            lag1 (yesterday), lag2 (2d ago), lag3 (3d ago), roll7 (7-day mean)

    Parameters
    ----------
    buffers : dict
        Dictionary with keys 'amb_temp', 'humidity', 'cloud_cover',
        'wind_speed', 'irradiance' — each a list of recent values.

    Returns
    -------
    np.ndarray of shape (1, 20) — one row of 20 features
    """
    features = []
    # Order must match: amb_temp, humidity, cloud_cover, wind_speed, irradiance
    for var in ["amb_temp", "humidity", "cloud_cover", "wind_speed", "irradiance"]:
        buf = buffers[var]
        lag1 = buf[-1]                      # Yesterday's value
        lag2 = buf[-2]                      # 2 days ago
        lag3 = buf[-3]                      # 3 days ago
        roll7 = np.mean(buf[-7:])           # 7-day rolling average
        features.extend([lag1, lag2, lag3, roll7])

    return np.array(features).reshape(1, -1)


def _build_solar_features(predicted_weather, buffers):
    """
    Construct the 9-feature vector for Model 2 (solar predictor)
    from predicted weather + sliding buffers.

    Feature order (MUST match training order):
        [predicted_temp, predicted_humidity, predicted_cloud_cover,
         solar_lag1, solar_lag2, solar_lag3, solar_roll7,
         irradiance_lag1, wind_speed_lag1]

    Parameters
    ----------
    predicted_weather : array of 3 values
        [predicted_temp, predicted_humidity, predicted_cloud_cover]
    buffers : dict
        Contains 'solar_kwh', 'irradiance', 'wind_speed' lists.

    Returns
    -------
    np.ndarray of shape (1, 9) — one row of 9 features
    """
    pred_temp, pred_humidity, pred_cloud = predicted_weather

    solar_buf = buffers["solar_kwh"]
    solar_lag1 = solar_buf[-1]
    solar_lag2 = solar_buf[-2]
    solar_lag3 = solar_buf[-3]
    solar_roll7 = np.mean(solar_buf[-7:])

    irr_lag1 = buffers["irradiance"][-1]
    wind_lag1 = buffers["wind_speed"][-1]

    features = [
        pred_temp, pred_humidity, pred_cloud,
        solar_lag1, solar_lag2, solar_lag3, solar_roll7,
        irr_lag1, wind_lag1
    ]
    return np.array(features).reshape(1, -1)


def _estimate_irradiance(cloud_cover, max_irradiance):
    """
    Estimate irradiance from predicted cloud cover.

    Since our weather model predicts cloud cover but NOT irradiance,
    we use the inverse of the formula used to generate synthetic cloud cover:
        cloud_cover ~ 100 - (irradiance / max_irr * 100)
    =>  irradiance  ~ max_irr * (100 - cloud_cover) / 100

    This is approximate but physically reasonable:
    clear sky (cloud=0%) -> max irradiance, overcast (cloud=100%) -> 0 irradiance.

    VIVA TIP: "This is called an inverse model — we reverse the formula
    that was used to create the synthetic data."
    """
    # Clip cloud cover to valid range
    cloud_cover = np.clip(cloud_cover, 0, 100)
    estimated = max_irradiance * (100 - cloud_cover) / 100
    return max(estimated, 0.0)


def predict_next_n_days(n=6, daily_df=None, weather_model=None,
                         solar_model=None):
    """
    Predict the next N days of weather and solar power RECURSIVELY.

    *** COMPOUNDING ERROR WARNING ***
    Each day's prediction depends on the PREVIOUS day's PREDICTED values
    (not actual measurements). This means:
    - Day+1: Based on actual recent data -> most accurate
    - Day+2: Based on Day+1's prediction -> less accurate
    - Day+3: Based on Day+2's prediction -> even less accurate
    - ...
    - Day+N: Most error accumulated -> least accurate

    This is an inherent limitation of recursive/autoregressive forecasting.
    The confidence levels in CONFIDENCE_LEVELS reflect this degradation.

    Parameters
    ----------
    n : int
        Number of days to forecast (default: 6)
    daily_df : pd.DataFrame or None
        The daily features dataset. If None, loaded from data/daily_features.csv.
    weather_model : model or None
        Trained weather model. If None, loaded from models/weather_model.pkl.
    solar_model : model or None
        Trained solar model. If None, loaded from models/solar_model.pkl.

    Returns
    -------
    list of dicts, each with keys:
        - day_offset: int (1 to n)
        - date: pd.Timestamp (the predicted date)
        - predicted_temp: float (degrees C)
        - predicted_humidity: float (%)
        - predicted_cloud_cover: float (%)
        - predicted_irradiance: float (W/m2, estimated)
        - predicted_wind_speed: float (m/s, persisted from recent avg)
        - predicted_solar_kwh: float (daily kWh)
        - confidence: dict with label, color, icon, pct
        - synthetic_load_kwh: float (the synthetic load for reference)
    """
    # --- Load data and models if not provided --------------------------------
    if daily_df is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "daily_features.csv")
        daily_df = pd.read_csv(data_path)
        daily_df["date"] = pd.to_datetime(daily_df["date"])

    if weather_model is None:
        weather_path = os.path.join(PROJECT_ROOT, "models", "weather_model.pkl")
        weather_model = joblib.load(weather_path)

    if solar_model is None:
        solar_path = os.path.join(PROJECT_ROOT, "models", "solar_model.pkl")
        solar_model = joblib.load(solar_path)

    # --- Initialize sliding buffers from the last 10 rows of actual data -----
    # We need at least 7 values for the rolling mean, so we take the last 10
    # as a safety margin.
    tail = daily_df.tail(10)

    buffers = {
        "amb_temp":    tail["avg_amb_temp"].tolist(),
        "humidity":    tail["synth_humidity"].tolist(),
        "cloud_cover": tail["synth_cloud_cover"].tolist(),
        "wind_speed":  tail["avg_wind_speed"].tolist(),
        "irradiance":  tail["avg_irradiance"].tolist(),
        "solar_kwh":   tail["daily_solar_kwh"].tolist(),
    }

    # Max irradiance from the full dataset (for estimating irradiance from cloud cover)
    max_irradiance = daily_df["avg_irradiance"].max()

    # Last known date — we'll forecast forward from here
    last_date = pd.to_datetime(daily_df["date"].iloc[-1])

    # Reference synthetic load values (last N from the dataset, cycled if needed)
    load_values = daily_df["load_kwh"].tail(30).tolist()

    # --- Recursive Forecast Loop ---------------------------------------------
    # *** COMPOUNDING ERROR: each iteration uses the PREVIOUS iteration's
    # *** predictions as input, causing errors to accumulate over time.
    results = []

    for step in range(1, n + 1):
        # 1. Build weather feature vector from current buffers
        weather_features = _build_weather_features(buffers)

        # 2. Predict tomorrow's weather using Model 1
        #    Returns: [predicted_temp, predicted_humidity, predicted_cloud_cover]
        weather_pred = weather_model.predict(weather_features)[0]
        pred_temp = float(weather_pred[0])
        pred_humidity = float(np.clip(weather_pred[1], 10, 100))
        pred_cloud = float(np.clip(weather_pred[2], 0, 100))

        # 3. Estimate irradiance from predicted cloud cover
        #    (weather model doesn't predict irradiance directly)
        pred_irradiance = _estimate_irradiance(pred_cloud, max_irradiance)

        # 4. Persist wind speed as rolling average of recent values
        #    (weather model doesn't predict wind speed either)
        pred_wind = float(np.mean(buffers["wind_speed"][-7:]))

        # 5. Build solar feature vector using predicted weather + buffers
        solar_features = _build_solar_features(weather_pred, buffers)

        # 6. Predict tomorrow's solar generation using Model 2
        pred_solar = float(solar_model.predict(solar_features)[0])
        pred_solar = max(pred_solar, 0.0)  # Solar can't be negative

        # 7. *** COMPOUNDING ERROR STEP ***
        #    Append PREDICTED values to buffers. From now on, the next iteration
        #    will use these PREDICTED values as if they were real measurements.
        #    This is where error compounds — each prediction builds on the
        #    previous prediction's error.
        buffers["amb_temp"].append(pred_temp)
        buffers["humidity"].append(pred_humidity)
        buffers["cloud_cover"].append(pred_cloud)
        buffers["wind_speed"].append(pred_wind)
        buffers["irradiance"].append(pred_irradiance)
        buffers["solar_kwh"].append(pred_solar)

        # 8. Record the result for this day
        forecast_date = last_date + timedelta(days=step)

        # Use a cycled synthetic load for reference
        ref_load = load_values[step % len(load_values)]

        results.append({
            "day_offset": step,
            "date": forecast_date,
            "predicted_temp": round(pred_temp, 1),
            "predicted_humidity": round(pred_humidity, 1),
            "predicted_cloud_cover": round(pred_cloud, 1),
            "predicted_irradiance": round(pred_irradiance, 1),
            "predicted_wind_speed": round(pred_wind, 1),
            "predicted_solar_kwh": round(pred_solar, 1),
            "confidence": CONFIDENCE_LEVELS.get(step, CONFIDENCE_LEVELS[6]),
            "synthetic_load_kwh": round(ref_load, 1),
        })

    return results


# --- Run as standalone script ------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  6-DAY RECURSIVE FORECAST")
    print("=" * 60)

    forecasts = predict_next_n_days(n=6)

    print(f"\n  {'Day':>6} {'Date':>12} {'Temp':>8} {'Humid':>8} "
          f"{'Cloud':>8} {'Solar':>10} {'Confidence':>20}")
    print("  " + "-" * 78)

    for f in forecasts:
        conf = f["confidence"]
        print(f"  Day+{f['day_offset']:<2} {f['date'].strftime('%Y-%m-%d'):>12} "
              f"{f['predicted_temp']:>7.1f}C {f['predicted_humidity']:>7.1f}% "
              f"{f['predicted_cloud_cover']:>7.1f}% {f['predicted_solar_kwh']:>9.1f} "
              f"{conf['icon']} {conf['label']}")

    print("\n  *** Note: Confidence decreases with each day due to")
    print("  *** compounding forecast error (recursive prediction).")
