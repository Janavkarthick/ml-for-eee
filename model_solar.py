"""
model_solar.py — Model 2: Random Forest Solar Power Predictor
===============================================================
PURPOSE:
    Predict TOMORROW's total solar generation (in kWh) using:
    - Model 1's predicted weather (temperature, humidity, cloud cover)
    - Historical solar generation lag features (yesterday, 2d ago, 3d ago, 7d avg)

HOW IT WORKS:
    1. Features: predicted weather from Model 1 + solar lag/rolling features
    2. Target: next day's daily_solar_kwh
    3. Algorithm: RandomForestRegressor (single output this time)
    4. Split: Same chronological 80/20 as Model 1

WHY COMBINE WEATHER + SOLAR LAGS?
    - Weather directly affects solar output (clouds block sunlight, temp
      affects panel efficiency, wind cools panels)
    - Solar lag features capture the panel's recent behaviour — if the
      panel produced well yesterday, it likely will today too (unless
      weather changes drastically)

VIVA TIP: "Model 2 is a 'chained model' — it uses Model 1's output as
input. This is called a pipeline or stacked prediction. It's like asking
'given tomorrow's weather forecast, how much solar power will we get?'"
"""

# ─── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluate import evaluate_model


def prepare_solar_data(daily: pd.DataFrame, weather_predictions=None):
    """
    Prepare features (X) and target (y) for the solar model.

    FEATURES:
    - Weather columns (actual for training, predicted for testing):
      avg_amb_temp, synth_humidity, synth_cloud_cover
    - Solar lag features:
      daily_solar_kwh_lag1, lag2, lag3, roll7
    - Additional weather lags for context:
      avg_irradiance_lag1, avg_wind_speed_lag1

    TARGET:
    - daily_solar_kwh (today's actual solar generation)
    """
    # Solar-specific lag and rolling features
    solar_feature_cols = [col for col in daily.columns
                          if ("_lag" in col or "_roll" in col)
                          and "solar_kwh" in col]

    # Additional helpful features
    extra_feature_cols = [col for col in daily.columns
                          if ("_lag1" in col)
                          and any(w in col for w in ["irradiance", "wind_speed"])]

    # Weather columns that Model 1 predicts
    weather_cols = ["avg_amb_temp", "synth_humidity", "synth_cloud_cover"]

    # Combine all feature columns
    feature_cols = weather_cols + solar_feature_cols + extra_feature_cols

    X = daily[feature_cols].values
    y = daily["daily_solar_kwh"].values

    print(f"[SOLAR] Features: {len(feature_cols)} columns")
    print(f"[SOLAR] Target: daily_solar_kwh")

    return X, y, feature_cols


def train_solar_model(daily: pd.DataFrame,
                       weather_model=None,
                       weather_feature_cols=None,
                       model_path: str = "models/solar_model.pkl"):
    """
    Train the solar power prediction model.

    KEY DESIGN DECISION:
    For the TEST set, we use Model 1's PREDICTED weather instead of actual
    weather. This simulates real-world usage where we don't know tomorrow's
    actual weather — we only have our model's forecast.

    For the TRAINING set, we use ACTUAL weather (the model needs accurate
    inputs during learning to learn the right patterns).

    Steps:
    1. Prepare features with actual weather
    2. Chronological 80/20 split
    3. Replace test set weather columns with Model 1's predictions
    4. Train and evaluate
    """
    print("=" * 60)
    print("  STEP 4: MODEL 2 — SOLAR POWER PREDICTOR")
    print("=" * 60)

    X, y, feature_cols = prepare_solar_data(daily)

    # ─── Chronological Split ──────────────────────────────────────────────
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx].copy(), X[split_idx:].copy()
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[SOLAR] Train: {len(X_train)} days | Test: {len(X_test)} days")

    # ─── Replace test weather with Model 1's predictions ─────────────────
    # In real use, we wouldn't know tomorrow's actual weather.
    # So we feed Model 1's predictions into Model 2.
    if weather_model is not None and weather_feature_cols is not None:
        print("[SOLAR] Using Model 1's predicted weather for test set...")

        # Get weather features for the test period
        X_weather_test = daily[weather_feature_cols].values[split_idx:]

        # Model 1 predicts: [temp, humidity, cloud_cover]
        weather_pred = weather_model.predict(X_weather_test)

        # Replace the first 3 columns (weather) in X_test with predictions
        # Columns 0, 1, 2 = avg_amb_temp, synth_humidity, synth_cloud_cover
        X_test[:, 0] = weather_pred[:, 0]  # Predicted temperature
        X_test[:, 1] = weather_pred[:, 1]  # Predicted humidity
        X_test[:, 2] = weather_pred[:, 2]  # Predicted cloud cover
        print("[SOLAR] Test set weather replaced with Model 1 predictions ✓")

    # ─── Train the Model ──────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=100,   # 100 trees
        max_depth=15,       # Limit depth to prevent overfitting
        random_state=42,    # Reproducibility
        n_jobs=-1           # Use all CPU cores
    )

    print("[SOLAR] Training Random Forest (100 trees, max_depth=15)...")
    model.fit(X_train, y_train)
    print("[SOLAR] Training complete ✓")

    # ─── Evaluate on Test Set ─────────────────────────────────────────────
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Solar Power (kWh)")

    # ─── Save the Model ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[SOLAR] Model saved to: {model_path}")

    # ─── Sample Predictions ──────────────────────────────────────────────
    print("\n[SOLAR] Sample predictions (last 5 test days):")
    print(f"  {'Actual (kWh)':>14} {'Predicted (kWh)':>16} {'Error (kWh)':>14}")
    for j in range(-5, 0):
        error = y_test[j] - y_pred[j]
        print(f"  {y_test[j]:14.2f} {y_pred[j]:16.2f} {error:14.2f}")
    print()

    return model, X_test, y_test, y_pred, metrics


# ─── Run as standalone script ─────────────────────────────────────────────────
if __name__ == "__main__":
    daily = pd.read_csv("data/daily_features.csv")
    train_solar_model(daily)
