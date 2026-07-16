"""
model_weather.py — Model 1: Random Forest Weather Predictor
==============================================================
PURPOSE:
    Predict TOMORROW's weather conditions from historical weather patterns:
    - Tomorrow's ambient temperature (°C)
    - Tomorrow's humidity (%)
    - Tomorrow's cloud cover (%)

HOW IT WORKS:
    1. Features (inputs): past 1–3 day lag values + 7-day rolling averages
       of temperature, humidity, cloud cover, wind speed, and irradiance.
    2. Targets (outputs): next day's temp, humidity, cloud cover.
    3. Algorithm: RandomForestRegressor with MultiOutputRegressor wrapper
       (because we're predicting 3 outputs at once).
    4. Split: Chronological 80/20 (first 80% of days for training,
       last 20% for testing — no shuffle, because this is time-series).

WHY RANDOM FOREST?
    - Works well with tabular data (numbers in columns)
    - Handles non-linear relationships automatically
    - Doesn't need feature scaling (normalisation)
    - Resistant to overfitting when properly tuned

VIVA TIP: "A Random Forest is a collection of many decision trees that
each vote on the prediction. The final answer is the average of all votes.
This 'wisdom of the crowd' approach reduces errors."
"""

# ─── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib  # For saving/loading the trained model to disk
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import our custom evaluation function
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluate import evaluate_model


def prepare_weather_data(daily: pd.DataFrame):
    """
    Prepare features (X) and targets (y) for the weather model.

    FEATURES (what the model sees):
    - Lag 1/2/3 and 7-day rolling mean of:
      avg_amb_temp, synth_humidity, synth_cloud_cover,
      avg_wind_speed, avg_irradiance

    TARGETS (what the model predicts):
    - avg_amb_temp (tomorrow's temperature)
    - synth_humidity (tomorrow's humidity)
    - synth_cloud_cover (tomorrow's cloud cover)

    We create targets by shifting: today's actual value becomes
    yesterday's target. So target[i] = actual[i], and the features
    at row[i] are the lags of row[i] (which look at days i-1, i-2, i-3).

    VIVA TIP: "The lag features at row i already represent 'yesterday's data'
    relative to day i. So the target for row i is just day i's actual value."
    """
    # Feature columns: all the lag and rolling columns for weather variables
    feature_cols = [col for col in daily.columns
                    if ("_lag" in col or "_roll" in col)
                    and any(w in col for w in ["amb_temp", "humidity",
                                                "cloud_cover", "wind_speed",
                                                "irradiance"])]

    # Target columns: the actual weather values for that day
    target_cols = ["avg_amb_temp", "synth_humidity", "synth_cloud_cover"]

    X = daily[feature_cols].values  # Features as numpy array
    y = daily[target_cols].values   # Targets as numpy array

    print(f"[WEATHER] Features: {len(feature_cols)} columns")
    print(f"[WEATHER] Targets: {target_cols}")

    return X, y, feature_cols, target_cols


def train_weather_model(daily: pd.DataFrame, model_path: str = "models/weather_model.pkl"):
    """
    Train the weather prediction model and save it.

    Steps:
    1. Prepare X (features) and y (targets)
    2. Chronological train/test split (80/20)
    3. Train RandomForestRegressor
    4. Evaluate on test set
    5. Save model to disk

    Returns
    -------
    tuple: (model, X_test, y_test, y_pred, metrics_dict)
    """
    print("=" * 60)
    print("  STEP 3: MODEL 1 — WEATHER PREDICTOR")
    print("=" * 60)

    X, y, feature_cols, target_cols = prepare_weather_data(daily)

    # ─── Chronological Split ──────────────────────────────────────────────
    # WHY NO SHUFFLE? In time series, future data must not leak into training.
    # We train on the first 80% of days and test on the last 20%.
    split_idx = int(len(X) * 0.8)  # Index where training ends

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[WEATHER] Train: {len(X_train)} days | Test: {len(X_test)} days")

    # ─── Train the Model ──────────────────────────────────────────────────
    # MultiOutputRegressor wraps a single-output model to handle 3 targets
    # n_estimators=100 → 100 decision trees in the forest
    # random_state=42 → reproducible results
    # n_jobs=-1 → use all CPU cores for speed
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,   # Number of trees in the forest
            max_depth=15,       # Max depth of each tree (prevents overfitting)
            random_state=42,    # Seed for reproducibility
            n_jobs=-1           # Parallel processing
        )
    )

    print("[WEATHER] Training Random Forest (100 trees, max_depth=15)...")
    model.fit(X_train, y_train)  # This is where the model learns!
    print("[WEATHER] Training complete ✓")

    # ─── Evaluate on Test Set ─────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # Evaluate each target separately
    all_metrics = {}
    for i, target_name in enumerate(target_cols):
        metrics = evaluate_model(y_test[:, i], y_pred[:, i],
                                  f"Weather → {target_name}")
        all_metrics[target_name] = metrics

    # ─── Save the Model ──────────────────────────────────────────────────
    # joblib.dump saves the trained model as a .pkl file
    # You can load it later with joblib.load() without retraining
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[WEATHER] Model saved to: {model_path}")

    # ─── Sample Predictions ──────────────────────────────────────────────
    print("\n[WEATHER] Sample predictions (last 5 test days):")
    print(f"  {'Actual Temp':>12} {'Pred Temp':>12} {'Actual Hum':>12} "
          f"{'Pred Hum':>12} {'Actual Cloud':>12} {'Pred Cloud':>12}")
    for j in range(-5, 0):
        print(f"  {y_test[j, 0]:12.2f} {y_pred[j, 0]:12.2f} "
              f"{y_test[j, 1]:12.2f} {y_pred[j, 1]:12.2f} "
              f"{y_test[j, 2]:12.2f} {y_pred[j, 2]:12.2f}")
    print()

    return model, X_test, y_test, y_pred, all_metrics, feature_cols


# ─── Run as standalone script ─────────────────────────────────────────────────
if __name__ == "__main__":
    daily = pd.read_csv("data/daily_features.csv")
    train_weather_model(daily)
