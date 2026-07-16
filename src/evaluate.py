"""
evaluate.py — Evaluation metrics helper for ML models
=======================================================
PURPOSE:
    Provide a reusable function to calculate and print standard
    regression evaluation metrics:
    - MAE  (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R²   (Coefficient of Determination)

VIVA TIPS:
    - MAE  = Average of |actual − predicted|. Easy to interpret: "on average,
             our prediction is off by X kWh."
    - RMSE = Square root of average of (actual − predicted)². Penalises large
             errors more than MAE. Good when big mistakes are costly.
    - R²   = How much variance the model explains. 1.0 = perfect, 0.0 = model
             is no better than predicting the mean every time, negative = worse
             than the mean.
"""

# ─── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Calculate and print MAE, RMSE, R² for a model's predictions.

    Parameters
    ----------
    y_true : array-like
        Actual (ground truth) values.
    y_pred : array-like
        Model's predicted values.
    model_name : str
        Name of the model (for display purposes).

    Returns
    -------
    dict
        Dictionary with keys 'MAE', 'RMSE', 'R2' and their float values.

    VIVA TIP: "We use three metrics because each tells a different story.
    MAE is the simplest, RMSE punishes big errors, and R² shows overall fit."
    """
    # Convert to numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate each metric
    mae = mean_absolute_error(y_true, y_pred)        # Average absolute error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root mean squared error
    r2 = r2_score(y_true, y_pred)                     # R-squared (0 to 1 ideally)

    # Print a nice summary
    print(f"\n{'─' * 40}")
    print(f"  📊 {model_name} — Evaluation Metrics")
    print(f"{'─' * 40}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    print(f"{'─' * 40}\n")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
