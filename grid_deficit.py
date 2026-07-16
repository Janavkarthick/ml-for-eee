"""
grid_deficit.py — Module 3: Grid Deficit Calculator (NO ML)
=============================================================
PURPOSE:
    Calculate how much extra power the factory needs from the grid
    when solar generation is not enough to meet demand.

    This is PURE PYTHON — no machine learning, no scikit-learn.
    It's a simple arithmetic function.

FORMULA:
    extra_power = load_requirement − predicted_solar
    BUT only if the result is positive (you can't have negative extra power).
    If solar exceeds load, extra_power = 0 (surplus, not deficit).

REAL-WORLD CONTEXT:
    In Tamil Nadu, TNPDCL (Tamil Nadu Power Distribution Corporation Limited)
    supplies grid power when renewable sources fall short. This module
    calculates exactly how much grid supply is needed.

VIVA TIP: "This is just subtraction with a floor of zero. If the factory
needs 500 kW and solar provides 300 kW, the grid supplies 200 kW. If
solar provides 600 kW, the grid supplies 0 kW (and the extra 100 kW
could be exported back or stored in batteries)."
"""

# ─── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


def calculate_extra_power(load_kw: float, predicted_solar_kw: float) -> float:
    """
    Calculate extra grid power needed for a SINGLE day.

    Parameters
    ----------
    load_kw : float
        Factory's electricity demand for the day (in kW or kWh).
    predicted_solar_kw : float
        Predicted solar generation for the day (in kW or kWh).

    Returns
    -------
    float
        Extra power needed from the grid. Zero if solar covers the load.

    Examples
    --------
    >>> calculate_extra_power(500, 300)
    200.0
    >>> calculate_extra_power(500, 700)
    0.0
    >>> calculate_extra_power(500, 500)
    0.0

    VIVA TIP: "max(deficit, 0) ensures we never return a negative number.
    A negative deficit means surplus solar — but we don't model export here."
    """
    deficit = load_kw - predicted_solar_kw  # How much more do we need?
    return max(deficit, 0.0)  # Only return positive values (or zero)


def calculate_deficit_series(load_series, solar_series) -> pd.DataFrame:
    """
    Calculate grid deficit for a SERIES of days (vectorised).

    This is the batch version of calculate_extra_power(), applied to
    arrays of load and solar values simultaneously.

    Parameters
    ----------
    load_series : array-like
        Daily load values (one per day).
    solar_series : array-like
        Daily predicted solar values (one per day).

    Returns
    -------
    pd.DataFrame with columns:
        - load_kwh: the factory's demand
        - solar_kwh: predicted solar generation
        - grid_deficit_kwh: extra power from grid (0 if solar is sufficient)
        - solar_sufficient: True/False flag

    VIVA TIP: "np.maximum is the array version of max(). It compares
    element-by-element and returns the larger value. We compare the deficit
    against 0 to clip negative values."
    """
    load_arr = np.array(load_series)
    solar_arr = np.array(solar_series)

    # Calculate deficit for every day at once (vectorised — fast!)
    deficit = load_arr - solar_arr
    grid_power = np.maximum(deficit, 0.0)  # Clip negatives to zero

    result = pd.DataFrame({
        "load_kwh": load_arr,
        "solar_kwh": solar_arr,
        "grid_deficit_kwh": grid_power,
        "solar_sufficient": solar_arr >= load_arr  # True if solar covers load
    })

    # Print summary statistics
    n_sufficient = result["solar_sufficient"].sum()
    n_total = len(result)
    avg_deficit = result["grid_deficit_kwh"].mean()

    print(f"\n{'─' * 40}")
    print(f"  ⚡ Grid Deficit Analysis")
    print(f"{'─' * 40}")
    print(f"  Days analysed:         {n_total}")
    print(f"  Solar sufficient:      {n_sufficient}/{n_total} "
          f"({100 * n_sufficient / n_total:.1f}%)")
    print(f"  Avg grid deficit:      {avg_deficit:.2f} kWh")
    print(f"  Max grid deficit:      {result['grid_deficit_kwh'].max():.2f} kWh")
    print(f"  Days needing grid:     {n_total - n_sufficient}")
    print(f"{'─' * 40}\n")

    return result


# ─── Run as standalone script ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick demonstration with sample data
    print("Grid Deficit Calculator — Demo")
    print(f"  500 kW load, 300 kW solar → deficit: {calculate_extra_power(500, 300)} kW")
    print(f"  500 kW load, 700 kW solar → deficit: {calculate_extra_power(500, 700)} kW")
    print(f"  500 kW load, 500 kW solar → deficit: {calculate_extra_power(500, 500)} kW")
