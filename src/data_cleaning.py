"""
data_cleaning.py — Step 1: Load and clean the raw solar generation data
==========================================================================
PURPOSE:
    - Load the raw CSV from data/Generation_data.csv
    - Create SYNTHETIC daylight-only timestamps (6 AM – 6 PM, 5-min intervals)
    - Handle missing values (fill with column median)
    - Remove duplicate rows
    - Clip outliers using IQR method
    - Save cleaned data to data/cleaned_data.csv

WHY SYNTHETIC TIMESTAMPS?
    The raw CSV has no timestamp column. We know from inspecting the data that
    every row has AC Power > 0 (minimum ~394 W), meaning the logger only
    recorded during sunlight hours. So we assign 5-minute timestamps from
    06:00 to 18:00 each day (that's 144 readings per day), cycling through
    synthetic calendar days starting 2023-01-01.

VIVA TIP:
    "We synthesized timestamps because the raw data had none. We assumed
    5-minute sampling during daylight (6 AM–6 PM) based on ~118 k rows
    giving us ~825 days of data."
"""

# ─── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd   # DataFrame manipulation
import numpy as np    # Numerical operations
import os             # File path handling


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file (e.g. 'data/Generation_data.csv').

    Returns
    -------
    pd.DataFrame
        Raw data with original columns.

    VIVA TIP: "pd.read_csv() reads a CSV into a table called a DataFrame.
    Each row is one sensor reading, each column is a measurement."
    """
    print(f"[CLEAN] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[CLEAN] Raw data shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def add_synthetic_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic daylight-only timestamps for each row.

    LOGIC:
    - Each synthetic day has 144 slots (6:00 AM to 5:55 PM, every 5 minutes)
    - Row 0 → 2023-01-01 06:00, Row 1 → 2023-01-01 06:05, ...
    - Row 144 → 2023-01-02 06:00, etc.
    - We assign day_number = row_index // 144, slot = row_index % 144
    - timestamp = start_date + day_number days + slot × 5 minutes

    VIVA TIP: "We used integer division (//) to figure out which day each row
    belongs to, and modulo (%) for the time-slot within that day."
    """
    READINGS_PER_DAY = 144  # 12 hours × 12 readings/hour (every 5 min)
    START_DATE = pd.Timestamp("2023-01-01 06:00:00")  # First day, 6 AM
    INTERVAL = pd.Timedelta(minutes=5)  # 5-minute gap between readings

    n_rows = len(df)

    # Calculate the day number and slot number for each row
    day_numbers = np.arange(n_rows) // READINGS_PER_DAY  # Which day? (0, 0, ..., 1, 1, ...)
    slot_numbers = np.arange(n_rows) % READINGS_PER_DAY   # Which slot within the day? (0–143)

    # Build the full timestamp array
    # Each timestamp = start_date + (day_number * 1 day) + (slot_number * 5 minutes)
    timestamps = (
        START_DATE
        + pd.to_timedelta(day_numbers, unit="D")   # Add the day offset
        + pd.to_timedelta(slot_numbers * 5, unit="m")  # Add the minute offset
    )

    df = df.copy()  # Don't modify the original
    df.insert(0, "timestamp", timestamps)  # Put timestamp as the first column

    n_days = day_numbers[-1] + 1  # Total synthetic days created
    print(f"[CLEAN] Added synthetic timestamps: {n_days} days, "
          f"6:00 AM – 5:55 PM, 5-min intervals")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill any missing (NaN) values with the column median.

    WHY MEDIAN (not mean)?
    - The median is robust to outliers. If one sensor spiked to 99999,
      the mean would be pulled way up, but the median stays stable.

    VIVA TIP: "Median is the middle value when sorted. It's better than
    mean for noisy sensor data because outliers don't affect it."
    """
    # Only fill numeric columns (skip 'timestamp')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_missing_before = df[numeric_cols].isnull().sum().sum()  # Total NaN count

    if n_missing_before > 0:
        # fillna(median) replaces each NaN with that column's median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(f"[CLEAN] Filled {n_missing_before} missing values with column medians")
    else:
        print("[CLEAN] No missing values found ✓")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows (all column values identical).

    WHY?
    - Sensor loggers sometimes double-write the same reading.
    - Duplicates inflate our dataset and bias the model.

    VIVA TIP: "df.duplicated() returns True for rows that are exact copies
    of an earlier row. We drop those to avoid training on repeated data."
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"[CLEAN] Removed {n_removed} duplicate rows")
    else:
        print("[CLEAN] No duplicate rows found ✓")
    return df.reset_index(drop=True)


def clip_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outlier values using the IQR (Interquartile Range) method.

    HOW IT WORKS:
    1. Q1 = 25th percentile, Q3 = 75th percentile
    2. IQR = Q3 − Q1 (the "middle 50%" spread)
    3. Lower fence = Q1 − 1.5 × IQR
    4. Upper fence = Q3 + 1.5 × IQR
    5. Any value below the lower fence → set to lower fence
       Any value above the upper fence → set to upper fence

    WHY 1.5?
    - It's a standard statistical convention (Tukey's rule).
    - Catches extreme values while keeping normal variation.

    VIVA TIP: "IQR clipping doesn't delete rows — it caps extreme values
    to a reasonable range. Think of it like setting min/max limits on a sensor."
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_clipped = 0

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1                # Interquartile range

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        # Count how many values will be clipped
        n_clipped = ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()
        total_clipped += n_clipped

        # Clip: values below lower_fence → lower_fence, above upper → upper
        df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)

    print(f"[CLEAN] Clipped {total_clipped} outlier values using IQR method")
    return df


def clean_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Master cleaning function — runs all cleaning steps in order.

    Steps:
    1. Load raw CSV
    2. Add synthetic daylight timestamps
    3. Fill missing values
    4. Remove duplicates
    5. Clip outliers
    6. Save cleaned CSV

    Parameters
    ----------
    input_path : str
        Path to raw CSV (e.g. 'data/Generation_data.csv')
    output_path : str
        Where to save the cleaned CSV (e.g. 'data/cleaned_data.csv')

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame (also saved to disk)
    """
    print("=" * 60)
    print("  STEP 1: DATA CLEANING")
    print("=" * 60)

    # Run each step in sequence
    df = load_raw_data(input_path)
    df = add_synthetic_timestamps(df)
    df = fill_missing_values(df)
    df = remove_duplicates(df)
    df = clip_outliers_iqr(df)

    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"[CLEAN] Saved cleaned data to: {output_path}")
    print(f"[CLEAN] Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print()

    return df


# ─── Run as standalone script ─────────────────────────────────────────────────
if __name__ == "__main__":
    # When you run: python src/data_cleaning.py
    clean_data("data/Generation_data.csv", "data/cleaned_data.csv")
