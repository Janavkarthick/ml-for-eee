"""
weather_api.py - Open-Meteo 5-Day Weather Forecast Fetcher
==========================================================
PURPOSE:
    Fetches real-time, 5-day hourly weather forecast data using the 
    free Open-Meteo API (no API key required). It then aggregates 
    this hourly data into daily daytime averages to be used for 
    solar power prediction.

VIVA TIP:
    "We use Open-Meteo because it doesn't require an API key and 
    provides high-resolution hourly data. We aggregate the hourly 
    data into daytime averages because solar panels only generate 
    power during the day, making nighttime weather irrelevant to 
    our predictions."
"""

# Import the requests library to make HTTP calls to the internet (APIs)
import requests
# Import pandas to process and tabulate our data easily
import pandas as pd
# Import joblib to load our pre-trained machine learning model
import joblib
# Import numpy for array manipulation needed by the ML model
import numpy as np
import os

def get_5day_weather():
    """
    Fetches the 5-day forecast for a specific location and returns 
    a pandas DataFrame with daily daytime averages.
    """
    
    # The endpoint URL for the Open-Meteo free Forecast API
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Define the parameters for our API request
    params = {
        # Latitude for Chennai (or specified location)
        "latitude": 13.0827,
        # Longitude for Chennai (or specified location)
        "longitude": 80.2707,
        # The specific hourly weather features we want to retrieve
        "hourly": [
            "temperature_2m", 
            "relative_humidity_2m", 
            "cloud_cover", 
            "wind_speed_10m", 
            "precipitation_probability", 
            "shortwave_radiation", 
            "weather_code"
        ],
        # Ensure the time returned matches our local timezone
        "timezone": "Asia/Kolkata",
        # Limit the forecast to exactly 5 days
        "forecast_days": 5
    }
    
    # Send an HTTP GET request to the Open-Meteo API with our parameters
    response = requests.get(url, params=params)
    
    # Raise an error automatically if the request failed (e.g., 404 or 500 error)
    response.raise_for_status()
    
    # Parse the returned JSON text into a Python dictionary
    data = response.json()
    
    # Extract the 'hourly' dictionary from the parsed JSON response
    hourly_data = data["hourly"]
    
    # Convert the extracted hourly dictionary into a pandas DataFrame
    df_hourly = pd.DataFrame(hourly_data)
    
    # Convert the 'time' column from plain strings into actual datetime objects
    df_hourly['time'] = pd.to_datetime(df_hourly['time'])
    
    # Extract just the 'date' part from the datetime, so we can group by day
    df_hourly['date'] = df_hourly['time'].dt.date
    
    # Extract just the 'hour' (0-23) from the datetime to filter for daytime only
    df_hourly['hour'] = df_hourly['time'].dt.hour
    
    # Filter the DataFrame to keep ONLY daytime hours (e.g., 6 AM to 6 PM inclusive)
    # This is crucial because solar generation only happens during the day!
    daytime_df = df_hourly[(df_hourly['hour'] >= 6) & (df_hourly['hour'] <= 18)]
    
    # Group the filtered daytime rows by the 'date' column
    # and calculate the average (mean) for all the weather features
    daily_avg_df = daytime_df.groupby('date').mean(numeric_only=True).reset_index()
    
    # Drop the 'hour' column since it is no longer meaningful after averaging a whole day
    daily_avg_df = daily_avg_df.drop(columns=['hour'])
    
    # Round all numerical columns to 2 decimal places for cleaner presentation
    daily_avg_df = daily_avg_df.round(2)
    
    # Return the final 5-row DataFrame (one row for each forecast day)
    return daily_avg_df


def predict_solar_from_api(weather_df):
    """
    Integrates the real-time weather API data with our trained Random Forest 
    solar prediction model to predict the next 5 days of solar generation.
    """
    
    # Determine the root path to easily load our model and historical data
    project_root = os.path.join(os.path.dirname(__file__), "..")
    
    # 1. Load the pre-trained solar prediction model (Random Forest)
    model_path = os.path.join(project_root, "models", "solar_model.pkl")
    solar_model = joblib.load(model_path)
    
    # 2. Load historical data to initialize our lag features (the 'buffers')
    # Our model needs to know what happened yesterday to predict tomorrow!
    data_path = os.path.join(project_root, "data", "daily_features.csv")
    historical_df = pd.read_csv(data_path)
    
    # Take the last 10 days to be safe, since we need up to 7 days for rolling averages
    tail = historical_df.tail(10)
    solar_buf = tail["daily_solar_kwh"].tolist()
    irr_buf = tail["avg_irradiance"].tolist()
    wind_buf = tail["avg_wind_speed"].tolist()
    
    # We will store our final prediction rows in a list before converting to a DataFrame
    results = []
    
    # Loop through each of the 5 days in the weather_df
    for index, row in weather_df.iterrows():
        # Read one day's weather features from the API
        date = row["date"]
        temp = row["temperature_2m"]
        humidity = row["relative_humidity_2m"]
        cloud = row["cloud_cover"]
        wind = row["wind_speed_10m"]
        irr = row["shortwave_radiation"]
        
        # Calculate lag features from our historical buffers
        solar_lag1 = solar_buf[-1]
        solar_lag2 = solar_buf[-2]
        solar_lag3 = solar_buf[-3]
        solar_roll7 = np.mean(solar_buf[-7:])
        irr_lag1 = irr_buf[-1]
        wind_lag1 = wind_buf[-1]
        
        # Build the exact 9-feature array the model expects:
        # [temp, humidity, cloud, solar_lag1, solar_lag2, solar_lag3, solar_roll7, irr_lag1, wind_lag1]
        features = np.array([
            temp, humidity, cloud, 
            solar_lag1, solar_lag2, solar_lag3, solar_roll7, 
            irr_lag1, wind_lag1
        ]).reshape(1, -1)
        
        # Predict solar generation using the ML model
        predicted_solar = solar_model.predict(features)[0]
        # Prevent negative solar predictions
        predicted_solar = max(predicted_solar, 0.0)
        
        # Update our historical buffers so the next day's lag features are correct
        # This is where we feed today's prediction back in as yesterday's reality!
        solar_buf.append(predicted_solar)
        irr_buf.append(irr)
        wind_buf.append(wind)
        
        # Save the prediction along with the weather data
        results.append({
            "Date": date,
            "Temperature": round(temp, 1),
            "Humidity": round(humidity, 1),
            "Cloud Cover": round(cloud, 1),
            "Wind Speed": round(wind, 1),
            "Solar Radiation": round(irr, 1),
            "Predicted Solar Generation": round(predicted_solar, 1)
        })
        
    # Convert the list of results into the final required DataFrame format
    return pd.DataFrame(results)


# This block only runs if we execute this script directly (not if imported)
if __name__ == "__main__":
    
    # Print a divider line for a clean console output
    print("=" * 70)
    
    # Print an informational message indicating what we are doing
    print("Fetching 5-Day Weather Forecast from Open-Meteo API...")
    
    # Call our function to fetch and process the weather data
    forecast_df = get_5day_weather()
    
    # Print a success message
    print("Data fetched successfully! Now running through ML model...")
    print("=" * 70)
    
    # Now integrate the weather data with our trained Random Forest model
    final_predictions_df = predict_solar_from_api(forecast_df)
    
    # Print the resulting 5-day DataFrame to the console
    print(final_predictions_df.to_string(index=False))
    
    # Print a final divider line
    print("=" * 70)
