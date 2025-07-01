"""
Warehouse SKU Sales Forecasting Module using Toto Model

This module provides functionality to forecast warehouse sales/inventory data using 
the Datadog Toto model, a state-of-the-art time series forecasting model.

Pipeline:
1. Loads the pre-trained Toto model (from Datadog/Toto-Open-Base-1.0)
2. Processes input CSV data with historical sales/inventory information
3. Builds input structure using MaskedTimeseries (data/util/dataset.py)
4. Runs prediction using TotoForecaster (inference/forecaster.py)
5. Returns the forecast as a CSV string (suitable for API responses or file downloads)

The module is designed to work with the Streamlit web application but can also
be used independently for batch processing or API integration.

Author: R2Talk Team
Date: July 2025
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pandas as pd
import numpy as np
from io import StringIO
from model.toto import Toto
from data.util.dataset import MaskedTimeseries
from inference.forecaster import TotoForecaster

def execute_forecast(csv_content: str, forecast_length: int = 30) -> str:
    """
    Runs the Toto forecasting pipeline on provided CSV content to generate time series forecasts.

    Args:
        csv_content (str): The raw CSV content of historical sales/inventory data, as a string.
            - Expected format: semicolon (;) separated
            - Decimal separator: comma (,)
            - Required columns: 'DATE' (in DD/MM/YYYY format) and 'VALUE' (numeric)
            - Data should be in chronological order (will be sorted if not)
        forecast_length (int, optional): Number of future time steps to forecast (the prediction horizon).
            Defaults to 30. Valid range is 7-90 days.

    Returns:
        str: CSV string with forecasted values for each future time step.
            - Format: semicolon (;) separated with comma (,) as decimal separator
            - Columns: 'DATE' (in DD/MM/YYYY format), 'VALUE' (forecasted quantity)
            - Each row represents one forecasted day
            - Values are rounded to integers and negative values are clipped to zero

    Steps:
        1. Parses and validates the input CSV content
        2. Converts data to tensor format required by the model
        3. Loads the pre-trained Toto model from Datadog
        4. Creates a MaskedTimeseries object with appropriate masks and timestamps
        5. Runs the forecast for the specified horizon
        6. Processes model output and formats as CSV string

    Notes:
        - The function is designed for in-memory operation (no files are read or written)
        - Uses GPU acceleration if available, otherwise falls back to CPU
        - Raises exceptions with descriptive messages if the input is malformed or forecasting fails
        - Forecast dates continue from the last date in the input data with daily frequency
    """
    # Step 1: Parse the CSV content
    try:
        # Read the CSV content with semicolon separator and comma as decimal
        df = pd.read_csv(StringIO(csv_content), sep=';', decimal=',')
        
        # Convert the 'DATE' column to datetime
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
        
        # Sort by date to ensure chronological order
        df = df.sort_values('DATE')
        
        # Extract date series and feature columns
        date_series = df['DATE']
        feature_cols = [col for col in df.columns if col != 'DATE']
        
        # Convert sales data to tensor format (shape: [num_features, num_timesteps])
        sales_data = df[feature_cols].values.T  # Transpose to get [features, timesteps]
        series_tensor = torch.tensor(sales_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension [1, features, timesteps]
        
    except Exception as e:
        raise Exception(f"Error reading or processing CoffeeSales.csv: {str(e)}")

    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    toto.to(device)
    toto.compile()
      
    # Create forecaster
    forecaster = TotoForecaster(toto.model)

    print(f"[DEBUG] series_tensor sample: {series_tensor[0,:, :5]}")  # Show the first 5 samples of each feature
    
    # Prepare masks and timestamps from the CSV data
    input_series = series_tensor.to(device)
    
    # padding_mask: all True (no padding)
    padding_mask = torch.ones_like(input_series, dtype=torch.bool)
    
    # id_mask: all zeros (one series)
    id_mask = torch.zeros_like(input_series, dtype=torch.int64)
    
    # timestamp_seconds: convert dates to seconds since epoch
    timestamp_seconds = torch.tensor(
        [(dt - pd.Timestamp("1970-01-01")).total_seconds() for dt in date_series],
        dtype=torch.int64
    ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, N)
    
    # time_interval_seconds: average difference between dates
    if len(date_series) > 1:
        interval = int((date_series.iloc[1] - date_series.iloc[0]).total_seconds())
    else:
        interval = 0
    time_interval_seconds = torch.full((1, 1), interval, dtype=torch.int64).to(device)

    # Create MaskedTimeseries
    inputs = MaskedTimeseries(
        series=input_series,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )
    print("[INFO] Input data prepared successfully!")
    
    # Make prediction
    # Execute the forecast for the next forecast_length steps (days)
    prediction_length = forecast_length
    forecast = forecaster.forecast(
        inputs,
        prediction_length=prediction_length,
        num_samples=None,
        samples_per_batch=1,
    )
    
    # Show results
    if forecast.samples is not None:
        print(f"Samples shape: {forecast.samples.shape}")
        print(f"Quantile 10%: {forecast.quantile(0.1).shape}")
        print(f"Quantile 90%: {forecast.quantile(0.9).shape}")

    # Generate CSV output for predictions
    print("[INFO] Generating CSV output for predictions...")
    
    # === Output forecast in the same format as input: DATE;VALUE ===
    forecast_arr = forecast.mean.cpu().numpy()
    # Remove batch and variate dims if present
    if forecast_arr.ndim == 3 and forecast_arr.shape[0] == 1:
        forecast_arr = forecast_arr[0]
    if forecast_arr.ndim == 2 and forecast_arr.shape[0] == 1:
        forecast_arr = forecast_arr[0]  # Now shape (num_variates, prediction_length)
    # Clip negatives to zero and round to nearest integer
    forecast_arr = np.round(np.clip(forecast_arr, 0, None)).astype(int)

    # Debug: print actual dtype and sample
    print(f"[DEBUG] date_series dtype before conversion: {getattr(date_series, 'dtype', type(date_series))}")
    # Only convert if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, dayfirst=True)

    last_date = date_series.iloc[-1]
    forecast_length_actual = forecast_arr.shape[-1]
    # Always use daily frequency for forecast dates
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_length_actual,
        freq='D'
    )

    # Build DataFrame for output
    data = {"DATE": forecast_dates.strftime("%d/%m/%Y")}
    for i, feature in enumerate(feature_cols):
        if i >= forecast_arr.shape[0]:
            continue
        if forecast_arr.ndim == 1:
            arr = forecast_arr[:len(forecast_dates)]
        else:
            arr = forecast_arr[i, :len(forecast_dates)]
        # Only add if length matches forecast_dates
        if len(arr) != len(forecast_dates):
            continue
        data[feature] = arr
    df_forecast = pd.DataFrame(data)
    forecast_csv = df_forecast.to_csv(index=False, sep=';', decimal=',', float_format='%.1f')
    print("[INFO] Forecast CSV generated successfully!")
    return forecast_csv
