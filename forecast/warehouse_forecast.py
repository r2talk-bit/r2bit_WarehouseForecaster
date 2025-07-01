"""
Warehouse SKU Sales Forecasting Module using Toto Model

This module provides functionality to forecast warehouse sales/inventory data using 
the Datadog Toto model, a state-of-the-art time series forecasting model.

What this code does:
- Takes historical sales/inventory data (past numbers)
- Uses AI to predict future sales/inventory (future numbers)
- Returns these predictions in a simple format

Pipeline (step-by-step process):
1. Loads the pre-trained Toto model (from Datadog/Toto-Open-Base-1.0)
   - This is the AI model that makes predictions
2. Processes input CSV data with historical sales/inventory information
   - CSV is a common format for storing data in tables
3. Builds input structure using MaskedTimeseries (data/util/dataset.py)
   - Prepares the data in a format the AI can understand
4. Runs prediction using TotoForecaster (inference/forecaster.py)
   - Makes the actual predictions for future days
5. Returns the forecast as a CSV string (suitable for API responses or file downloads)
   - Gives back results in the same format as the input

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
    In simple terms: Takes your past sales data and predicts future sales.

    Args (what you need to provide):
        csv_content (str): Your historical sales/inventory data as a CSV string.
            - Expected format: semicolon (;) separated values (like "01/01/2025;10")
            - Decimal separator: comma (,) (European format like "10,5" not "10.5")
            - Required columns: 'DATE' (in DD/MM/YYYY format) and 'VALUE' (the numbers)
            - Data should be in time order (oldest to newest, but will be sorted if not)
        forecast_length (int, optional): How many days into the future to predict.
            Defaults to 30 days. You can choose between 7-90 days.

    Returns (what you get back):
        str: A CSV string containing your forecasted values for each future day.
            - Same format as your input: semicolon (;) separated with comma (,) for decimals
            - Contains columns: 'DATE' (in DD/MM/YYYY format) and 'VALUE' (predicted numbers)
            - Each row is one day's prediction
            - All predictions are whole numbers (no decimals) and never negative

    How it works (step by step):
        1. Reads and checks your CSV data
        2. Converts your data to a special format the AI model can understand
        3. Loads the Toto AI model (either from your computer or downloads it)
        4. Prepares your data with special markers that help the AI understand time patterns
        5. Asks the AI to make predictions for the number of days you specified
        6. Formats the AI's predictions back into a simple CSV you can use

    Good to know:
        - Everything happens in memory (no files are saved to your computer)
        - Uses your graphics card (GPU) if available for faster processing
        - Gives helpful error messages if something goes wrong
        - Predictions start from the day after your last data point
    """
    # Step 1: Parse the CSV content (convert text data into a format we can work with)
    try:
        # Read the CSV content with semicolon separator and comma as decimal
        # This creates a table (DataFrame) from your CSV text
        df = pd.read_csv(StringIO(csv_content), sep=';', decimal=',')
        
        # Convert the 'DATE' column to datetime objects
        # This helps Python understand these are dates, not just text
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)  # day first for DD/MM/YYYY format
        
        # Sort by date to ensure chronological order (oldest to newest)
        df = df.sort_values('DATE')
        
        # Extract date series and feature columns
        # date_series = all the dates in your data
        date_series = df['DATE']
        # feature_cols = all columns except DATE (these contain your actual values)
        feature_cols = [col for col in df.columns if col != 'DATE']
        
        # Convert sales data to tensor format (special format for AI models)
        # Think of tensors as multi-dimensional arrays that AI can process
        sales_data = df[feature_cols].values.T  # Transpose to get [features, timesteps]
        series_tensor = torch.tensor(sales_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension [1, features, timesteps]
        
    except Exception as e:
        # If anything goes wrong, show a helpful error message
        raise Exception(f"Error reading or processing CoffeeSales.csv: {str(e)}")

    # Step 2: Set up the AI model
    
    # Choose the best hardware to run the AI model
    # 'cuda' means using the graphics card (GPU) which is much faster
    # 'cpu' is used as a fallback if no GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device} for model processing")
    
    # Load the AI model - first check if we already have it, otherwise download it
    import os
    
    # Define where the model should be stored on the computer
    local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "pretrained")
    
    # Try to load the model using these steps:
    # 1. Check if it's already on the computer
    # 2. If not, download it from the internet
    # 3. If that fails, try a different download method
    try:
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            # The model is already on the computer, so use it
            print(f"[INFO] Loading model from local path: {local_model_path}")
            toto = Toto.from_pretrained(local_model_path)
        else:
            # The model isn't on the computer, so download it
            print("[INFO] Local model not found, downloading from Hugging Face Hub...")
            # Create a folder to store the model
            os.makedirs(local_model_path, exist_ok=True)
            # Download the model and save it to the folder
            toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0', cache_dir=local_model_path)
    except Exception as e:
        # If something went wrong with the download, try one more method
        print(f"[WARNING] Error loading/downloading model: {str(e)}")
        print("[INFO] Falling back to direct model loading...")
        # Try to download directly without saving
        toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    
    # Move the model to the right hardware (GPU or CPU)
    toto.to(device)
    # Optimize the model for faster processing
    toto.compile()
      
    # Step 3: Set up the forecasting tool and prepare the data
    
    # Create a forecaster object that will use our AI model to make predictions
    forecaster = TotoForecaster(toto.model)

    # Show a sample of our data (helpful for debugging)
    print(f"[DEBUG] series_tensor sample: {series_tensor[0,:, :5]}")  # Show the first 5 samples of each feature
    
    # Move our data to the right device (GPU or CPU)
    input_series = series_tensor.to(device)
    
    # The next few steps create special markers (masks) that help the AI understand our data
    
    # padding_mask: tells the model which data points are real vs. padding
    # (all True means all our data points are valid - no padding)
    padding_mask = torch.ones_like(input_series, dtype=torch.bool)
    
    # id_mask: used when you have multiple time series (we have just one, so all zeros)
    id_mask = torch.zeros_like(input_series, dtype=torch.int64)
    
    # timestamp_seconds: convert our dates to seconds since Jan 1, 1970
    # This helps the model understand the exact timing of each data point
    timestamp_seconds = torch.tensor(
        [(dt - pd.Timestamp("1970-01-01")).total_seconds() for dt in date_series],
        dtype=torch.int64
    ).unsqueeze(0).unsqueeze(0).to(device)  # Add dimensions to match expected format
    
    # time_interval_seconds: calculate how many seconds between data points
    # This helps the model understand if data is daily, weekly, etc.
    if len(date_series) > 1:
        interval = int((date_series.iloc[1] - date_series.iloc[0]).total_seconds())
    else:
        interval = 0  # Default if we only have one data point
    time_interval_seconds = torch.full((1, 1), interval, dtype=torch.int64).to(device)

    # Create a MaskedTimeseries object that packages all our data in the format the AI needs
    inputs = MaskedTimeseries(
        series=input_series,            # The actual values
        padding_mask=padding_mask,      # Which values are real
        id_mask=id_mask,               # Series identifier
        timestamp_seconds=timestamp_seconds,           # When each value occurred
        time_interval_seconds=time_interval_seconds,   # Time between values
    )
    print("[INFO] Input data prepared successfully!")
    
    # Step 4: Make the forecast
    
    # Ask the AI model to predict the next 'forecast_length' days
    prediction_length = forecast_length
    forecast = forecaster.forecast(
        inputs,                      # Our prepared data
        prediction_length=prediction_length,  # How many days to predict
        num_samples=None,            # Use default number of samples
        samples_per_batch=1,         # Process one sample at a time
    )
    print("[INFO] Forecast calculation complete!")
    
    
    # Step 5: Process the forecast results
    
    # Show information about the forecast results (helpful for debugging)
    if forecast.samples is not None:
        print(f"Samples shape: {forecast.samples.shape}")
        print(f"Quantile 10%: {forecast.quantile(0.1).shape}")  # Lower estimate (pessimistic)
        print(f"Quantile 90%: {forecast.quantile(0.9).shape}")  # Upper estimate (optimistic)

    # Step 6: Format the results as a CSV
    print("[INFO] Generating CSV output for predictions...")
    
    # Get the mean (average) prediction values and convert from GPU/tensor to regular numpy array
    forecast_arr = forecast.mean.cpu().numpy()
    
    # Clean up the array dimensions to make it easier to work with
    # (The model outputs in a specific format that we need to simplify)
    if forecast_arr.ndim == 3 and forecast_arr.shape[0] == 1:
        forecast_arr = forecast_arr[0]  # Remove batch dimension
    if forecast_arr.ndim == 2 and forecast_arr.shape[0] == 1:
        forecast_arr = forecast_arr[0]  # Remove feature dimension if only one feature
    
    # Make the predictions more practical:
    # 1. Round to whole numbers (can't have 1.5 items)
    # 2. Remove negative values (can't have -2 items)
    # 3. Convert to integers
    forecast_arr = np.round(np.clip(forecast_arr, 0, None)).astype(int)

    # Make sure our dates are in the right format
    print(f"[DEBUG] date_series dtype before conversion: {getattr(date_series, 'dtype', type(date_series))}")
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, dayfirst=True)

    # Get the last date from our input data
    last_date = date_series.iloc[-1]
    
    # Determine how many days we're actually forecasting
    forecast_length_actual = forecast_arr.shape[-1]
    
    # Create a series of dates for our forecast, starting from the day after our last data point
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),  # Start from the next day
        periods=forecast_length_actual,         # How many days to create
        freq='D'                               # Daily frequency
    )

    # Create a dictionary to hold our forecast data
    data = {"DATE": forecast_dates.strftime("%d/%m/%Y")}  # Format dates as DD/MM/YYYY
    
    # Add each feature's forecast values to the dictionary
    for i, feature in enumerate(feature_cols):
        # Skip if we don't have forecast data for this feature
        if i >= forecast_arr.shape[0]:
            continue
            
        # Get the forecast values for this feature
        if forecast_arr.ndim == 1:
            arr = forecast_arr[:len(forecast_dates)]  # Single feature case
        else:
            arr = forecast_arr[i, :len(forecast_dates)]  # Multiple features case
            
        # Only add if the number of values matches the number of dates
        if len(arr) != len(forecast_dates):
            continue
            
        # Add this feature's forecast to our data dictionary
        data[feature] = arr
    
    # Create a DataFrame (table) from our dictionary
    df_forecast = pd.DataFrame(data)
    
    # Convert the DataFrame to a CSV string with the same format as the input
    forecast_csv = df_forecast.to_csv(index=False, sep=';', decimal=',', float_format='%.1f')
    
    print("[INFO] Forecast CSV generated successfully!")
    # Return the CSV string so it can be used by the application
    return forecast_csv
