# Warehouse Forecaster

A powerful time series forecasting application for warehouse inventory management, built with Streamlit and the Toto forecasting model.

## Overview

Warehouse Forecaster is a web application that allows users to upload historical sales/inventory data and generate accurate forecasts for future periods. The application uses the Toto model, a state-of-the-art time series forecasting model from Datadog, to provide reliable predictions.

## Features

- **Easy-to-use interface**: Upload your data and get forecasts with just a few clicks
- **Interactive visualization**: View your forecast results as both tables and charts
- **Configurable forecast horizon**: Choose how many days into the future you want to forecast
- **Example data included**: Try the application with included sample data
- **Secure processing**: Your data is processed securely and not stored

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd r2bit_WarehouseForecaster
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit app with:

```
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501` by default.

## Data Format

Your input CSV file must:
- Use semicolon (`;`) as the separator
- Use comma (`,`) as the decimal separator
- Contain two columns:
  - `DATE`: Formatted as DD/MM/YYYY
  - `VALUE`: Numeric values representing sales or inventory quantities

Example:
```
DATE;VALUE
01/01/2023;120
02/01/2023;135
03/01/2023;142
...
```

## Configuration

The application allows you to configure:

- **Forecast Length**: The number of days to forecast (7-90 days)

## Model Details

The forecasting is powered by the Toto model from Datadog, which is a transformer-based time series forecasting model. The model pipeline:

1. Loads the Toto model
2. Processes the input data
3. Runs prediction using TotoForecaster
4. Returns the forecast as a CSV string

## Security

- The application runs domain validation to ensure it's only accessible from authorized domains
- Data is processed in-memory and not stored
- No external API calls are made with your data

## Example Usage

1. Open the application in your web browser
2. Use the provided example data or upload your own CSV file
3. Set the desired forecast length using the slider
4. Click "Generate Forecast"
5. View the results and download the forecast as a CSV file

## Troubleshooting

If you encounter issues:

1. Ensure your CSV file follows the required format
2. Check that your data is chronologically ordered
3. Verify that you have all required dependencies installed

## License

MIT License

## Contact

For support or inquiries, please contact [R2Talk](https://waapp.r2talk.com.br).
