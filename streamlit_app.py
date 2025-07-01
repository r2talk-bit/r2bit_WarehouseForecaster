"""
Warehouse Forecaster - Streamlit Web Application

This application provides a user-friendly web interface for forecasting warehouse inventory
or sales data using the Toto forecasting model. Users can upload their historical data
in CSV format or use the provided example data to generate forecasts.

Features:
- CSV file upload with validation
- Configurable forecast horizon (7-90 days)
- Interactive data visualization
- Example data for demonstration
- Downloadable forecast results
- Domain-restricted access for security

The application is built with Streamlit and integrates with the warehouse_forecast module
which handles the actual forecasting using the Datadog Toto model.

Author: R2Talk Team
Date: July 2025
"""

import streamlit as st
import pandas as pd
import io
from forecast.warehouse_forecast import execute_forecast

# Hide the hamburger menu, Deploy button, and sidebar close button
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="collapsedControl"] {display: none;}

/* Override primary button color to blue */
button[kind="primary"],
button.st-emotion-cache-19rxjzo,
button.st-emotion-cache-1cpxm7s,
button.st-bw,
div.stButton button[data-baseweb="button"] {
    background-color: #0066cc !important;
    border-color: #0066cc !important;
}

button[kind="primary"]:hover,
button.st-emotion-cache-19rxjzo:hover,
button.st-emotion-cache-1cpxm7s:hover,
button.st-bw:hover,
div.stButton button[data-baseweb="button"]:hover {
    background-color: #0052a3 !important;
    border-color: #0052a3 !important;
}
</style>
"""

# --- Helper Functions ---

def validate_csv(file):
    """
    Validate that the uploaded CSV file meets the required format specifications.
    
    Args:
        file: A file-like object containing the uploaded CSV data
        
    Returns:
        tuple: A 3-element tuple containing:
            - is_valid (bool): True if the file is valid, False otherwise
            - error_message (str): Empty string if valid, otherwise contains error details
            - csv_content (str): The file content as a string if valid, None otherwise
            
    Validation checks:
        - File can be decoded as UTF-8
        - File is valid CSV with semicolon separator and comma decimal
        - Required columns (DATE, VALUE) are present
    """
    try:
        # Read the file content
        content = file.getvalue().decode('utf-8')
        
        # Parse the CSV to check columns
        df = pd.read_csv(io.StringIO(content), sep=';', decimal=',')
        
        # Check if required columns exist
        required_columns = ['DATE', 'VALUE']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", None
        
        return True, "", content
    except Exception as e:
        return False, f"Error validating CSV file: {str(e)}", None

# --- Main App ---

def main():
    """
    Main application function that sets up and runs the Streamlit web interface.
    
    This function handles:
    1. Page configuration and styling
    2. Domain validation for security
    3. User interface layout and components
    4. File upload and validation
    5. Example data processing
    6. Forecast generation and visualization
    7. Result download functionality
    
    The application follows a sidebar + main panel layout pattern with
    inputs in the sidebar and results/instructions in the main panel.
    
    State is managed through Streamlit's session_state to persist
    forecast results between interactions.
    """
    
    # Set page config first as it must be the first Streamlit command
    st.set_page_config(
        page_title="Warehouse Forecaster",
        page_icon="📊",
        layout="wide"
    )
    
    # Check if the request is coming from the allowed domain (only in production)
    try:
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if ctx and hasattr(ctx, 'request') and ctx.request:
            host = ctx.request.headers.get('host', '')
            if host and not host.endswith('r2talk.com.br'):
                st.error("Access denied: This application can only be accessed from r2talk.com.br domains")
                st.stop()
    except Exception as e:
        # If there's any error in domain checking, log it but continue
        import logging
        logging.warning(f"Domain check warning: {str(e)}")
    
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("📊 Warehouse Forecaster")
    st.caption("Upload a CSV file with historical data to generate a forecast.")

    # --- Layout: Sidebar for input, main for output ---
    with st.sidebar:
        
        # Forecast length slider - moved above the example button
        forecast_length = st.slider(
            "Forecast Length (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=1
        )
        
        # Example file buttons
        load_example = st.button("Use CoffeeSales Example Data", use_container_width=True, key="load_example")
        
        # Add download example file button to sidebar
        import os
        example_path = os.path.join(os.path.dirname(__file__), "example", "CoffeeSales.csv")
        with open(example_path, "r") as f:
            example_content = f.read()
        
        st.download_button(
            label="Download CoffeeSales Example File",
            data=example_content,
            file_name="CoffeeSales.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Process example data when Load Example Data is clicked
        if load_example:
            # Process example data immediately
            with st.spinner("Processing example data..."):
                try:
                    # Call the execute_forecast function with the example content
                    forecast_csv = execute_forecast(example_content, forecast_length=forecast_length)
                    
                    # Store the result in session state
                    st.session_state.forecast_result = forecast_csv
                    
                    # Also parse it as a DataFrame for display
                    forecast_df = pd.read_csv(io.StringIO(forecast_csv), sep=';', decimal=',')
                    st.session_state.forecast_df = forecast_df
                    st.session_state.error_message = None
                    
                    # Force a rerun to update the UI
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing example data: {str(e)}")
                    st.session_state.error_message = str(e)
        
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            accept_multiple_files=False
        )
        
        run_forecast = st.button("Generate Forecast", type="primary", use_container_width=True)

    # --- Main area: Results ---
    st.subheader("Instructions")
    
    st.markdown("""
    1. Prepare a CSV file with two columns: `DATE` and `VALUE`
    2. Use semicolon (;) as separator and comma (,) as decimal
    3. Format dates as DD/MM/YYYY
    4. Upload the file and click 'Generate Forecast'
    """)
    
    st.subheader("Forecast Results")

    if "forecast_result" not in st.session_state:
        st.session_state.forecast_result = None
        st.session_state.forecast_df = None
        st.session_state.error_message = None

    # Initialize session state variables if not exist
    # We don't need these anymore since we're processing example data immediately
    # but keeping the structure for possible future use
        
    if run_forecast:
        if not uploaded_file:
            st.warning("Please upload a CSV file first or use the example data.")
            return
        
        is_valid, error_message, csv_content = validate_csv(uploaded_file)
        
        if not is_valid:
            st.error(error_message)
            st.session_state.error_message = error_message
        else:
            with st.spinner("Generating forecast..."):
                try:
                    # Call the execute_forecast function with the CSV content
                    forecast_csv = execute_forecast(csv_content, forecast_length=forecast_length)
                    
                    # Store the result in session state
                    st.session_state.forecast_result = forecast_csv
                    
                    # Also parse it as a DataFrame for display
                    forecast_df = pd.read_csv(io.StringIO(forecast_csv), sep=';', decimal=',')
                    st.session_state.forecast_df = forecast_df
                    st.session_state.error_message = None
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    st.session_state.error_message = str(e)

    # Display results if available
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    
    elif st.session_state.forecast_result is not None:
        # Display the forecast as a table
        st.subheader("Forecast Data")
        st.dataframe(st.session_state.forecast_df, use_container_width=True)
        
        # Display the forecast as a chart
        st.subheader("Forecast Chart")
        chart_data = st.session_state.forecast_df.copy()
        chart_data['DATE'] = pd.to_datetime(chart_data['DATE'], dayfirst=True)
        chart_data = chart_data.set_index('DATE')
        st.line_chart(chart_data, use_container_width=True)
        
        # Provide download button for the forecast
        st.download_button(
            label="Download Forecast CSV",
            data=st.session_state.forecast_result,
            file_name="forecast_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a CSV file and click 'Generate Forecast' to see results here.")

    st.markdown("---")
    st.info("Your data is processed securely and not stored.")

if __name__ == "__main__":
    main()
