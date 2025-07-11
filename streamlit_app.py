"""
# Warehouse Forecaster - Streamlit Web Application

## What is this application?
This is a beginner-friendly web application that helps businesses predict their future 
warehouse inventory or sales data. It uses a machine learning model called 'Toto' to 
analyze historical data patterns and generate forecasts.

## How does it work?
1. Users upload their historical data as a CSV file
2. The app validates the data format
3. Users select how many days into the future they want to forecast
4. The app processes the data and generates predictions
5. Results are displayed as both tables and interactive charts

## Key Features:
- **Easy Data Upload**: Upload CSV files with your historical data
- **Flexible Forecasting**: Choose any forecast period from 1 to 90 days
- **Visual Results**: See your data and forecasts in interactive charts
- **Example Data**: Try the app with pre-loaded example data
- **Downloadable Results**: Save your forecast as a CSV file
- **Secure Access**: Domain-restricted for security

## Technical Details:
The application uses Streamlit for the web interface and connects to our custom
warehouse_forecast module which implements the Datadog Toto forecasting model.

Author: R2Talk Team
Date: July 2025
"""

# Import necessary libraries
import streamlit as st  # The main Streamlit library for creating web apps
import pandas as pd   # For data manipulation and analysis
import io            # For handling input/output operations
from forecast.warehouse_forecast import execute_forecast  # Our custom forecasting function
import streamlit.components.v1 as components  # For custom HTML/JS components

# --- Custom CSS to clean up the user interface ---
# This CSS code hides various Streamlit elements we don't need
# Beginners: CSS is a styling language used to control the appearance of web elements
hide_streamlit_style = """
<style>
/* Hide the default Streamlit menu */
#MainMenu {visibility: hidden;}

/* Hide the footer and header */
footer {visibility: hidden;}
header {visibility: hidden;}

/* Set primary button color to blue */
button[kind="primary"],
button.st-emotion-cache-19rxjzo,
button.st-emotion-cache-1cpxm7s,
button.st-bw,
div.stButton button[data-baseweb="button"][kind="primary"] {
    background-color: #0066cc !important;
    border-color: #0066cc !important;
}

/* Button hover state - slightly darker blue */
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
    Check if the uploaded CSV file has the correct format for our forecasting model.
    
    What this function does:
    1. Tries to read the uploaded file as a UTF-8 text file
    2. Checks if it's a valid CSV with semicolons (;) as separators
    3. Verifies it has the required columns: DATE and VALUE
    
    Parameters:
    -----------
    file : UploadedFile
        The file uploaded by the user through Streamlit's file_uploader
    
    Returns:
    --------
    tuple with three elements:
        - is_valid (bool): True if the file is correctly formatted
        - error_message (str): Empty if valid, otherwise explains what's wrong
        - csv_content (str): The file content as text if valid, None otherwise
    
    Example:
    --------
    >>> is_valid, error, content = validate_csv(uploaded_file)
    >>> if is_valid:
    >>>     # Process the content
    >>> else:
    >>>     st.error(error)
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
    The main function that creates our Streamlit web application.
    
    How a Streamlit app works:
    -------------------------
    - Streamlit runs this function from top to bottom
    - Each Streamlit command (st.something) adds an element to the web page
    - When a user interacts with an element, the entire script runs again
    
    What this function does step by step:
    -----------------------------------
    1. Sets up the page (title, icon, layout)
    2. Checks if the user is accessing from an allowed domain
    3. Creates the sidebar with input controls
    4. Creates the main panel with instructions and results
    5. Handles file uploads and validation
    6. Processes data and generates forecasts
    7. Displays results as tables and charts
    8. Provides download options for results
    
    App Structure:
    -------------
    - SIDEBAR: Contains inputs (forecast length, file upload, buttons)
    - MAIN PANEL: Contains instructions and forecast results
    
    Data Flow:
    ---------
    Upload/Example Data → Validate → Process → Display Results
    
    State Management:
    ---------------
    We use st.session_state to remember data between user interactions
    """
    
    # --- Page Configuration ---
    # This must be the first Streamlit command in your app
    # It sets up the browser tab title, icon, and layout
    st.set_page_config(
        page_title="Warehouse Forecaster",  # The title shown in the browser tab
        page_icon="📊",                    # The icon shown in the browser tab
        layout="wide"                      # Use the full width of the screen
    )
    
    # --- Initialize Session State Variables ---
    # Initialize session state variables if they don't exist
    if 'forecast_result' not in st.session_state:
        st.session_state.forecast_result = None
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    # Note: Domain restriction has been removed to allow access from any domain
    
    # --- Apply our custom CSS ---
    # This injects our CSS into the page to customize the appearance
    # The unsafe_allow_html=True parameter is needed to allow HTML/CSS code
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # --- Page Title ---
    # st.title adds a large, prominent title to the page
    st.title("📊 Warehouse Forecaster")
    
    # st.caption adds a smaller, gray text below the title
    st.caption("Upload a CSV file with historical data to generate a forecast.")

    # --- Layout: Sidebar for input, main for output ---
    # The 'with st.sidebar:' creates a sidebar on the left side of the app
    # Everything indented under this will appear in the sidebar
    
    # Initialize variables that need to be accessed outside the sidebar
    forecast_length = 30  # Default value
    uploaded_file = None
    run_forecast = False
    load_example = False
    
    # Import os module to work with file paths
    import os
    
    # Find the path to our example CSV file
    example_path = os.path.join(os.path.dirname(__file__), "example", "CoffeeSales.csv")
    
    # Read the example file content
    with open(example_path, "r") as f:
        example_content = f.read()
    
    # Sidebar for inputs
    with st.sidebar:
            
            # --- Forecast Length Input ---
            # st.number_input creates a field where users can type a number or use +/- buttons
            # This replaces the slider we had before for more precise input
            forecast_length = st.number_input(
                "Forecast Length (days)",           # Label shown above the input
                min_value=1,                       # Minimum allowed value
                max_value=90,                      # Maximum allowed value
                value=30,                          # Default value
                step=1,                            # Increment when using +/- buttons
                help="Enter the number of days to forecast (between 1 and 90)"  # Tooltip text
            )
            
            # --- Example Data Section ---
            # This section provides example data for users to try the app
            
            # Button to load example data directly into the app
            # When clicked, this will process the example data without requiring upload
            load_example = st.button(
                "Use CoffeeSales Example Data",     # Button text
                use_container_width=True,           # Make button full width of sidebar
                key="load_example"                  # Unique identifier for this button
            )
            
            # Button to download the example file to the user's computer
            # This helps users understand the required format
            st.download_button(
                label="Download CoffeeSales Example File",  # Button text
                data=example_content,                      # The file content to download
                file_name="CoffeeSales.csv",               # The default filename when downloading
                mime="text/csv",                           # File type (CSV in this case)
                use_container_width=True                   # Make button full width of sidebar
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
            
            # --- File Upload Section ---
            # This creates a file upload area where users can drag & drop or select files
            uploaded_file = st.file_uploader(
                "Upload CSV File",                # Label shown above the uploader
                type=["csv"],                     # Only allow CSV files
                accept_multiple_files=False      # Only allow one file at a time
            )
            
            # --- Generate Forecast Button ---
            # This is the main action button that processes the uploaded file
            # The type="primary" makes it blue and stand out as the main action
            run_forecast = st.button(
                "Generate Forecast",              # Button text
                type="primary",                   # Make it a primary (highlighted) button
                use_container_width=True         # Make button full width of sidebar
            )

    # --- Main Area: Instructions and Results ---
    # Everything outside the sidebar appears in the main panel
    
    # Instructions section with a subheader
    st.subheader("Instructions")
    
    # Markdown allows us to format text with lists, bold, etc.
    # Here we provide step-by-step instructions for users
    st.markdown("""
    1. **IMPORTANT**: This application does not work on mobile devices. A desktop or laptop is required.
    2. Prepare a CSV file with two columns: `DATE` and `VALUE`
    3. Use semicolon (;) as separator and comma (,) as decimal
    4. Format dates as DD/MM/YYYY
    5. Upload the file and click 'Generate Forecast'
    """)
    
    # Results section header
    st.subheader("Forecast Results")

    # --- Initialize Session State ---
    # Session state is how Streamlit remembers values between reruns
    # Here we're setting up variables to store our forecast results
    
    # If this is the first time running the app, set up our state variables
    if "forecast_result" not in st.session_state:
        st.session_state.forecast_result = None    # Will store the CSV forecast result
        st.session_state.forecast_df = None        # Will store the pandas DataFrame
        st.session_state.error_message = None      # Will store any error messages

    # --- Process User-Uploaded File ---
    # This section runs when the user clicks the "Generate Forecast" button
    if run_forecast:
        # First check if a file was uploaded
        if not uploaded_file:
            # Show a warning if no file was uploaded
            st.warning("Please upload a CSV file first or use the example data.")
            return  # Exit the function early
        
        # Validate the uploaded CSV file format
        is_valid, error_message, csv_content = validate_csv(uploaded_file)
        
        # Handle invalid files
        if not is_valid:
            # Display the error message to the user
            st.error(error_message)
            # Store the error in session state so it persists between reruns
            st.session_state.error_message = error_message
        else:
            # File is valid, proceed with forecasting
            # Show a spinner while processing (gives visual feedback)
            with st.spinner("Generating forecast..."):
                try:
                    # Call our forecasting function with the CSV content
                    forecast_csv = execute_forecast(csv_content, forecast_length=forecast_length)
                    
                    # Store the raw CSV result in session state
                    st.session_state.forecast_result = forecast_csv
                    
                    # Convert the CSV to a DataFrame for display and analysis
                    forecast_df = pd.read_csv(io.StringIO(forecast_csv), sep=';', decimal=',')
                    st.session_state.forecast_df = forecast_df
                    st.session_state.error_message = None  # Clear any previous errors
                    
                except Exception as e:
                    # Handle any errors that occur during forecasting
                    st.error(f"Error generating forecast: {str(e)}")
                    st.session_state.error_message = str(e)

    # --- Display Results Section ---
    # This section shows either errors, results, or instructions based on the app state
    
    # If there was an error, display it
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    
    # If we have forecast results, display them
    elif st.session_state.forecast_result is not None:
        # --- Table View ---
        # Display the forecast data as an interactive table
        st.subheader("Forecast Data")
        # The use_container_width=True makes the table use the full width available
        st.dataframe(st.session_state.forecast_df, use_container_width=True)
        
        # --- Chart View ---
        # Display the forecast as a line chart for visual analysis
        st.subheader("Forecast Chart")
        # Make a copy of the data to avoid modifying the original
        chart_data = st.session_state.forecast_df.copy()
        # Convert the DATE column to datetime format (dayfirst=True means DD/MM/YYYY format)
        chart_data['DATE'] = pd.to_datetime(chart_data['DATE'], dayfirst=True)
        # Set the DATE column as the index for the chart
        chart_data = chart_data.set_index('DATE')
        # Create an interactive line chart
        st.line_chart(chart_data, use_container_width=True)
        
        # --- Download Option ---
        # Provide a button to download the forecast results as a CSV file
        st.download_button(
            label="Download Forecast CSV",           # Button text
            data=st.session_state.forecast_result,  # The data to download (CSV string)
            file_name="forecast_results.csv",       # Default filename
            mime="text/csv",                        # File type
        )
    else:
        # If no forecast has been generated yet, show instructions
        st.info("Upload a CSV file and click 'Generate Forecast' to see results here.")

    # --- Footer ---
    # Add a separator line
    st.markdown("---")
    # Add a security notice
    st.info("Your data is processed securely and not stored.")

# --- App Entry Point ---
# This is where Python starts executing when you run this file directly
# It checks if this file is being run directly (not imported by another file)
if __name__ == "__main__":
    # Call our main function to start the Streamlit app
    main()
