import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load ARIMA model
with open("models/ARIMA_Model_Store1041.t5", "rb") as f:
    arima_model = pickle.load(f)

# Load SARIMA model
with open("models/SARIMA_Model_Store1041.t5", "rb") as f:
    sarima_model = pickle.load(f)

# Function to perform ARIMA forecasting
def arima_forecast(data):
    forecast = arima_model.forecast(steps=len(data))
    return forecast

# Function to perform SARIMA forecasting
def sarima_forecast(data):
    forecast = sarima_model.forecast(steps=len(data))
    return forecast

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def main():
    st.title('Real-time Forecasting Web App')

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file for forecasting", type="csv")
    
    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded file:")

            # Validate data format
            if (not any(col.lower() == 'date' for col in df.columns) or 
                not any(col.lower() == 'store_customers' for col in df.columns) or
                len(df.columns) != 2):
                st.error("""Error: Data format is not valid. 
                Please ensure the CSV file contains columns 'Date' and 'Store_Customers' for forecasting.""")
            else:
                # Perform forecasting based on user selection
                inference_method = st.radio("File Uploaded, Choose inference method:", ('ARIMA', 'SARIMA'))
                data = df['Store_Customers'].values.astype(float)

                if st.button("Generate Forecast"):
                    if inference_method == 'ARIMA':
                        forecast = arima_forecast(data)
                    elif inference_method == 'SARIMA':
                        forecast = sarima_forecast(data)

                    forecast_df = pd.DataFrame({
                        'Date': df['Date'],  # Use dates from the uploaded file
                        'Actual': df['Store_Customers'],  # Actual store customers
                        'Forecast': forecast.values #forecast from arima or sarima
                    })

                    st.write(inference_method + " Forecast:")
                    st.write(forecast_df)

        else:
            st.error("Error: File format not allowed. Please upload a CSV file.")

if __name__ == "__main__":
    main()
