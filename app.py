import streamlit as st
import pandas as pd
import pickle
import os

# Set up the Streamlit app
st.title("NYC Accidents Analysis and Prediction")
st.sidebar.title("Navigation")

# Load data
DATA_PATH = 'NYC Accidents 2020.csv'
MODEL_PATH = 'sarima_model.pkl'

def load_data(data_path):
    try:
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {data_path}")
        return None

# Load SARIMA model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None

# Display data
st.header("Exploratory Data Analysis")
data = load_data(DATA_PATH)
if data is not None:
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    st.write("### Basic Statistics")
    st.write(data.describe())

# SARIMA model predictions
st.header("Accident Predictions")
model = load_model(MODEL_PATH)
if model is not None:
    st.write("### Predict Future Accidents")

    # Interactive input for prediction
    forecast_steps = st.slider("Select number of days to forecast", min_value=1, max_value=30, value=7)

    if st.button("Generate Forecast"):
        try:
            forecast = model.forecast(steps=forecast_steps)
            st.write(f"Forecast for next {forecast_steps} days:")
            st.line_chart(forecast)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.write("---")
st.markdown("Developed by AKASH RAJPURIA")
