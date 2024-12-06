import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Title and Description
st.title("Time Series Sales Prediction")
st.write("This app predicts future sales using an LSTM model trained on historical data.")

# Step 1: Upload File
uploaded_file = st.file_uploader("Upload your dataset (Excel file):", type=["xlsx"])

if uploaded_file:
    # Step 2: Load and Display Data
    data = pd.read_excel(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(data.head())

    # Step 3: Aggregate Sales Data by Month
    data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'])
    monthly_aggregated_data = data.groupby(data['Date of Purchase'].dt.to_period('M'))['Count'].sum().reset_index()
    monthly_aggregated_data['Date of Purchase'] = monthly_aggregated_data['Date of Purchase'].dt.to_timestamp()
    monthly_aggregated_data.set_index('Date of Purchase', inplace=True)
    
    st.write("### Monthly Aggregated Data")
    st.line_chart(monthly_aggregated_data['Count'])

    # Step 4: Normalize Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(monthly_aggregated_data[['Count']])

    # Step 5: Create Sequences
    def create_sequences(data, sequence_length=6):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    sequence_length = 6
    X, y = create_sequences(scaled_data)

    # Step 6: Load Trained Model
    model = load_model("lstm_sales_model.h5")
    st.success("Model Loaded Successfully!")

    # Step 7: Predict Future Sales
    future_months = st.slider("Number of Months to Predict", min_value=1, max_value=24, value=6)
    last_sequence = scaled_data[-sequence_length:]
    future_predictions = []

    for _ in range(future_months):
        next_prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
        future_predictions.append(next_prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction[0, 0]).reshape(-1, 1)

    # Inverse Transform Predictions
    future_predictions_original = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Generate Future Dates
    last_date = monthly_aggregated_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='M')

    # Step 8: Display Predictions
    st.write("### Future Sales Predictions")
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": future_predictions_original.flatten()
    }).set_index("Date")
    st.line_chart(future_df["Predicted Sales"])

    # Step 9: Download Predictions
    st.write("### Download Predictions")
    st.download_button(
        label="Download Predictions as CSV",
        data=future_df.to_csv().encode('utf-8'),
        file_name="future_sales_predictions.csv",
        mime="text/csv"
    )
