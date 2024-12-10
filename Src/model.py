import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'])
    data['YearMonth'] = data['Date of Purchase'].dt.to_period('M')
    monthly_data = data.groupby('YearMonth')['Count'].sum().reset_index()
    monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.to_timestamp()
    monthly_data.set_index('YearMonth', inplace=True)

    # Add seasonality features
    monthly_data['sin_month'] = np.sin(2 * np.pi * monthly_data.index.month / 12)
    monthly_data['cos_month'] = np.cos(2 * np.pi * monthly_data.index.month / 12)
    return monthly_data

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # Predict only 'Count'
    return np.array(X), np.array(y)

# Build LSTM model
def build_lstm_model(sequence_length, n_features):
    model = Sequential([
        Bidirectional(LSTM(128, activation='tanh', return_sequences=True, input_shape=(sequence_length, n_features))),
        Dropout(0.2),
        LSTM(64, activation='tanh'),
        Dropout(0.2),
        Dense(1)  # Output layer for predicting 'Count'
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train, validate, and save the model
def train_and_validate(file_path, sequence_length=12, future_steps=6):
    # Load and preprocess the data
    data = load_and_preprocess_data(file_path)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Save scaler for future use
    joblib.dump(scaler, 'scaler.pkl')

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train the model
    model = build_lstm_model(sequence_length, X.shape[2])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model.save('final_lstm_model.keras')

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_test_rescaled = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1))])
    )[:, 0]
    y_pred_rescaled = scaler.inverse_transform(
        np.hstack([y_pred, np.zeros((len(y_pred), scaled_data.shape[1] - 1))])
    )[:, 0]

    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

    # Future forecasting
    last_sequence = scaled_data[-sequence_length:, :]
    future_predictions = []

    for step in range(future_steps):
        pred = model.predict(last_sequence.reshape(1, sequence_length, -1))
        future_predictions.append(pred[0, 0])
        next_step = np.hstack([pred[0, 0], last_sequence[-1, 1:]])
        last_sequence = np.vstack([last_sequence[1:], next_step])

    # Rescale future predictions
    future_predictions_rescaled = scaler.inverse_transform(
        np.hstack([
            np.array(future_predictions).reshape(-1, 1),
            np.zeros((len(future_predictions), scaled_data.shape[1] - 1))
        ])
    )[:, 0]

    # Get predicted month names
    last_date = data.index[-1]
    future_months = pd.date_range(last_date, periods=future_steps + 1, freq='M')[1:]
    predicted_months = [date.strftime('%B %Y') for date in future_months]

    return model, history, future_predictions_rescaled, y_test_rescaled, y_pred_rescaled, predicted_months

# Visualize results
def visualize_results(data, y_test_rescaled, y_pred_rescaled, future_predictions_rescaled, predicted_months):
    # Plot historical and forecasted sales
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test_rescaled):], y_test_rescaled, label="Historical Sales", color="blue")
    plt.plot(data.index[-len(y_test_rescaled):], y_pred_rescaled, label="Predicted Sales", color="orange")
    future_dates = pd.date_range(data.index[-1], periods=len(future_predictions_rescaled) + 1, freq='M')[1:]
    plt.plot(future_dates, future_predictions_rescaled, label="Forecasted Sales", color="green")
    plt.title("Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.show()

    # Display future predictions
    future_df = pd.DataFrame({'Month': predicted_months, 'Predicted Sales': future_predictions_rescaled})
    print("\nFuture Predictions:")
    print(future_df)

# Main execution
if __name__ == "__main__":
    file_path = r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\EDA\Cleaned_Data.xlsx"
    model, history, predictions, y_test_rescaled, y_pred_rescaled, predicted_months = train_and_validate(file_path)
    print("Predicted Sales for Future Months:", predictions)
    data = load_and_preprocess_data(file_path)
    visualize_results(data, y_test_rescaled, y_pred_rescaled, predictions, predicted_months)
