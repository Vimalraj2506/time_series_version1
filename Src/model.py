import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    
    # Aggregate sales by date
    aggregated_data = data.groupby('Date of Purchase')['Count'].sum().reset_index()
    aggregated_data['Date of Purchase'] = pd.to_datetime(aggregated_data['Date of Purchase'])
    aggregated_data = aggregated_data.sort_values('Date of Purchase')
    aggregated_data.set_index('Date of Purchase', inplace=True)
    
    # Fill missing dates with forward fill then backward fill
    aggregated_data = aggregated_data.asfreq('D')
    # Using newer pandas methods instead of deprecated fillna(method=)
    aggregated_data['Count'] = aggregated_data['Count'].ffill().bfill()
    
    # Add time-based features
    aggregated_data['day_of_week'] = aggregated_data.index.dayofweek
    aggregated_data['month'] = aggregated_data.index.month
    aggregated_data['day_of_month'] = aggregated_data.index.day
    
    return aggregated_data

# Create sequences with multiple features
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # Only predict the Count column
    return np.array(X), np.array(y)

# Build enhanced LSTM model
def build_model(sequence_length, n_features):
    model = Sequential()
    
    # Add Input layer explicitly
    model.add(Input(shape=(sequence_length, n_features)))
    
    # LSTM layers
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='huber')
    return model

# Main execution
def train_and_predict(file_path, sequence_length=30, future_steps=600):
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    
    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences with all features
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train model
    model = build_model(sequence_length, X.shape[2])
    
    # Add callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')  # Changed to .keras extension
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Make predictions
    predicted = model.predict(X_test)
    
    # Prepare for inverse transform
    pred_copies = np.repeat(predicted, scaled_data.shape[1], axis=1)
    pred_rescaled = scaler.inverse_transform(pred_copies)[:, 0]
    
    y_test_copies = np.repeat(y_test.reshape(-1, 1), scaled_data.shape[1], axis=1)
    y_test_rescaled = scaler.inverse_transform(y_test_copies)[:, 0]
    
    # Plot test predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Actual', color='blue')
    plt.plot(pred_rescaled, label='Predicted', color='red')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    
    # Future predictions
    last_sequence = scaled_data[-sequence_length:]
    future_predictions = []
    
    for _ in range(future_steps):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, -1))
        
        # Create a full feature set for the next prediction
        next_date = data.index[-1] + pd.Timedelta(days=len(future_predictions) + 1)
        next_features = np.zeros((1, scaled_data.shape[1]))
        next_features[0, 0] = next_pred[0, 0]  # Sales prediction
        next_features[0, 1] = next_date.dayofweek / 6  # Normalized day of week
        next_features[0, 2] = next_date.month / 12  # Normalized month
        next_features[0, 3] = next_date.day / 31  # Normalized day of month
        
        future_predictions.append(next_features[0])
        last_sequence = np.vstack([last_sequence[1:], next_features])
    
    # Rescale predictions
    future_predictions = np.array(future_predictions)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)[:, 0]
    
    # Plot historical and future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Count'], label='Historical', color='blue')
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps)
    plt.plot(future_dates, future_predictions_rescaled, label='Forecast', color='green')
    plt.title('Historical and Forecasted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    
    return model, history, future_predictions_rescaled

# Example usage
file_path = r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\EDA\Cleaned_Data.xlsx"
model, history, predictions = train_and_predict(file_path)