import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scaler
model = load_model(r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\time_series_version1\Src\final_lstm_model.keras")
scaler = joblib.load(r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\time_series_version1\Src\scaler.pkl")

# Function to handle file upload and make predictions
def upload_and_predict():
    # Open file dialog to select a file
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")],
        title="Select a CSV or Excel File"
    )
    
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    try:
        # Load the uploaded file
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            messagebox.showerror("Error", "Unsupported file format.")
            return

        # Ensure the required 'Date of Purchase' and 'Count' columns are present
        if 'Date of Purchase' not in data.columns or 'Count' not in data.columns:
            messagebox.showerror("Error", "'Date of Purchase' or 'Count' column is missing in the file.")
            return
        
        # Preprocess the data
        data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'])
        data['YearMonth'] = data['Date of Purchase'].dt.to_period('M')
        monthly_data = data.groupby('YearMonth')['Count'].sum().reset_index()
        monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.to_timestamp()
        monthly_data.set_index('YearMonth', inplace=True)
        monthly_data['sin_month'] = np.sin(2 * np.pi * monthly_data.index.month / 12)
        monthly_data['cos_month'] = np.cos(2 * np.pi * monthly_data.index.month / 12)

        # Scale the data
        scaled_data = scaler.transform(monthly_data)

        # Predict future sales
        sequence_length = 12
        last_sequence = scaled_data[-sequence_length:, :]
        future_steps = 6
        future_predictions = []

        for _ in range(future_steps):
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

        # Display the predictions
        result_text = "Future Predictions:\n"
        for i, pred in enumerate(future_predictions_rescaled, start=1):
            result_text += f"Month {i}: {pred:.2f}\n"

        result_label.config(text=result_text)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# Create the GUI
app = tk.Tk()
app.title("Sales Forecast Application")
app.geometry("500x400")

# Create and place widgets
welcome_label = tk.Label(app, text="Welcome to the Sales Forecast Application", font=("Arial", 14))
welcome_label.pack(pady=10)

upload_button = tk.Button(app, text="Upload Sales Data File", command=upload_and_predict, font=("Arial", 12))
upload_button.pack(pady=20)

result_label = tk.Label(app, text="", font=("Arial", 12), justify="left")
result_label.pack(pady=20)

# Run the app
app.mainloop()
