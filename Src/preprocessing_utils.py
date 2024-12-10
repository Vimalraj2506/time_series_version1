import pandas as pd
import numpy as np
import os

def preprocess_dataset(file_path, output_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # Load the dataset
    print(f"Loading data from: {file_path}")
    data = pd.read_excel(file_path)
    
    # Display the first few rows to verify the structure
    print("Initial Data Sample:")
    print(data.head())
    
    # Keep only relevant columns
    if 'Date of Purchase' not in data.columns or 'Count' not in data.columns:
        raise KeyError("Required columns 'Date of Purchase' or 'Count' are missing in the dataset.")
    
    data = data[['Date of Purchase', 'Count']]
    
    # Drop rows with missing or invalid values
    data.dropna(subset=['Date of Purchase', 'Count'], inplace=True)
    
    # Ensure 'Count' is numeric
    data['Count'] = pd.to_numeric(data['Count'], errors='coerce')
    data.dropna(subset=['Count'], inplace=True)  # Drop rows where 'Count' is non-numeric
    
    # Convert 'Date of Purchase' to datetime
    data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'], errors='coerce')
    data.dropna(subset=['Date of Purchase'], inplace=True)
    
    # Aggregate sales by month
    data['YearMonth'] = data['Date of Purchase'].dt.to_period('M')
    monthly_data = data.groupby('YearMonth')['Count'].sum().reset_index()
    monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.to_timestamp()
    monthly_data.set_index('YearMonth', inplace=True)
    
    # Add seasonality features
    monthly_data['sin_month'] = np.sin(2 * np.pi * (monthly_data.index.month / 12))
    monthly_data['cos_month'] = np.cos(2 * np.pi * (monthly_data.index.month / 12))
    
    # Save the preprocessed data to a file
    monthly_data.to_csv(output_path)
    print(f"Preprocessed data saved to {output_path}")
    
    return monthly_data

# Example usage
if __name__ == "__main__":
    input_path = r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\EDA\Cleaned_Data.xlsx"
    output_path = r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\EDA\Preprocessed_Data.csv"
    
    try:
        preprocess_dataset(input_path, output_path)
    except Exception as e:
        print(f"An error occurred: {e}")
