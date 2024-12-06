import pandas as pd

# Load the Excel file
file_path = r"C:\Users\vimal\OneDrive\Desktop\Data_cleaned.xlsx"  # Update with the correct file path if needed
excel_data = pd.ExcelFile(file_path)

# Load the "Data" sheet
data = excel_data.parse('Data')

# Step 1: Clean the column names
data.columns = data.columns.str.strip()

# Step 2: Drop rows with missing critical values
cleaned_data = data.dropna(subset=["Date of Purchase", "Count"])

# Step 3: Convert "Cost_Price" to numeric and handle invalid entries
cleaned_data['Cost_Price'] = pd.to_numeric(cleaned_data['Cost_Price'], errors='coerce')

# Step 4: Save the cleaned data to a new Excel file
output_path = r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\EDA\Cleaned_Data.xlsx"
cleaned_data.to_excel(output_path, index=False)

print(f"Cleaned data saved to {output_path}")
