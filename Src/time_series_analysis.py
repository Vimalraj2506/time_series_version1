import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data
file_path = r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\EDA\Cleaned_Data.xlsx"  # Replace with your file name
data = pd.read_excel(file_path)

# Aggregate sales by 'Date of Purchase' and sum the 'Count' column
aggregated_data = data.groupby('Date of Purchase')['Count'].sum().reset_index()

# Set the 'Date of Purchase' column as the index for time-series analysis
aggregated_data.set_index('Date of Purchase', inplace=True)

# Ensure the data is in proper time-series format
aggregated_data.index = pd.to_datetime(aggregated_data.index)

# Fill any missing dates with 0 sales (optional, depending on your data)
aggregated_data = aggregated_data.asfreq('D', fill_value=0)

# Decompose the time series into trend, seasonality, and residuals
result = seasonal_decompose(aggregated_data['Count'], model='additive', period=30)  # Adjust period as needed

# Plot the decomposed components with extra details
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
fig.suptitle('Detailed Decomposition of Sales Data', fontsize=18, weight='bold', color='darkblue')

# Plot original data
axes[0].plot(aggregated_data.index, aggregated_data['Count'], label='Original Sales Data', color='blue')
axes[0].set_title('Original Sales Data', fontsize=14)
axes[0].legend(loc='upper left')
axes[0].grid(True)

# Plot trend
axes[1].plot(result.trend, label='Trend Component', color='orange')
axes[1].set_title('Trend Component', fontsize=14)
axes[1].legend(loc='upper left')
axes[1].grid(True)

# Plot seasonality
axes[2].plot(result.seasonal, label='Seasonality Component', color='green')
axes[2].set_title('Seasonality Component', fontsize=14)
axes[2].legend(loc='upper left')
axes[2].grid(True)

# Plot residuals
axes[3].plot(result.resid, label='Residuals (Noise)', color='red')
axes[3].set_title('Residuals (Noise)', fontsize=14)
axes[3].legend(loc='upper left')
axes[3].grid(True)

# Add x-axis label for shared axis
axes[3].set_xlabel('Date', fontsize=12)

# Tight layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make room for suptitle
plt.show()

# Display message
print("Decomposition completed with enhanced visualization.")
#Autocorrelation Function (ACF) Plot for Residuals
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import os

# Set the file path and check if it exists
file_path = r"C:\Users\vimal\OneDrive\Documents\AI Assingments\Capstone\EDA\Cleaned_Data.xlsx"
if not os.path.exists(file_path):
    print(f"Error: The file {file_path} does not exist.")
    exit()  # Stop the script if the file does not exist

# Load the data
data = pd.read_excel(file_path)

# Aggregate sales by 'Date of Purchase' and sum the 'Count' column
aggregated_data = data.groupby('Date of Purchase')['Count'].sum().reset_index()

# Set the 'Date of Purchase' column as the index for time-series analysis
aggregated_data.set_index('Date of Purchase', inplace=True)

# Ensure the data is in proper time-series format
aggregated_data.index = pd.to_datetime(aggregated_data.index)

# Fill any missing dates with 0 sales (optional, depending on your data)
aggregated_data = aggregated_data.asfreq('D', fill_value=0)

# Decompose the time series into trend, seasonality, and residuals
result = seasonal_decompose(aggregated_data['Count'], model='additive', period=30)

# Visualize all components of the decomposition
decomp_result = result.plot()
plt.suptitle("Time Series Decomposition", fontsize=16)
plt.show()

# Plot residuals (noise) component
residuals = result.resid

# Plot the ACF of residuals to check for autocorrelation
plt.figure(figsize=(12, 6))
plot_acf(residuals.dropna(), lags=50)  # Drop NA values before plotting
plt.title("Autocorrelation Function (ACF) of Residuals", fontsize=16)
plt.show()
