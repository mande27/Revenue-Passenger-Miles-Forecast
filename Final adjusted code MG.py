# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:59:29 2024

@author: maria
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:57:29 2024
@author: maria
"""

# Step 1: Import Libraries and Load Data
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import numpy as np

# Set the working directory
os.chdir(r'C:\Users\maria\OneDrive\Desktop\BANA 7350\Data')
print("Current Working Directory:", os.getcwd())

# Load the FMFINAL dataset
df = pd.read_csv('FMFINAL.csv', parse_dates=['DATE'], index_col='DATE')
df.rename(columns={
    'RPMD11': 'PassengerMiles',
    'CPIAUCSL': 'CPI',
    'CES4348100001': 'AirTransportationEmployees',
    'DSPIC96': 'DisposableIncome'
}, inplace=True)

# Remove any columns with "Unnamed" in the name
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Split data into training and test sets
train = df[:'2022']
test = df['2023':]

# Create a dictionary to store the metrics for each model
metrics = {}

# Function to calculate and store metrics
def calculate_metrics(model_name, actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mape = mean_absolute_percentage_error(actual, forecast) * 100
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, forecast)
    
    # Store the metrics in the dictionary
    metrics[model_name] = {
        'MAE': mae,
        'MAPE (%)': mape,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }
    
    # Print the metrics
    print(f"\n{model_name} Model Metrics:")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

# Step 3: Holt's Linear Trend Model
holt_model = ExponentialSmoothing(train['PassengerMiles'], trend='add', seasonal=None).fit()
holt_forecast = holt_model.forecast(steps=len(test))
test['Holt_Forecast'] = holt_forecast

# Calculate metrics for Holt's Linear Trend Model
calculate_metrics("Holt's Linear Trend", test['PassengerMiles'], test['Holt_Forecast'])

# Step 4: ARIMA Model
arima_model = ARIMA(train['PassengerMiles'], order=(1, 1, 1))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=len(test))
test['ARIMA_Forecast'] = arima_forecast

# Calculate metrics for ARIMA Model
calculate_metrics("ARIMA", test['PassengerMiles'], test['ARIMA_Forecast'])

# Step 5: ETS Model (Additive)
ets_model = ETSModel(
    train['PassengerMiles'],
    error='add',           # Additive error
    trend='add',           # Additive trend
    seasonal='add',        # Additive seasonality
    seasonal_periods=12,   # Monthly data, so 12 periods per year
    initialization_method="estimated"
).fit()

ets_forecast = ets_model.forecast(steps=len(test))
test['ETS_Forecast'] = ets_forecast

# Calculate metrics for ETS Model
calculate_metrics("ETS (A,A,A)", test['PassengerMiles'], test['ETS_Forecast'])

# Step 6: VAR Model
var_model = VAR(train)
fit = var_model.fit(maxlags=15, ic='aic')
best_lag = fit.k_ar
print(f"Optimal lag order chosen: {best_lag}")

# Forecast for the test period
forecast = fit.forecast(train.values[-best_lag:], steps=len(test))
forecast_df = pd.DataFrame(forecast, index=test.index, columns=df.columns)

# Calculate metrics for VAR model (only PassengerMiles for comparison)
calculate_metrics("VAR", test['PassengerMiles'], forecast_df['PassengerMiles'])

# Step 7: Visualizations
# Plot the forecasts vs actuals with confidence intervals for each model
plt.figure(figsize=(14, 8))
plt.plot(train['PassengerMiles'], label='Training Data', color='blue')
plt.plot(test['PassengerMiles'], label='Actual Test Data', color='orange')
plt.plot(test['Holt_Forecast'], label="Holt's Linear Trend", color='purple')
plt.plot(test['ARIMA_Forecast'], label="ARIMA", color='green')
plt.plot(test['ETS_Forecast'], label="ETS (A,A,A)", color='red')
plt.plot(forecast_df['PassengerMiles'], label="VAR", color='brown')
plt.title("Forecast Comparison Across Models")
plt.xlabel('Date')
plt.ylabel('Passenger Miles')
plt.legend()
plt.show()

# Step 8: Long-Term Forecast for VAR model
future_forecast = fit.forecast(df.values[-best_lag:], steps=24)
future_forecast_index = pd.date_range(start=df.index[-1], periods=24+1, freq='M')[1:]
future_forecast_df = pd.DataFrame(future_forecast, index=future_forecast_index, columns=df.columns)

# Plot long-term forecast for Passenger Miles
plt.figure(figsize=(14, 8))
plt.plot(df['PassengerMiles'], label='Historical Data', color='blue')
plt.plot(future_forecast_df['PassengerMiles'], label="Future Forecast (VAR)", color='green')
plt.title("2-Year Future Forecast for Passenger Miles using VAR Model")
plt.xlabel('Date')
plt.ylabel('Passenger Miles')
plt.legend()
plt.show()

# Step 9: Compare All Model Metrics in a DataFrame
metrics_df = pd.DataFrame(metrics).T
print("\nAll Model Metrics Comparison:")
print(metrics_df)

# Optional: Display the comparison table as a heatmap for visualization

plt.figure(figsize=(14, 8))  # You can change the width (14) and height (8) to make it larger or smaller
sns.heatmap(metrics_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Comparison of Model Metrics")
plt.show()


# Concatenate the forecasted values and intervals into a single DataFrame for easier export
forecast_export = pd.concat([forecast_mid, forecast_lower, forecast_upper], axis=1)
forecast_export.columns = [
    'PassengerMiles_Forecast', 'CPI_Forecast', 'AirTransportationEmployees_Forecast', 'DisposableIncome_Forecast',
    'PassengerMiles_Lower', 'CPI_Lower', 'AirTransportationEmployees_Lower', 'DisposableIncome_Lower',
    'PassengerMiles_Upper', 'CPI_Upper', 'AirTransportationEmployees_Upper', 'DisposableIncome_Upper'
]

# Export to Excel
forecast_export.to_excel("VAR_Forecast_Output_Future.xlsx", sheet_name="VAR Forecast Future")

print("VAR forecast successfully exported to 'VAR_Forecast_Output_Future.xlsx'")
