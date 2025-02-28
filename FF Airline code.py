# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:57:29 2024

@author: maria
"""

# Step 1: Import Libraries and Load Data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set Seaborn style
sns.set(style="whitegrid")

# Load and preprocess data
df = pd.read_csv('FMFINAL.csv', parse_dates=['DATE'], index_col='DATE')
df.rename(columns={'RPMD11': 'PassengerMiles', 'CPIAUCSL': 'CPI', 'CES4348100001': 'AirTransportationEmployees', 'DSPIC96': 'DisposableIncome'}, inplace=True)

# Remove unnecessary columns
df = df[['PassengerMiles', 'CPI', 'AirTransportationEmployees', 'DisposableIncome']]

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Variables")
plt.show()

# Split data into training and test sets
train = df[:'2022']
test = df['2023':]

# Function to calculate metrics
def calculate_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{name} Model Metrics:")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    return mae, mape, mse, rmse, r2

# Step 2: Holt-Winters Model
holt_model = ExponentialSmoothing(train['PassengerMiles'], trend='add', seasonal='add', seasonal_periods=12).fit()
holt_forecast = holt_model.forecast(steps=len(test))
test['Holt_Forecast'] = holt_forecast
calculate_metrics("Holt-Winters", test['PassengerMiles'], test['Holt_Forecast'])

# Plot Holt-Winters forecast
plt.figure(figsize=(10, 6))
sns.lineplot(data=train['PassengerMiles'], label='Training Data', color='blue')
sns.lineplot(data=test['PassengerMiles'], label='Test Data', color='orange')
sns.lineplot(data=test['Holt_Forecast'], label="Holt-Winters Forecast", color='red')
plt.title("Holt-Winters Model Forecast vs Actuals")
plt.xlabel('Date')
plt.ylabel('Passenger Miles')
plt.legend()
plt.show()

# Step 3: SARIMAX Model
sarimax_model = SARIMAX(train['PassengerMiles'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
sarimax_forecast = sarimax_model.forecast(steps=len(test))
test['SARIMAX_Forecast'] = sarimax_forecast
calculate_metrics("SARIMAX", test['PassengerMiles'], test['SARIMAX_Forecast'])

# Plot SARIMAX forecast
plt.figure(figsize=(10, 6))
sns.lineplot(data=train['PassengerMiles'], label='Training Data', color='blue')
sns.lineplot(data=test['PassengerMiles'], label='Test Data', color='orange')
sns.lineplot(data=test['SARIMAX_Forecast'], label="SARIMAX Forecast", color='green')
plt.title("SARIMAX Model Forecast vs Actuals")
plt.xlabel('Date')
plt.ylabel('Passenger Miles')
plt.legend()
plt.show()

# Step 4: ETS Model
ets_model = ETSModel(train['PassengerMiles'], error='add', trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated").fit()
ets_forecast = ets_model.forecast(steps=len(test))
test['ETS_Forecast'] = ets_forecast
calculate_metrics("ETS (A,A,A)", test['PassengerMiles'], test['ETS_Forecast'])

# Plot ETS forecast
plt.figure(figsize=(10, 6))
sns.lineplot(data=train['PassengerMiles'], label='Training Data', color='blue')
sns.lineplot(data=test['PassengerMiles'], label='Test Data', color='orange')
sns.lineplot(data=test['ETS_Forecast'], label="ETS (A,A,A) Forecast", color='purple')
plt.title("ETS Model (Additive) Forecast vs Actuals")
plt.xlabel('Date')
plt.ylabel('Passenger Miles')
plt.legend()
plt.show()

# Step 5: VAR Model
var_model = VAR(train)
fit = var_model.fit(maxlags=15, ic='aic')
best_lag = fit.k_ar
print(f"Optimal lag order chosen: {best_lag}")
forecast = fit.forecast(train.values[-best_lag:], steps=len(test))
forecast_df = pd.DataFrame(forecast, index=test.index, columns=df.columns)
test['VAR_Forecast'] = forecast_df['PassengerMiles']
calculate_metrics("VAR", test['PassengerMiles'], test['VAR_Forecast'])

# Plot VAR forecast
plt.figure(figsize=(10, 6))
sns.lineplot(data=train['PassengerMiles'], label='Training Data', color='blue')
sns.lineplot(data=test['PassengerMiles'], label='Test Data', color='orange')
sns.lineplot(data=test['VAR_Forecast'], label="VAR Forecast", color='brown')
plt.title("VAR Model Forecast vs Actuals")
plt.xlabel('Date')
plt.ylabel('Passenger Miles')
plt.legend()
plt.show()

# Step 5: VAR Model - Updated with Detailed Forecasts for Each Variable
var_model = VAR(train)
fit = var_model.fit(maxlags=15, ic='aic')
best_lag = fit.k_ar
print(f"Optimal lag order chosen: {best_lag}")

# Forecast for the test period
forecast = fit.forecast(train.values[-best_lag:], steps=len(test))
forecast_df = pd.DataFrame(forecast, index=test.index, columns=df.columns)

# Obtain Prediction Intervals for the Forecast
mid, lower, upper = fit.forecast_interval(train.values[-best_lag:], steps=len(test), alpha=0.05)

# Convert intervals to DataFrames for easier plotting
forecast_mid = pd.DataFrame(mid, index=test.index, columns=df.columns)
forecast_lower = pd.DataFrame(lower, index=test.index, columns=df.columns)
forecast_upper = pd.DataFrame(upper, index=test.index, columns=df.columns)

# Plot Forecast with Prediction Intervals for each variable
for column in df.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train[column], label='Training Data', color='blue')
    sns.lineplot(data=test[column], label='Actual Test Data', color='orange')
    sns.lineplot(data=forecast_mid[column], label='Forecast', color='red')
    plt.fill_between(forecast_mid.index, forecast_lower[column], forecast_upper[column], color='pink', alpha=0.3)
    plt.title(f'{column} Forecast with Prediction Intervals')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.show()



# Step 6: Compare All Model Forecasts
plt.figure(figsize=(12, 8))
sns.lineplot(data=train['PassengerMiles'], label='Training Data', color='blue')
sns.lineplot(data=test['PassengerMiles'], label='Actual Test Data', color='orange')
sns.lineplot(data=test['Holt_Forecast'], label="Holt-Winters Forecast", color='red')
sns.lineplot(data=test['SARIMAX_Forecast'], label="SARIMAX Forecast", color='green')
sns.lineplot(data=test['ETS_Forecast'], label="ETS (A,A,A) Forecast", color='purple')
sns.lineplot(data=test['VAR_Forecast'], label="VAR Forecast", color='brown')
plt.title("Forecast Comparison Across Models")
plt.xlabel('Date')
plt.ylabel('Passenger Miles')
plt.legend()
plt.show()

# Step 7: Forecast 24 months into the future with the best model (assuming SARIMAX here)
future_forecast = sarimax_model.get_forecast(steps=24)
future_mean = future_forecast.predicted_mean
future_conf_int = future_forecast.conf_int()

# Export forecast to Excel
forecast_df = pd.DataFrame({
    'Forecast': future_mean,
    'Lower CI': future_conf_int.iloc[:, 0],
    'Upper CI': future_conf_int.iloc[:, 1]
})

# Set index as date range for the future forecast period
forecast_df.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=24, freq='M')

# Save to Excel
forecast_df.to_excel('SARIMAX_Future_Forecast.xlsx', sheet_name='Forecast_24_Months')

# Display confirmation message
print("The 24-month forecast has been saved to 'SARIMAX_Future_Forecast.xlsx'.")
