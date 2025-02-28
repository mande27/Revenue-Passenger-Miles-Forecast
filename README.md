# Forecasting Airline Demand: Revenue Passenger Miles (RPM)

## Overview
This project aims to forecast U.S. airline revenue passenger miles (RPM) over the next 24 months using time series forecasting models. The analysis incorporates economic indicators such as CPI, disposable income, and air transportation employment to improve forecast accuracy.

## Problem Statement
The airline industry is highly sensitive to economic fluctuations. Accurate demand forecasting helps airlines optimize resources, plan capacity, and improve profitability.

## Data Sources
- **Revenue Passenger Miles (RPM)**: [FRED](https://fred.stlouisfed.org/series/RPMD11)
- **Consumer Price Index (CPI)**: [FRED](https://fred.stlouisfed.org/series/CPIAUCSL)
- **Disposable Income**: [FRED](https://fred.stlouisfed.org/series/DSPIC96)
- **Air Transportation Employees**: [FRED](https://fred.stlouisfed.org/series/CES4348100001)

## Methodology
- **Data Preprocessing**: Cleaned and prepared monthly data from Jan 2000 to July 2024.
- **Model Selection**: Compared four models:
  - Holt-Winters Seasonal Model
  - SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)
  - ETS (Error, Trend, Seasonal)
  - VAR (Vector AutoRegression)
- **Evaluation Metrics**: Used MAE, MAPE, MSE, RMSE, and R² to compare model performance.

## Results
- **Best Model**: SARIMAX performed the best, with the lowest error metrics.
- **Forecast**: Predicted RPM for the next 24 months, showing a gradual recovery in passenger demand.

## Business Impact
This forecast can help airlines:
- Plan capacity and staffing.
- Optimize pricing strategies.
- Anticipate changes in demand due to economic factors.

## Files in This Repository
Here’s a breakdown of the files included in this repository:

### Code
- **`Final adjusted code MG.py`**: The main Python script for data analysis and forecasting.
- **`FF Airline code.py`**: Additional Python script for data processing and visualization.

### Data
- **`FM/FINAL.csv`**: The dataset used for analysis, containing monthly data from Jan 2000 to July 2024.

### Visualizations
- **`Holt-Winters Seasonal Model.jpg`**: Forecast results from the Holt-Winters Seasonal Model.
- **`VAR Model Graphs.png.jpg`**: Forecast results from the VAR Model.
- **`Forecast Comparison Across Models.jpg`**: Comparison of forecasts from all models.
- **`ETS (Error, Trend, Seasonal) Model.jpg`**: Forecast results from the ETS Model.
- **`SARIMAX Model.jpg`**: Forecast results from the SARIMAX Model.
- **`Graph2 Correlation.png`**: Correlation heatmap between variables.
- **`Primary Target Variables Data Graph1.jpg`**: Time series plot of the primary target variable (RPM).

### Reports and Outputs
- **`Forecasting Airline Demand- Maria Isabel Gutierrez Port.pdf`**: A detailed report summarizing the project, including methodology, results, and business impact.
- **`SARIMAX_Future_Forecast.xlsx`**: Excel file containing the SARIMAX model’s future forecast results.

