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

## Visualizations
![Forecast Comparison](forecast_comparison.png)
![Long-Term Forecast](long_term_forecast.png)

## Code
- Python script: `Final_adjusted_code_MG.py`
- Jupyter Notebook: `Forecasting_Airline_Demand.ipynb`

## How to Run
1. Clone this repository.
2. Install the required libraries: `pip install -r requirements.txt`.
3. Run the Python script or Jupyter Notebook.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
