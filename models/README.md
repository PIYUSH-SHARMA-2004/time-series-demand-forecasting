# Saved forecasting model

`demand_forecast.pkl` contains a dictionary with the fitted random forest, full-history mean baseline, ordered input columns, forecast horizon, training cutoff, target description, model role, and package versions.

- Cutoff: 29 December 2024.
- Horizon: 13 Monday-Sunday weeks, labelled by Sunday.
- Inputs: `horizon_weeks`, `forecast_month`, `units_latest_week`, `mean_units_4weeks`, in that order.
- Forest: 200 trees, maximum depth 3, minimum leaf size 20, random seed 42.
- Saved versions: Python 3.13.7, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.6.

The forest is an experimental comparison model. The historical mean had lower MAE in the last historical comparison. Read the root README for the evaluation and its retrospective limitations.

Load only a trusted pickle file. The Streamlit app loads this repository's fixed model path and never retrains the model.
