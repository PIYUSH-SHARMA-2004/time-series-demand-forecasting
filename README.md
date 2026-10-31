# Time Series Demand Forecasting

I built this project to forecast weekly purchased units from shopping transaction data. I compared a historical average, ARIMA, linear regression, and a small random forest, then built a Streamlit app to display the forecasts.

The main result was that the random forest did slightly better during validation, but the historical average performed better in the last historical period. The app includes both so the comparison stays visible.

## About the data

The analysis uses **104 completed weeks**, ending **8 January 2023 to 29 December 2024**. Each week runs from Monday to Sunday.

The target, `weekly_units`, is the total purchased quantity across all categories and both online and offline channels. Transactions marked as returned remain included, so this measures recorded purchases rather than net units after returns or unmet demand.

The repository contains the weekly totals and prepared features. The original transaction-level CSVs are kept locally and are needed to rerun EDA and feature engineering from scratch.

## What I worked on

- **EDA:** checked data quality, purchase patterns, category and channel differences, and date coverage.
- **Feature engineering:** created weekly totals and examples for forecasting the next 13 weeks from a completed-week cutoff.
- **Modeling:** compared five forecasting methods using chronological validation.
- **App:** saved the fitted forest with `pickle` and loaded it in Streamlit, without training the model again each time the app runs.

The forest uses four inputs, in this order:

| Input | Description |
| --- | --- |
| `horizon_weeks` | How many weeks ahead the prediction is for, from 1 to 13 |
| `forecast_month` | Month of the forecast week's ending date |
| `units_latest_week` | Purchased units in the latest completed week |
| `mean_units_4weeks` | Average purchased units over the last four completed weeks |

All 13 predictions use the same cutoff. The purchase-history inputs stay fixed throughout that forecast window. Training rows are included only when their target week has already ended by the training cutoff.

The feature table contains **1,313 overlapping cutoff/horizon examples**, not 1,313 independent weeks.

## Results

I used three chronological validation rounds, each forecasting 13 weeks. The rounds started with 52, 65, and 78 completed weeks of history.

| Model | Average validation MAE | Last 13-week historical period MAE |
| --- | ---: | ---: |
| Small random forest | **12.63** | 16.91 |
| Linear regression | 12.67 | — |
| Historical mean | 12.92 | **13.10** |
| ARIMA(0,1,1) | 12.92 | — |
| ARIMA(1,0,0) | 12.95 | — |

MAE measures the average absolute error in weekly purchased units; lower is better. The last-period comparison covers weeks ending 6 October to 29 December 2024. Only the forest and historical mean were compared in that stage.

I had already explored the full dataset and inspected the last period during earlier work. These results are therefore a **retrospective comparison**, not a fresh unseen test. The forest's small validation advantage did not translate into a consistent improvement over the average.

The forest has 200 trees, a maximum depth of 3, a minimum of 20 samples per leaf, and a random seed of 42. After the comparisons, I refitted it on all 1,222 labelled feature rows through 29 December 2024 and saved it with the input details and package versions.

## Streamlit app

The app shows the completed purchase history, the historical-mean baseline, and the experimental forest forecasts. It also includes a forecast table and a CSV download.

The example forecast covers weeks ending **5 January to 30 March 2025**. There are no actual purchases for those weeks in this dataset. The week selector changes how many of these predictions are displayed; the app uses a fixed historical cutoff and does not make forecasts from today's date.

## Project structure

```text
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── processed/
│       ├── weekly_target.csv
│       └── features.csv
├── models/
│   └── demand_forecast.pkl
└── notebooks/
    ├── EDA.ipynb
    ├── Feature_engineering.ipynb
    └── Modeling.ipynb
```

`Modeling.ipynb` contains the final three-round validation plan. The feature-engineering notebook contains the earlier split preparation.

## Run locally

From the project folder, using Python 3.13:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The saved model uses scikit-learn 1.9.1, NumPy 2.5.3, and pandas 3.0.6. These versions are pinned in `requirements.txt`, which also includes the notebook packages. Notebook paths assume the working directory is `notebooks/`.

## What I would work on next

New completed weeks would allow a fresh evaluation of forecasts made before the actual purchases become known. I would also investigate additional ARIMA settings, check the remaining errors, and assess forecast uncertainty before expanding the app.

## Tools used

Python, pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Seaborn, Jupyter, pickle, and Streamlit.

Related project: [E-Commerce Product Recommendation System](https://github.com/PIYUSH-SHARMA-2004/ml-ecommerce-product-recommendation).
