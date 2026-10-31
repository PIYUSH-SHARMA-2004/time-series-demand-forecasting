# Time Series Demand Forecasting

A Python machine learning project that compares simple forecasting methods with a small random forest, then displays the results in a Streamlit app. The goal is to estimate purchased units for the next 13 weeks from one completed-week cutoff.

**Main finding:** the random forest had the lowest average validation MAE by a small margin, but the historical mean performed better in the last historical period. The app keeps both models visible rather than claiming that the forest is consistently better.

## What the app shows

- A chart of 104 completed weeks of purchases.
- The historical-mean baseline alongside the saved experimental forest.
- A selector to display 1 to 13 forecast weeks.
- A forecast table and CSV download.
- The evaluation results and the limits of the comparison.

The saved data ends on **29 December 2024**. The example forecasts cover weeks ending **5 January to 30 March 2025**. These are historical examples, not forecasts from today's date. There are no actual purchases for those forecast weeks in this dataset.

## Target and data

`weekly_units` is the sum of purchased quantities across **all categories and both online and offline channels**, including transactions marked as returned. It is not net revenue, net units after returns, or a measure of demand that includes missed purchases due to stockouts.

- Week definition: Monday to Sunday, labelled by the Sunday ending the week.
- Completed weekly observations: **104**, ending 8 January 2023 through 29 December 2024.
- Forecast horizon: **13 weeks**.
- Feature-table rows: **1,313**. These are overlapping cutoff/horizon examples, not 1,313 separate weeks.

The repository includes weekly aggregates and the prepared feature table. The transaction-level source and cleaned data stay local. The original dataset download URL and redistribution licence have not been recorded in this project, so no external data source or licence is claimed here. Running the EDA notebook from scratch requires the original local CSV.

## What I did

1. **EDA:** checked data quality, purchase patterns, categories, channels, and time coverage.
2. **Feature engineering:** aggregated completed weeks and created inputs available at each forecast cutoff.
3. **Model comparison:** compared the historical mean, two ARIMA models, linear regression, and a small random forest using chronological validation.
4. **Model saving:** refitted the selected forest on all labelled rows through 29 December 2024, then saved the fitted model and its metadata with `pickle`.
5. **App:** loaded the saved model in Streamlit without repeating model training.

## Inputs and leakage prevention

The model receives these columns in this exact order:

| Input | Meaning |
| --- | --- |
| `horizon_weeks` | Number of weeks ahead, from 1 to 13 |
| `forecast_month` | Calendar month of the forecast week's ending date |
| `units_latest_week` | Units in the latest completed week at the cutoff |
| `mean_units_4weeks` | Mean units across the last four completed weeks at the cutoff |

All 13 predictions share one cutoff. The two purchase-history inputs stay fixed across the forecast window; actual purchases from future weeks are never supplied as inputs. At each validation cutoff, a training row is included only when its target week has already ended.

## Model results

Validation used three chronological 13-week rounds, with **52, 65, and 78 completed weeks** available at the respective cutoffs. Each model's validation MAE below is the average of its three round scores.

| Model | Average validation MAE |
| --- | ---: |
| Small random forest | 12.63 |
| Linear regression | 12.67 |
| Historical mean | 12.92 |
| ARIMA(0,1,1) | 12.92 |
| ARIMA(1,0,0) | 12.95 |

MAE is the average absolute difference between predicted and actual weekly purchased units. Lower is better.

For the last 13-week historical period, ending 6 October through 29 December 2024:

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| Historical mean | **13.10** | 16.02 | 9.50% |
| Small random forest | 16.91 | 19.51 | 12.26% |

**This is a retrospective comparison.** EDA covered the full dataset, and the last period had already been inspected during earlier work. It is not a fresh unseen test. The forest's small validation advantage does not establish a reliable improvement over the mean. New actual purchases are needed for a fresh evaluation.

The forest uses 200 trees, `max_depth=3`, `min_samples_leaf=20`, and `random_state=42`. The saved forest was refitted on **1,222 labelled feature rows**. Its accompanying full-history mean is approximately **136.80 purchased units per week**; the last-period evaluation used the mean available at that earlier cutoff.

## Project structure

```text
time-series-demand-forecasting/
├── README.md
├── requirements.txt
├── requirements-notebooks.txt
├── .gitignore
├── data/
│   └── processed/
│       ├── weekly_target.csv
│       └── features.csv
├── models/
│   └── demand_forecast.pkl
└── notebooks/
    ├── app.py
    ├── EDA.ipynb
    ├── Feature_engineering.ipynb
    └── Modeling.ipynb
```

The app needs only `weekly_target.csv` and `demand_forecast.pkl`. The feature table and notebooks document the analysis. The feature-engineering notebook contains the initial split preparation; `Modeling.ipynb` defines the final three-round validation plan reported above.

## Run the app locally

Use **Python 3.13**. The model was saved with Python 3.13.7, scikit-learn 1.9.1, NumPy 2.5.3, and pandas 3.0.6. Keep the pinned package versions when loading this pickle.

```bash
git clone https://github.com/PIYUSH-SHARMA-2004/time-series-demand-forecasting.git
cd time-series-demand-forecasting
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the app packages and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run notebooks/app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`.

To work through the notebooks, install their additional packages with `python -m pip install -r requirements-notebooks.txt`. The EDA and feature-engineering stages also require the local transaction data described above. Keep the notebook working directory inside `notebooks/` so their `../data/` paths resolve correctly.

Only load pickle files you trust: unpickling can execute Python code. The app loads the fixed model file included in this repository and does not accept uploaded models.

## Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
2. Create an app from `PIYUSH-SHARMA-2004/time-series-demand-forecasting`.
3. Select branch **main** and entrypoint **notebooks/app.py**.
4. In **Advanced settings**, select **Python 3.13**.
5. Deploy. Community Cloud installs the pinned packages from the root `requirements.txt`.

The app runs on the hosting service while your laptop is off. Free Community Cloud apps sleep after 12 hours without traffic, and visitors can wake them. See the official [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [app hibernation details](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

No live app URL is listed yet; creating this repository does not deploy the app automatically.

## Tech used

Python, pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Seaborn, Jupyter, pickle, Streamlit, and GitHub.

## What I would improve next

- Collect new completed weeks and evaluate forecasts made before their actuals become known.
- Investigate additional useful inputs only when they are known at prediction time.
- Compare prediction intervals and errors across more forecast periods.
- Record the original dataset source and its licence before distributing transaction data.

Related project: [E-Commerce Product Recommendation System](https://github.com/PIYUSH-SHARMA-2004/ml-ecommerce-product-recommendation).
