# Analysis notebooks

Read the notebooks in this order:

1. `EDA.ipynb`: transaction checks and exploratory analysis.
2. `Feature_engineering.ipynb`: completed weekly totals and cutoff/horizon examples.
3. `Modeling.ipynb`: final chronological validation, retrospective comparison, example forecasts, and model saving.

The notebooks are published as saved locally, without editing their cells or outputs. `Modeling.ipynb` defines the final three-round validation plan; the feature-engineering notebook contains the earlier split preparation.

Install notebook packages using `python -m pip install -r requirements-notebooks.txt` from the repository root. When running notebook cells, keep the working directory inside `notebooks/` because their paths begin with `../data/`.

EDA requires `data/raw/customer_shopping_behavior.csv`. Feature engineering requires `data/processed/cleaned_data.csv`. Those transaction-level files are kept local. Modeling can use the included `weekly_target.csv` and `features.csv` files. The Streamlit app needs no notebook execution.
