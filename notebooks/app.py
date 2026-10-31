import os
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Ecommerce Purchase Forecast", layout="wide")
st.title("Ecommerce Purchase Forecast")
st.write("Weekly purchased units across all categories and sales channels.")

# app.py is inside notebooks; the project folder is one level above it.
notebooks_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(notebooks_folder)
model_path = os.path.join(project_folder, "models", "demand_forecast.pkl")
weekly_path = os.path.join(project_folder, "data", "processed", "weekly_target.csv")

for file_path in [model_path, weekly_path]:
    if not os.path.isfile(file_path):
        st.error("A required file is missing: " + os.path.relpath(file_path, project_folder))
        st.stop()


# Keep the fitted model in memory when a visitor changes the display controls.
@st.cache_resource
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


model_bundle = load_model(model_path)
cutoff = pd.Timestamp(model_bundle["trained_through"])

# Read completed history. The app never fits or retrains the model.
weekly = pd.read_csv(weekly_path, parse_dates=["week_end"])
weekly = weekly.sort_values("week_end").set_index("week_end")
weekly_units = weekly.loc[:cutoff, "weekly_units"]
expected_weeks = pd.date_range(weekly_units.index.min(), cutoff, freq="W-SUN")
if (
    len(weekly_units) < 4
    or weekly_units.isna().any()
    or not weekly_units.index.equals(expected_weeks)
):
    st.error("The weekly history must contain consecutive completed weeks through the model cutoff.")
    st.stop()

st.info(
    f"Historical example: the data ends on {cutoff:%d %B %Y}. "
    "These are example forecasts after that date, not forecasts for today. "
    "The dataset has no actual purchases for the forecast weeks."
)
st.caption("Weeks run Monday to Sunday. Purchased units do not measure unmet demand.")

left, right = st.columns(2)
left.metric("Completed weeks", len(weekly_units))
right.metric("Historical mean: units per week", f"{model_bundle['historical_mean_units']:.2f}")

st.subheader("Completed weekly purchases")
st.line_chart(weekly_units.rename("Purchased units"))

# All 13 rows use the same last completed week and last-four-week average.
horizon = model_bundle["horizon_weeks"]
future_weeks = pd.date_range(
    start=cutoff + pd.Timedelta(weeks=1),
    periods=horizon,
    freq=model_bundle["week_frequency"],
)
forecast_inputs = pd.DataFrame({
    "horizon_weeks": range(1, horizon + 1),
    "forecast_month": future_weeks.month,
    "units_latest_week": float(weekly_units.iloc[-1]),
    "mean_units_4weeks": float(weekly_units.tail(4).mean()),
})

# Preserve the exact input-column order used when fitting the saved model.
x_future = forecast_inputs[model_bundle["feature_columns"]]
forest_prediction = model_bundle["model"].predict(x_future)
forecast = pd.DataFrame({
    "Week ending": future_weeks,
    "Historical mean": model_bundle["historical_mean_units"],
    "Experimental forest": forest_prediction,
})

st.subheader("Example forecasts: baseline and experimental model")
weeks_to_show = st.slider("Weeks to display", 1, horizon, horizon)
visible_forecast = forecast.head(weeks_to_show)
st.line_chart(visible_forecast.set_index("Week ending"))
st.caption("Both lines show predicted purchased units per week. Predictions can be fractional.")

display_table = visible_forecast.copy()
display_table["Week ending"] = display_table["Week ending"].dt.strftime("%Y-%m-%d")
st.dataframe(display_table.round(2), hide_index=True)
st.download_button(
    "Download displayed forecasts (CSV)",
    data=display_table.to_csv(index=False).encode("utf-8"),
    file_name="example_purchase_forecasts.csv",
    mime="text/csv",
)

with st.expander("How the models performed and how this app works"):
    st.write("Notebook results: MAE is the average absolute error, measured in purchased units.")
    st.dataframe(pd.DataFrame({
        "Model": ["Historical mean", "Experimental forest"],
        "Average validation MAE (three 13-week rounds)": [12.92, 12.63],
        "Last 13-week historical period MAE": [13.10, 16.91],
    }), hide_index=True)
    st.write(
        "The mean performed better in the last historical period. That period had "
        "already been inspected, so this is a retrospective comparison, not a fresh unseen test. "
        "The forest is an experimental comparison model."
    )
    st.write(
        "The app loads the fitted model using pickle. It prepares the number of weeks ahead, "
        "forecast month, latest completed week's units, and mean of the last four completed weeks. "
        "All predictions use one cutoff, without actual purchases from the forecast window."
    )
    st.caption("MAE values above are recorded notebook results, not recalculated by the app.")
