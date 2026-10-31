# Forecast inputs

- `weekly_target.csv`: 104 completed Monday-Sunday weeks with `week_end` and `weekly_units`.
- `features.csv`: 1,313 cutoff/horizon examples used in the modeling notebook, including 13 rows with unknown future targets after 29 December 2024.

The app reads the weekly totals and prepares its four model inputs from the saved model's cutoff. It does not use future actual purchases. Transaction-level `cleaned_data.csv` is kept local and excluded from this repository.
