"""Training, evaluating, saving and using the Random Forest model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import data_utils
import features as ft

RANDOM_SEED = 42
TRAIN_FRACTION = 0.8
MIN_TRAINING_ROWS = 100


class MissingHistoryError(ValueError):
    """Raised when the readings needed to make a prediction are not available."""


def split_chronologically(table, train_fraction=TRAIN_FRACTION):
    """First 80% of time -> training, last 20% -> testing (never shuffled).

    Shuffling would let the model "see the future" during training.
    """
    table = table.sort_index()
    cut = min(max(int(len(table) * train_fraction), 1), len(table) - 1)
    return table.iloc[:cut], table.iloc[cut:]


def compute_metrics(y_true, y_pred):
    """MAE, RMSE and R2 (all computed from real predictions)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def train_model(df, use_temperature=None):
    """Train on the first 80% of the data, evaluate on the final 20%."""
    if use_temperature is None:
        use_temperature = "temperature_c" in df.columns
    table = ft.build_feature_table(df, use_temperature)
    if len(table) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Only {len(table)} usable rows after building lag features; at least "
            f"{MIN_TRAINING_ROWS} are needed to train and test a model. Upload more data "
            "or fill gaps in the readings.")

    columns = ft.feature_columns(use_temperature)
    train, test = split_chronologically(table)

    # Random Forest = many decision trees, each trained on a random sample of the
    # data; the final answer is the average of all trees' predictions.
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                  random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(train[columns], train[ft.TARGET])

    predicted = pd.Series(model.predict(test[columns]), index=test.index, name="predicted_kwh")
    train_mean = float(train[ft.TARGET].mean())
    baseline = np.full(len(test), train_mean)  # "always guess the training average"
    return {
        "model": model,
        "feature_columns": columns,
        "use_temperature": use_temperature,
        "metrics": {"random_forest": compute_metrics(test[ft.TARGET], predicted),
                    "baseline": compute_metrics(test[ft.TARGET], baseline)},
        "test_actual": test[ft.TARGET],
        "test_predicted": predicted,
        "importances": pd.Series(model.feature_importances_, index=columns).sort_values(),
        "n_train": len(train), "n_test": len(test),
        "train_start": train.index.min(), "train_end": train.index.max(),
        "test_start": test.index.min(), "test_end": test.index.max(),
        "train_mean": train_mean,
        "fingerprint": data_utils.fingerprint(df),
    }


def save_model(bundle, path):
    try:
        joblib.dump(bundle, path)
    except OSError:
        pass  # saving is a convenience; the app still works without it


def load_model(path):
    """Return a saved bundle, or None if it is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def get_or_train(df, path, force=False):
    """Reuse the saved model if it was trained on exactly this data."""
    if not force:
        saved = load_model(path)
        if saved is not None and saved.get("fingerprint") == data_utils.fingerprint(df):
            return saved, "loaded"
    bundle = train_model(df)
    save_model(bundle, path)
    return bundle, "trained"


def predict_for_timestamp(bundle, hourly_usage, timestamp, temperature=None, manual_lags=None):
    """Predict kWh for one hour. Returns (prediction, lags_used)."""
    timestamp = pd.Timestamp(timestamp).floor("h")
    if manual_lags is None:
        manual_lags, missing = ft.lags_from_history(hourly_usage, timestamp)
        if missing:
            raise MissingHistoryError(
                f"Cannot predict for {timestamp:%Y-%m-%d %H:%M} from the dataset: "
                f"{len(missing)} of the 24 previous hourly readings are missing "
                f"(from {missing[0]:%Y-%m-%d %H:%M} to {missing[-1]:%Y-%m-%d %H:%M}). "
                "Choose a time within or just after the recorded data, or enter recent usage manually.")
    if bundle["use_temperature"] and temperature is None:
        raise ValueError("This model uses temperature, so a temperature value is required.")
    row = ft.make_input_row(timestamp, temperature, manual_lags, bundle["use_temperature"])
    row = row[bundle["feature_columns"]]  # guarantees the training column order
    return float(bundle["model"].predict(row)[0]), manual_lags
