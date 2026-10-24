"""Data helpers: synthetic data generation, CSV validation and cleaning."""
import hashlib

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "usage_kwh"]
MIN_ROWS = 72  # three days of hourly data is the least we can sensibly model


class DataValidationError(ValueError):
    """Raised with a friendly message when an uploaded CSV cannot be used."""


# ----------------------------------------------------------------------------
# Synthetic demonstration data
# ----------------------------------------------------------------------------
def generate_synthetic_data(days=240, seed=42, start="2025-01-01"):
    """Create SYNTHETIC hourly electricity data (not real measurements).

    Patterns included: morning and evening peaks, higher weekend daytime use,
    heating in cold weather, cooling in hot weather, random noise and a few
    short high-consumption periods.
    """
    rng = np.random.default_rng(seed)  # fixed seed -> same data every time
    timestamps = pd.date_range(start=start, periods=days * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    is_weekend = (timestamps.dayofweek.to_numpy() >= 5).astype(float)

    # Temperature: yearly cycle (coldest in January) + daily cycle + smooth noise
    yearly = -8 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    daily = 3.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather_noise = pd.Series(rng.normal(0, 1.5, len(timestamps))).ewm(span=24).mean().to_numpy() * 2
    temperature = 11 + yearly + daily + weather_noise

    # Usage: a base load plus bumps for typical household routines
    base = 0.25
    morning_peak = 0.6 * np.exp(-(((hour - 7.5) / 1.5) ** 2))
    evening_peak = 1.0 * np.exp(-(((hour - 19) / 2.0) ** 2))
    weekend_daytime = 0.35 * np.exp(-(((hour - 13) / 3.0) ** 2)) * is_weekend
    heating = 0.06 * np.clip(15 - temperature, 0, None)
    cooling = 0.04 * np.clip(temperature - 22, 0, None)
    usage = base + morning_peak + evening_peak + weekend_daytime + heating + cooling
    usage = usage * rng.lognormal(0, 0.10, len(usage)) + rng.normal(0, 0.03, len(usage))

    # A handful of short "high consumption" periods (guests, heavy appliance use...)
    for _ in range(max(3, days // 30)):
        begin = int(rng.integers(0, len(usage) - 12))
        length = int(rng.integers(3, 10))
        usage[begin:begin + length] *= rng.uniform(1.6, 2.2)

    return pd.DataFrame({
        "timestamp": timestamps,
        "usage_kwh": np.round(np.clip(usage, 0.05, None), 3),
        "temperature_c": np.round(temperature, 1),
    })


# ----------------------------------------------------------------------------
# CSV loading and validation
# ----------------------------------------------------------------------------
def load_csv(file_or_path):
    """Read a CSV and return (clean_dataframe, report). Raises DataValidationError."""
    try:
        raw = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        raise DataValidationError("The file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError) as err:
        raise DataValidationError(f"The file could not be read as a CSV: {err}")
    return validate_and_clean(raw)


def validate_and_clean(raw):
    """Check an uploaded table and return (clean_dataframe, report)."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Your file has: {', '.join(df.columns) or 'no columns'}. "
            "Required: timestamp, usage_kwh. Optional: temperature_c.")

    keep = REQUIRED_COLUMNS + (["temperature_c"] if "temperature_c" in df.columns else [])
    df = df[keep]
    report = {"rows_in": len(df), "messages": []}
    report["missing_summary"] = (
        df.isna().sum().rename("missing values in upload").rename_axis("column").reset_index())

    # 1. Timestamps (offsets are converted to UTC, then made timezone-naive)
    parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    bad_dates = int(parsed.isna().sum())
    df = df.assign(timestamp=parsed).dropna(subset=["timestamp"])
    if df.empty:
        raise DataValidationError("No valid dates found in the 'timestamp' column. "
                                  "Try a format such as 2025-03-01 14:00.")
    _note(report, "invalid_dates_removed", bad_dates, "row(s) with unreadable dates removed")

    # 2. Usage values must be numeric, present and not negative
    usage = pd.to_numeric(df["usage_kwh"], errors="coerce")
    missing_usage = int(usage.isna().sum())
    df = df.assign(usage_kwh=usage).dropna(subset=["usage_kwh"])
    _note(report, "missing_usage_removed", missing_usage, "row(s) with missing or non-numeric usage removed")
    negative = int((df["usage_kwh"] < 0).sum())
    df = df[df["usage_kwh"] >= 0]
    _note(report, "negative_usage_removed", negative, "row(s) with negative usage removed")

    # 3. The app models HOURLY data, so check the spacing before tidying timestamps
    df = df.sort_values("timestamp")
    if len(df) >= 2:
        median_gap = df["timestamp"].diff().dropna().median()
        if not pd.Timedelta(minutes=55) <= median_gap <= pd.Timedelta(minutes=65):
            raise DataValidationError(
                f"Readings must be hourly, but the typical gap in your file is {median_gap}. "
                "Please aggregate your data to one reading per hour.")

    # 4. Align to the hour and remove duplicate timestamps (keep the first)
    df["timestamp"] = df["timestamp"].dt.floor("h")
    duplicates = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates(subset="timestamp", keep="first")
    _note(report, "duplicates_removed", duplicates, "duplicate timestamp(s) removed (first reading kept)")

    # 5. Optional temperature: fill small gaps, or drop the column if mostly empty
    if "temperature_c" in df.columns:
        df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
        missing_temp = int(df["temperature_c"].isna().sum())
        if missing_temp > 0.5 * len(df):
            df = df.drop(columns="temperature_c")
            report["messages"].append("Temperature column is mostly empty, so the model will run without it.")
        elif missing_temp:
            df = df.set_index("timestamp")
            df["temperature_c"] = df["temperature_c"].interpolate(method="time", limit_direction="both")
            df = df.reset_index()
            report["messages"].append(f"{missing_temp} missing temperature value(s) filled by interpolation.")

    df = df.reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise DataValidationError(
            f"Only {len(df)} usable hourly rows remain after cleaning; at least {MIN_ROWS} "
            "(3 days) are needed.")

    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    gap_hours = len(full_range) - len(df)
    if gap_hours:
        report["messages"].append(
            f"{gap_hours} hour(s) are missing inside the date range. They are left as gaps; "
            "rows that need them for lag features are skipped during training.")
    report["gap_hours"] = gap_hours
    report["rows_out"] = len(df)
    if len(report["messages"]) == 0:
        report["messages"].append("No problems found - no rows needed cleaning.")
    return df, report


def _note(report, key, count, text):
    report[key] = count
    if count:
        report["messages"].append(f"{count} {text}.")


def fingerprint(df):
    """Short hash of a dataset, used to know if a saved model matches the data."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed + ",".join(df.columns).encode()).hexdigest()[:16]
