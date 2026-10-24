"""Feature engineering shared by training AND prediction (so they always match)."""
import pandas as pd

TARGET = "usage_kwh"
CALENDAR_COLUMNS = ["hour", "weekday", "month", "is_weekend", "season"]
LAG_COLUMNS = ["lag_1", "lag_2", "lag_24", "roll_mean_3", "roll_mean_24"]
LAG_DESCRIPTIONS = {
    "lag_1": "usage 1 hour earlier",
    "lag_2": "usage 2 hours earlier",
    "lag_24": "usage 24 hours earlier (same hour yesterday)",
    "roll_mean_3": "average of the previous 3 hours",
    "roll_mean_24": "average of the previous 24 hours",
}


def feature_columns(use_temperature):
    """The exact, ordered list of model inputs."""
    temp = ["temperature_c"] if use_temperature else []
    return CALENDAR_COLUMNS + temp + LAG_COLUMNS


def season_from_month(month):
    """0 = winter, 1 = spring, 2 = summer, 3 = autumn."""
    return (month % 12) // 3


def add_calendar_features(frame):
    """Add hour, weekday, month, weekend flag and season from a DatetimeIndex."""
    frame = frame.copy()
    index = frame.index
    frame["hour"] = index.hour
    frame["weekday"] = index.dayofweek  # Monday = 0
    frame["month"] = index.month
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    frame["season"] = season_from_month(index.month)
    return frame


def to_hourly_grid(df):
    """Index by timestamp on a complete hourly grid (gaps become NaN)."""
    return df.set_index("timestamp").sort_index().asfreq("h")


def to_hourly_series(df):
    return to_hourly_grid(df)[TARGET]


def add_lag_features(frame, usage):
    """Add lag and rolling features.

    LEAKAGE PREVENTION: every feature is built from usage.shift(1), i.e. only
    readings BEFORE the hour being predicted. The current reading is never used.
    """
    frame = frame.copy()
    previous = usage.shift(1)
    frame["lag_1"] = usage.shift(1)
    frame["lag_2"] = usage.shift(2)
    frame["lag_24"] = usage.shift(24)
    frame["roll_mean_3"] = previous.rolling(3).mean()
    frame["roll_mean_24"] = previous.rolling(24).mean()
    return frame


def build_feature_table(df, use_temperature):
    """Full training table: model inputs plus the target, rows with gaps dropped."""
    grid = to_hourly_grid(df)
    table = add_calendar_features(pd.DataFrame(index=grid.index))
    table = add_lag_features(table, grid[TARGET])
    if use_temperature:
        table["temperature_c"] = grid["temperature_c"]
    table[TARGET] = grid[TARGET]
    return table[feature_columns(use_temperature) + [TARGET]].dropna()


def lags_from_history(hourly_usage, timestamp):
    """Look up lag features for `timestamp` from recorded data.

    Returns (lags_dict, missing_timestamps). If any of the 24 hours before
    `timestamp` is not recorded, lags_dict is None and the missing hours are listed.
    """
    window_index = pd.date_range(timestamp - pd.Timedelta(hours=24),
                                 timestamp - pd.Timedelta(hours=1), freq="h")
    window = hourly_usage.reindex(window_index)
    missing = list(window.index[window.isna()])
    if missing:
        return None, missing
    return {
        "lag_1": window.iloc[-1],
        "lag_2": window.iloc[-2],
        "lag_24": window.iloc[0],
        "roll_mean_3": window.iloc[-3:].mean(),
        "roll_mean_24": window.mean(),
    }, []


def lags_from_manual(usage_1h, usage_2h, usage_3h, usage_24h, average_24h):
    """Build the lag dictionary from values typed in by the user."""
    return {"lag_1": usage_1h, "lag_2": usage_2h, "lag_24": usage_24h,
            "roll_mean_3": (usage_1h + usage_2h + usage_3h) / 3,
            "roll_mean_24": average_24h}


def make_input_row(timestamp, temperature, lags, use_temperature):
    """One-row table with the same columns, in the same order, as training."""
    row = add_calendar_features(pd.DataFrame(index=pd.DatetimeIndex([timestamp])))
    for name, value in lags.items():
        row[name] = float(value)
    if use_temperature:
        row["temperature_c"] = float(temperature)
    return row[feature_columns(use_temperature)]
