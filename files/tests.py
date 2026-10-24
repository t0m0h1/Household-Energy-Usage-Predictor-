"""Basic automated tests. Run with:  python -m unittest tests -v"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import data_utils
import features as ft
import model as ml


def small_csv_frame(hours=200):
    stamps = pd.date_range("2025-03-01", periods=hours, freq="h")
    return pd.DataFrame({"timestamp": stamps.astype(str), "usage_kwh": np.linspace(0.5, 2, hours)})


class TestSyntheticData(unittest.TestCase):
    def test_shape_columns_and_values(self):
        df = data_utils.generate_synthetic_data(days=60)
        self.assertEqual(list(df.columns), ["timestamp", "usage_kwh", "temperature_c"])
        self.assertEqual(len(df), 60 * 24)
        self.assertTrue((df["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all())
        self.assertFalse(df.isna().any().any())
        self.assertTrue((df["usage_kwh"] > 0).all())

    def test_reproducible_with_seed(self):
        a = data_utils.generate_synthetic_data(days=10, seed=1)
        b = data_utils.generate_synthetic_data(days=10, seed=1)
        c = data_utils.generate_synthetic_data(days=10, seed=2)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a["usage_kwh"].equals(c["usage_kwh"]))

    def test_evening_higher_than_night(self):
        df = data_utils.generate_synthetic_data(days=60)
        by_hour = df.groupby(df["timestamp"].dt.hour)["usage_kwh"].mean()
        self.assertGreater(by_hour[19], by_hour[3])


class TestCsvValidation(unittest.TestCase):
    def test_valid_file_passes(self):
        clean, report = data_utils.validate_and_clean(small_csv_frame())
        self.assertEqual(len(clean), 200)
        self.assertNotIn("temperature_c", clean.columns)  # optional

    def test_missing_column(self):
        with self.assertRaisesRegex(data_utils.DataValidationError, "Missing required"):
            data_utils.validate_and_clean(small_csv_frame().drop(columns="usage_kwh"))

    def test_invalid_dates_removed_and_reported(self):
        df = small_csv_frame()
        df.loc[5, "timestamp"] = "not a date"
        clean, report = data_utils.validate_and_clean(df)
        self.assertEqual(report["invalid_dates_removed"], 1)
        self.assertEqual(len(clean), 199)

    def test_all_dates_invalid(self):
        df = small_csv_frame()
        df["timestamp"] = "nope"
        with self.assertRaisesRegex(data_utils.DataValidationError, "No valid dates"):
            data_utils.validate_and_clean(df)

    def test_duplicates_and_missing_values(self):
        df = small_csv_frame()
        df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
        df.loc[20, "usage_kwh"] = np.nan
        clean, report = data_utils.validate_and_clean(df)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["missing_usage_removed"], 1)
        self.assertFalse(clean["timestamp"].duplicated().any())
        self.assertFalse(clean["usage_kwh"].isna().any())

    def test_temperature_gaps_filled(self):
        df = small_csv_frame()
        df["temperature_c"] = 10.0
        df.loc[[30, 31], "temperature_c"] = np.nan
        clean, _ = data_utils.validate_and_clean(df)
        self.assertFalse(clean["temperature_c"].isna().any())

    def test_too_few_rows_and_non_hourly(self):
        with self.assertRaisesRegex(data_utils.DataValidationError, "at least"):
            data_utils.validate_and_clean(small_csv_frame(hours=20))
        daily = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=100, freq="D").astype(str),
                              "usage_kwh": 1.0})
        with self.assertRaisesRegex(data_utils.DataValidationError, "hourly"):
            data_utils.validate_and_clean(daily)


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data_utils.generate_synthetic_data(days=30)

    def test_calendar_features(self):
        frame = ft.add_calendar_features(pd.DataFrame(index=pd.DatetimeIndex(["2025-01-04 14:00"])))
        row = frame.iloc[0]
        self.assertEqual((row.hour, row.weekday, row.month, row.is_weekend, row.season), (14, 5, 1, 1, 0))

    def test_feature_columns_with_and_without_temperature(self):
        self.assertIn("temperature_c", ft.feature_columns(True))
        self.assertNotIn("temperature_c", ft.feature_columns(False))

    def test_lag_values_use_only_the_past(self):
        table = ft.build_feature_table(self.df, True)
        usage = self.df.set_index("timestamp")["usage_kwh"]
        ts = table.index[100]
        self.assertAlmostEqual(table.loc[ts, "lag_1"], usage[ts - pd.Timedelta(hours=1)])
        self.assertAlmostEqual(table.loc[ts, "lag_24"], usage[ts - pd.Timedelta(hours=24)])
        previous_24 = usage[ts - pd.Timedelta(hours=24): ts - pd.Timedelta(hours=1)]
        self.assertAlmostEqual(table.loc[ts, "roll_mean_24"], previous_24.mean())

    def test_changing_the_future_does_not_change_the_past(self):
        # Make one reading enormous; features for that hour and earlier must not move.
        original = ft.build_feature_table(self.df, True)
        changed_df = self.df.copy()
        position = 400
        changed_df.loc[position, "usage_kwh"] = 999.0
        changed = ft.build_feature_table(changed_df, True)
        cutoff = self.df.loc[position, "timestamp"]
        columns = ft.feature_columns(True)
        pd.testing.assert_frame_equal(original.loc[:cutoff, columns], changed.loc[:cutoff, columns])
        # ...but later hours legitimately do change (they can see this reading)
        self.assertNotEqual(original["lag_1"].loc[cutoff + pd.Timedelta(hours=1)],
                            changed["lag_1"].loc[cutoff + pd.Timedelta(hours=1)])

    def test_history_lookup_matches_training_table(self):
        table = ft.build_feature_table(self.df, True)
        hourly = ft.to_hourly_series(self.df)
        ts = table.index[150]
        lags, missing = ft.lags_from_history(hourly, ts)
        self.assertEqual(missing, [])
        for name in ft.LAG_COLUMNS:
            self.assertAlmostEqual(lags[name], table.loc[ts, name])

    def test_history_lookup_reports_missing(self):
        hourly = ft.to_hourly_series(self.df)
        lags, missing = ft.lags_from_history(hourly, hourly.index[5])
        self.assertIsNone(lags)
        self.assertGreater(len(missing), 0)


class TestSplitAndModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = data_utils.generate_synthetic_data(days=60)
        cls.bundle = ml.train_model(cls.df)

    def test_chronological_split(self):
        table = ft.build_feature_table(self.df, True)
        train, test = ml.split_chronologically(table)
        self.assertEqual(len(train) + len(test), len(table))
        self.assertAlmostEqual(len(train) / len(table), 0.8, places=2)
        self.assertLess(train.index.max(), test.index.min())

    def test_model_configuration(self):
        model = self.bundle["model"]
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.min_samples_leaf, 2)
        self.assertEqual(model.random_state, ml.RANDOM_SEED)
        self.assertLess(self.bundle["train_end"], self.bundle["test_start"])

    def test_metrics_exist_and_beat_baseline(self):
        rf, base = self.bundle["metrics"]["random_forest"], self.bundle["metrics"]["baseline"]
        self.assertTrue(all(np.isfinite(v) for v in rf.values()))
        self.assertLess(rf["mae"], base["mae"])
        self.assertGreater(rf["r2"], base["r2"])

    def test_metric_formulas(self):
        result = ml.compute_metrics([1, 2, 3], [1, 2, 5])
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(4 / 3))
        self.assertAlmostEqual(result["r2"], -1.0)

    def test_works_without_temperature(self):
        bundle = ml.train_model(self.df.drop(columns="temperature_c"))
        self.assertFalse(bundle["use_temperature"])
        self.assertNotIn("temperature_c", bundle["feature_columns"])

    def test_small_dataset_gives_clear_error(self):
        tiny = data_utils.generate_synthetic_data(days=3)
        with self.assertRaisesRegex(ValueError, "at least"):
            ml.train_model(tiny)

    def test_prediction_output(self):
        hourly = ft.to_hourly_series(self.df)
        next_hour = self.df["timestamp"].max() + pd.Timedelta(hours=1)
        value, lags = ml.predict_for_timestamp(self.bundle, hourly, next_hour, temperature=8.0)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0)
        row = ft.make_input_row(next_hour, 8.0, lags, True)
        self.assertAlmostEqual(value, float(self.bundle["model"].predict(row)[0]))

    def test_prediction_without_history_explains_what_is_missing(self):
        hourly = ft.to_hourly_series(self.df)
        far_future = self.df["timestamp"].max() + pd.Timedelta(days=10)
        with self.assertRaises(ml.MissingHistoryError) as ctx:
            ml.predict_for_timestamp(self.bundle, hourly, far_future, temperature=8.0)
        self.assertIn("missing", str(ctx.exception))

    def test_manual_lags_allow_future_prediction(self):
        hourly = ft.to_hourly_series(self.df)
        far_future = self.df["timestamp"].max() + pd.Timedelta(days=10)
        lags = ft.lags_from_manual(1.0, 1.1, 1.2, 1.0, 0.9)
        value, _ = ml.predict_for_timestamp(self.bundle, hourly, far_future, 8.0, manual_lags=lags)
        self.assertTrue(np.isfinite(value))

    def test_save_load_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            _, first = ml.get_or_train(self.df, path)
            _, second = ml.get_or_train(self.df, path)
            _, forced = ml.get_or_train(self.df, path, force=True)
            self.assertEqual((first, second, forced), ("trained", "loaded", "trained"))
            other = data_utils.generate_synthetic_data(days=60, seed=7)
            _, changed = ml.get_or_train(other, path)
            self.assertEqual(changed, "trained")  # different data -> retrain


if __name__ == "__main__":
    unittest.main()
