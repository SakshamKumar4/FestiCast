"""
src/train_models.py  –  Model training, persistence & inference
Implements: Naive, Moving Average, SARIMA, Prophet, XGBoost
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engineering import build_features, get_feature_cols
from src.evaluation import compute_all_metrics

logger = logging.getLogger(__name__)


# ── Model file path ───────────────────────────────────────────────────────────
def _model_path(model_name: str, store: int, item: int) -> str:
    slug = model_name.lower().replace(" ", "_")
    fname = f"{slug}_s{store}_i{item}.joblib"
    return os.path.join(config.MODEL_DIR, fname)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Naive Forecast
# ══════════════════════════════════════════════════════════════════════════════
class NaiveModel:
    """Last-value naive forecast."""
    def __init__(self):
        self.last_value = None
        self.name = "Naive"

    def fit(self, series: pd.Series):
        self.last_value = float(series.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_value)

    def predict_in_sample(self, series: pd.Series) -> np.ndarray:
        return series.shift(1).fillna(method="bfill").values


# ══════════════════════════════════════════════════════════════════════════════
# 2. Moving Average
# ══════════════════════════════════════════════════════════════════════════════
class MovingAverageModel:
    def __init__(self, window: int = config.DEFAULT_MA_WINDOW):
        self.window = window
        self.last_values = None
        self.name = "Moving Average"

    def fit(self, series: pd.Series):
        self.last_values = series.values[-self.window:]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        vals = list(self.last_values)
        preds = []
        for _ in range(horizon):
            p = np.mean(vals[-self.window:])
            preds.append(p)
            vals.append(p)
        return np.array(preds)

    def predict_in_sample(self, series: pd.Series) -> np.ndarray:
        return series.rolling(self.window, min_periods=1).mean().shift(1).fillna(method="bfill").values


# ══════════════════════════════════════════════════════════════════════════════
# 3. SARIMA
# ══════════════════════════════════════════════════════════════════════════════
class SARIMAModel:
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,0,7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.name = "SARIMA"

    def fit(self, series: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # Use weekly data to keep SARIMA tractable
        weekly = series.resample("W").mean()
        self.model = SARIMAX(
            weekly,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        self._series_weekly = weekly
        return self

    def predict(self, horizon: int) -> np.ndarray:
        # Forecast in weeks, upsample to days
        weeks = max(1, horizon // 7 + 2)
        fc = self.model.get_forecast(steps=weeks)
        mean_w = fc.predicted_mean.values

        # Repeat each week's value for 7 days then trim
        daily = np.repeat(mean_w, 7)[:horizon]
        return np.maximum(daily, 0)

    def predict_in_sample(self, series: pd.Series) -> np.ndarray:
        weekly = series.resample("W").mean()
        in_sample = self.model.fittedvalues
        # Upsample to daily length of series
        daily = in_sample.reindex(series.index, method="ffill")
        return daily.fillna(method="bfill").values


# ══════════════════════════════════════════════════════════════════════════════
# 4. Prophet
# ══════════════════════════════════════════════════════════════════════════════
class ProphetModel:
    def __init__(self, include_events: bool = True):
        self.model = None
        self.include_events = include_events
        self.name = "Prophet"
        self._train_df = None

    def fit(self, series: pd.Series):
        from prophet import Prophet
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["y"] = df["y"].clip(lower=0)

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
        )

        if self.include_events:
            from src.event_calendar import get_calendar
            cal = get_calendar()
            start = df["ds"].min().date()
            end   = (df["ds"].max() + pd.Timedelta(days=180)).date()

            holidays_list = []
            for ev in cal._events:
                for peak in ev.dates_in_range(start, end):
                    holidays_list.append({
                        "holiday":         ev.name,
                        "ds":              pd.Timestamp(peak),
                        "lower_window":   -ev.window_before,
                        "upper_window":    ev.window_after,
                    })
            if holidays_list:
                holidays_df = pd.DataFrame(holidays_list)
                m = Prophet(
                    holidays=holidays_df,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10,
                )

        m.fit(df)
        self.model = m
        self._train_df = df
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=horizon, freq="D")
        fc = self.model.predict(future)
        return np.maximum(fc["yhat"].values[-horizon:], 0)

    def predict_in_sample(self, series: pd.Series) -> np.ndarray:
        fc = self.model.predict(self._train_df.rename(columns={"y": "yhat"})[["ds"]])
        return np.maximum(fc["yhat"].values, 0)

    def get_forecast_df(self, horizon: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon, freq="D")
        return self.model.predict(future).tail(horizon)


# ══════════════════════════════════════════════════════════════════════════════
# 5. XGBoost
# ══════════════════════════════════════════════════════════════════════════════
class XGBoostModel:
    def __init__(self, include_events: bool = True, n_estimators: int = 300,
                 max_depth: int = 6, learning_rate: float = 0.05):
        self.include_events = include_events
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None
        self.name = "XGBoost"

    def _prepare(self, series: pd.Series):
        from src.feature_engineering import build_features, get_feature_cols
        feat_df = build_features(series.to_frame(), include_events=self.include_events)
        self.feature_cols = get_feature_cols(self.include_events)
        # Keep only cols that exist
        self.feature_cols = [c for c in self.feature_cols if c in feat_df.columns]
        feat_df = feat_df.dropna(subset=self.feature_cols)
        X = feat_df[self.feature_cols]
        y = feat_df[config.SALES_COL]
        return X, y

    def fit(self, series: pd.Series):
        import xgboost as xgb
        X, y = self._prepare(series)
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        self.model.fit(X, y)
        self._last_known = series.copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        from src.feature_engineering import build_features, get_feature_cols, add_event_features, add_calendar_features
        series = self._last_known.copy()
        last_date = series.index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        preds = []
        extended = series.copy()

        for fd in future_dates:
            temp = pd.concat([extended, pd.Series([0.0], index=[fd])])
            feat_df = build_features(temp.to_frame(), include_events=self.include_events)
            feat_df = feat_df[feat_df[config.DATE_COL] == fd]
            if feat_df.empty:
                preds.append(extended.iloc[-1])
                continue
            cols = [c for c in self.feature_cols if c in feat_df.columns]
            X = feat_df[cols]
            # fill missing feature cols with 0
            for c in self.feature_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[self.feature_cols]
            pred = float(self.model.predict(X)[0])
            pred = max(0.0, pred)
            preds.append(pred)
            extended = pd.concat([extended, pd.Series([pred], index=[fd])])

        return np.array(preds)

    def predict_in_sample(self, series: pd.Series) -> np.ndarray:
        from src.feature_engineering import build_features
        feat_df = build_features(series.to_frame(), include_events=self.include_events)
        feat_df = feat_df.dropna(subset=self.feature_cols)
        X = feat_df[self.feature_cols]
        preds = self.model.predict(X)
        # Align length
        pad = len(series) - len(preds)
        if pad > 0:
            preds = np.concatenate([np.full(pad, preds[0]), preds])
        return np.maximum(preds, 0)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        scores = self.model.feature_importances_
        return pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": scores,
        }).sort_values("importance", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Registry & persistence
# ══════════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "Naive":          lambda ie: NaiveModel(),
    "Moving Average": lambda ie: MovingAverageModel(),
    "SARIMA":         lambda ie: SARIMAModel(),
    "Prophet":        lambda ie: ProphetModel(include_events=ie),
    "XGBoost":        lambda ie: XGBoostModel(include_events=ie),
}


def get_or_train_model(series: pd.Series, model_name: str,
                       store: int, item: int,
                       include_events: bool = True,
                       force_retrain: bool = False) -> Any:
    """
    Load model from disk if available, otherwise train and save.
    """
    path = _model_path(model_name, store, item)

    if not force_retrain and os.path.exists(path):
        logger.info("Loading cached model: %s", path)
        try:
            return joblib.load(path)
        except Exception as e:
            logger.warning("Cache load failed (%s), retraining.", e)

    logger.info("Training %s for store=%d item=%d", model_name, store, item)
    t0 = time.time()
    model = MODEL_REGISTRY[model_name](include_events)

    if model_name in ("Naive", "Moving Average"):
        model.fit(series[config.SALES_COL] if config.SALES_COL in series else series)
    elif model_name == "SARIMA":
        s = series[config.SALES_COL] if config.SALES_COL in series.columns else series
        s.index = pd.DatetimeIndex(s.index)
        model.fit(s)
    else:
        s = series[config.SALES_COL] if config.SALES_COL in series.columns else series
        s.index = pd.DatetimeIndex(s.index)
        model.fit(s)

    elapsed = time.time() - t0
    logger.info("Trained %s in %.1fs", model_name, elapsed)
    joblib.dump(model, path)
    return model


def train_all_models(series: pd.Series, store: int, item: int,
                     include_events: bool = True,
                     force_retrain: bool = False) -> Dict[str, Any]:
    """Train (or load) all 5 models for one store-item pair."""
    result = {}
    for name in config.MODELS:
        try:
            result[name] = get_or_train_model(
                series, name, store, item, include_events, force_retrain)
        except Exception as e:
            logger.error("Model %s failed: %s", name, e)
    return result


def benchmark_models(series: pd.Series, store: int, item: int,
                     include_events: bool = True) -> pd.DataFrame:
    """
    Train all models and collect in-sample metrics + timing.
    Returns a comparison DataFrame.
    """
    from src.evaluation import compute_all_metrics

    s = series[config.SALES_COL] if config.SALES_COL in series.columns else series
    s.index = pd.DatetimeIndex(s.index)

    rows = []
    for name in config.MODELS:
        t0 = time.time()
        try:
            model = get_or_train_model(series, name, store, item, include_events)
            train_time = round(time.time() - t0, 2)

            t1 = time.time()
            preds = model.predict_in_sample(s)
            infer_time = round(time.time() - t1, 4)

            # Align
            n = min(len(s), len(preds))
            metrics = compute_all_metrics(s.values[-n:], preds[-n:])
            rows.append({
                "Model":          name,
                "MAE":            metrics["MAE"],
                "RMSE":           metrics["RMSE"],
                "MAPE":           metrics["MAPE"],
                "SMAPE":          metrics["SMAPE"],
                "R²":             metrics["R2"],
                "Train Time (s)": train_time,
                "Infer Time (s)": infer_time,
            })
        except Exception as e:
            logger.error("Benchmark %s failed: %s", name, e)
            rows.append({"Model": name, "MAE": None, "RMSE": None,
                         "MAPE": None, "SMAPE": None, "R²": None,
                         "Train Time (s)": None, "Infer Time (s)": None})

    return pd.DataFrame(rows)
