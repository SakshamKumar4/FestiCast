"""
src/forecasting.py  –  Forecast generation with confidence intervals
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.train_models import get_or_train_model
from src.event_calendar import get_calendar

logger = logging.getLogger(__name__)


def generate_forecast(
    series: pd.DataFrame,
    store: int,
    item: int,
    model_name: str = config.DEFAULT_MODEL,
    horizon: int = config.DEFAULT_FORECAST_HORIZON,
    include_events: bool = True,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Generate horizon-day forecast for a single store-item pair.

    Returns DataFrame with columns:
      date, yhat, yhat_lower, yhat_upper,
      is_event, event_name, event_category, event_window
    """
    s = series[config.SALES_COL] if config.SALES_COL in series.columns else series
    s = s.astype(float)
    s.index = pd.DatetimeIndex(s.index)

    # Train or load model
    model = get_or_train_model(
        series, model_name, store, item, include_events
    )

    # Predict
    yhat = model.predict(horizon)
    yhat = np.maximum(yhat, 0)

    # Confidence intervals using historical residual std
    in_sample = model.predict_in_sample(s)
    n = min(len(s), len(in_sample))
    residuals = s.values[-n:] - in_sample[-n:]
    std = float(np.std(residuals)) if len(residuals) > 1 else float(np.std(s))

    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    margin = z * std

    last_date = s.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    fc_df = pd.DataFrame({
        "date":        future_dates,
        "yhat":        yhat,
        "yhat_lower":  np.maximum(yhat - margin, 0),
        "yhat_upper":  yhat + margin,
        "store":       store,
        "item":        item,
        "model":       model_name,
    })

    # Attach event info
    if include_events:
        cal = get_calendar()
        date_idx = pd.DatetimeIndex(future_dates)
        ev_df = cal.build_event_dataframe(date_idx)
        ev_df = ev_df.reset_index().rename(columns={"index": "date"})
        fc_df = fc_df.merge(ev_df[["date", "is_event", "event_name",
                                    "event_category", "event_window",
                                    "festival_score"]], on="date", how="left")
    else:
        fc_df["is_event"]       = 0
        fc_df["event_name"]     = ""
        fc_df["event_category"] = "none"
        fc_df["event_window"]   = 0
        fc_df["festival_score"] = 0.0

    fc_df["is_event"]   = fc_df["is_event"].fillna(0).astype(int)
    fc_df["event_name"] = fc_df["event_name"].fillna("")

    return fc_df


def generate_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str = config.DEFAULT_MODEL,
    include_events: bool = True,
) -> pd.DataFrame:
    """
    Generate submission-ready forecasts for every store-item pair in test.csv
    """
    results = []
    pairs = test[[config.STORE_COL, config.ITEM_COL]].drop_duplicates().values

    for store, item in pairs:
        # Get train series for this pair
        mask = (train[config.STORE_COL] == store) & (train[config.ITEM_COL] == item)
        series = (train[mask]
                  .set_index(config.DATE_COL)[[config.SALES_COL]]
                  .sort_index())

        # Get test dates for this pair
        tmask = (test[config.STORE_COL] == store) & (test[config.ITEM_COL] == item)
        test_rows = test[tmask].sort_values(config.DATE_COL)
        horizon = len(test_rows)

        try:
            fc = generate_forecast(
                series, store, item, model_name, horizon, include_events
            )
            test_rows = test_rows.reset_index(drop=True)
            fc = fc.reset_index(drop=True)
            results.append(pd.DataFrame({
                config.ID_COL:    test_rows[config.ID_COL],
                config.SALES_COL: fc["yhat"].round().astype(int),
            }))
        except Exception as e:
            logger.error("Forecast failed store=%d item=%d: %s", store, item, e)
            results.append(pd.DataFrame({
                config.ID_COL:    test_rows[config.ID_COL],
                config.SALES_COL: int(series[config.SALES_COL].mean()),
            }))

    submission = pd.concat(results, ignore_index=True).sort_values(config.ID_COL)
    return submission
