"""
src/feature_engineering.py  –  All feature creation for FestiCast
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.event_calendar import get_calendar

logger = logging.getLogger(__name__)


# ── Calendar features ─────────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame, date_col: str = config.DATE_COL) -> pd.DataFrame:
    df = df.copy()
    dt = df[date_col] if date_col in df.columns else df.index.to_series()
    dt = pd.to_datetime(dt)

    df["day_of_week"]  = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["day_of_year"]  = dt.dt.dayofyear
    df["week"]         = dt.dt.isocalendar().week.astype(int)
    df["month"]        = dt.dt.month
    df["quarter"]      = dt.dt.quarter
    df["year"]         = dt.dt.year
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

    # Cyclical encoding for day-of-week and month
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ── Lag features ──────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame,
                     target: str = config.SALES_COL,
                     lags: list[int] = config.LAG_FEATURES) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    return df


# ── Rolling features ──────────────────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame,
                         target: str = config.SALES_COL,
                         windows: list[int] = config.ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"roll_mean_{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"]  = df[target].shift(1).rolling(w, min_periods=1).std().fillna(0)
    df["roll_mean_30"] = df[target].shift(1).rolling(30, min_periods=1).mean()
    return df


# ── Event features ────────────────────────────────────────────────────────────
def add_event_features(df: pd.DataFrame,
                       date_col: str = config.DATE_COL) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[date_col] if date_col in df.columns else df.index)
    date_idx = pd.DatetimeIndex(dates.values)

    cal = get_calendar()
    event_df = cal.build_event_dataframe(date_idx)

    df = df.reset_index(drop=False) if date_col not in df.columns else df
    df[date_col] = dates.values

    df = df.merge(
        event_df.reset_index().rename(columns={"index": date_col}),
        on=date_col, how="left"
    )

    # Fill defaults
    df["is_event"]         = df["is_event"].fillna(0).astype(int)
    df["event_window"]     = df["event_window"].fillna(0).astype(int)
    df["event_importance"] = df["event_importance"].fillna(0.0)
    df["festival_score"]   = df["festival_score"].fillna(0.0)
    df["days_until_event"] = df["days_until_event"].fillna(999)
    df["days_since_event"] = df["days_since_event"].fillna(999)
    df["event_category"]   = df["event_category"].fillna("none")
    df["event_name"]       = df["event_name"].fillna("")

    return df


# ── Full feature pipeline ─────────────────────────────────────────────────────
def build_features(series: pd.DataFrame,
                   include_events: bool = True,
                   lags: list[int] = config.LAG_FEATURES,
                   rolling_windows: list[int] = config.ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Input: single store-item series with DatetimeIndex and 'sales' column
    Output: feature DataFrame ready for modelling
    """
    df = series.reset_index()
    df.columns = [config.DATE_COL, config.SALES_COL]

    df = add_calendar_features(df)
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=rolling_windows)

    if include_events:
        df = add_event_features(df)

    return df


# ── Feature column lists ──────────────────────────────────────────────────────
CALENDAR_FEATURES = [
    "day_of_week", "day_of_month", "day_of_year", "week",
    "month", "quarter", "year", "is_weekend",
    "is_month_end", "is_month_start", "is_quarter_end",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]

LAG_FEAT_COLS = [f"lag_{l}" for l in config.LAG_FEATURES]

ROLLING_FEAT_COLS = [
    f"roll_mean_{w}" for w in config.ROLLING_WINDOWS
] + [f"roll_std_{w}" for w in config.ROLLING_WINDOWS] + ["roll_mean_30"]

EVENT_FEAT_COLS = [
    "is_event", "event_window", "event_importance",
    "festival_score", "days_until_event", "days_since_event",
]

ALL_FEATURES = CALENDAR_FEATURES + LAG_FEAT_COLS + ROLLING_FEAT_COLS


def get_feature_cols(include_events: bool = True) -> list[str]:
    base = CALENDAR_FEATURES + LAG_FEAT_COLS + ROLLING_FEAT_COLS
    if include_events:
        base = base + EVENT_FEAT_COLS
    return base
