"""
src/preprocessing.py  –  Data loading & cleaning for FestiCast
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ── Load raw CSVs ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, submission) DataFrames."""
    for path in [config.TRAIN_FILE, config.TEST_FILE, config.SUBMISSION_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    train = pd.read_csv(config.TRAIN_FILE, parse_dates=[config.DATE_COL])
    test  = pd.read_csv(config.TEST_FILE,  parse_dates=[config.DATE_COL])
    sub   = pd.read_csv(config.SUBMISSION_FILE)

    logger.info("Loaded train=%s test=%s", train.shape, test.shape)
    return train, test, sub


# ── Validate schema ───────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


# ── Clean train ───────────────────────────────────────────────────────────────
def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_schema(df, [config.DATE_COL, config.STORE_COL,
                         config.ITEM_COL, config.SALES_COL], "train")

    # Type coercion
    df[config.DATE_COL]  = pd.to_datetime(df[config.DATE_COL])
    df[config.STORE_COL] = df[config.STORE_COL].astype(int)
    df[config.ITEM_COL]  = df[config.ITEM_COL].astype(int)
    df[config.SALES_COL] = pd.to_numeric(df[config.SALES_COL], errors="coerce")

    # Remove negatives
    df[config.SALES_COL] = df[config.SALES_COL].clip(lower=0)

    # Drop rows with null dates or sales
    before = len(df)
    df = df.dropna(subset=[config.DATE_COL, config.SALES_COL])
    if len(df) < before:
        logger.warning("Dropped %d rows with nulls", before - len(df))

    # Sort
    df = df.sort_values([config.STORE_COL, config.ITEM_COL, config.DATE_COL])
    df = df.reset_index(drop=True)

    return df


# ── Filter for one store/item ─────────────────────────────────────────────────
def filter_series(df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
    mask = (df[config.STORE_COL] == store) & (df[config.ITEM_COL] == item)
    series = df[mask][[config.DATE_COL, config.SALES_COL]].copy()
    series = series.set_index(config.DATE_COL).sort_index()
    # Fill any missing dates with 0
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_idx, fill_value=0)
    series.index.name = config.DATE_COL
    return series


# ── Aggregate helpers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(config.DATE_COL)[config.SALES_COL]
              .sum().reset_index().sort_values(config.DATE_COL))


@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[config.DATE_COL].dt.to_period("M").astype(str)
    return df.groupby("month")[config.SALES_COL].sum().reset_index()


@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def aggregate_by_store(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(config.STORE_COL)[config.SALES_COL]
              .sum().reset_index().sort_values(config.SALES_COL, ascending=False))


@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def aggregate_by_item(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(config.ITEM_COL)[config.SALES_COL]
              .sum().reset_index().sort_values(config.SALES_COL, ascending=False))


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_summary(df: pd.DataFrame) -> dict:
    return {
        "total_sales":    int(df[config.SALES_COL].sum()),
        "mean_daily":     float(df.groupby(config.DATE_COL)[config.SALES_COL].sum().mean()),
        "max_daily":      int(df.groupby(config.DATE_COL)[config.SALES_COL].sum().max()),
        "num_stores":     int(df[config.STORE_COL].nunique()),
        "num_items":      int(df[config.ITEM_COL].nunique()),
        "date_start":     df[config.DATE_COL].min(),
        "date_end":       df[config.DATE_COL].max(),
        "num_days":       int((df[config.DATE_COL].max() - df[config.DATE_COL].min()).days + 1),
    }
