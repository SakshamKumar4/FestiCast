"""
src/evaluation.py  –  Metrics & walk-forward validation
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ── Point metrics ─────────────────────────────────────────────────────────────
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE":   round(mae(y_true, y_pred), 4),
        "RMSE":  round(rmse(y_true, y_pred), 4),
        "MAPE":  round(mape(y_true, y_pred), 4),
        "SMAPE": round(smape(y_true, y_pred), 4),
        "R2":    round(r2(y_true, y_pred), 4),
    }


# ── Event-period metrics ───────────────────────────────────────────────────────
def compute_event_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          event_mask: np.ndarray) -> dict:
    em = event_mask.astype(bool)
    if em.sum() == 0:
        return {k: float("nan") for k in ["MAE", "RMSE", "MAPE", "SMAPE", "R2"]}
    return compute_all_metrics(y_true[em], y_pred[em])


# ── Walk-forward validation ───────────────────────────────────────────────────
def walk_forward_validate(model_fn, X: pd.DataFrame,
                          y: pd.Series, n_splits: int = config.N_SPLITS) -> dict:
    """
    model_fn: callable(X_train, y_train) -> fitted model with .predict(X_test)
    Returns averaged metrics across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = model_fn(X_tr, y_tr)
        preds = model.predict(X_te)
        preds = np.maximum(preds, 0)

        m = compute_all_metrics(y_te.values, preds)
        m["fold"] = fold + 1
        fold_metrics.append(m)
        logger.debug("Fold %d: %s", fold + 1, m)

    df = pd.DataFrame(fold_metrics)
    agg = {col: float(df[col].mean()) for col in ["MAE", "RMSE", "MAPE", "SMAPE", "R2"]}
    agg["folds"] = fold_metrics
    return agg


# ── Accuracy score (1 - MAPE/100) ────────────────────────────────────────────
def forecast_accuracy(mape_val: float) -> float:
    return max(0.0, 100.0 - mape_val)
