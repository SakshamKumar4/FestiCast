"""
src/utils.py  –  Shared helper functions for FestiCast
"""

from __future__ import annotations

import os
import io
import logging
import hashlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt_number(n: float, decimals: int = 0) -> str:
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.{decimals}f}"


def fmt_pct(n: float, decimals: int = 2) -> str:
    return f"{n:.{decimals}f}%"


def fmt_currency(n: float) -> str:
    return f"₹{fmt_number(n)}"


# ── Date helpers ──────────────────────────────────────────────────────────────
def to_datetime(series: pd.Series, fmt: str = "%Y-%m-%d") -> pd.Series:
    return pd.to_datetime(series, format=fmt, errors="coerce")


def date_range_str(start: Any, end: Any) -> str:
    s = pd.Timestamp(start).strftime("%d %b %Y")
    e = pd.Timestamp(end).strftime("%d %b %Y")
    return f"{s} → {e}"


# ── File helpers ──────────────────────────────────────────────────────────────
def safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── DataFrame helpers ─────────────────────────────────────────────────────────
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# ── Streamlit metric card ─────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str | None = None,
                delta_positive: bool = True, icon: str = "") -> None:
    delta_color = "#17A34A" if delta_positive else "#DC2626"
    delta_html = ""
    if delta:
        arrow = "▲" if delta_positive else "▼"
        delta_html = (
            f'<div style="color:{delta_color};font-size:12px;margin-top:2px;">'
            f"{arrow} {delta}</div>"
        )
    st.markdown(
        f"""
        <div style="
            background:#fff;
            border:1px solid #E2E8F0;
            border-radius:12px;
            padding:18px 20px;
            box-shadow:0 1px 4px rgba(0,0,0,.06);
        ">
            <div style="color:#64748B;font-size:12px;font-weight:600;
                        letter-spacing:.5px;text-transform:uppercase;">
                {icon} {label}
            </div>
            <div style="color:#1E293B;font-size:26px;font-weight:700;
                        margin-top:6px;line-height:1.1;">
                {value}
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Progress / spinner wrapper ─────────────────────────────────────────────────
def run_with_spinner(msg: str, fn, *args, **kwargs):
    with st.spinner(msg):
        return fn(*args, **kwargs)


# ── Colour palette for Plotly ─────────────────────────────────────────────────
PLOT_COLORS = [
    "#0066CC", "#17A34A", "#D97706", "#DC2626",
    "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
    "#EA580C", "#6366F1",
]


def get_color(idx: int) -> str:
    return PLOT_COLORS[idx % len(PLOT_COLORS)]


# ── Confidence band helper ────────────────────────────────────────────────────
def add_confidence_band(
    fig,
    x,
    yhat: np.ndarray,
    std: np.ndarray,
    z: float = 1.96,
    name: str = "95% CI",
    color: str = "rgba(0,102,204,0.12)",
    row: int = 1,
    col: int = 1,
):
    import plotly.graph_objects as go

    upper = yhat + z * std
    lower = yhat - z * std
    lower = np.maximum(lower, 0)

    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            showlegend=True,
            name=name,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    return fig
