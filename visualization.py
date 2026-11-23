"""
src/visualization.py  –  All Plotly charts for FestiCast
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

T = config.CHART_TEMPLATE
H = config.CHART_HEIGHT
PC = config.PRIMARY_COLOR


def _base_layout(title: str = "", height: int = H) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=config.TEXT_PRIMARY), x=0.01),
        template=T,
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        font=dict(family="Inter, Segoe UI, Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )


# ── 1. Forecast Chart ─────────────────────────────────────────────────────────
def plot_forecast(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    store: int,
    item: int,
    model_name: str,
    tail_days: int = 90,
) -> go.Figure:
    fig = go.Figure()

    hist = history.tail(tail_days)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="Actual Sales",
        line=dict(color=PC, width=2),
        hovertemplate="%{x|%b %d, %Y}<br>Sales: <b>%{y:.0f}</b><extra></extra>",
    ))

    # CI band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(0,102,204,0.10)",
        line=dict(width=0), name="95% CI", showlegend=True,
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["yhat"],
        mode="lines", name=f"{model_name} Forecast",
        line=dict(color="#D97706", width=2.5, dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>Forecast: <b>%{y:.0f}</b><extra></extra>",
    ))

    # Event markers
    ev = forecast_df[forecast_df["is_event"] == 1]
    if not ev.empty:
        fig.add_trace(go.Scatter(
            x=ev["date"], y=ev["yhat"],
            mode="markers", name="Event Day",
            marker=dict(color="#DC2626", size=9, symbol="diamond"),
            hovertemplate="%{x|%b %d}<br><b>%{text}</b><extra></extra>",
            text=ev["event_name"],
        ))

    fig.update_layout(
        **_base_layout(f"Store {store} · Item {item} — {model_name} Forecast"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", title="Date")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", title="Units Sold")
    return fig


# ── 2. Daily Sales Overview ───────────────────────────────────────────────────
def plot_daily_sales(daily_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df[config.DATE_COL], y=daily_df[config.SALES_COL],
        mode="lines", name="Daily Sales",
        line=dict(color=PC, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,102,204,0.07)",
        hovertemplate="%{x|%b %d, %Y}<br>Sales: <b>%{y:,}</b><extra></extra>",
    ))
    # 30-day MA
    ma = daily_df[config.SALES_COL].rolling(30, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=daily_df[config.DATE_COL], y=ma,
        mode="lines", name="30-day MA",
        line=dict(color="#D97706", width=2, dash="dot"),
        hovertemplate="%{x|%b %d}<br>30d MA: <b>%{y:.0f}</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("Total Daily Sales (All Stores · All Items)"), hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", title="Units")
    return fig


# ── 3. Monthly Sales Bar ──────────────────────────────────────────────────────
def plot_monthly_sales(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=monthly_df["month"], y=monthly_df[config.SALES_COL],
        marker_color=PC, marker_line_width=0,
        hovertemplate="Month: %{x}<br>Sales: <b>%{y:,}</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("Monthly Sales Totals"))
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title="Units", showgrid=True, gridcolor="#F1F5F9")
    return fig


# ── 4. Store Comparison ───────────────────────────────────────────────────────
def plot_store_comparison(store_df: pd.DataFrame) -> go.Figure:
    colors = [PC if i == 0 else "#93C5FD"
              for i in range(len(store_df))]
    fig = go.Figure(go.Bar(
        x=store_df[config.STORE_COL].astype(str).apply(lambda x: f"Store {x}"),
        y=store_df[config.SALES_COL],
        marker_color=colors, marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Total Sales: <b>%{y:,}</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("Sales by Store"))
    fig.update_yaxes(title="Total Units", showgrid=True, gridcolor="#F1F5F9")
    return fig


# ── 5. Item Comparison ────────────────────────────────────────────────────────
def plot_item_comparison(item_df: pd.DataFrame) -> go.Figure:
    top = item_df.head(20)
    fig = go.Figure(go.Bar(
        x=top[config.ITEM_COL].astype(str).apply(lambda x: f"Item {x}"),
        y=top[config.SALES_COL],
        marker_color="#0891B2", marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Total Sales: <b>%{y:,}</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("Top 20 Items by Total Sales"))
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title="Total Units", showgrid=True, gridcolor="#F1F5F9")
    return fig


# ── 6. Model Comparison ───────────────────────────────────────────────────────
def plot_model_comparison(bench_df: pd.DataFrame) -> go.Figure:
    metrics = ["MAE", "RMSE", "MAPE"]
    fig = make_subplots(rows=1, cols=len(metrics),
                        subplot_titles=metrics)

    colors = ["#0066CC","#17A34A","#D97706","#DC2626","#7C3AED"]
    b = bench_df.dropna()

    for i, m in enumerate(metrics, 1):
        fig.add_trace(go.Bar(
            x=b["Model"], y=b[m],
            marker_color=colors[:len(b)],
            showlegend=False,
            hovertemplate=f"<b>%{{x}}</b><br>{m}: <b>%{{y:.2f}}</b><extra></extra>",
        ), row=1, col=i)

    fig.update_layout(
        **_base_layout("Model Comparison — Error Metrics", height=380),
        showlegend=False,
    )
    for i in range(1, len(metrics)+1):
        fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", row=1, col=i)
    return fig


def plot_model_radar(bench_df: pd.DataFrame) -> go.Figure:
    b = bench_df.dropna()
    # Normalise metrics (lower=better → invert)
    cats = ["MAE", "RMSE", "MAPE", "SMAPE"]
    fig = go.Figure()
    for _, row in b.iterrows():
        vals = []
        for c in cats:
            mx = b[c].max()
            vals.append(1 - row[c]/mx if mx > 0 else 0)
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=row["Model"],
        ))
    fig.update_layout(
        **_base_layout("Model Performance Radar", height=420),
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    )
    return fig


# ── 7. Seasonality ────────────────────────────────────────────────────────────
def plot_seasonality(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["dow"]   = df[config.DATE_COL].dt.day_name()
    df["month"] = df[config.DATE_COL].dt.month_name()

    dow_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

    dow_avg = (df.groupby("dow")[config.SALES_COL].mean()
                 .reindex(dow_order).reset_index())
    mon_avg = (df.groupby("month")[config.SALES_COL].mean()
                 .reindex(month_order).reset_index())

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Avg Sales by Day of Week",
                                        "Avg Sales by Month"))
    fig.add_trace(go.Bar(
        x=dow_avg["dow"], y=dow_avg[config.SALES_COL],
        marker_color=PC, showlegend=False,
        hovertemplate="<b>%{x}</b><br>Avg: <b>%{y:.1f}</b><extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=mon_avg["month"], y=mon_avg[config.SALES_COL],
        marker_color="#17A34A", showlegend=False,
        hovertemplate="<b>%{x}</b><br>Avg: <b>%{y:.1f}</b><extra></extra>",
    ), row=1, col=2)

    fig.update_layout(**_base_layout("Seasonality Patterns", height=380))
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", title="Avg Units")
    return fig


# ── 8. Trend Decomposition ────────────────────────────────────────────────────
def plot_decomposition(series: pd.Series) -> go.Figure:
    from statsmodels.tsa.seasonal import seasonal_decompose
    try:
        weekly = series.resample("W").mean().dropna()
        if len(weekly) < 14:
            raise ValueError("Too short")
        result = seasonal_decompose(weekly, model="additive", period=52, extrapolate_trend="freq")

        fig = make_subplots(rows=4, cols=1,
                            subplot_titles=("Observed","Trend","Seasonal","Residual"),
                            shared_xaxes=True, vertical_spacing=0.06)
        components = [
            (result.observed,  PC,       "Observed"),
            (result.trend,     "#17A34A","Trend"),
            (result.seasonal,  "#D97706","Seasonal"),
            (result.resid,     "#DC2626","Residual"),
        ]
        for i, (comp, color, name) in enumerate(components, 1):
            fig.add_trace(go.Scatter(
                x=comp.index, y=comp.values,
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
                showlegend=False,
            ), row=i, col=1)

        fig.update_layout(**_base_layout("Trend Decomposition (Weekly Aggregation)", height=580))
        for i in range(1, 5):
            fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", row=i, col=1)
        return fig
    except Exception:
        # fallback simple chart
        fig = go.Figure()
        ma = series.rolling(30, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                  mode="lines", name="Observed", line=dict(color=PC)))
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values,
                                  mode="lines", name="Trend (30-day MA)",
                                  line=dict(color="#17A34A", dash="dash")))
        fig.update_layout(**_base_layout("Trend (Decomposition unavailable — insufficient data)"))
        return fig


# ── 9. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance(fi_df: pd.DataFrame) -> go.Figure:
    top = fi_df.head(15)
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"],
        orientation="h",
        marker_color=PC, marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Importance: <b>%{x:.4f}</b><extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("XGBoost Feature Importance (Top 15)", height=420),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(title="Importance Score", showgrid=True, gridcolor="#F1F5F9")
    return fig


# ── 10. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame) -> go.Figure:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = [c for c in num_cols if c not in [config.ID_COL]][:12]
    corr = df[keep].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout("Feature Correlation Heatmap", height=480))
    return fig


# ── 11. Event Calendar Gantt ──────────────────────────────────────────────────
def plot_event_calendar(year: int = 2017) -> go.Figure:
    from src.event_calendar import get_calendar
    import datetime
    cal = get_calendar()
    start = datetime.date(year, 1, 1)
    end   = datetime.date(year, 12, 31)

    rows = []
    for ev in cal._events:
        for peak in ev.dates_in_range(start, end):
            win_s = peak - datetime.timedelta(days=ev.window_before)
            win_e = peak + datetime.timedelta(days=ev.window_after)
            rows.append({
                "Event":    ev.name,
                "Category": ev.category.title(),
                "Start":    pd.Timestamp(win_s),
                "End":      pd.Timestamp(win_e),
                "Peak":     pd.Timestamp(peak),
            })

    if not rows:
        fig = go.Figure()
        fig.update_layout(**_base_layout(f"Event Calendar {year}"))
        return fig

    df = pd.DataFrame(rows).sort_values("Start")
    cat_colors = {
        "Festival": "#DC2626", "Holiday": "#7C3AED",
        "Business": "#0891B2", "Custom": "#17A34A",
    }
    fig = px.timeline(
        df, x_start="Start", x_end="End", y="Event",
        color="Category",
        color_discrete_map=cat_colors,
        hover_data={"Peak": True, "Category": True},
        title=f"Event Calendar {year}",
    )
    fig.update_layout(**_base_layout(f"Event Calendar — {year}", height=max(380, len(rows)*22)))
    fig.update_yaxes(autorange="reversed")
    return fig


# ── 12. Weekly Heatmap ────────────────────────────────────────────────────────
def plot_weekly_heatmap(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["week"] = df[config.DATE_COL].dt.isocalendar().week.astype(int)
    df["dow"]  = df[config.DATE_COL].dt.dayofweek
    pivot = df.groupby(["dow","week"])[config.SALES_COL].mean().unstack(fill_value=0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        colorscale="Blues",
        hovertemplate="Week %{x}<br>%{y}<br>Avg Sales: <b>%{z:.0f}</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("Weekly Sales Heatmap (Avg Units)", height=280))
    fig.update_xaxes(title="Week of Year")
    return fig
