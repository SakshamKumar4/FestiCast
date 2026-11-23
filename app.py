"""
app.py  –  FestiCast Main Application Entry Point
AI-Powered Event-Aware Retail Demand Forecasting Platform
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="FestiCast — Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }

/* ── App background ── */
.stApp { background: #F8FAFC; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94A3B8 !important; font-size: 11px; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0066CC 0%, #004A99 100%);
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 24px;
    color: white;
}
.page-header h1 { margin: 0; font-size: 22px; font-weight: 700; color: white; }
.page-header p  { margin: 4px 0 0 0; font-size: 13px; color: rgba(255,255,255,0.75); }

/* ── Cards ── */
.fc-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
    margin-bottom: 16px;
}
.fc-section-title {
    font-size: 13px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-bottom: 12px;
}

/* ── Metric cards ── */
.metric-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px; }
.metric-box {
    flex:1; min-width:140px;
    background:white;
    border:1px solid #E2E8F0;
    border-radius:12px;
    padding:16px 18px;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
}
.metric-box .label {
    font-size:11px; font-weight:600;
    color:#64748B; text-transform:uppercase; letter-spacing:.5px;
}
.metric-box .value { font-size:26px; font-weight:700; color:#1E293B; margin-top:4px; }
.metric-box .delta { font-size:11px; margin-top:2px; }

/* ── Tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stButton > button[kind="primary"] {
    background: #0066CC !important;
    border: none !important;
    color: white !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    padding: 20px 16px 16px 16px;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 8px;
}
.sidebar-brand .logo { font-size:20px; font-weight:700; color:#F8FAFC !important; }
.sidebar-brand .sub  { font-size:11px; color:#64748B !important; margin-top:2px; }

/* ── Nav items ── */
.nav-label {
    font-size:10px; font-weight:600;
    color:#475569 !important; text-transform:uppercase;
    letter-spacing:.6px; padding:12px 16px 4px 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Imports (after page config) ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io

import config
from src.preprocessing   import load_raw_data, clean_train, aggregate_daily, aggregate_monthly, aggregate_by_store, aggregate_by_item, compute_summary, filter_series
from src.feature_engineering import build_features
from src.visualization   import (plot_forecast, plot_daily_sales, plot_monthly_sales,
                                  plot_store_comparison, plot_item_comparison,
                                  plot_model_comparison, plot_model_radar,
                                  plot_seasonality, plot_decomposition,
                                  plot_feature_importance, plot_correlation,
                                  plot_event_calendar, plot_weekly_heatmap)
from src.train_models    import benchmark_models, get_or_train_model, XGBoostModel
from src.forecasting     import generate_forecast, generate_submission
from src.event_calendar  import get_calendar, Event
from src.utils           import fmt_number, fmt_pct, df_to_csv_bytes, df_to_excel_bytes


# ── PDF export helper ─────────────────────────────────────────────────────────
def _make_pdf(title: str, content_lines: list[str]) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "FestiCast — Retail Demand Forecasting", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.set_draw_color(0, 102, 204)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)
    for line in content_lines:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, str(line))
    return pdf.output(dest="S").encode("latin-1")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="logo">📈 FestiCast</div>
            <div class="sub">Event-Aware Retail Forecasting</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            label="",
            options=["🏠 Dashboard", "🔮 Forecast", "📊 Analytics",
                     "🏆 Model Comparison", "📄 Reports", "⚙️ Settings"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown('<div class="nav-label">Quick Filters</div>', unsafe_allow_html=True)
        store = st.selectbox("Store", list(range(1, config.NUM_STORES + 1)), index=0)
        item  = st.selectbox("Item",  list(range(1, config.NUM_ITEMS  + 1)), index=0)

        st.markdown("---")
        st.markdown(
            '<div style="font-size:10px;color:#475569;padding:0 16px;">'
            'FestiCast v1.0 · Train: 2013–2017<br>10 Stores · 50 Items</div>',
            unsafe_allow_html=True,
        )
    return page, store, item


# ── Page header helper ────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        {'<p>' + subtitle + '</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(train, store, item):
    page_header("🏠 Executive Dashboard",
                "Real-time overview of retail performance and forecast accuracy")

    summary = compute_summary(train)

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Total Sales",      fmt_number(summary["total_sales"]),   "🛒"),
        (c2, "Daily Avg",        fmt_number(summary["mean_daily"], 0), "📅"),
        (c3, "Peak Day",         fmt_number(summary["max_daily"]),      "🔥"),
        (c4, "Stores",           str(summary["num_stores"]),            "🏪"),
        (c5, "Products",         str(summary["num_items"]),             "📦"),
        (c6, "Days of History",  str(summary["num_days"]),              "📆"),
    ]
    for col, label, val, icon in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="label">{icon} {label}</div>
                <div class="value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Daily sales + Event calendar
    col_a, col_b = st.columns([2, 1])
    with col_a:
        daily = aggregate_daily(train)
        st.plotly_chart(plot_daily_sales(daily), use_container_width=True)

    with col_b:
        st.markdown('<div class="fc-card">', unsafe_allow_html=True)
        st.markdown('<div class="fc-section-title">📅 Upcoming Events (2017)</div>', unsafe_allow_html=True)
        cal = get_calendar()
        import datetime
        events = []
        for ev in cal._events:
            for peak in ev.dates_in_range(datetime.date(2017,1,1), datetime.date(2017,12,31)):
                events.append({"Event": ev.name, "Date": peak.strftime("%b %d"), "Category": ev.category.title()})
        if events:
            ev_df = pd.DataFrame(events).sort_values("Date")
            st.dataframe(ev_df, use_container_width=True, height=340, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Monthly bar + Store comparison
    col_c, col_d = st.columns(2)
    with col_c:
        monthly = aggregate_monthly(train)
        st.plotly_chart(plot_monthly_sales(monthly), use_container_width=True)
    with col_d:
        store_df = aggregate_by_store(train)
        st.plotly_chart(plot_store_comparison(store_df), use_container_width=True)

    # Row 3: Quick forecast preview
    st.markdown('<div class="fc-section-title">🔮 Forecast Preview — Selected Store & Item</div>',
                unsafe_allow_html=True)
    try:
        series = filter_series(train, store, item)
        with st.spinner(f"Generating XGBoost forecast for Store {store} · Item {item}…"):
            fc_df = generate_forecast(series, store, item, "XGBoost", horizon=90)
        fig = plot_forecast(series[config.SALES_COL], fc_df, store, item, "XGBoost", tail_days=120)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecast preview unavailable: {e}")

    # Key insights
    st.markdown('<div class="fc-section-title">💡 Key Insights</div>', unsafe_allow_html=True)
    top_store = int(aggregate_by_store(train).iloc[0][config.STORE_COL])
    top_item  = int(aggregate_by_item(train).iloc[0][config.ITEM_COL])
    daily2 = aggregate_daily(train)
    best_day = daily2.loc[daily2[config.SALES_COL].idxmax(), config.DATE_COL]

    col_i1, col_i2, col_i3 = st.columns(3)
    for col, icon, txt in [
        (col_i1, "🏆", f"Best-performing store is **Store {top_store}** with highest total units sold."),
        (col_i2, "📦", f"Top-selling product is **Item {top_item}** across all stores."),
        (col_i3, "📅", f"Peak sales day: **{pd.Timestamp(best_day).strftime('%b %d, %Y')}**."),
    ]:
        with col:
            st.info(f"{icon} {txt}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
def page_forecast(train, test, store_default, item_default):
    page_header("🔮 Demand Forecast",
                "Generate and download AI-powered demand forecasts by store and product")

    with st.form("forecast_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            store = st.selectbox("Store", list(range(1, config.NUM_STORES + 1)),
                                 index=store_default - 1)
            item  = st.selectbox("Item",  list(range(1, config.NUM_ITEMS  + 1)),
                                 index=item_default - 1)
        with col2:
            model_name = st.selectbox("Model", config.MODELS,
                                      index=config.MODELS.index(config.DEFAULT_MODEL))
            horizon    = st.slider("Forecast Horizon (days)", 7, 180,
                                   config.DEFAULT_FORECAST_HORIZON, step=7)
        with col3:
            include_events = st.toggle("Enable Event Awareness", value=True)
            confidence     = st.select_slider("Confidence Level",
                                              options=[0.90, 0.95, 0.99], value=0.95)
            force_retrain  = st.checkbox("Force Retrain", value=False)

        submitted = st.form_submit_button("🚀 Generate Forecast", type="primary",
                                          use_container_width=True)

    if submitted or ("last_fc" in st.session_state and st.session_state.get("fc_store") == store and st.session_state.get("fc_item") == item):
        if submitted:
            series = filter_series(train, store, item)
            with st.spinner(f"Training {model_name} and generating {horizon}-day forecast…"):
                try:
                    if force_retrain:
                        import joblib, glob
                        slug = model_name.lower().replace(" ", "_")
                        for f in glob.glob(os.path.join(config.MODEL_DIR, f"{slug}_s{store}_i{item}*")):
                            os.remove(f)
                    fc_df = generate_forecast(series, store, item, model_name,
                                              horizon, include_events, confidence)
                    st.session_state["last_fc"]  = fc_df
                    st.session_state["last_ser"] = series
                    st.session_state["fc_store"] = store
                    st.session_state["fc_item"]  = item
                    st.session_state["fc_model"] = model_name
                    st.success(f"✅ Forecast generated for Store {store} · Item {item} · {model_name}")
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    return

        fc_df  = st.session_state.get("last_fc")
        series = st.session_state.get("last_ser")
        m_name = st.session_state.get("fc_model", model_name)

        if fc_df is not None:
            # Chart
            fig = plot_forecast(series[config.SALES_COL], fc_df, store, item, m_name, tail_days=120)
            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Avg Forecasted Sales", f"{fc_df['yhat'].mean():.1f} units/day")
            with col_m2:
                st.metric("Peak Forecast",        f"{fc_df['yhat'].max():.0f} units")
            with col_m3:
                ev_days = fc_df["event_window"].sum()
                st.metric("Event Days in Horizon", f"{int(ev_days)} days")

            # Forecast table
            st.markdown("#### 📋 Forecast Table")
            show_df = fc_df[["date","yhat","yhat_lower","yhat_upper",
                              "event_name","event_category","festival_score"]].copy()
            show_df.columns = ["Date","Forecast","Lower CI","Upper CI",
                                "Event","Category","Festival Score"]
            show_df["Forecast"]       = show_df["Forecast"].round(1)
            show_df["Lower CI"]       = show_df["Lower CI"].round(1)
            show_df["Upper CI"]       = show_df["Upper CI"].round(1)
            show_df["Festival Score"] = show_df["Festival Score"].round(3)
            st.dataframe(show_df, use_container_width=True, height=300, hide_index=True)

            # Downloads
            st.markdown("#### 💾 Export Forecast")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("⬇ Download CSV", df_to_csv_bytes(show_df),
                                   f"forecast_s{store}_i{item}_{m_name}.csv",
                                   "text/csv", use_container_width=True)
            with d2:
                st.download_button("⬇ Download Excel", df_to_excel_bytes(show_df, "Forecast"),
                                   f"forecast_s{store}_i{item}_{m_name}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            with d3:
                lines = [f"Store: {store}", f"Item: {item}", f"Model: {m_name}",
                         f"Horizon: {horizon} days", "",
                         "Date | Forecast | Event"] + \
                        [f"{r['Date']} | {r['Forecast']} | {r['Event']}"
                         for _, r in show_df.iterrows()]
                st.download_button("⬇ Download PDF", _make_pdf("Forecast Report", lines),
                                   f"forecast_s{store}_i{item}.pdf",
                                   "application/pdf", use_container_width=True)
    else:
        st.info("👆 Configure the forecast parameters above and click **Generate Forecast**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics(train, store, item):
    page_header("📊 Analytics",
                "Deep dive into sales patterns, seasonality, and feature relationships")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📅 Monthly", "🗓 Weekly Heatmap", "📈 Seasonality",
         "📉 Decomposition", "🔗 Correlation", "🏪 Store vs Item"])

    with tab1:
        monthly = aggregate_monthly(train)
        st.plotly_chart(plot_monthly_sales(monthly), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_weekly_heatmap(train), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_seasonality(train), use_container_width=True)

    with tab4:
        series = filter_series(train, store, item)
        s = series[config.SALES_COL]
        s.index = pd.DatetimeIndex(s.index)
        st.plotly_chart(plot_decomposition(s), use_container_width=True)

    with tab5:
        try:
            feat_df = build_features(filter_series(train, store, item),
                                     include_events=True)
            num_df = feat_df.select_dtypes(include=[float, int])
            st.plotly_chart(plot_correlation(num_df), use_container_width=True)
        except Exception as e:
            st.warning(f"Correlation unavailable: {e}")

    with tab6:
        col_s, col_i = st.columns(2)
        with col_s:
            st.plotly_chart(plot_store_comparison(aggregate_by_store(train)),
                            use_container_width=True)
        with col_i:
            st.plotly_chart(plot_item_comparison(aggregate_by_item(train)),
                            use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def page_model_comparison(train, store, item):
    page_header("🏆 Model Comparison",
                "Head-to-head benchmarking of all 5 forecasting models")

    col_opt1, col_opt2 = st.columns([1,3])
    with col_opt1:
        inc_ev = st.toggle("Include Event Features", value=True)
        run_btn = st.button("▶ Run Benchmark", type="primary", use_container_width=True)

    cache_key = f"bench_{store}_{item}_{inc_ev}"
    if run_btn or cache_key not in st.session_state:
        series = filter_series(train, store, item)
        with st.spinner("Training all 5 models and computing metrics…"):
            try:
                bench = benchmark_models(series, store, item, include_events=inc_ev)
                st.session_state[cache_key] = bench
            except Exception as e:
                st.error(f"Benchmark failed: {e}")
                return

    bench = st.session_state.get(cache_key)
    if bench is not None:
        st.markdown("#### 📊 Benchmark Results")
        st.dataframe(
            bench.style.format({
                "MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%",
                "SMAPE": "{:.2f}%", "R²": "{:.4f}",
                "Train Time (s)": "{:.2f}", "Infer Time (s)": "{:.4f}",
            }).background_gradient(subset=["MAE","RMSE","MAPE"], cmap="RdYlGn_r")
             .background_gradient(subset=["R²"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.plotly_chart(plot_model_comparison(bench), use_container_width=True)
        with col_ch2:
            st.plotly_chart(plot_model_radar(bench), use_container_width=True)

        # Feature importance (XGBoost)
        try:
            xgb_model = get_or_train_model(filter_series(train, store, item),
                                           "XGBoost", store, item, inc_ev)
            if hasattr(xgb_model, "feature_importance"):
                fi = xgb_model.feature_importance()
                if not fi.empty:
                    st.plotly_chart(plot_feature_importance(fi), use_container_width=True)
        except Exception:
            pass

        # Download benchmark
        st.download_button("⬇ Download Benchmark CSV", df_to_csv_bytes(bench),
                           "model_benchmark.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
def page_reports(train, test, store, item):
    page_header("📄 Reports & Exports",
                "Download forecasts, analytics summaries, and submission files")

    tab_a, tab_b, tab_c = st.tabs(
        ["📊 Summary Report", "📁 Submission File", "📅 Event Calendar Report"])

    # ── Summary Report ────────────────────────────────────────────────────────
    with tab_a:
        summary = compute_summary(train)
        st.markdown("### 📊 Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.table(pd.DataFrame({
                "Metric": ["Total Sales","Daily Average","Peak Day Sales",
                           "Stores","Products","Training Days"],
                "Value":  [fmt_number(summary["total_sales"]),
                           f"{summary['mean_daily']:.0f}",
                           fmt_number(summary["max_daily"]),
                           summary["num_stores"],
                           summary["num_items"],
                           summary["num_days"]],
            }))
        with col2:
            top5s = aggregate_by_store(train).head(5)
            top5s.columns = ["Store", "Total Sales"]
            st.table(top5s)

        lines = [
            f"Total Sales: {fmt_number(summary['total_sales'])}",
            f"Daily Average: {summary['mean_daily']:.0f}",
            f"Stores: {summary['num_stores']}",
            f"Products: {summary['num_items']}",
            f"Training Period: {summary['date_start'].date()} to {summary['date_end'].date()}",
        ]
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("⬇ CSV Summary", df_to_csv_bytes(pd.DataFrame({
                "Metric": ["Total Sales","Daily Avg","Stores","Products"],
                "Value":  [summary["total_sales"], round(summary["mean_daily"],1),
                           summary["num_stores"], summary["num_items"]],
            })), "summary_report.csv", "text/csv", use_container_width=True)
        with d2:
            monthly = aggregate_monthly(train)
            st.download_button("⬇ Monthly Excel", df_to_excel_bytes(monthly, "Monthly Sales"),
                               "monthly_sales.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with d3:
            st.download_button("⬇ PDF Report", _make_pdf("Summary Report", lines),
                               "summary_report.pdf", "application/pdf",
                               use_container_width=True)

    # ── Submission File ───────────────────────────────────────────────────────
    with tab_b:
        st.markdown("### 📁 Generate Submission File")
        st.info("ℹ️ This generates forecasts for all 500 store-item pairs in test.csv using the selected model.")

        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            sub_model = st.selectbox("Model for Submission", config.MODELS,
                                     index=config.MODELS.index("XGBoost"))
        with col_sb2:
            sub_events = st.toggle("Use Event Features", value=True)

        if st.button("🚀 Generate Full Submission", type="primary", use_container_width=True):
            with st.spinner("Forecasting all 500 store-item pairs… (this may take a few minutes)"):
                try:
                    sub_df = generate_submission(train, test, sub_model, sub_events)
                    st.session_state["submission"] = sub_df
                    st.success(f"✅ Submission generated: {len(sub_df):,} rows")
                except Exception as e:
                    st.error(f"Submission failed: {e}")

        if "submission" in st.session_state:
            sub = st.session_state["submission"]
            st.dataframe(sub.head(20), use_container_width=True, hide_index=True)
            sb1, sb2 = st.columns(2)
            with sb1:
                st.download_button("⬇ Download submission.csv", df_to_csv_bytes(sub),
                                   "submission.csv", "text/csv", use_container_width=True)
            with sb2:
                st.download_button("⬇ Download submission.xlsx",
                                   df_to_excel_bytes(sub, "Submission"),
                                   "submission.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

    # ── Event Calendar ────────────────────────────────────────────────────────
    with tab_c:
        year = st.selectbox("Calendar Year", [2013,2014,2015,2016,2017,2018], index=4)
        st.plotly_chart(plot_event_calendar(year), use_container_width=True)

        import datetime
        cal = get_calendar()
        ev_rows = []
        for ev in cal._events:
            for peak in ev.dates_in_range(
                    datetime.date(year, 1, 1), datetime.date(year, 12, 31)):
                ev_rows.append({"Event": ev.name, "Category": ev.category,
                                "Date": str(peak), "Importance": ev.importance,
                                "Window Before": ev.window_before,
                                "Window After": ev.window_after})
        ev_df = pd.DataFrame(ev_rows).sort_values("Date")
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Event Calendar CSV", df_to_csv_bytes(ev_df),
                           f"event_calendar_{year}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
def page_settings():
    page_header("⚙️ Settings & Configuration",
                "Manage event calendar, model defaults, and application preferences")

    tab_ev, tab_mod, tab_app = st.tabs(
        ["📅 Event Manager", "🤖 Model Defaults", "🎨 App Preferences"])

    # ── Event Manager ─────────────────────────────────────────────────────────
    with tab_ev:
        cal = get_calendar()
        st.markdown("### 📅 Registered Events")
        all_names = cal.list_events()

        col_ev1, col_ev2 = st.columns([2, 1])
        with col_ev1:
            import datetime
            rows = []
            for ev in cal._events:
                rows.append({
                    "Event": ev.name,
                    "Category": ev.category,
                    "Importance": ev.importance,
                    "Window Before": ev.window_before,
                    "Window After":  ev.window_after,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with col_ev2:
            st.markdown("**➕ Add Custom Event**")
            with st.form("add_event_form"):
                ev_name = st.text_input("Event Name", placeholder="e.g. Onam")
                ev_cat  = st.selectbox("Category", ["festival","holiday","business","custom"])
                ev_date = st.date_input("Event Date", value=datetime.date(2018, 9, 5))
                ev_wb   = st.number_input("Window Before (days)", 1, 30, 7)
                ev_wa   = st.number_input("Window After (days)",  1, 14, 3)
                ev_imp  = st.slider("Importance (0–1)", 0.0, 1.0, 0.75, 0.05)
                add_ok  = st.form_submit_button("Add Event", type="primary")

            if add_ok and ev_name.strip():
                new_ev = Event(
                    name=ev_name.strip(), category=ev_cat,
                    base_date=ev_date, window_before=int(ev_wb),
                    window_after=int(ev_wa), importance=ev_imp,
                )
                cal.add_event(new_ev)
                st.success(f"✅ Event '{ev_name}' added!")
                st.rerun()

            st.markdown("**🗑 Remove Event**")
            with st.form("remove_event_form"):
                to_remove = st.selectbox("Select event to remove", all_names)
                rem_ok = st.form_submit_button("Remove", type="secondary")
            if rem_ok:
                cal.remove_event(to_remove)
                st.success(f"✅ Removed '{to_remove}'")
                st.rerun()

    # ── Model Defaults ────────────────────────────────────────────────────────
    with tab_mod:
        st.markdown("### 🤖 Model Configuration")
        col_md1, col_md2 = st.columns(2)
        with col_md1:
            st.number_input("Default Forecast Horizon (days)",
                            7, 365, config.DEFAULT_FORECAST_HORIZON, 7)
            st.selectbox("Default Model", config.MODELS,
                         index=config.MODELS.index(config.DEFAULT_MODEL))
            st.number_input("Moving Average Window", 5, 90,
                            config.DEFAULT_MA_WINDOW, 5)
        with col_md2:
            st.number_input("Event Window (days)", 1, 30, config.EVENT_WINDOW_DAYS)
            st.number_input("CV Folds", 2, 10, config.N_SPLITS)
            st.toggle("Enable Event Awareness by Default", value=True)

        # Clear model cache
        st.markdown("---")
        if st.button("🗑 Clear All Cached Models", type="secondary"):
            import glob
            files = glob.glob(os.path.join(config.MODEL_DIR, "*.joblib"))
            for f in files:
                os.remove(f)
            st.success(f"✅ Cleared {len(files)} cached model(s).")

    # ── App Preferences ───────────────────────────────────────────────────────
    with tab_app:
        st.markdown("### 🎨 Application Preferences")
        st.info("ℹ️ These preferences are shown for display. Theme is applied via CSS.")
        st.selectbox("Chart Theme", ["plotly_white","plotly","ggplot2","seaborn"])
        st.number_input("Chart Height (px)", 300, 800, config.CHART_HEIGHT, 20)
        st.selectbox("Date Format", ["%Y-%m-%d","%d/%m/%Y","%b %d, %Y"])
        st.toggle("Show Confidence Intervals by Default", value=True)
        st.toggle("Auto-Save Models", value=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page, store, item = render_sidebar()

    # Load data once
    try:
        with st.spinner("Loading data…"):
            train_raw, test_raw, sub_raw = load_raw_data()
            train = clean_train(train_raw)
    except FileNotFoundError as e:
        st.error(f"""
        **Data files not found.**

        Please ensure the following files exist in `data/`:
        - `train.csv`
        - `test.csv`
        - `sample_submission.csv`

        Error: `{e}`
        """)
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Route to pages
    if   page == "🏠 Dashboard":        page_dashboard(train, store, item)
    elif page == "🔮 Forecast":          page_forecast(train, test_raw, store, item)
    elif page == "📊 Analytics":         page_analytics(train, store, item)
    elif page == "🏆 Model Comparison":  page_model_comparison(train, store, item)
    elif page == "📄 Reports":           page_reports(train, test_raw, store, item)
    elif page == "⚙️ Settings":          page_settings()


if __name__ == "__main__":
    main()
