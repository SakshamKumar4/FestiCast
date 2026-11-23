# FestiCast 📈
### AI-Powered Event-Aware Retail Demand Forecasting Platform

---

## 🌟 Project Overview

**FestiCast** (Festival + Forecast) is an enterprise-grade AI platform that forecasts retail demand by integrating historical sales data with external event calendars — festivals, holidays, promotional events, and business milestones.

The key innovation is the **Event Calendar Engine**: any new event calendar (regional, religious, promotional) can be plugged in without modifying the forecasting pipeline.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FestiCast Platform                    │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Upload  │──▶│ Preprocessing│──▶│Feature Engineer│  │
│  │  CSV     │   │  & Cleaning  │   │   + Events     │  │
│  └──────────┘   └──────────────┘   └───────┬────────┘  │
│                                            │            │
│  ┌─────────────────────────────────────────▼──────────┐ │
│  │               Model Training Pipeline              │ │
│  │  Naive │ Moving Avg │ SARIMA │ Prophet │ XGBoost   │ │
│  │              Walk-Forward Validation               │ │
│  └─────────────────────────────────────────┬──────────┘ │
│                                            │            │
│  ┌─────────────────────────────────────────▼──────────┐ │
│  │            Forecast + Confidence Intervals         │ │
│  └─────────────────────────────────────────┬──────────┘ │
│                                            │            │
│  ┌────────────┐  ┌────────────┐  ┌─────────▼─────────┐  │
│  │  Dashboard │  │ Analytics  │  │  Reports/Export   │  │
│  └────────────┘  └────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Folder Structure

```
FestiCast_data/
│
├── app.py                    ← Streamlit main application
├── config.py                 ← Global configuration
├── requirements.txt          ← Python dependencies
├── README.md
│
├── data/                     ← Dataset CSVs (train, test, submission)
├── models/                   ← Saved model files (.joblib)
├── outputs/                  ← Generated forecast outputs
├── reports/                  ← Exported PDF/Excel reports
├── assets/                   ← Static assets
│
└── src/
    ├── __init__.py
    ├── preprocessing.py      ← Data loading & cleaning
    ├── feature_engineering.py← Lag, rolling, calendar, event features
    ├── event_calendar.py     ← Event registry (festivals, holidays, business)
    ├── train_models.py       ← 5 model implementations + persistence
    ├── forecasting.py        ← Forecast generation + CI
    ├── evaluation.py         ← MAE, RMSE, MAPE, SMAPE, R², walk-forward
    ├── visualization.py      ← All Plotly charts
    └── utils.py              ← Shared helpers
```

---

## ⚙️ Installation

```bash
# 1. Clone / open the project folder in VS Code
cd FestiCast_data

# 2. Create a virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🚀 Usage

1. Open `http://localhost:8501` in your browser
2. Navigate using the left sidebar
3. Select a Store and Item from the sidebar filters
4. Go to **Forecast** → choose model → click Generate
5. Download results as CSV / Excel / PDF

---

## 📊 Dataset

| File | Rows | Columns |
|------|------|---------|
| train.csv | 913,000 | date, store, item, sales |
| test.csv | 45,000 | id, date, store, item |
| sample_submission.csv | 45,000 | id, sales |

- **Training period:** 2013-01-01 → 2017-12-31
- **Test period:** 2018-01-01 → 2018-03-31
- **10 stores × 50 items** = 500 unique time-series

---

## 🤖 Models Implemented

| Model | Type | Event-Aware |
|-------|------|------------|
| Naive | Baseline | ❌ |
| Moving Average | Statistical | ❌ |
| SARIMA | Statistical | ❌ |
| Prophet | Bayesian | ✅ |
| XGBoost | ML/Gradient Boost | ✅ |

---

## 🎪 Supported Events

### Indian Festivals
Diwali, Holi, Navratri, Dussehra, Eid, Ganesh Chaturthi, Raksha Bandhan

### Global Holidays
Christmas, New Year, Halloween, Black Friday, Cyber Monday, Easter

### Business Events
Weekend, End of Month, End of Quarter, Independence Day, Valentine's Day

### Custom Events
Add any event via **Settings → Event Manager** without code changes.

---

## 📐 Feature Engineering

| Category | Features |
|----------|---------|
| Lag | lag_1, lag_7, lag_14, lag_30 |
| Rolling | roll_mean_7, roll_mean_14, roll_mean_30, roll_std_7, roll_std_14 |
| Calendar | day_of_week, week, month, quarter, year, is_weekend, is_month_end |
| Cyclical | dow_sin, dow_cos, month_sin, month_cos |
| Events | is_event, event_window, days_until_event, days_since_event, festival_score, event_importance |

---

## 📏 Validation Strategy

- ✅ **Walk-Forward Validation** (TimeSeriesSplit, 5 folds)
- ❌ No random train-test split
- Metrics: MAE, RMSE, MAPE, SMAPE, R²
- Event-period metrics computed separately

---

## 📤 Exports

| Format | Available |
|--------|---------|
| CSV | ✅ |
| Excel (.xlsx) | ✅ |
| PDF Report | ✅ |
| PNG Charts | ✅ (via Plotly toolbar) |

---

## 🔮 Future Enhancements

- [ ] LightGBM and CatBoost models
- [ ] Deep learning (LSTM, N-BEATS, TFT)
- [ ] Store-level clustering
- [ ] Automated hyperparameter tuning (Optuna)
- [ ] Weather and economic indicator integration
- [ ] Real-time streaming data support
- [ ] Multi-currency and multi-region support

---

## 👩‍💻 Tech Stack

- **Frontend:** Streamlit 1.32
- **ML Models:** XGBoost, Prophet, Statsmodels (SARIMA)
- **Visualization:** Plotly
- **Data:** Pandas, NumPy
- **Persistence:** Joblib
- **Exports:** openpyxl, fpdf2

---

*FestiCast — Where Festivals Meet Forecasting*
