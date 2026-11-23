"""
FestiCast - Global Configuration
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
ASSET_DIR   = os.path.join(BASE_DIR, "assets")

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, REPORT_DIR, ASSET_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset files ──────────────────────────────────────────────────────────────
TRAIN_FILE      = os.path.join(DATA_DIR, "train.csv")
TEST_FILE       = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

# ── Dataset schema (discovered from inspection) ───────────────────────────────
DATE_COL  = "date"
STORE_COL = "store"
ITEM_COL  = "item"
SALES_COL = "sales"
ID_COL    = "id"

DATE_FORMAT = "%Y-%m-%d"

NUM_STORES = 10
NUM_ITEMS  = 50

TRAIN_START = "2013-01-01"
TRAIN_END   = "2017-12-31"
TEST_START  = "2018-01-01"
TEST_END    = "2018-03-31"

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = ["Naive", "Moving Average", "SARIMA", "Prophet", "XGBoost"]

DEFAULT_MODEL           = "XGBoost"
DEFAULT_FORECAST_HORIZON = 90   # days
DEFAULT_MA_WINDOW        = 30

# ── Feature Engineering ────────────────────────────────────────────────────────
LAG_FEATURES       = [1, 7, 14, 30]
ROLLING_WINDOWS    = [7, 14]
EVENT_WINDOW_DAYS  = 7          # days before/after event

# ── UI / Theme ─────────────────────────────────────────────────────────────────
PRIMARY_COLOR   = "#0066CC"
ACCENT_COLOR    = "#004A99"
SUCCESS_COLOR   = "#17A34A"
WARNING_COLOR   = "#D97706"
DANGER_COLOR    = "#DC2626"
BG_COLOR        = "#F8FAFC"
CARD_BG         = "#FFFFFF"
TEXT_PRIMARY    = "#1E293B"
TEXT_SECONDARY  = "#64748B"
BORDER_COLOR    = "#E2E8F0"

CHART_TEMPLATE  = "plotly_white"
CHART_HEIGHT    = 420

# ── Validation ─────────────────────────────────────────────────────────────────
N_SPLITS = 5          # TimeSeriesSplit folds

# ── Caching ────────────────────────────────────────────────────────────────────
CACHE_TTL = 3600      # seconds
