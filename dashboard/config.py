"""Dashboard configuration — paths, constants, theme settings."""

import os

# Project root (parent of dashboard/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
PERMANENT_DATA = os.path.join(PROJECT_ROOT, "permanent_data")
LEDGER_PATH = os.path.join(PERMANENT_DATA, "bet_ledger.parquet")
SG_DIAGNOSTIC_PATH = os.path.join(PERMANENT_DATA, "sg_diagnostic.parquet")

# Sportsbook lists (mirrors round_sim.py)
SHARP_BOOKS = ["pinnacle", "betonline", "betcris"]
RETAIL_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "barstool", "espn"]
BOOKS_TO_USE = ["betcris", "betmgm", "betonline", "bovada", "caesars", "draftkings", "fanduel", "kalshi", "novig", "pinnacle", "unibet"]

# Edge thresholds (matches round_sim.py)
EDGE_THRESHOLD_WIN = 2.0
EDGE_THRESHOLD_TOPN = 2.0
EMAIL_MIN_PRED = 0.75
EMAIL_MIN_SAMPLE = 20

# Edge color thresholds
EDGE_COLOR_HIGH = 10.0   # green
EDGE_COLOR_MID = 5.0     # yellow
EDGE_COLOR_LOW = -5.0    # red

# Colors
COLOR_GREEN = "#d4edda"
COLOR_YELLOW = "#fff3cd"
COLOR_RED = "#f8d7da"
COLOR_WIN = "#2ca02c"
COLOR_LOSS = "#d62728"
COLOR_PUSH = "#7f7f7f"
COLOR_SHARP = "#1f77b4"
COLOR_RETAIL = "#ff7f0e"

# Cache TTL (seconds)
CACHE_TTL = 300  # 5 minutes
