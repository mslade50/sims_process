import os
import requests
import datetime
import pandas as pd
import gspread
from dotenv import load_dotenv

from sim_inputs import dewpoint_wave, baseline_dew, tourney, course_id
try:
    from sim_inputs import lat_override, lon_override
except ImportError:
    lat_override = None
    lon_override = None
from sheets_storage import get_spreadsheet
from api_utils import fetch_historical_hourly_wind

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Auto-detect course coordinates from permanent_data/course_coordinates.csv
# ══════════════════════════════════════════════════════════════════════════════

COURSE_COORDS_CSV = os.path.join(os.path.dirname(__file__), "permanent_data", "course_coordinates.csv")


def get_course_coordinates(cid):
    """Look up lat/lon for a course_id from course_coordinates.csv."""
    if not os.path.exists(COURSE_COORDS_CSV):
        print(f"  WARNING: {COURSE_COORDS_CSV} not found")
        return None, None
    df = pd.read_csv(COURSE_COORDS_CSV)
    match = df[df["course_id"] == cid]
    if match.empty:
        print(f"  WARNING: course_id {cid} not found in course_coordinates.csv")
        return None, None
    row = match.iloc[0]
    return float(row["lat"]), float(row["lon"])


# ══════════════════════════════════════════════════════════════════════════════
# Auto-detect tournament dates from DataGolf field-updates API
# ══════════════════════════════════════════════════════════════════════════════

DATAGOLF_BASE = "https://feeds.datagolf.com"


def get_tournament_info():
    """Fetch dates and course codes from DataGolf field-updates API.

    Returns (dates, course_codes) where dates is a list of 4 datetime objects
    and course_codes is a sorted list of unique course code strings.
    """
    api_key = os.getenv("DATAGOLF_API_KEY")
    if not api_key:
        print("  WARNING: DATAGOLF_API_KEY not set, cannot auto-detect dates")
        return None, []
    params = {"tour": "pga", "file_format": "json", "key": api_key}
    resp = requests.get(f"{DATAGOLF_BASE}/field-updates", params=params, timeout=15)
    if resp.status_code != 200:
        print(f"  WARNING: field-updates API returned {resp.status_code}")
        return None, []
    data = resp.json()
    date_start = data.get("event_start_date") or data.get("date_start")
    if not date_start:
        for key in data:
            if "date" in str(key).lower() and "start" in str(key).lower():
                date_start = data[key]
                break
    if not date_start:
        print("  WARNING: Could not find start date in field-updates response")
        print(f"  Available top-level keys: {list(data.keys())}")
        return None, []

    # Extract unique course codes from field
    courses = set()
    for player in data.get("field", []):
        course = player.get("course")
        if course and str(course).strip():
            courses.add(str(course).strip())
    course_codes = sorted(courses)

    r1 = datetime.datetime.strptime(str(date_start)[:10], "%Y-%m-%d")
    dates = [r1 + datetime.timedelta(days=i) for i in range(4)]
    return dates, course_codes


# ══════════════════════════════════════════════════════════════════════════════
# Resolve coordinates and dates
# ══════════════════════════════════════════════════════════════════════════════

if lat_override is not None and lon_override is not None:
    LATITUDE = float(lat_override)
    LONGITUDE = float(lon_override)
    print(f"Course {course_id} ({tourney}): lat/lon OVERRIDE from sim_inputs "
          f"-> lat={LATITUDE}, lon={LONGITUDE}")
else:
    lat, lon = get_course_coordinates(course_id)
    if lat is None:
        print("FATAL: Could not resolve course coordinates. Exiting.")
        raise SystemExit(1)
    LATITUDE = lat
    LONGITUDE = lon
    print(f"Course {course_id} ({tourney}): lat={LATITUDE}, lon={LONGITUDE}")

tournament_dates, course_codes = get_tournament_info()
if tournament_dates is None:
    print("FATAL: Could not resolve tournament dates. Exiting.")
    raise SystemExit(1)
if course_codes:
    print(f"Course codes: {course_codes}")

START_DATE = tournament_dates[0].strftime("%Y-%m-%d")
END_DATE = tournament_dates[3].strftime("%Y-%m-%d")
print(f"Tournament dates: {START_DATE} to {END_DATE}")

date_list = tournament_dates

# ══════════════════════════════════════════════════════════════════════════════
# Existing analysis logic (preserved from original)
# ══════════════════════════════════════════════════════════════════════════════

# Fetch historical base values
start_year, end_year = 2019, 2025
month = tournament_dates[0].month

# Adjust dewpoint wave factor
dewpoint_wave = dewpoint_wave * 0.6 + baseline_dew * 0.4

# Blended dew coefficient: same formula as sim_inputs.dew_calculation
dew_factor = 0.6 * baseline_dew + 0.4 * dewpoint_wave
print(f"Dew factor (for sheet): {dew_factor:.4f}  (baseline_dew={baseline_dew}, dewpoint_wave={dewpoint_wave:.4f})")

# Create dictionaries to store weather data dynamically
dewpoint_data = {f"dewpoint_{i+1}": [] for i in range(len(date_list))}
wind_data = {f"wind_{i+1}": [] for i in range(len(date_list))}


def get_historical_weather(start_year, end_year, month, variable):
    all_data = []
    for year in range(start_year, end_year + 1):
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={LATITUDE}&longitude={LONGITUDE}"
               f"&start_date={year}-{month:02d}-01&end_date={year}-{month:02d}-31"
               f"&hourly={variable}&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto")
        response = requests.get(url, timeout=10)
        data = response.json()
        if "hourly" in data and variable in data["hourly"]:
            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                variable: data["hourly"][variable]
            })
            df["hour"] = df["datetime"].dt.hour
            df = df[(df["hour"] >= 7) & (df["hour"] <= 19)]
            df["year"] = df["datetime"].dt.year
            monthly_avg = df.groupby("year")[variable].mean().reset_index()
            all_data.append(monthly_avg)
    final_df = pd.concat(all_data)
    base_value = round(final_df[variable].mean(), 1)
    return base_value


dewpoint_base = get_historical_weather(start_year, end_year, month, "dewpoint_2m")
wind_base = get_historical_weather(start_year, end_year, month, "wind_speed_10m")

print("Dewpoint Base (°F) =", dewpoint_base)
print("Wind Base (mph) =", wind_base)

# ══════════════════════════════════════════════════════════════════════════════
# Fetch forecast from Open-Meteo
# ══════════════════════════════════════════════════════════════════════════════

api_url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": ["dewpoint_2m", "wind_speed_10m"],
    "temperature_unit": "fahrenheit",
    "timezone": "America/New_York",
    "start_date": START_DATE,
    "end_date": END_DATE,
}

response = requests.get(api_url, params=params)
data = response.json()

dewpoint_values = data["hourly"]["dewpoint_2m"]
wind_values = data["hourly"]["wind_speed_10m"]
timestamps = data["hourly"]["time"]

# Store hourly values per round (6 AM - 8 PM = 15 hours)
# Also build rounded integer arrays for sheet writing
dew_by_round = {i: [] for i in range(1, 5)}   # {1: [...], 2: [...], ...}
wind_by_round = {i: [] for i in range(1, 5)}

for time_str, dewpoint, wind in zip(timestamps, dewpoint_values, wind_values):
    dt_obj = datetime.datetime.fromisoformat(time_str)
    if 6 <= dt_obj.hour <= 20:  # 6 AM to 8 PM = 15 values
        date_str = dt_obj.date().strftime("%Y-%m-%d")
        date_strs = [d.strftime("%Y-%m-%d") for d in date_list]
        if date_str in date_strs:
            index = date_strs.index(date_str)
            rd = index + 1
            dewpoint_data[f"dewpoint_{rd}"].append(dewpoint)
            wind_mph = round(wind * 0.621371, 1)
            wind_data[f"wind_{rd}"].append(wind_mph)
            dew_by_round[rd].append(round(dewpoint))
            wind_by_round[rd].append(round(wind_mph))

# Print raw values
print("\n*** Raw Dewpoint Values ***")
for key, values in dewpoint_data.items():
    print(f"{key} =", values)

for key, values in wind_data.items():
    print(f"{key} =", values)

# Compute average dewpoints dynamically
avg_dewpoints = {key: round(sum(values) / len(values), 1) if values else None for key, values in dewpoint_data.items()}

# Compute delta dewpoints relative to historical base values
delta_dewpoints = {key: round(avg - dewpoint_base, 1) if avg is not None else None for key, avg in avg_dewpoints.items()}

# Apply the wave adjustment factor
dewpoint_wave_adjustments = {key: round(delta * dewpoint_wave, 2) if delta is not None else None for key, delta in delta_dewpoints.items()}

# Print original adjustments
print("\n*** Score Adjustments (Based on Historical Averages) ***")
for i, (key, adjustment) in enumerate(dewpoint_wave_adjustments.items(), start=1):
    print(f"score_adj_r{i} =", adjustment)

# Compute and print average wind speeds dynamically
avg_wind_data = {key: round(sum(values) / len(values), 1) if values else None for key, values in wind_data.items()}
print("\n*** Average Wind Speeds ***")
for key, avg in avg_wind_data.items():
    print(f"Average {key} =", avg)

# ======================
# RECENT RAINFALL FETCH
# ======================

today = datetime.date.today()
one_week_ago = today - datetime.timedelta(days=7)
one_month_ago = today - datetime.timedelta(days=30)


def fetch_recent_rainfall(start_date, end_date):
    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={LATITUDE}&longitude={LONGITUDE}"
           f"&start_date={start_date}&end_date={end_date}"
           f"&daily=precipitation_sum&timezone=auto")
    response = requests.get(url, timeout=10)
    data = response.json()
    if "daily" in data and "precipitation_sum" in data["daily"]:
        valid_precip = [x for x in data["daily"]["precipitation_sum"] if x is not None]
        total_rainfall = sum(valid_precip)
        return round(total_rainfall, 2)
    else:
        return None


rainfall_last_7d = fetch_recent_rainfall(one_week_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
rainfall_last_30d = fetch_recent_rainfall(one_month_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))

print("\n*** Recent Rainfall ***")
print(f"Rainfall Last 7 Days (in) = {rainfall_last_7d}")
print(f"Rainfall Last 30 Days (in) = {rainfall_last_30d}")

# ==================================
# HISTORICAL RAINFALL BASELINE FETCH
# ==================================


def fetch_historical_rainfall(start_year, end_year, month):
    monthly_totals = []
    for year in range(start_year, end_year + 1):
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={LATITUDE}&longitude={LONGITUDE}"
               f"&start_date={year}-{month:02d}-01&end_date={year}-{month:02d}-30"
               f"&daily=precipitation_sum&timezone=auto")
        response = requests.get(url, timeout=10)
        data = response.json()
        if "daily" in data and "precipitation_sum" in data["daily"]:
            total_monthly_rainfall = sum(data["daily"]["precipitation_sum"])
            monthly_totals.append(total_monthly_rainfall)
    historical_avg_rainfall = round(sum(monthly_totals) / len(monthly_totals), 2) if monthly_totals else None
    return historical_avg_rainfall


historical_avg_rainfall_april = fetch_historical_rainfall(start_year, end_year, 4)

print("\n*** Historical Baseline Rainfall ***")
print(f"Historical Average Rainfall for Current Month (in) = {historical_avg_rainfall_april}")


def get_historical_variance(start_year, end_year, month, variable):
    """
    Returns the pooled hourly standard deviation (σ) of <variable>
    over 07-19 local time for the given month across all years.
    """
    all_vals = []
    for year in range(start_year, end_year + 1):
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={LATITUDE}&longitude={LONGITUDE}"
               f"&start_date={year}-{month:02d}-01&end_date={year}-{month:02d}-31"
               f"&hourly={variable}&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto")
        data = requests.get(url, timeout=10).json()
        if "hourly" not in data or variable not in data["hourly"]:
            continue
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            variable: data["hourly"][variable]
        })
        df["hour"] = df["datetime"].dt.hour
        df = df[(df["hour"] >= 7) & (df["hour"] <= 19)]
        all_vals.extend(df[variable].tolist())
    return round(pd.Series(all_vals).std(ddof=0), 2)


sigma_wind = get_historical_variance(start_year, end_year, month, "wind_speed_10m")
sigma_dewpt = get_historical_variance(start_year, end_year, month, "dewpoint_2m")

print(f"sigma_wind  (mph) = {sigma_wind}")
print(f"sigma_dewpt (F)  = {sigma_dewpt}")

# ══════════════════════════════════════════════════════════════════════════════
# Write forecast weather to Google Sheet (round_config tab)
# ══════════════════════════════════════════════════════════════════════════════
#
# Column mapping (1-indexed for gspread):
#   R1: Dew = col 7  (G)
#   R2: Dew = col 11 (K)
#   R3: Wind = col 14 (N), Dew = col 15 (O)
#   R4: Wind = col 18 (R), Dew = col 19 (S)
#
# R1/R2 wind columns (F, J) are manually populated in the Sheet.
# Rows 3-17 = 15 hours (6 AM - 8 PM)
# Rows 18-21: formula-generated comma-separated summaries (do NOT overwrite)

DEW_COLS = {1: 7, 2: 11, 3: 15, 4: 19}   # round -> column for dew
WIND_COLS = {3: 14, 4: 18}                 # round -> column for wind (R3/R4 only)
DATA_START_ROW = 3
DATA_END_ROW = 17   # 15 rows total


def write_weather_to_sheet():
    """Write forecast dew and wind values to the round_config tab."""
    print("\n*** Writing weather to Google Sheet ***")

    spreadsheet = get_spreadsheet()
    ws = spreadsheet.worksheet("round_config")

    cells_to_update = []

    # Write dew values for all 4 rounds
    for rd in range(1, 5):
        col = DEW_COLS[rd]
        values = dew_by_round.get(rd, [])
        if not values:
            print(f"  WARNING: No dew data for R{rd}, skipping")
            continue
        # Pad or truncate to exactly 15 values
        values = values[:15]
        while len(values) < 15:
            values.append("")
        for i, val in enumerate(values):
            row = DATA_START_ROW + i
            cells_to_update.append(gspread.Cell(row=row, col=col, value=val))
        print(f"  R{rd} dew -> col {col} ({len([v for v in values if v != ''])} values)")

    # Write wind values for R3 and R4 only (R1/R2 wind is manual)
    for rd in [3, 4]:
        col = WIND_COLS[rd]
        values = wind_by_round.get(rd, [])
        if not values:
            print(f"  WARNING: No wind data for R{rd}, skipping")
            continue
        values = values[:15]
        while len(values) < 15:
            values.append("")
        for i, val in enumerate(values):
            row = DATA_START_ROW + i
            cells_to_update.append(gspread.Cell(row=row, col=col, value=val))
        print(f"  R{rd} wind -> col {col} ({len([v for v in values if v != ''])} values)")

    # Rows 18-21 contain formula-generated comma-separated summaries
    # (auto-generated from the grid columns above) — do NOT overwrite them.

    if cells_to_update:
        ws.update_cells(cells_to_update, value_input_option="USER_ENTERED")
        print(f"  Batch updated {len(cells_to_update)} cells in round_config")
    else:
        print("  No cells to update")

    # Write course_codes and dew_calculation to column B
    all_values = ws.get("A:B")
    param_rows = {}
    for i, row in enumerate(all_values):
        if row and row[0].strip():
            param_rows[row[0].strip().lower()] = i + 1

    if course_codes:
        row_idx = param_rows.get("course_codes")
        if row_idx:
            ws.update_cell(row_idx, 2, ",".join(course_codes))
            print(f"  course_codes -> {','.join(course_codes)}")

    row_idx = param_rows.get("dew_calculation")
    if row_idx:
        ws.update_cell(row_idx, 2, str(round(dew_factor, 4)))
        print(f"  dew_calculation -> {dew_factor:.4f}")

    # Write course lat/lon so it's easy to copy into Windy
    coord_str = f"{LATITUDE},{LONGITUDE}"
    row_idx = param_rows.get("course_lat_lon")
    if row_idx:
        ws.update_cell(row_idx, 2, coord_str)
    else:
        next_row = len(all_values) + 1
        ws.update_cell(next_row, 1, "course_lat_lon")
        ws.update_cell(next_row, 2, coord_str)
    print(f"  course_lat_lon -> {coord_str}")

    # Write dewpoint_base to config tab (used by live_stats_engine for actuals)
    row_idx = param_rows.get("dewpoint_base")
    if row_idx:
        ws.update_cell(row_idx, 2, str(round(dewpoint_base, 1)))
    else:
        next_row = len(all_values) + 1
        ws.update_cell(next_row, 1, "dewpoint_base")
        ws.update_cell(next_row, 2, str(round(dewpoint_base, 1)))
        all_values.append(["dewpoint_base", str(round(dewpoint_base, 1))])
    print(f"  dewpoint_base -> {dewpoint_base:.1f}")

    # NOTE: wind_r1..wind_r4 and dew_r1..dew_r4 in the A:B params area are
    # FORMULA cells (e.g. =TEXTJOIN(",",TRUE,F3:F17)) that pull from the grid
    # columns. Do NOT overwrite them here — the grid writes above are sufficient.

    # Write historical hourly wind base rates as Python list to D44
    hist_wind = fetch_historical_hourly_wind(LATITUDE, LONGITUDE, month)
    if hist_wind:
        wind_base_str = f"wind_base={hist_wind}"
        ws.update_cell(44, 4, wind_base_str)  # D44
        print(f"  wind_base -> D44 ({len(hist_wind)} hours, avg {sum(hist_wind)/len(hist_wind):.1f} mph)")


write_weather_to_sheet()
