"""Read LIVE Google Sheet round_config values and check freshness."""
import json
import sys
import sheet_config

PARAMS_OF_INTEREST = [
    "cat_mult_ott", "cat_mult_app", "cat_mult_arg", "cat_mult_putt",
    "cat_skew_ott", "cat_skew_app", "cat_skew_arg", "cat_skew_putt",
]
FRESH_PARAMS = ["event_id", "tourney", "course_id"]

result = {"accessible": False}

try:
    ws = sheet_config._connect_sheet()
except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}"
    print("RESULT_JSON_START")
    print(json.dumps(result, indent=2, default=str))
    print("RESULT_JSON_END")
    sys.exit(0)

result["accessible"] = True

# Read columns A:B (param name / value)
all_values = ws.get("A:B")

# Build param -> (row_number, value) map. Row number is 1-indexed (gspread).
rows = {}
raw_by_row = {}
for i, row in enumerate(all_values):
    rownum = i + 1
    name = row[0].strip() if len(row) >= 1 and row[0] else ""
    val = row[1].strip() if len(row) >= 2 and row[1] is not None else ""
    raw_by_row[rownum] = (name, val)
    if name:
        rows[name.strip().lower()] = (rownum, val)

# Report params of interest with row numbers
report = {}
for p in PARAMS_OF_INTEREST:
    if p in rows:
        rn, v = rows[p]
        report[p] = {"row": rn, "value": v}
    else:
        report[p] = {"row": None, "value": "<MISSING>"}
result["params_of_interest"] = report

# Literal B28..B35 dump (cells), with the A-column label next to each
b28_b35 = []
for rn in range(28, 36):
    name, val = raw_by_row.get(rn, ("", ""))
    b28_b35.append(f"A{rn}={name!r} B{rn}={val!r}")
result["cells_b28_b35"] = b28_b35

# Freshness params
fresh = {}
for p in FRESH_PARAMS:
    if p in rows:
        rn, v = rows[p]
        fresh[p] = {"row": rn, "value": v}
    else:
        fresh[p] = {"row": None, "value": "<MISSING>"}
result["freshness_params"] = fresh

# Now load_config to see what the sim actually consumes
try:
    cfg = sheet_config.load_config()
    result["loaded_course_cat_mults"] = cfg.get("course_cat_mults")
    result["loaded_course_cat_skew"] = cfg.get("course_cat_skew")
    result["loaded_event_id"] = cfg.get("event_id")
    result["loaded_tourney"] = cfg.get("tourney")
    result["loaded_course_id"] = cfg.get("course_id")
except Exception as e:
    result["load_config_error"] = f"{type(e).__name__}: {e}"

print("RESULT_JSON_START")
print(json.dumps(result, indent=2, default=str))
print("RESULT_JSON_END")
