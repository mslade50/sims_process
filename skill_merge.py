"""
skill_merge.py — Build the final combined skill file for etr-golf-sims.

Ported from ~/OneDrive/skill_merge.py (IGNORE_SEAN_DATA path) so it runs from
this repo with the current week's sim_inputs. Chained automatically at the end
of mkt_regress.py; safe to re-run standalone as forecasts/odds update.

Steps:
  1. Fetch this week's field updates (r1/r2 tee times) from DataGolf
  2. Read final_predictions_{tourney}.csv (mkt_regress output)
  3. Merge fresh tee times
  4. 50/50 blend my_pred with DataGolf's dg_final_pred where available
  5. Write final_predictions_{tourney}_combined.csv (+ _combineddetail.csv)
     and copy the combined file to ~/OneDrive/etr-golf-sims
"""

import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from api_utils import fetch_player_decompositions
from sim_inputs import tourney, name_replacements

load_dotenv()
API_KEY = os.getenv("DATAGOLF_API_KEY")

ETR_REPO = os.path.join(os.path.expanduser("~"), "OneDrive", "etr-golf-sims")

# ---------------------------------------------------------------------------
# 1. Fetch field updates (tee times)
# ---------------------------------------------------------------------------
print("Fetching field updates...")
resp = requests.get(
    "https://feeds.datagolf.com/field-updates",
    params={"tour": "pga", "file_format": "json", "key": API_KEY},
    timeout=60,
)
resp.raise_for_status()
field_df = pd.DataFrame(resp.json()["field"])

if "teetimes" in field_df.columns:
    def _unpack_teetimes(teetimes):
        result = {}
        if isinstance(teetimes, list):
            for entry in teetimes:
                rnd = entry.get("round_num")
                if rnd is not None:
                    result[f"r{rnd}_teetime"] = entry.get("teetime")
        return pd.Series(result)

    parsed = field_df["teetimes"].apply(_unpack_teetimes)
    if not parsed.empty and len(parsed.columns) > 0:
        field_df = pd.concat([field_df, parsed], axis=1)
    field_df = field_df.drop(columns=["teetimes"])

default_time = datetime.strptime(
    "6/15/2025  10:09:00 AM", "%m/%d/%Y  %I:%M:%S %p"
).strftime("%Y-%m-%d %H:%M")

field_df["player_name"] = field_df["player_name"].str.lower().replace(name_replacements)
for col in ("r1_teetime", "r2_teetime"):
    if col in field_df.columns:
        missing_mask = field_df[col].replace("", np.nan).isna()
        if missing_mask.any():
            print(f"Filling defaults for {col} for players: "
                  f"{field_df.loc[missing_mask, 'player_name'].tolist()}")
        field_df[col] = field_df[col].replace("", np.nan).fillna(default_time)
    else:
        field_df[col] = default_time

tee_time_df = field_df[["player_name", "r1_teetime", "r2_teetime"]].copy()

# ---------------------------------------------------------------------------
# 2. Load regressed predictions + merge tee times
# ---------------------------------------------------------------------------
print(f"Loading predictions for {tourney}...")
df = pd.read_csv(f"final_predictions_{tourney}.csv")
df["player_name"] = df["player_name"].str.lower().replace(name_replacements)

df = df.drop(columns=["r1_teetime", "r2_teetime"], errors="ignore")
df = df.merge(tee_time_df, on="player_name", how="left")
df["r1_teetime"] = df["r1_teetime"].fillna(default_time)
df["r2_teetime"] = df["r2_teetime"].fillna(default_time)

# ---------------------------------------------------------------------------
# 3. DG 50/50 blend (for the etr-golf-sims Rust sim)
# ---------------------------------------------------------------------------
dg_decomp = fetch_player_decompositions(API_KEY)
if not dg_decomp.empty and "dg_final_pred" in dg_decomp.columns:
    dg_decomp = dg_decomp[["player_name", "dg_final_pred"]].copy()
    dg_decomp["player_name"] = dg_decomp["player_name"].str.lower().replace(name_replacements)
    df = df.merge(dg_decomp, on="player_name", how="left")
    has_dg = df["dg_final_pred"].notna()
    if has_dg.sum() > 0:
        old_preds = df.loc[has_dg, "my_pred"].copy()
        df.loc[has_dg, "my_pred"] = (
            0.5 * df.loc[has_dg, "my_pred"] + 0.5 * df.loc[has_dg, "dg_final_pred"]
        )
        avg_shift = (df.loc[has_dg, "my_pred"] - old_preds).abs().mean()
        print(f"[DG blend] 50/50 blended {has_dg.sum()} players (avg shift: {avg_shift:.3f} SG)")
    df = df.drop(columns=["dg_final_pred"])
else:
    print("[DG blend] No decomposition data from API — skipping blend")

# ---------------------------------------------------------------------------
# 4. Export + copy to etr-golf-sims
# ---------------------------------------------------------------------------
df.to_csv(f"final_predictions_{tourney}_combineddetail.csv", index=False)

output_filename = f"final_predictions_{tourney}_combined.csv"
out = df[["player_name", "my_pred", "std_dev", "r1_teetime", "r2_teetime"]]
out.to_csv(output_filename, index=False)
print(f"[ok] Saved {output_filename} ({len(out)} players)")

destination_file = os.path.join(ETR_REPO, output_filename)
try:
    os.makedirs(ETR_REPO, exist_ok=True)
    shutil.copyfile(output_filename, destination_file)
    print(f"[ok] Copied {output_filename} -> {ETR_REPO}")
except Exception as e:
    print(f"[err] Could not copy {output_filename} to {ETR_REPO}: {e}")
    raise
