import requests
import pandas as pd
import numpy as np
import os
import plotly.express as px
from patsy import dmatrix
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# --- INPUTS FROM YOUR LOCAL FILES ---
from sim_inputs import course_par, name_replacements, tourney
from sim_inputs import coefficients_r1_high, coefficients_r1_midh, coefficients_r1_midl, coefficients_r1_low

def fetch_live_tournament_stats_all_to_df(round=None, display="value", file_format="json", api_key="your_api_key_here"):
    # Base URLs
    base_url_stats = "https://feeds.datagolf.com/preds/live-tournament-stats"
    base_url_field_updates = "https://feeds.datagolf.com/field-updates"
    
    # ADDED "score" to this list to ensure we can calculate manual SG if needed
    all_stats = [
        "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_bs", "sg_total",
        "distance", "accuracy", "gir", "prox_fw", "prox_rgh", "scrambling", 
        "great_shots", "poor_shots", "score"
    ]
    
    # Fetch live stats
    params_stats = {
        "stats": ",".join(all_stats),
        "round": round,
        "display": display,
        "file_format": file_format,
        "key": api_key
    }
    
    response_stats = requests.get(base_url_stats, params=params_stats)
    if response_stats.status_code == 200:
        data_stats = response_stats.json()
        if 'live_stats' in data_stats:
            df_stats = pd.DataFrame(data_stats['live_stats'])
            
            # Add metadata fields
            df_stats['course_name'] = data_stats.get('course_name', None)
            df_stats['event_name'] = data_stats.get('event_name', None)
            df_stats['last_updated'] = data_stats.get('last_updated', None)
            
            # Reorder columns
            columns = ['player_name'] + [col for col in df_stats.columns if col != 'player_name']
            df_stats = df_stats[columns]
            df_stats['player_name'] = df_stats['player_name'].str.lower().replace(name_replacements)
        else:
            print("No 'live_stats' field found in response.")
            return None
    else:
        print(f"Error fetching stats: {response_stats.status_code} - {response_stats.text}")
        return None
    
    # Fetch field updates for course information
    params_field_updates = {
        "tour": "pga", 
        "file_format": file_format,
        "key": api_key
    }
    
    response_field_updates = requests.get(base_url_field_updates, params=params_field_updates)
    if response_field_updates.status_code == 200:
        data_field_updates = response_field_updates.json()
        
        if 'field' in data_field_updates:
            df_field = pd.DataFrame(data_field_updates['field'])
            
            # Extract course information
            if 'course' in df_field.columns:
                # Keep relevant columns
                df_field = df_field[['player_name', 'course','r1_teetime']].dropna(subset=['course'])
                df_field['player_name'] = df_field['player_name'].str.lower()
            else:
                print("No 'course' field found in field updates response.")
                return df_stats
        else:
            print("No 'field' data found in response.")
            return df_stats
    else:
        print(f"Error fetching field updates: {response_field_updates.status_code} - {response_field_updates.text}")
        return df_stats

    # Merge the course information into the live stats DataFrame
    # Note: This will result in 'course' from df_field matching into df_stats
    df_merged = pd.merge(df_stats, df_field, on='player_name', how='left') 

    return df_merged

# --- MAIN SCRIPT EXECUTION ---

# 1. Setup and Fetch
api_key = "c05ee5fd8f2f3b14baab409bd83c"  # Replace with actual key
df_tournament_stats = fetch_live_tournament_stats_all_to_df(round=1, api_key=api_key)

# Clean names
df_tournament_stats['player_name'] = df_tournament_stats['player_name'].str.lower().replace(name_replacements)

# 2. Load Pre-Tournament Predictions
model_predictions = pd.read_csv('model_predictions_r1.csv')
model_predictions['player_name'] = model_predictions['player_name'].str.lower().replace(name_replacements)
model_predictions['pred'] = model_predictions['my_pred']

# 3. Merge Predictions
df_tournament_stats = df_tournament_stats.merge(
    model_predictions[['player_name', 'wind_adj1', 'dew_adj1', 'pred']], 
    on='player_name', 
    how='left'
)

# Ensure tee times are correct format for Spline
df_tournament_stats['r1_teetime'] = pd.to_datetime(df_tournament_stats['r1_teetime'])
df_tournament_stats['teetime_numeric'] = df_tournament_stats['r1_teetime'].astype(np.int64)

# ---------------------------------------------------------
# NEW LOGIC: SPLIT PROCESSING BY COURSE_X
# ---------------------------------------------------------

processed_frames = []
course_col = 'course_x'  # As requested

# Get unique courses, ignoring NaNs
courses = [c for c in df_tournament_stats[course_col].unique() if pd.notna(c)]

if not courses:
    # Fallback if no course info is found (shouldn't happen if merge worked)
    print("Warning: No course info found. Processing as single batch.")
    courses = [None]

for course_id in courses:
    if course_id is None:
        df_course = df_tournament_stats.copy()
    else:
        df_course = df_tournament_stats[df_tournament_stats[course_col] == course_id].copy()
    
    print(f"Processing Course: {course_id} | Players: {len(df_course)}")

    # --- A. HANDLE MISSING SG_TOTAL (Satellite Logic) ---
    # If sg_total is NaN, calculate it: (Avg Score of Field on this course - Player Score)
    if 'sg_total' not in df_course.columns or df_course['sg_total'].isna().all():
        print(f"  -> Missing sg_total for {course_id}, calculating proxy from score...")
        if 'score' in df_course.columns:
            avg_score = df_course['score'].mean()
            df_course['sg_total'] = avg_score - df_course['score']
            df_course['sg_total'] = df_course['sg_total'].fillna(0)
        else:
            print("  -> ERROR: No 'score' column found to calculate proxy.")
            df_course['sg_total'] = 0

    # Fill remaining NaNs in sg_total just in case
    df_course['sg_total'] = df_course['sg_total'].fillna(0)

    # --- B. STANDARD ADJUSTMENTS ---
    my_pred_avg = df_course['pred'].mean()
    df_course['sg_total_adj'] = df_course['sg_total'] + my_pred_avg
    
    wind_adjust_avg = df_course['wind_adj1'].mean()
    dew_adj_avg = df_course['dew_adj1'].mean()
    
    df_course['player_wind_benefit'] = df_course['wind_adj1'] - wind_adjust_avg
    df_course['player_dew_benefit'] = df_course['dew_adj1'] - dew_adj_avg

    df_course['residual'] = (
        df_course['sg_total_adj'] 
        - df_course['pred'] 
        + df_course['player_wind_benefit'] 
        + df_course['dew_adj1']
    )

    # --- C. COURSE-SPECIFIC SPLINE ---
    # We fit the weather/wave spline specifically for this course's tee times
    if len(df_course) > 10:
        try:
            X_spline = dmatrix("bs(teetime_numeric, df=4, degree=3, include_intercept=False)",
                               {"teetime_numeric": df_course['teetime_numeric']},
                               return_type='dataframe')
            
            spline_model = sm.OLS(df_course['residual'].fillna(0), X_spline).fit()
            df_course['weather_signal'] = spline_model.fittedvalues
        except Exception as e:
            print(f"  -> Spline failed for {course_id}: {e}")
            df_course['weather_signal'] = 0
    else:
        df_course['weather_signal'] = 0

    df_course['residual_w_adj'] = df_course['residual'] - df_course['weather_signal']
    df_course['residual2'] = df_course['residual']**2

    # --- D. APPLY COEFFICIENTS (Conditional on ShotLink) ---
    # Check if this course actually has SG_OTT data (Host Course vs Satellite)
    # If sum of sg_ott is 0 or all NaN, we assume no ShotLink.
    has_shotlink = False
    if 'sg_ott' in df_course.columns:
        if df_course['sg_ott'].abs().sum() > 0:
            has_shotlink = True
            
    print(f"  -> Has ShotLink Data? {has_shotlink}")

    # Fill granular stats with 0 for calculation safety
    cols_to_fill = ['sg_ott', 'sg_putt']
    for c in cols_to_fill:
        if c in df_course.columns:
            df_course[c] = df_course[c].fillna(0)
        else:
            df_course[c] = 0.0

    # Initialize Adjustments
    df_course['ott_adj'] = 0.0
    df_course['putt_adj'] = 0.0
    df_course['residual_adj'] = 0.0
    df_course['residual2_adj'] = 0.0

    # Define Masks
    high_mask = df_course['pred'] > 1
    midh_mask = (df_course['pred'] > 0.5) & (df_course['pred'] <= 1)
    midl_mask = (df_course['pred'] > -0.5) & (df_course['pred'] <= 0.5)
    low_mask = df_course['pred'] <= -0.5

    def apply_coeffs(mask, coeffs):
        # Always apply residual adjustments
        df_course.loc[mask, 'residual_adj'] = df_course.loc[mask, 'residual'] * coeffs['residual']
        df_course.loc[mask, 'residual2_adj'] = df_course.loc[mask, 'residual2'] * coeffs['residual2']
        
        # Only apply granular adjustments if ShotLink exists
        if has_shotlink:
            df_course.loc[mask, 'ott_adj'] = df_course.loc[mask, 'sg_ott'] * coeffs['ott']
            df_course.loc[mask, 'putt_adj'] = df_course.loc[mask, 'sg_putt'] * coeffs['putt']
        else:
            # Explicitly 0 out if no shotlink (redundant but safe)
            df_course.loc[mask, 'ott_adj'] = 0
            df_course.loc[mask, 'putt_adj'] = 0

    apply_coeffs(high_mask, coefficients_r1_high)
    apply_coeffs(midh_mask, coefficients_r1_midh)
    apply_coeffs(midl_mask, coefficients_r1_midl)
    apply_coeffs(low_mask, coefficients_r1_low)

    # --- E. TOTALS & CAPS ---
    df_course['tot_resid_adj'] = df_course['residual_adj'] + df_course['residual2_adj']
    
    # Cap logic
    df_course['tot_resid_adj'] = np.minimum(
        np.where(
            (df_course['tot_resid_adj'] > 0.2) & (df_course['residual'] < 0),
            0.2,
            df_course['tot_resid_adj']
        ),
        0.5
    )

    df_course['total_adjustment'] = (
        df_course['ott_adj'] 
        + df_course['putt_adj'] 
        + df_course['tot_resid_adj']
    )

    df_course['updated_pred'] = df_course['pred'] + df_course['total_adjustment']
    
    processed_frames.append(df_course)

# --- 4. RECOMBINE ---
if processed_frames:
    df_tournament_stats = pd.concat(processed_frames)
else:
    print("Error: No data processed.")

# Filter WD/DQ
df_tournament_stats = df_tournament_stats[~df_tournament_stats['position'].str.contains("WD|DQ", na=False)]
df_tournament_stats['position'] = df_tournament_stats['position'].astype(str).str.replace("T", "").astype(int)

# Position Adjustments (Leaderboard Gravity)
position_adjustments_default = {1: -0.07, 2: -0.03, 3: -0.02, 4: -0.01, 5:-0.01}
df_tournament_stats.loc[df_tournament_stats['updated_pred'] < 0.5, 'updated_pred'] += (
    df_tournament_stats['position'].map(position_adjustments_default).fillna(0)
)

df_tournament_stats['Score'] = df_tournament_stats['round'] + course_par

# --- 5. SUMMARY METRICS (Global) ---
# We calculate RMSE on the whole dataset to see overall fit
valid_mask = df_tournament_stats[['sg_total_adj', 'pred', 'residual']].dropna()
if not valid_mask.empty:
    avg_abs_residual = valid_mask['residual'].abs().mean()
    rmse = np.sqrt((valid_mask['residual'] ** 2).mean())
    r_squared = r2_score(valid_mask['sg_total_adj'], valid_mask['pred'])

    print(f"Average absolute residual: {avg_abs_residual:.4f}")
    print(f"Root mean squared error (RMSE): {rmse:.4f}")
    print(f"R² between sg_total_adj and pred: {r_squared:.4f}")

    # Append to Summary CSV
    summary_row = pd.DataFrame([{
        'event_name': df_tournament_stats['event_name'].iloc[0] if not df_tournament_stats.empty else "Unknown",
        'round_num': 1,
        'average_residual': avg_abs_residual,
        'rmse': rmse,
        'r_squared': r_squared,
        'year': datetime.now().year
    }])

    summary_path = "residual_summary.csv"
    if os.path.exists(summary_path):
        summary_row.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        summary_row.to_csv(summary_path, index=False)

# --- 6. EXPORTS & PLOTTING ---
print(df_tournament_stats[['player_name', 'course_x', 'sg_total', 'updated_pred']].head())

df_tournament_stats.to_csv('r1_live_model.csv', index=False)
df_tournament_stats.to_csv(f'r1_live_model_{tourney}.csv', index=False)

columns_to_keep = ['player_name', 'residual', 'weather_signal', 'residual_w_adj', 'tot_resid_adj', 'total_adjustment', 'updated_pred', 'course_x']
df_tournament_stats[columns_to_keep].to_csv('r1_live_summary.csv', index=False)

# Plotting (Colored by Course to see different weather/scoring environments)
fig = px.line(
    df_tournament_stats.sort_values('r1_teetime'),
    x='r1_teetime',
    y='weather_signal',
    color='course_x', # Separate lines for separate courses
    title='Spline-Smoothed Residual by Tee Time (Per Course)',
    labels={'r1_teetime': 'Tee Time', 'weather_signal': 'Estimated Weather Signal'}
)

fig.update_layout(
    xaxis_tickformat="%H:%M",
    xaxis_title='Tee Time',
    yaxis_title='Smoothed Residual',
    template='plotly_white'
)

fig.show()