import requests
import pandas as pd
from datetime import datetime
import numpy as np
from sim_inputs import course_par, wind_3, event_ids, wind_override, baseline_wind, name_replacements
from sim_inputs import tee_time_start, tee_time_end, tourney, coefficients_r2, coefficients_r2_30_up, coefficients_r2_6_30
from patsy import dmatrix
import statsmodels.api as sm
import numpy as np
import os
import plotly.express as px
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score

def fetch_live_tournament_stats_all_to_df(
    round=None, 
    display="value", 
    file_format="json", 
    api_key="your_api_key_here", 
    ignore_field_updates=False,  # Skip field updates
    teetime_start=tee_time_start,  # Default start time
    teetime_end=tee_time_end  # Default end time
):
    base_url_stats = "https://feeds.datagolf.com/preds/live-tournament-stats"
    base_url_field_updates = "https://feeds.datagolf.com/field-updates"
    
    # All available stats
    all_stats = [
        "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_bs", "sg_total",
        "distance", "accuracy", "gir", "prox_fw", "prox_rgh", "scrambling", 
        "great_shots", "poor_shots"
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
            
            # Reorder columns to make 'player_name' the first column
            columns = ['player_name'] + [col for col in df_stats.columns if col != 'player_name']
            df_stats = df_stats[columns]
            
        else:
            print("No 'live_stats' field found in response.")
            return None
    else:
        print(f"Error fetching stats: {response_stats.status_code} - {response_stats.text}")
        return None

    # If ignore_field_updates is True, process the position column and assign r3_teetime
    if ignore_field_updates:
        if 'position' in df_stats.columns:
            df_stats['position'] = df_stats['position'].str.replace("T", "", regex=False)

            # Remove rows where 'position' is not numeric (e.g., 'WD')
            df_stats = df_stats[~df_stats['position'].str.contains(r'\D', na=True)]

            # Convert 'position' to float after cleaning
            df_stats['position'] = df_stats['position'].astype(float)

            # Remove rows where position > 200
            df_stats = df_stats[df_stats['position'] <= 200]

            # Calculate percentile rank
            df_stats['percentile'] = df_stats['position'].rank(pct=True, method="max", ascending=True)

            # Hardcoded round date
            round_date = "8/9/2025"

            # Convert start and end times to datetime
            start_time = datetime.strptime(f"{round_date} {teetime_start}", "%m/%d/%Y %H:%M")
            end_time = datetime.strptime(f"{round_date} {teetime_end}", "%m/%d/%Y %H:%M")
            total_minutes = int((end_time - start_time).total_seconds() // 60)

            # Function to format without leading zero in month/day/hour
            def fmt_no_leading_zero(dt):
                return f"{dt.month}/{dt.day}/{dt.year} {dt.hour}:{dt.strftime('%M')}"

            # Calculate r3_teetime based on percentile rank
            df_stats['r3_teetime'] = df_stats['percentile'].apply(
                lambda p: fmt_no_leading_zero(start_time + pd.Timedelta(minutes=p * total_minutes))
            )
            df_stats['player_name'] = df_stats['player_name'].str.lower().replace(name_replacements)

        else:
            print("Position column not found in stats data.")
        return df_stats

    # Fetch field updates for R3 tee times
    params_field_updates = {
        "tour": "pga",  # Change tour if needed
        "file_format": file_format,
        "key": api_key
    }
    
    response_field_updates = requests.get(base_url_field_updates, params=params_field_updates)
  
    if response_field_updates.status_code == 200:
        data_field_updates = response_field_updates.json()
 
        if 'field' in data_field_updates:
            df_field = pd.DataFrame(data_field_updates['field'])
            df_field.to_csv('test.csv')
            # Extract only R3 tee times
            if 'r3_teetime' in df_field.columns:
                df_field = df_field[['player_name', 'r3_teetime', 'course']].dropna(subset=['r3_teetime'])
                df_field['player_name'] = df_field['player_name'].str.lower().replace(name_replacements)

            else:
                print("No 'r3_teetime' field found in response.")
                return df_stats
        else:
            print("No 'field' data found in response.")
            return df_stats
    else:
        print(f"Error fetching field updates: {response_field_updates.status_code} - {response_field_updates.text}")
        return df_stats

    # Merge R3 tee times into live stats
    df_stats['player_name']=df_stats['player_name'].str.lower().replace(name_replacements)
    df_field['player_name']=df_field['player_name'].str.lower().replace(name_replacements)
    df_merged = pd.merge(df_stats, df_field, on='player_name', how='left')  # Left join to keep all players from df_stats # Inner join to keep only players with R3 tee times

    return df_merged


api_key = "c05ee5fd8f2f3b14baab409bd83c"  # Replace with your actual API key
df_tournament_stats = fetch_live_tournament_stats_all_to_df(round=2, api_key=api_key)

df_tournament_stats['player_name'] = df_tournament_stats['player_name'].str.lower()

# model_predictions = pd.read_csv("r1_live_model.csv")
model_predictions = pd.read_csv("model_predictions_r2.csv")
r1_live_model = pd.read_csv("r1_live_model.csv")
r1_cols = ['player_name', 'great_shots', 'poor_shots', 'sg_app', 'sg_arg', 'sg_ott', 'sg_putt','sg_adj_r1']

# Rename the stat columns with _r1 suffix
r1_renamed = r1_live_model[r1_cols].rename(columns={
    'great_shots': 'great_shots_r1',
    'poor_shots': 'poor_shots_r1',
    'sg_app': 'sg_app_r1',
    'sg_arg': 'sg_arg_r1',
    'sg_ott': 'sg_ott_r1',
    'sg_putt': 'sg_putt_r1'
})

df_tournament_stats = df_tournament_stats.merge(r1_renamed, on='player_name', how='left')

# Average each pair of stats: live (current) + round 1 version
df_tournament_stats['great_shots_avg'] = df_tournament_stats[['great_shots', 'great_shots_r1']].mean(axis=1)
df_tournament_stats['poor_shots_avg'] = df_tournament_stats[['poor_shots', 'poor_shots_r1']].mean(axis=1)
df_tournament_stats['sg_app_avg'] = df_tournament_stats[['sg_app', 'sg_app_r1']].mean(axis=1)
df_tournament_stats['sg_arg_avg'] = df_tournament_stats[['sg_arg', 'sg_arg_r1']].mean(axis=1)
df_tournament_stats['sg_ott_avg'] = df_tournament_stats[['sg_ott', 'sg_ott_r1']].mean(axis=1)
df_tournament_stats['sg_putt_avg'] = df_tournament_stats[['sg_putt', 'sg_putt_r1']].mean(axis=1)
df_tournament_stats['sg_app_delta'] = df_tournament_stats['sg_app'] - df_tournament_stats['sg_app_r1']

# Ensure 'player_name' in model_predictions is also lowercase for consistent merging
model_predictions['player_name'] = model_predictions['player_name'].str.lower()

df_tournament_stats = df_tournament_stats.merge(
    model_predictions[['player_name','r2_teetime','wind_adj2','dew_adj2', 'my_pred2']],
    on='player_name',
    how='left'
).rename(columns={'my_pred2': 'updated_pred'})

# Calculate the average of 'my_pred2'
my_pred2_avg = df_tournament_stats['updated_pred'].mean()

# Create 'sg_total_adj' by adding the average of 'my_pred2' to each value in 'sg_total'
df_tournament_stats['sg_total_adj'] = df_tournament_stats['sg_total'] + my_pred2_avg

df_tournament_stats = df_tournament_stats.dropna(subset=['updated_pred', 'sg_total_adj'])

wind_adjust_avg= df_tournament_stats['wind_adj2'].mean()
dew_adj_avg = df_tournament_stats['dew_adj2'].mean()
df_tournament_stats['player_wind_benefit'] = df_tournament_stats['wind_adj2'] - wind_adjust_avg
df_tournament_stats['player_dew_benefit'] = df_tournament_stats['dew_adj2'] - dew_adj_avg

df_tournament_stats['residual'] = (
    df_tournament_stats['sg_total_adj']
    - df_tournament_stats['updated_pred']
    + df_tournament_stats['player_wind_benefit']
    + df_tournament_stats['player_dew_benefit']
)  

avg_abs_residual = df_tournament_stats['residual'].abs().mean()
rmse = np.sqrt((df_tournament_stats['residual'] ** 2).mean())
r_squared = r2_score(df_tournament_stats['sg_total_adj'], df_tournament_stats['updated_pred'])

print(f"Average absolute residual: {avg_abs_residual:.4f}")
print(f"Root mean squared error (RMSE): {rmse:.4f}")
print(f"R² between sg_total_adj and updated_pred: {r_squared:.4f}")

summary_row = pd.DataFrame([{
    'event_name': df_tournament_stats['event_name'].iloc[0],
    'round_num': 2,
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

# Ensure tee time is in datetime format
df_tournament_stats['r2_teetime'] = pd.to_datetime(df_tournament_stats['r2_teetime'])

# Convert tee time to integer timestamps (needed for spline)
df_tournament_stats['teetime_numeric'] = df_tournament_stats['r2_teetime'].astype(np.int64)

# Create spline basis (df controls smoothness)
X_spline = dmatrix("bs(teetime_numeric, df=4, degree=3, include_intercept=False)",
                   {"teetime_numeric": df_tournament_stats['teetime_numeric']},
                   return_type='dataframe')

# Fit OLS model: residual ~ spline(teetime)
spline_model = sm.OLS(df_tournament_stats['residual'], X_spline).fit()

# Fitted values give smoothed tee-time-based signal
df_tournament_stats['weather_signal'] = spline_model.fittedvalues

# Optional: adjust residuals using estimated wave effect
df_tournament_stats['residual_w_adj'] = (
    df_tournament_stats['residual'] - df_tournament_stats['weather_signal']
)


fig = px.line(
    df_tournament_stats.sort_values('r2_teetime'),
    x='r2_teetime',
    y='weather_signal',
    title='Smoothed Residual by Tee Time',
    labels={'r2_teetime': 'Tee Time', 'weather_signal': 'Estimated Weather Signal'},
    line_group=None  # prevents grouping by other columns
)

fig.update_layout(
    xaxis_tickformat="%H:%M",
    xaxis_title='Tee Time',
    yaxis_title='Smoothed Residual',
    template='plotly_white',
    showlegend=False
)

fig.show()

df_tournament_stats['residual2'] = df_tournament_stats['residual']**2
df_tournament_stats['residual3'] = df_tournament_stats['residual']**3

df_tournament_stats['position'] = pd.to_numeric(
    df_tournament_stats['position'].astype(str).str.replace('T', '', regex=False),
    errors='coerce'
)

df_tournament_stats['position'] = pd.to_numeric(df_tournament_stats['position'], errors='coerce')

column_mapping = {
    'residual': 'residual',
    'residual2': 'residual2',
    'residual3': 'residual3',
    'avg_ott': 'sg_ott_avg',
    'avg_putt': 'sg_putt_avg',
    'avg_app': 'sg_app_avg',
    'avg_arg': 'sg_arg_avg',
    'delta_app': 'sg_app_delta'
}

pos_lt_6_mask = df_tournament_stats['position'].astype(float) < 6
pos_6_30_mask = (df_tournament_stats['position'].astype(float) >= 6) & (df_tournament_stats['position'].astype(float) <= 30)
pos_gt_30_mask = df_tournament_stats['position'].astype(float) > 30

# Apply coefficients_r2 where position < 6
for key, col in column_mapping.items():
    df_tournament_stats.loc[pos_lt_6_mask, f'{key}_adj'] = (
        df_tournament_stats.loc[pos_lt_6_mask, col] * coefficients_r2[key]
    )

# Apply coefficients_r2_6_30 where position is between 6 and 30
for key, col in column_mapping.items():
    df_tournament_stats.loc[pos_6_30_mask, f'{key}_adj'] = (
        df_tournament_stats.loc[pos_6_30_mask, col] * coefficients_r2_6_30[key]
    )

# Apply coefficients_r2_30_up where position > 30
for key, col in column_mapping.items():
    df_tournament_stats.loc[pos_gt_30_mask, f'{key}_adj'] = (
        df_tournament_stats.loc[pos_gt_30_mask, col] * coefficients_r2_30_up[key]
    )


df_tournament_stats['tot_resid_adj'] = (
    df_tournament_stats['residual_adj'] +
    df_tournament_stats['residual2_adj'] +
    df_tournament_stats['residual3_adj']
).clip(lower=-0.5)

df_tournament_stats['tot_sg_adj'] = (
    df_tournament_stats['avg_arg_adj'] +
    df_tournament_stats['avg_putt_adj'] +
    df_tournament_stats['avg_ott_adj'] +
    df_tournament_stats['avg_app_adj'] +
    df_tournament_stats['delta_app_adj'] 
)

# Sum all adjustment columns
adj_cols = [f'{key}_adj' for key in column_mapping.keys()]
df_tournament_stats['total_adjustment'] = df_tournament_stats[adj_cols].sum(axis=1) -df_tournament_stats['sg_adj_r1']

# Drop individual residual adjustment columns
df_tournament_stats = df_tournament_stats.drop(columns=[
    'residual_adj', 'residual2_adj', 'residual3_adj'
])

df_tournament_stats['Score']=df_tournament_stats['round']+course_par
scoring_avg = df_tournament_stats['Score'].mean()


df_tournament_stats['updated_pred_r3']=df_tournament_stats['updated_pred'] + df_tournament_stats['total_adjustment'] 
skill_avg = df_tournament_stats.loc[df_tournament_stats['r3_teetime'].notna(), 'updated_pred_r3'].mean()


# Display the updated DataFrame
print(df_tournament_stats)
print(f"Scoring Average: {scoring_avg:.2f}")
print(f"Next rd skill Average: {skill_avg:.2f}")

df_tournament_stats.to_csv('r2_live_model.csv')
df_tournament_stats.to_csv(f'r2_live_model_{tourney}.csv')
selected_columns = [
    'player_name',
    'tot_resid_adj',
    'total_adjustment',
    'avg_ott_adj',
    'avg_putt_adj',
    'avg_app_adj',
    'avg_arg_adj',
    'delta_app_adj',
    'updated_pred_r3'
]

# Filter and save
df_tournament_stats[selected_columns].to_csv('r2_live_summary.csv', index=False)
