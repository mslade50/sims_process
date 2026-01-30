import requests
import pandas as pd
from datetime import datetime
from scipy.stats import percentileofscore
from sim_inputs import course_par, wind_3, event_ids, wind_override, baseline_wind
from sim_inputs import coefficients_r3, coefficients_r3_high,coefficients_r3_mid, tourney, name_replacements
import pandas as pd
import requests
from patsy import dmatrix
import statsmodels.api as sm
import numpy as np
import os
import plotly.express as px
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import requests
import pandas as pd

def fetch_live_tournament_stats_all_to_df(round=None, display="value", file_format="json", api_key="your_api_key_here"):
    base_url_stats = "https://feeds.datagolf.com/preds/live-tournament-stats"
    
    all_stats = [
        "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_bs", "sg_total",
        "distance", "accuracy", "gir", "prox_fw", "prox_rgh", "scrambling", "great_shots", "poor_shots"
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
            print(df_stats)
            
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

    round_columns = [
        "accuracy", "distance", "gir", "great_shots", "poor_shots",
        "position", "prox_fw", "prox_rgh", "round", "scrambling", "sg_app", "sg_arg",
        "sg_bs", "sg_ott", "sg_putt", "sg_t2g", "sg_total"
    ]
    df_stats[round_columns] = df_stats[round_columns].round(2)
    return df_stats


    # Merge R3 tee times into live stats
    df_stats['player_name'] = df_stats['player_name'].str.lower().replace(name_replacements)  # Match casing for merging
    df_merged = pd.merge(df_stats, df_field, on='player_name', how='inner')  # Inner join to keep only players with R3 tee times

    round_columns = [
        "accuracy", "distance", "gir", "great_shots", "poor_shots",
        "position", "prox_fw", "prox_rgh", "round", "scrambling", "sg_app", "sg_arg",
        "sg_bs", "sg_ott", "sg_putt", "sg_t2g", "sg_total"
    ]
    df_merged[round_columns] = df_merged[round_columns].round(2)
    return df_merged

# Example usage
api_key = "c05ee5fd8f2f3b14baab409bd83c"  # Replace with your actual API key
df_tournament_stats = fetch_live_tournament_stats_all_to_df(round=4, api_key=api_key)
print(df_tournament_stats)
df_tournament_stats['player_name'] = df_tournament_stats['player_name'].str.lower()

# Import the CSV file
model_predictions = pd.read_csv("r3_live_model.csv")
model_predictions_r3 = pd.read_csv("model_predictions_r4.csv")

# Ensure 'player_name' in both DataFrames is lowercase for consistent merging
model_predictions['player_name'] = model_predictions['player_name'].str.lower()
model_predictions_r3['player_name'] = model_predictions_r3['player_name'].str.lower()

# Merge the 'updated_pred_r3' from model_predictions and 'wind_adj3' from model_predictions_r3
df_tournament_stats = df_tournament_stats.merge(
    model_predictions[['player_name', 'updated_pred_r3','sg_app_avg','sg_ott_avg',
    'sg_arg_avg','sg_putt_avg','tot_sg_adj']], 
    on='player_name', 
    how='left'
).merge(
    model_predictions_r3[['player_name', 'wind_adj4', 'dew_adj4','updated_pred_r4','r4_teetime']], 
    on='player_name', 
    how='left'
)

for sg in ['app', 'ott', 'arg', 'putt']:
    df_tournament_stats[f'sg_{sg}_avg'] = (
        df_tournament_stats[f'sg_{sg}_avg'] * 0.75 +
        df_tournament_stats[f'sg_{sg}'] * 0.25
    )
# Calculate the average of 'my_pred2'
my_pred2_avg = df_tournament_stats['updated_pred_r4'].mean()

# Create 'sg_total_adj' by adding the average of 'my_pred2' to each value in 'sg_total'
df_tournament_stats['sg_total_adj'] = df_tournament_stats['sg_total'] + my_pred2_avg
wind_adjust_avg= df_tournament_stats['wind_adj4'].mean()
df_tournament_stats['player_wind_benefit'] = df_tournament_stats['wind_adj4'] - wind_adjust_avg

dew_adj_avg= df_tournament_stats['dew_adj4'].mean()
df_tournament_stats['player_dew_benefit'] = df_tournament_stats['dew_adj4'] - dew_adj_avg

# Create 'outperformance_r1' as the difference between 'sg_total_adj' and 'my_pred2'

df_tournament_stats['residual'] = (
    df_tournament_stats['sg_total_adj']
    - df_tournament_stats['updated_pred_r4']
    + df_tournament_stats['player_wind_benefit']
    + df_tournament_stats['player_dew_benefit']
)

# Drop NaNs
valid = df_tournament_stats.dropna(subset=['residual', 'sg_total_adj', 'updated_pred_r4'])

# Metrics
avg_abs_residual = valid['residual'].abs().mean()
rmse = np.sqrt(mean_squared_error(valid['sg_total_adj'], valid['updated_pred_r4']))
r_squared = r2_score(valid['sg_total_adj'], valid['updated_pred_r4'])

print(f"Average absolute residual: {avg_abs_residual:.4f}")
print(f"Root mean squared error: {rmse:.4f}")
print(f"R² between sg_total_adj and updated_pred_r4: {r_squared:.4f}")

# Summary row
summary_row = pd.DataFrame([{
    'event_name': valid['event_name'].iloc[0],
    'round_num': 4,
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
df_tournament_stats['r4_teetime'] = pd.to_datetime(df_tournament_stats['r4_teetime'])

# Convert tee time to integer timestamps (needed for spline)
df_tournament_stats['teetime_numeric'] = df_tournament_stats['r4_teetime'].astype(np.int64)

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
    df_tournament_stats.sort_values('r4_teetime'),
    x='r4_teetime',
    y='weather_signal',
    title='Smoothed Residual by Tee Time',
    labels={'r4_teetime': 'Tee Time', 'weather_signal': 'Estimated Weather Signal'},
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

df_tournament_stats.to_csv('test_r4.csv')