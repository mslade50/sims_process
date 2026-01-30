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

def fetch_live_tournament_stats_all_to_df(round=None, display="value", file_format="json", api_key="your_api_key_here"):
    base_url_stats = "https://feeds.datagolf.com/preds/live-tournament-stats"
    base_url_field_updates = "https://feeds.datagolf.com/field-updates"
    
    # All available stats
    all_stats = [
        "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_bs", "sg_total",
        "distance", "accuracy", "gir", "prox_fw", "prox_rgh", "scrambling","great_shots",'poor_shots'
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
            
            # Extract only R3 tee times
            if 'r4_teetime' in df_field.columns:
                df_field = df_field[['player_name', 'r4_teetime']].dropna(subset=['r4_teetime'])
                df_field['player_name'] = df_field['player_name'].str.lower().replace(name_replacements)  # Match casing for merging
            else:
                print("No 'r4_teetime' field found in response.")
                return df_stats
        else:
            print("No 'field' data found in response.")
            return df_stats
    else:
        print(f"Error fetching field updates: {response_field_updates.status_code} - {response_field_updates.text}")
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
df_tournament_stats = fetch_live_tournament_stats_all_to_df(round=3, api_key=api_key)
df_tournament_stats['player_name'] = df_tournament_stats['player_name'].str.lower()

# Import the CSV file
model_predictions = pd.read_csv("r2_live_model.csv")
model_predictions_r3 = pd.read_csv("model_predictions_r3.csv")

# Ensure 'player_name' in both DataFrames is lowercase for consistent merging
model_predictions['player_name'] = model_predictions['player_name'].str.lower()
model_predictions_r3['player_name'] = model_predictions_r3['player_name'].str.lower()

# Merge the 'updated_pred_r3' from model_predictions and 'wind_adj3' from model_predictions_r3
df_tournament_stats = df_tournament_stats.merge(
    model_predictions[['player_name', 'updated_pred_r3','sg_app_avg','sg_ott_avg','tot_resid_adj',
    'sg_arg_avg','sg_putt_avg','tot_sg_adj']], 
    on='player_name', 
    how='left'
).merge(
    model_predictions_r3[['player_name', 'wind_adj3', 'dew_adj3','r3_teetime']], 
    on='player_name', 
    how='left'
)
df_tournament_stats.to_csv('test.csv')
for sg in ['app', 'ott', 'arg', 'putt']:
    df_tournament_stats[f'sg_{sg}_avg'] = (
        df_tournament_stats[f'sg_{sg}_avg'] * 0.66 +
        df_tournament_stats[f'sg_{sg}'] * 0.34
    )
# Calculate the average of 'my_pred2'
my_pred2_avg = df_tournament_stats['updated_pred_r3'].mean()

# Create 'sg_total_adj' by adding the average of 'my_pred2' to each value in 'sg_total'
df_tournament_stats['sg_total_adj'] = df_tournament_stats['sg_total'] + my_pred2_avg
wind_adjust_avg= df_tournament_stats['wind_adj3'].mean()
df_tournament_stats['player_wind_benefit'] = df_tournament_stats['wind_adj3'] - wind_adjust_avg

dew_adj_avg= df_tournament_stats['dew_adj3'].mean()
df_tournament_stats['player_dew_benefit'] = df_tournament_stats['dew_adj3'] - dew_adj_avg

# Create 'outperformance_r1' as the difference between 'sg_total_adj' and 'my_pred2'

# Adjust the outperformance formula with weighted strokes gained
df_tournament_stats['residual'] = (
    df_tournament_stats['sg_total_adj']
    - df_tournament_stats['updated_pred_r3']
    + df_tournament_stats['player_wind_benefit']
    + df_tournament_stats['player_dew_benefit']

)  
cols = ['sg_total', 'updated_pred_r3', 'wind_adj3', 'dew_adj3']
df_tournament_stats = df_tournament_stats.dropna(subset=cols)

avg_abs_residual = df_tournament_stats['residual'].abs().mean()
print(f"Average absolute residual: {avg_abs_residual:.4f}")

mse = np.sqrt((df_tournament_stats['residual'] ** 2).mean())

print(f"Mean squared error: {mse:.4f}")
r_squared = r2_score(df_tournament_stats['sg_total_adj'], df_tournament_stats['updated_pred_r3'])
print(f"R² between sg_total_adj and updated_pred_r3: {r_squared:.4f}")

summary_row = pd.DataFrame([{
    'event_name': df_tournament_stats['event_name'].iloc[0],
    'round_num': 3,
    'average_residual': avg_abs_residual,
    'mse': mse,
    'r_squared': r_squared,
    'year': datetime.now().year
}])
summary_path = "residual_summary.csv"

if os.path.exists(summary_path):
    summary_row.to_csv(summary_path, mode='a', header=False, index=False)
else:
    summary_row.to_csv(summary_path, index=False)

# Ensure tee time is in datetime format
df_tournament_stats['r3_teetime'] = pd.to_datetime(df_tournament_stats['r3_teetime'])

# Convert tee time to integer timestamps (needed for spline)
df_tournament_stats['teetime_numeric'] = df_tournament_stats['r3_teetime'].astype(np.int64)

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
    df_tournament_stats.sort_values('r3_teetime'),
    x='r3_teetime',
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

df_tournament_stats['position'] = df_tournament_stats['position'].str.replace('T', '', regex=False)
df_tournament_stats['position'] = pd.to_numeric(df_tournament_stats['position'], errors='coerce')


# Define masks
pos_lt_6 = df_tournament_stats['position'] < 6
pos_6_30 = (df_tournament_stats['position'] >= 6) & (df_tournament_stats['position'] <= 20)
pos_gt_30 = df_tournament_stats['position'] > 20

# Apply adjustments based on each coefficient block
for col, coeff in coefficients_r3.items():
    df_tournament_stats.loc[pos_lt_6, f'{col}_adj_r3'] = df_tournament_stats.loc[pos_lt_6, col] * coeff

for col, coeff in coefficients_r3_mid.items():
    df_tournament_stats.loc[pos_6_30, f'{col}_adj_r3'] = df_tournament_stats.loc[pos_6_30, col] * coeff

for col, coeff in coefficients_r3_high.items():
    df_tournament_stats.loc[pos_gt_30, f'{col}_adj_r3'] = df_tournament_stats.loc[pos_gt_30, col] * coeff

adj_cols = [f'{col}_adj_r3' for col in coefficients_r3.keys()]
df_tournament_stats['total_adjustment'] = df_tournament_stats[adj_cols].sum(axis=1)

# Update R4 prediction
df_tournament_stats['updated_pred_r4'] = (
    df_tournament_stats['updated_pred_r3'] 
    - df_tournament_stats['tot_sg_adj'] 
    - df_tournament_stats['tot_resid_adj']
    + df_tournament_stats['total_adjustment']
)
df_tournament_stats['Score']=df_tournament_stats['round']+course_par
# Display the updated DataFrame
print(df_tournament_stats)


df_tournament_stats.to_csv('r3_live_model.csv')
df_tournament_stats.to_csv(f'r3_live_model_{tourney}.csv')

# Define columns to keep
adj_cols = [f'{col}_adj_r3' for col in ['sg_ott_avg', 'sg_putt_avg', 'sg_app_avg', 'sg_arg_avg']]
columns_to_keep = ['player_name'] + adj_cols + ['updated_pred_r3', 'updated_pred_r4']
df_tournament_stats = df_tournament_stats[columns_to_keep]

# Filter the DataFrame
df_tournament_stats = df_tournament_stats[columns_to_keep]
df_tournament_stats['delta'] = df_tournament_stats['updated_pred_r4']- df_tournament_stats['updated_pred_r3']
df_tournament_stats.to_csv('r3_live_model_summary.csv')