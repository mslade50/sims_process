import pandas as pd
import numpy as np
import requests
from sim_inputs import tourney
import os

# Define constants
API_KEY = 'c05ee5fd8f2f3b14baab409bd83c'
URL = 'https://feeds.datagolf.com/betting-tools/matchups'

sample = pd.read_csv(f'pre_sim_summary_{tourney}.csv')
round_var = 'r2'  # Assuming we're simulating for round 1
NUM_SIMULATIONS = 100000
STD_DEV = 2.73
PAR = 72

model_preds = pd.read_csv(f"model_predictions_{round_var}.csv")
print(round_var)

pred_col = {
    'r1': 'my_pred',
    'r2': 'my_pred2',
    'r3': 'my_pred3',
    'r4': 'my_pred4',
}[round_var]


# Helper functions
def american_to_implied_probability(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100) * 100
    elif american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100) * 100
    return None

def implied_prob_to_american_odds(prob):
    """Convert implied probability to American odds."""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob+0.0000001))
    else:
        return int(100 * (1 - prob) / prob+0.00000001)

# Fetch tournament matchups
params = {
    'tour': 'pga',
    'market': 'round_matchups',
    'odds_format': 'american',
    'file_format': 'json',
    'key': API_KEY
}
response = requests.get(URL, params=params)
if response.status_code != 200:
    raise Exception(f"API request failed with status {response.status_code}: {response.text}")
data_tournament = response.json()

# Process matchup data
rows_tournament = []
for match in data_tournament['match_list']:
    p1 = match['p1_player_name'].lower()
    p2 = match['p2_player_name'].lower()
    ties_handling = match.get('ties', 'unknown')  # Default to 'unknown' if 'ties' is missing
    for bookmaker, odds in match['odds'].items():
        if bookmaker == 'datagolf':
            continue
        rows_tournament.append({
            'Player 1': p1,
            'Player 2': p2,
            'Bookmaker': bookmaker,
            'P1 Odds': odds['p1'],
            'P2 Odds': odds['p2'],
            'Datagolf Odds (P1)': match['odds']['datagolf']['p1'],
            'Datagolf Odds (P2)': match['odds']['datagolf']['p2'],
            'matchup_tag': '-'.join(sorted([p1, p2])),
            'Ties': ties_handling  # Add ties handling information
        })

df_2 = pd.DataFrame(rows_tournament)
df_2 = df_2.drop_duplicates(subset=['Player 1', 'Player 2', 'Bookmaker'], keep='first')

simulations = []
for _, row in model_preds.iterrows():
    sim_scores = np.round(PAR - np.random.normal(loc=row[f"scores_{round_var}"], scale=STD_DEV, size=NUM_SIMULATIONS)).astype(int)
    sim_df = pd.DataFrame({
        'player_name': row['player_name'],
        f'simulated_score_{round_var}': sim_scores
    })
    simulations.append(sim_df)

sim_scores = pd.concat(simulations, ignore_index=True)

# Calculate custom odds
my_odds_p1, my_odds_p2 = [], []
my_odds_p1_ties_loss, my_odds_p2_ties_loss = [], []  # New lists for odds considering ties as losses

for _, row in df_2.iterrows():
    p1 = row['Player 1']
    p2 = row['Player 2']
    p1_scores = sim_scores.loc[sim_scores['player_name'] == p1, f"simulated_score_{round_var}"]
    p2_scores = sim_scores.loc[sim_scores['player_name'] == p2, f"simulated_score_{round_var}"]

    if p1_scores.empty or p2_scores.empty:
        my_odds_p1.append(None)
        my_odds_p2.append(None)
        my_odds_p1_ties_loss.append(None)
        my_odds_p2_ties_loss.append(None)
        continue

    wins_p1 = sum(p1 < p2 for p1, p2 in zip(p1_scores, p2_scores))
    wins_p2 = sum(p1 > p2 for p1, p2 in zip(p1_scores, p2_scores))
    ties = sum(p1 == p2 for p1, p2 in zip(p1_scores, p2_scores))
    total = wins_p1 + wins_p2 + ties

    # Calculate probabilities (ties as 0.5)
    p1_prob = (wins_p1) / (total-ties)
    p2_prob = (wins_p2) / (total-ties)

    # Calculate probabilities (ties as losses)
    p1_prob_ties_loss = wins_p1 / total
    p2_prob_ties_loss = wins_p2 / total

    my_odds_p1.append(p1_prob)
    my_odds_p2.append(p2_prob)
    my_odds_p1_ties_loss.append(p1_prob_ties_loss)
    my_odds_p2_ties_loss.append(p2_prob_ties_loss)

# Add calculated columns to df_2
# Add calculated columns to df_2
df_2['my_odds_p1'] = my_odds_p1
df_2['my_odds_p2'] = my_odds_p2
df_2['my_odds_p1_ties_loss'] = my_odds_p1_ties_loss
df_2['my_odds_p2_ties_loss'] = my_odds_p2_ties_loss

# Convert bookmaker odds to numeric
df_2['P1 Odds'] = pd.to_numeric(df_2['P1 Odds'], errors='coerce')
df_2['P2 Odds'] = pd.to_numeric(df_2['P2 Odds'], errors='coerce')

# Process each bookmaker's data
dfs_by_bookmaker = {bookmaker: df for bookmaker, df in df_2.groupby('Bookmaker')}
for bookmaker, df in dfs_by_bookmaker.items():
    # Calculate implied probabilities
    df['p1_implied'] = df['P1 Odds'].apply(american_to_implied_probability).round(1)
    df['p2_implied'] = df['P2 Odds'].apply(american_to_implied_probability).round(1)

    # Add a column to indicate whether to use ties loss probabilities
    df['use_ties_loss'] = df['Ties'] == "separate bet offered"
    df['p1_decimal_odds'] = np.where(df['P1 Odds'] > 0, df['P1 Odds'] / 100 + 1, 100 / df['P1 Odds'].abs() + 1)
    df['p2_decimal_odds'] = np.where(df['P2 Odds'] > 0, df['P2 Odds'] / 100 + 1, 100 / df['P2 Odds'].abs() + 1)

    # Use the appropriate probabilities to calculate the edge
    df['edge_p1'] = np.where(
        df['use_ties_loss'],
        ((df['my_odds_p1_ties_loss'] * (df['p1_decimal_odds'] - 1)) - (1 - df['my_odds_p1_ties_loss'])) * 100,
        ((df['my_odds_p1'] * (df['p1_decimal_odds'] - 1)) - (1 - df['my_odds_p1'])) * 100
    )

    df['edge_p2'] = np.where(
        df['use_ties_loss'],
        ((df['my_odds_p2_ties_loss'] * (df['p2_decimal_odds'] - 1)) - (1 - df['my_odds_p2_ties_loss'])) * 100,
        ((df['my_odds_p2'] * (df['p2_decimal_odds'] - 1)) - (1 - df['my_odds_p2'])) * 100
    )

    # Other calculations
    df['half_shot_val_p1'] = (df['my_odds_p1'] - df['my_odds_p1_ties_loss']) * 400
    df['half_shot_val_p2'] = (df['my_odds_p2'] - df['my_odds_p2_ties_loss']) * 400

    df['p1_pushwins'] = 1 - df['my_odds_p2_ties_loss']
    df['p2_pushwins'] = 1 - df['my_odds_p1_ties_loss']

    if bookmaker == 'betonline':
        df['p1_pushwins_imp'] = (df['P1 Odds'] - 25).apply(american_to_implied_probability).round(1)
        df['p1_nopush_imp'] = (df['P1 Odds'] + 25).apply(american_to_implied_probability).round(1)
        df['p2_pushwins_imp'] = (df['P2 Odds'] - 25).apply(american_to_implied_probability).round(1)
        df['p2_nopush_imp'] = (df['P2 Odds'] + 25).apply(american_to_implied_probability).round(1)
    elif bookmaker == 'betcris':
        df['p1_pushwins_imp'] = (df['P1 Odds'] - 30).apply(american_to_implied_probability).round(1)
        df['p1_nopush_imp'] = (df['P1 Odds'] + 30).apply(american_to_implied_probability).round(1)
        df['p2_pushwins_imp'] = (df['P2 Odds'] - 30).apply(american_to_implied_probability).round(1)
        df['p2_nopush_imp'] = (df['P2 Odds'] + 30).apply(american_to_implied_probability).round(1)

    # Rename columns
    df.rename(columns={
        'my_odds_p1_ties_loss': 'p1_nopush',
        'my_odds_p2_ties_loss': 'p2_nopush',
        'half_shot_val_p1': 'half_shot1',
        'half_shot_val_p2': 'half_shot2',
        'Datagolf Odds (P1)': 'DG_p1',
        'Datagolf Odds (P2)': 'DG_p2'
    }, inplace=True)

    if bookmaker in ['betonline', 'betcris']:
        df['p1_+0.5'] = df['p1_pushwins'] * 100 - df['p1_pushwins_imp']
        df['p2_+0.5'] = df['p2_pushwins'] * 100 - df['p2_pushwins_imp']
        df['p1_-0.5'] = df['p1_nopush'] * 100 - df['p1_nopush_imp']
        df['p2_-0.5'] = df['p2_nopush'] * 100 - df['p2_nopush_imp']

    # Drop unnecessary columns
    df.drop(columns=["matchup_tag"], inplace=True)

    # Save to CSV
    output_file = f'{bookmaker}_odds_with_my_odds.csv'
    df = df.drop_duplicates(subset=['Player 1', 'Player 2', 'Bookmaker'], keep='first')
    df['Round'] = round_var
    preds_p1 = model_preds[['player_name', pred_col]].rename(columns={'player_name': 'Player 1', pred_col: 'p1_pred'})
    preds_p2 = model_preds[['player_name', pred_col]].rename(columns={'player_name': 'Player 2', pred_col: 'p2_pred'})
    df = df.merge(preds_p1, on='Player 1', how='left')
    df = df.merge(preds_p2, on='Player 2', how='left')
    df = df.dropna(subset=['my_odds_p1', 'my_odds_p2'])
    df['Fair_p1'] = df['my_odds_p1'].apply(implied_prob_to_american_odds)
    df['Fair_p2'] = df['my_odds_p2'].apply(implied_prob_to_american_odds)

    # Update final column order to include Fair_p1 and Fair_p2 after P1 Odds and P2 Odds
    final_column_order = [
        'Player 1', 'Player 2', 'Round', 'Bookmaker', 'Ties', 'P1 Odds', 'P2 Odds', 'Fair_p1', 'Fair_p2',
        'edge_p1', 'edge_p2', 'p1_implied', 'p2_implied', 'my_odds_p1', 'my_odds_p2',
        'p1_pred', 'p2_pred'
    ]

    df = df[[col for col in final_column_order if col in df.columns]]
    df['my_odds_p1'] = df['my_odds_p1'] * 100
    df['my_odds_p2'] = df['my_odds_p2'] * 100
    
    df.to_csv(output_file, index=False)

import os
from datetime import datetime

# Generate a timestamp for the combined CSV
timestamp = datetime.now().strftime('%H%M')

# Define the tournament folder
tourney_folder = f"./{tourney}"
os.makedirs(tourney_folder, exist_ok=True)

combined_csv_name = f"{tourney_folder}/matchups_{round_var}_{tourney}_{timestamp}.csv"

# Step 1: Collect all the CSVs to combine
csv_files = [f"{bookmaker}_odds_with_my_odds.csv" for bookmaker in dfs_by_bookmaker.keys()]

# Step 2: Re-import and combine
dfs_list = []
for csv_file in csv_files:
    if os.path.exists(csv_file):  # Check if the file exists
        df = pd.read_csv(csv_file)
        dfs_list.append(df)  # Add the DataFrame to the list
    else:
        print(f"File not found: {csv_file}")  # Debugging message for missing files

# Step 3: Combine all DataFrames into a single DataFrame
if dfs_list:  # Ensure there's data to combine
    combined_df = pd.concat(dfs_list, ignore_index=True)
    combined_df = combined_df[(combined_df['edge_p1'] > 3) | (combined_df['edge_p2'] > 3)]

    # Create sample lookup dictionary
    sample_lookup = dict(zip(sample['player_name'].str.lower(), sample['sample']))

    # Add sample values for Player 1 and Player 2
    combined_df['Sample_P1'] = combined_df['Player 1'].str.lower().map(sample_lookup)
    combined_df['Sample_P2'] = combined_df['Player 2'].str.lower().map(sample_lookup)

    # Add sample_on column
    combined_df['sample_on'] = combined_df.apply(
        lambda row: row['Sample_P1'] if row['edge_p1'] > row['edge_p2'] else row['Sample_P2'], axis=1
    )

    # Step 4: Save the combined DataFrame inside the tourney folder
    # combined_df.to_csv(combined_csv_name, index=False)
    print(f"Combined CSV saved as {combined_csv_name}")
else:
    print("No valid CSV files found to combine. Skipping save.")

# === Filter for sharp books and deduplicate matchups by highest edge_on ===
sharp_books = ['pinnacle', 'betonline', 'betcris']
filtered_df = combined_df[combined_df['Bookmaker'].str.lower().isin(sharp_books)].copy()

# Determine which side the edge is on and store it in a unified 'edge_on' column
filtered_df['edge_on'] = filtered_df.apply(
    lambda row: row['edge_p1'] if row['edge_p1'] > row['edge_p2'] else row['edge_p2'], axis=1
)

# Create matchup key to identify duplicates (order-independent)
filtered_df['matchup_key'] = filtered_df.apply(
    lambda row: '-'.join(sorted([row['Player 1'].lower(), row['Player 2'].lower()])), axis=1
)

# Keep only the row with the highest edge_on per matchup
filtered_df = filtered_df.sort_values(by='edge_on', ascending=False)
filtered_df = filtered_df.drop_duplicates(subset='matchup_key', keep='first')

# Drop the helper column
filtered_df = filtered_df.drop(columns='matchup_key')

# Save the filtered DataFrame
sharp_filtered_path = f"{tourney_folder}/sharp_filtered_{round_var}.csv"
# filtered_df.to_csv(sharp_filtered_path, index=False)
print(f"Sharp filtered CSV saved as {sharp_filtered_path}")


import xlsxwriter

# === Preprocessing for both DataFrames ===

# Determine edge_on before dropping original edge columns
combined_df['edge_on'] = combined_df[['edge_p1', 'edge_p2']].max(axis=1).round(1)
filtered_df['edge_on'] = filtered_df[['edge_p1', 'edge_p2']].max(axis=1).round(1)

# Determine bet_on and pred_on before dropping original edge columns
combined_df['bet_on'] = combined_df.apply(lambda row: row['Player 1'] if row['edge_p1'] > row['edge_p2'] else row['Player 2'], axis=1)
filtered_df['bet_on'] = filtered_df.apply(lambda row: row['Player 1'] if row['edge_p1'] > row['edge_p2'] else row['Player 2'], axis=1)

combined_df['pred_on'] = combined_df.apply(lambda row: row['p1_pred'] if row['edge_p1'] > row['edge_p2'] else row['p2_pred'], axis=1)
filtered_df['pred_on'] = filtered_df.apply(lambda row: row['p1_pred'] if row['edge_p1'] > row['edge_p2'] else row['p2_pred'], axis=1)

combined_df = combined_df[~((combined_df['edge_on'] < 5) & (combined_df['pred_on'] < 1))]
filtered_df = filtered_df[~((filtered_df['edge_on'] < 5) & (filtered_df['pred_on'] < 1))]

# Drop unneeded columns
cols_to_drop = ['edge_p1', 'edge_p2', 'p1_implied', 'p2_implied', 'my_odds_p1', 'my_odds_p2', 'Sample_P1', 'Sample_P2']
combined_df = combined_df.drop(columns=[col for col in cols_to_drop if col in combined_df.columns])
filtered_df = filtered_df.drop(columns=[col for col in cols_to_drop if col in filtered_df.columns])

# Round predictions
for df in [combined_df, filtered_df]:
    df['p1_pred'] = df['p1_pred'].round(2)
    df['p2_pred'] = df['p2_pred'].round(2)

# Export with Excel formatting
excel_path = f"{tourney_folder}/matchup_outputs_{round_var}_{timestamp}.xlsx"

# Create Excel writer and apply formatting
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    # Export WITH pred_on so we can use it for formatting
    combined_df.to_excel(writer, sheet_name='Combined', index=False)
    worksheet_comb = writer.sheets['Combined']

    filtered_df.to_excel(writer, sheet_name='Filtered_Sharp', index=False)
    worksheet_filt = writer.sheets['Filtered_Sharp']

    workbook = writer.book
    yellow_format = workbook.add_format({'bg_color': '#FFFF00'})

    # Get pred_on column index for both sheets (1-based index for Excel column letter)
    def get_column_letter(df, col_name):
        col_idx = df.columns.get_loc(col_name) + 1
        return chr(64 + col_idx) if col_idx <= 26 else f"{chr(64 + (col_idx - 1) // 26)}{chr(64 + (col_idx - 1) % 26 + 1)}"

    pred_on_col_comb = get_column_letter(combined_df, 'pred_on')
    pred_on_col_filt = get_column_letter(filtered_df, 'pred_on')

    def add_format(ws, col_letter, n_rows):
        ws.conditional_format(f"A2:Z{n_rows+1}", {
            'type': 'formula',
            'criteria': f'=${col_letter}2>1',
            'format': yellow_format
        })

    add_format(worksheet_comb, pred_on_col_comb, len(combined_df))
    add_format(worksheet_filt, pred_on_col_filt, len(filtered_df))

    # Drop pred_on before saving to avoid including it in file
    combined_df.drop(columns='pred_on', inplace=True)
    filtered_df.drop(columns='pred_on', inplace=True)

print(f"Excel file with formatted sheets saved to: {excel_path}")







# params = {
#     'tour': 'pga',
#     'market': '3_balls',
#     'odds_format': 'american',
#     'file_format': 'json',
#     'key': API_KEY
# }
# response = requests.get(URL, params=params)
# if response.status_code != 200:
#     raise Exception(f"API request failed with status {response.status_code}: {response.text}")
# data_tournament = response.json()

# rows_tournament_3ball = []
# for match in data_tournament['match_list']:
#     p1 = match['p1_player_name'].lower()
#     p2 = match['p2_player_name'].lower()
#     p3 = match['p3_player_name'].lower()
#     for bookmaker, odds in match['odds'].items():
#         if bookmaker == 'datagolf':
#             continue
#         rows_tournament_3ball.append({
#             'Player 1': p1,
#             'Player 2': p2,
#             'Player 3': p3,
#             'Bookmaker': bookmaker,
#             'P1 Odds': odds['p1'],
#             'P2 Odds': odds['p2'],
#             'P3 Odds': odds['p3'],
#             'Datagolf Odds (P1)': match['odds']['datagolf']['p1'],
#             'Datagolf Odds (P2)': match['odds']['datagolf']['p2'],
#             'Datagolf Odds (P3)': match['odds']['datagolf']['p3'],
#             'matchup_tag': '-'.join(sorted([p1, p2, p3]))
#         })

# # Create the DataFrame
# df_3 = pd.DataFrame(rows_tournament_3ball)

# # Drop duplicates based on the players and bookmaker
# df_3 = df_3.drop_duplicates(subset=['Player 1', 'Player 2', 'Player 3', 'Bookmaker'], keep='first')

# # Convert odds columns to numeric
# df_3['P1 Odds'] = pd.to_numeric(df_3['P1 Odds'], errors='coerce')
# df_3['P2 Odds'] = pd.to_numeric(df_3['P2 Odds'], errors='coerce')
# df_3['P3 Odds'] = pd.to_numeric(df_3['P3 Odds'], errors='coerce')

# # Helper function to calculate implied probabilities
# def calculate_implied_probabilities(df):
#     """Convert odds to implied probabilities."""
#     df['P1 Implied Prob'] = df['P1 Odds'].apply(american_to_implied_probability).round(2)
#     df['P2 Implied Prob'] = df['P2 Odds'].apply(american_to_implied_probability).round(2)
#     df['P3 Implied Prob'] = df['P3 Odds'].apply(american_to_implied_probability).round(2)
#     return df

# # Add implied probabilities for bookmaker odds
# df_3 = calculate_implied_probabilities(df_3)

# # Initialize lists for your custom probabilities
# my_probs_p1, my_probs_p2, my_probs_p3 = [], [], []

# # Calculate your custom probabilities
# for _, row in df_3.iterrows():
#     p1 = row['Player 1']
#     p2 = row['Player 2']
#     p3 = row['Player 3']

#     # Filter scores for the three players
#     p1_scores = sim_scores.loc[sim_scores['player_name'] == p1, f"simulated_score_{round_var}"]
#     p2_scores = sim_scores.loc[sim_scores['player_name'] == p2, f"simulated_score_{round_var}"]
#     p3_scores = sim_scores.loc[sim_scores['player_name'] == p3, f"simulated_score_{round_var}"]

#     if p1_scores.empty or p2_scores.empty or p3_scores.empty:
#         my_probs_p1.append(None)
#         my_probs_p2.append(None)
#         my_probs_p3.append(None)
#         continue

#     # Calculate wins, ties, and total outcomes
#     wins_p1 = sum((p1 < p2) & (p1 < p3) for p1, p2, p3 in zip(p1_scores, p2_scores, p3_scores))
#     wins_p2 = sum((p2 < p1) & (p2 < p3) for p1, p2, p3 in zip(p1_scores, p2_scores, p3_scores))
#     wins_p3 = sum((p3 < p1) & (p3 < p2) for p1, p2, p3 in zip(p1_scores, p2_scores, p3_scores))

#     ties_p1 = sum((p1 == p2) & (p1 < p3) for p1, p2, p3 in zip(p1_scores, p2_scores, p3_scores))
#     ties_p2 = sum((p2 == p3) & (p2 < p1) for p1, p2, p3 in zip(p1_scores, p2_scores, p3_scores))
#     ties_p3 = sum((p3 == p1) & (p3 < p2) for p1, p2, p3 in zip(p1_scores, p2_scores, p3_scores))

#     ties_all = sum((p1 == p2) & (p1 == p3) for p1, p2, p3 in zip(p1_scores, p2_scores, p3_scores))

#     # Total outcomes
#     total_outcomes = wins_p1 + wins_p2 + wins_p3 + ties_p1 + ties_p2 + ties_p3 + ties_all

#     # Calculate custom probabilities
#     p1_prob = (wins_p1 + (ties_p1 / 2) + (ties_all / 3)) / total_outcomes
#     p2_prob = (wins_p2 + (ties_p2 / 2) + (ties_all / 3)) / total_outcomes
#     p3_prob = (wins_p3 + (ties_p3 / 2) + (ties_all / 3)) / total_outcomes
#     re_weight=p1_prob+p2_prob+p3_prob
#     my_probs_p1.append(round(p1_prob, 3))
#     my_probs_p2.append(round(p2_prob, 3))
#     my_probs_p3.append(round(p3_prob, 3))

# # Add custom probabilities to the DataFrame
# df_3['My Implied Prob (P1)'] = my_probs_p1
# df_3['My Implied Prob (P2)'] = my_probs_p2
# df_3['My Implied Prob (P3)'] = my_probs_p3
# df_3.drop(columns=['matchup_tag'], inplace=True)

# # Calculate edge columns
# df_3['edge_p1'] = (df_3['My Implied Prob (P1)'] * 100) - df_3['P1 Implied Prob']
# df_3['edge_p2'] = (df_3['My Implied Prob (P2)'] * 100) - df_3['P2 Implied Prob']
# df_3['edge_p3'] = (df_3['My Implied Prob (P3)'] * 100) - df_3['P3 Implied Prob']

# # Filter out rows where all three edges are < 0
# df_3 = df_3[~((df_3['edge_p1'] < 1) & (df_3['edge_p2'] < 1) & (df_3['edge_p3'] < 1))].reset_index(drop=True)

# combined_csv_name_2 = f"{tourney_folder}/3balls_{round_var}_{tourney}_{timestamp}.csv"

# # Save the updated DataFrame to a CSV file
# df_3.to_csv(combined_csv_name_2, index=False)