import pandas as pd
from scipy.stats import norm
from bs4 import BeautifulSoup
import re
from fuzzywuzzy import process 
from sim_inputs import tourney, course_id
from scores_xtract_html import (
    extract_info_jazz,
    extract_info_fd,
    extract_info_dk,
    extract_info_bov,
    extract_info_pph,
    american_to_implied_probability,
    extract_info_pp,
    extract_info_drafters,
    extract_info_pick6,
    extract_info_ud,
    extract_info_proph,
    extract_info_buck
)

import pandas as pd
import numpy as np

# === Inputs ===
round_var = 'r2'
PAR = 68.31
NUM_SIMULATIONS = 10000
STD_DEV = 2.7
model_preds = pd.read_csv(f"model_predictions_{round_var}.csv")

# Step 1: Add a 'pred' column to model_preds based on round_var
pred_col = f"my_pred{round_var[-1]}" if round_var != 'r1' else "my_pred"
model_preds['pred'] = model_preds[pred_col]

# Simulate scores
simulated_scores = []
for _, row in model_preds.iterrows():
    pred = row[f"scores_{round_var}"]
    player = row["player_name"]
    teetime = row.get(f"{round_var}_teetime")  # e.g., r1_teetime

    sim_scores = np.round(PAR - np.random.normal(loc=pred, scale=STD_DEV, size=NUM_SIMULATIONS)).astype(int)
    simulated_scores.append(pd.DataFrame({
        'player_name': player,
        f'simulated_score_{round_var}': sim_scores,
        f'{round_var}_teetime': teetime  # repeated for all rows
    }))

sim_df = pd.concat(simulated_scores, ignore_index=True)
# Step 2: Merge 'pred' column into sim_df
sim_df = sim_df.merge(model_preds[['player_name', 'pred']], on='player_name', how='left')

round_var=2
tee_time_column=f"r{round_var}_teetime"

pd.set_option('display.max_columns', None)

score_info_df = pd.read_csv(f"final_predictions_{tourney}.csv")
r1_teetimes = score_info_df[['player_name', 'r1_teetime']]

golfer_names = score_info_df['player_name'].str.strip().str.lower().tolist() 
formatted_golfer_names = [' '.join(name.split(', ')[::-1]).lower() for name in golfer_names]
formatted_golfer_names_up = [' '.join(name.split(', ')[::-1]).upper() for name in golfer_names]

def format_player_name(name):
    parts = name.split()
    if len(parts) > 1:  # Ensure there are at least two parts in the name
        last_name = parts[-1]
        first_name = " ".join(parts[:-1])
        return f"{last_name}, {first_name}"
    return name 

def process_file(file_path, golfer_names=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')

    # Dispatch to the correct extractor
    if 'pph' in file_path:
        df = pd.DataFrame(extract_info_pph(soup))
    elif 'buckeye' in file_path:
        df = pd.DataFrame(extract_info_buck(soup))
    elif 'jazz' in file_path:
        results = extract_info_jazz(soup, golfer_names)
        df = pd.DataFrame(results)
        if not df.empty:
            df['Player'] = df['Player'].str.lower()
            df = df[df['Player'].isin(formatted_golfer_names)]
    elif 'fd_html' in file_path:
        if not golfer_names:
            raise ValueError("golfer_names parameter is required for fd_html extraction")
        df = pd.DataFrame(extract_info_fd(soup, golfer_names, round_var))
    elif 'dk_sb_html' in file_path:
        df = pd.DataFrame(extract_info_dk(soup))
    elif 'bov' in file_path:
        df = pd.DataFrame(extract_info_bov(soup))
        # Remove the "Round" column if it exists
        if 'Round' in df.columns:
            df.drop(columns=['Round'], inplace=True)
    elif 'pp_html' in file_path:  # Add PP integration
        df = pd.DataFrame(extract_info_pp(soup, golfer_names))
    elif 'drafters' in file_path:  # Add Drafters integration
        df = pd.DataFrame(extract_info_drafters(soup, golfer_names))
    elif 'pick_6' in file_path:  # Add Drafters integration
        df = pd.DataFrame(extract_info_pick6(soup, golfer_names))
    elif 'underdog' in file_path:  # Add Drafters integration
        df = pd.DataFrame(extract_info_ud(soup, golfer_names))
    elif 'proph' in file_path:  # Add Drafters integration
        df = pd.DataFrame(extract_info_proph(soup, golfer_names))
    else:
        raise ValueError(f"Unknown file type for {file_path}")

    # Convert player names to lowercase if the DataFrame is not empty
    if not df.empty:
        df['Player'] = df['Player'].str.lower()

    return df

# List of files to process
files = [
    ("bov.txt", None),
    ("fd_html.txt", formatted_golfer_names),
    ("dk_sb_html.txt", None),
    ("pph.txt", None),
    ("pp_html.txt", formatted_golfer_names),
    ("jazz.txt", None),
    ("drafters.txt", formatted_golfer_names),
    ("pick_6.txt", formatted_golfer_names),
    ("underdog.txt", formatted_golfer_names),
    ("proph.txt", formatted_golfer_names),
    ("buckeye.txt", None),
]

# Process each file and save results
for file_path, names in files:
    try:
        df = process_file(file_path, names)
        if not df.empty:
            output_file = file_path.split('.')[0] + "_scraped_data.csv"
            df.to_csv(output_file, index=False)
            print(f"Processed {file_path}:\n", df.head())
        else:
            print(f"No data extracted from {file_path}.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

betting_files = [
    "bov_scraped_data.csv",
    "fd_html_scraped_data.csv",
    "dk_sb_html_scraped_data.csv",
    "pph_scraped_data.csv",
    "pp_html_scraped_data.csv", 
    "jazz_scraped_data.csv",
    "drafters_scraped_data.csv",
    "pick_6_scraped_data.csv",
    "underdog_scraped_data.csv",
    "proph_scraped_data.csv",
    "buckeye_scraped_data.csv"
]

def implied_probability_to_american(prob):
    """
    Convert an implied probability (in percentage) to American odds.
    """
    if prob >= 50:
        return round(-100 * (prob / (100 - prob)))
    else:
        return round((100 - prob) / prob * 100)

def calculate_score_pct(betting_df, leaderboard_df, round_var):
    """
    Calculate the percentage of scores above and below the Total_Strokes line for each player,
    and compute additional metrics such as Kelly values and eg_sort.
    """
    # Column for the current round's scores
    score_column = "simulated_score_r1" if round_var == 1 else f"simulated_score_r{round_var}"
    betting_df['Total_Strokes'] = pd.to_numeric(betting_df['Total_Strokes'], errors='coerce')
    # Ensure scores are numeric
    leaderboard_df[score_column] = pd.to_numeric(leaderboard_df[score_column], errors='coerce')

    # Add avg_score and tee_time columns before existing columns
    avg_scores = []
    tee_times = []

    for index, row in betting_df.iterrows():
        player_name = row['Player']

        # Filter leaderboard for specific player
        player_data = leaderboard_df.loc[
            leaderboard_df['player_name'].str.strip().str.lower() == player_name.strip().lower()
        ]

        # Calculate avg_score
        avg_score = player_data[score_column].mean() if not player_data.empty else float('nan')
        avg_scores.append(avg_score)

        # Extract tee_time
        # Extract tee_time
        tee_time = player_data[tee_time_column].iloc[0] if tee_time_column in player_data.columns and not player_data.empty else None

        if tee_time:
            tee_time = pd.to_datetime(tee_time).strftime('%H:%M')  # Extract hour and minute
        tee_times.append(tee_time)

    # Insert the new columns before existing columns
    betting_df.insert(1, 'Avg_Score', avg_scores)
    betting_df.insert(2, 'Tee_Time', tee_times)
    # Add pred values to betting_df
    pred_map = dict(zip(leaderboard_df['player_name'].str.lower(), leaderboard_df['pred']))
    betting_df['Pred'] = betting_df['Player'].str.lower().map(pred_map)

    # Initialize columns to store the counts and additional metrics
    betting_df['Over_pct'] = float('nan')
    betting_df['Under_pct'] = float('nan')
    betting_df['over_fair'] = float('nan')
    betting_df['under_fair'] = float('nan')
    betting_df['Implied_o'] = float('nan')
    betting_df['Implied_u'] = float('nan')
    betting_df['Edge_o'] = float('nan')
    betting_df['Edge_u'] = float('nan')
    betting_df['Kelly_o'] = float('nan')
    betting_df['Kelly_u'] = float('nan')
    betting_df['Eg_sort'] = 0  # Default value for eg_sort

    # Iterate through each player in the betting DataFrame
    for index, row in betting_df.iterrows():
        try:
            player_name = row['Player']
            total_strokes = row['Total_Strokes']

            # Filter the leaderboard dataframe for the specific player
            player_scores = leaderboard_df.loc[
                leaderboard_df['player_name'].str.strip().str.lower() == player_name.strip().lower(),
                score_column
            ]

            # Count scores below and above the total strokes line
            scores_below = (player_scores < total_strokes).sum()
            scores_above = (player_scores > total_strokes).sum()
            total_scores = scores_below + scores_above

            # Calculate percentages
            over_pct = round((scores_above / total_scores) * 100, 1) if total_scores > 0 else 0
            under_pct = round((scores_below / total_scores) * 100, 1) if total_scores > 0 else 0

            # Update the betting dataframe
            betting_df.at[index, 'Over_pct'] = over_pct
            betting_df.at[index, 'Under_pct'] = under_pct

            over_fair = implied_probability_to_american(over_pct) if over_pct > 0 else None
            under_fair = implied_probability_to_american(under_pct) if under_pct > 0 else None
            betting_df.at[index, 'Over_fair'] = over_fair
            betting_df.at[index, 'Under_fair'] = under_fair

            # Calculate implied probabilities
            over_odds = row['Over_Odds']
            under_odds = row['Under_Odds']
            implied_o = american_to_implied_probability(over_odds) if pd.notna(over_odds) else None
            implied_u = american_to_implied_probability(under_odds) if pd.notna(under_odds) else None
            betting_df.at[index, 'Implied_o'] = implied_o
            betting_df.at[index, 'Implied_u'] = implied_u

            edge_o = round(over_pct - implied_o, 1) if implied_o is not None else None
            edge_u = round(under_pct - implied_u, 1) if implied_u is not None else None
            betting_df.at[index, 'Edge_o'] = edge_o
            betting_df.at[index, 'Edge_u'] = edge_u

            kelly_o = round(edge_o / over_pct, 3) if edge_o not in [None, 0] and implied_o > 0 and over_pct > 0 else float('nan')
            kelly_u = round(edge_u / under_pct, 3) if edge_u not in [None, 0] and implied_u > 0 and under_pct > 0 else float('nan')

            betting_df.at[index, 'Kelly_o'] = kelly_o if kelly_o > 0 else float('nan')
            betting_df.at[index, 'Kelly_u'] = kelly_u if kelly_u > 0 else float('nan')

            # Compute eg_sort
            positive_kelly = max(kelly_o if kelly_o > 0 else 0, kelly_u if kelly_u > 0 else 0)
            if positive_kelly > 0:
                edge_value = max(edge_o if edge_o > 0 else 0, edge_u if edge_u > 0 else 0)
                betting_df.at[index, 'Eg_sort'] = round(positive_kelly * (edge_value / 2), 5) if edge_value > 0 else 0
            else:
                betting_df.at[index, 'Eg_sort'] = 0

        except Exception as e:
            print(f"Error processing player: {row['Player']} in DataFrame: {betting_df}")
            print(f"Error details: {e}")

    return betting_df



# Process each betting data file
# Process each betting data file
for file_path in betting_files:
    try:
        # Load the betting DataFrame
        betting_df = pd.read_csv(file_path)

        # Normalize player names in the betting DataFrame
        betting_df['Player'] = betting_df['Player'].str.strip().str.lower()
        betting_df['Player'] = betting_df['Player'].apply(format_player_name)

        # Calculate scores above and below
        betting_df = calculate_score_pct(betting_df, sim_df, round_var)

        # === Final formatting ===

        # Move 'Pred' after 'Avg_Score'
        cols = betting_df.columns.tolist()
        if 'Pred' in cols and 'Avg_Score' in cols:
            cols.insert(cols.index('Avg_Score') + 1, cols.pop(cols.index('Pred')))
            betting_df = betting_df[cols]

        # Round Pred
        if 'Pred' in betting_df.columns:
            betting_df['Pred'] = betting_df['Pred'].round(2)
        def determine_bet(row):
            if row.get('Edge_o', -999) > 0:
                return 'Over'
            elif row.get('Edge_u', -999) > 0:
                return 'Under'
            else:
                return ''
        
        betting_df['Bet'] = betting_df.apply(determine_bet, axis=1)
        # Drop unnecessary columns
        cols_to_drop = ['Over_pct', 'Under_pct', 'over_fair', 'under_fair', 'Implied_o', 'Implied_u']
        betting_df.drop(columns=[col for col in cols_to_drop if col in betting_df.columns], inplace=True)

        # Drop rows where both edges are negative
        betting_df = betting_df[~((betting_df['Edge_o'] < 2) & (betting_df['Edge_u'] < 2))]

        # Sort by Eg_sort descending, then Pred < 1 to bottom
        betting_df['Pred_lt_1'] = betting_df['Pred'] < 1
        betting_df = betting_df.sort_values(by=['Eg_sort', 'Pred_lt_1'], ascending=[False, True])
        betting_df.drop(columns='Pred_lt_1', inplace=True)
        cols_to_drop = ['Edge_o', 'Edge_u', 'Kelly_o', 'Kelly_u']
        betting_df.drop(columns=[col for col in cols_to_drop if col in betting_df.columns], inplace=True)
        rename_map = {
            'Tee_Time': 'tt',
            'Total_Strokes': 'Line',
            'Over_Odds': 'Over',
            'Under_Odds': 'Under',
            'Over_fair': 'O_Fair',
            'Under_fair': 'U_Fair'
        }
        betting_df.rename(columns=rename_map, inplace=True)

        # Save the updated DataFrame
        output_file = file_path.replace("_scraped_data.csv", "_with_counts.csv")
        betting_df.to_csv(output_file, index=False)
        print(f"Processed and saved: {output_file}\n", betting_df.head())

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

betting_files = [
    "bov_with_counts.csv",
    "fd_html_with_counts.csv",
    "dk_sb_html_with_counts.csv",
    "pph_with_counts.csv",
    "pp_html_with_counts.csv",
    "jazz_with_counts.csv",
    "drafters_with_counts.csv",
    "pick_6_with_counts.csv",
    "underdog_with_counts.csv",
    "proph_with_counts.csv",
    "buckeye_with_counts.csv"
]

# Name of the output Excel file
output_excel_file = "round_scores.xlsx"

combined_dfs = []

# Create a Pandas Excel writer
with pd.ExcelWriter(output_excel_file, engine="xlsxwriter") as writer:
    for file_path in betting_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Get the worksheet name from the file name (without extension)
            worksheet_name = file_path.split("_")[0]

            # Add a "book" column to the DataFrame
            df["book"] = worksheet_name

            # Append the DataFrame to the list for combining
            combined_dfs.append(df)

            # Write the DataFrame to a worksheet
            df.to_excel(writer, sheet_name=worksheet_name, index=False)

            print(f"Added {worksheet_name} to {output_excel_file}.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Combine all the DataFrames
    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Write the combined DataFrame to a new worksheet
    combined_df.to_excel(writer, sheet_name="combined", index=False)

    print("Added combined sheet to the Excel file.")

print(f"Saved all worksheets to {output_excel_file}.")

def calculate_scoring_by_time_of_day(leaderboard_df, tee_time_column):
    """
    Calculate scoring averages by quartiles and pre/post 10 AM tee times.
    
    Parameters:
    - leaderboard_df: DataFrame containing player scores and tee times.
    - tee_time_column: Column name for the tee times.
    
    Returns:
    - scoring_summary: DataFrame containing scoring averages by time group.
    """
    import pandas as pd
    leaderboard_df[f"r{round_var}_teetime"] = leaderboard_df[f"r{round_var}_teetime"].str.strip()
    # Convert tee times to datetime for easier manipulation
    leaderboard_df[f"r{round_var}_teetime"] = pd.to_datetime(leaderboard_df[f"r{round_var}_teetime"], errors='coerce')

    # Add a column to classify before and after 10 AM
    leaderboard_df['time_group'] = leaderboard_df[tee_time_column].apply(
        lambda x: 'Before 10 AM' if x.hour < 10 else 'After 10 AM'
    )

    # Divide tee times into quartiles
    leaderboard_df['quartile'] = pd.qcut(
        leaderboard_df[tee_time_column].dt.hour * 60 + leaderboard_df[tee_time_column].dt.minute,
        q=4,
        labels=['Q1 (Early)', 'Q2', 'Q3', 'Q4 (Late)']
    )

    # Group by quartiles and time group, then calculate scoring averages
    quartile_avg = leaderboard_df.groupby('quartile')[f"simulated_score_r{round_var}"].mean().reset_index()
    quartile_avg.rename(columns={'simulated_score': 'Avg_Score'}, inplace=True)

    time_group_avg = leaderboard_df.groupby('time_group')[f"simulated_score_r{round_var}"].mean().reset_index()
    time_group_avg.rename(columns={'simulated_score': 'Avg_Score'}, inplace=True)

    # Combine results into a summary DataFrame
    scoring_summary = pd.concat([
        quartile_avg.assign(Group='Quartile'),
        time_group_avg.assign(Group='Time Group')
    ], ignore_index=True)

    return scoring_summary

# Example Usage
# Assuming leaderboard_df has columns 'tee_time_r1' and 'simulated_score'
scoring_summary = calculate_scoring_by_time_of_day(sim_df, tee_time_column)
print(scoring_summary)


import os

# List of intermediate CSV files to delete
intermediate_csv_files = [
    "bov_scraped_data.csv",
    "fd_html_scraped_data.csv",
    "dk_sb_html_scraped_data.csv",
    "pph_scraped_data.csv",
    "pp_html_scraped_data.csv",
    "jazz_scraped_data.csv",
    "drafters_scraped_data.csv",
    "bov_with_counts.csv",
    "fd_html_with_counts.csv",
    "dk_sb_html_with_counts.csv",
    "pph_with_counts.csv",
    "pp_html_with_counts.csv",
    "jazz_with_counts.csv",
    "drafters_with_counts.csv",
    "pick_6_with_counts.csv",
    "pick_6_scraped_data.csv",
    "underdog_with_counts.csv",
    "underdog_scraped_data.csv",
    "buckeye_with_counts.csv",
    "buckeye_scraped_data.csv",

]

# Delete each file in the list if it exists
for file_path in intermediate_csv_files:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted intermediate file: {file_path}")
        else:
            print(f"File not found, skipping: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

print("Cleanup complete. All intermediate CSV files have been removed.")
