import pandas as pd
from pso_model_pipeline import run_pso_pipeline
from config import TRACKING_DATA_URL

all_results = []

# Loop over all weeks
for week in range(1, 9 + 1):
    week_data = pd.read_csv(TRACKING_DATA_URL.format(week=week))

    # Get unique game and play IDs for this week
    unique_games = week_data['gameId'].unique()

    for game_id in unique_games:
        try:
            result = run_pso_pipeline(week, game_id)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing week {week}, game {game_id}: {e}")

# Concatenate all results
final_result = pd.concat(all_results)
final_result.to_csv('final_pso_pipeline_results.csv')
