import pandas as pd
import time
from reuse.pso_model_pipeline import run_pso_pipeline
from config import TRACKING_DATA_URL

def process_weeks(start_week, end_week):
    for week_num in range(start_week, end_week + 1):
        all_results = []
        start_week_time = time.time()

        # Load week data
        week_data = pd.read_csv(TRACKING_DATA_URL.format(week=week_num))

        # Get unique game IDs for this week
        unique_games = week_data['gameId'].unique()
        print(f"Processing Week {week_num}: Total Games = {len(unique_games)}")

        for game_id in unique_games:
            try:
                start_game_time = time.time()
                print(f"Processing Week {week_num}, Game ID: {game_id}")
                result = run_pso_pipeline(week_num, game_id)
                all_results.append(result)
                end_game_time = time.time()
                game_duration = end_game_time - start_game_time
                print(f"Game ID {game_id} processed in {game_duration:.2f} seconds.")
            except Exception as e:
                print(f"Error processing game ID {game_id}: {e}")

        # Concatenate all results for the week
        final_result = pd.concat(all_results)
        output_file = f'final_pso_pipeline_results_week_{week_num}.csv'
        final_result.to_csv(output_file)
        end_week_time = time.time()
        week_duration = end_week_time - start_week_time
        print(f"Week {week_num} processed in {week_duration:.2f} seconds.")
        print(f"Results saved to {output_file}")

# Run for weeks 1 to 9
start_week = 2
end_week = 2
process_weeks(start_week, end_week)




