import numpy as np
import pandas as pd

def calculate_distance(row, ball_position):
    """
    Calculate the Euclidean distance between two points.

    Args:
        row (pandas.Series): A row of a DataFrame representing a player's position.
        ball_position (dict): A dictionary with keys 'x' and 'y' representing the ball's position.

    Returns:
        float: Euclidean distance.
    """
    return np.sqrt((ball_position['x'] - row['x'])**2 + (ball_position['y'] - row['y'])**2)

def create_feature_space(frame_df):
    """
    Create a feature space for a given frame of play.

    Args:
        frame_df (pandas.DataFrame): DataFrame representing a single frame of play.

    Returns:
        dict: Dictionary of aggregated features.
    """
    gameId = frame_df['gameId'].iloc[0]
    playId = frame_df['playId'].iloc[0]
    frameId = frame_df['frameId'].iloc[0]

    offense = frame_df[frame_df['OFF'] == True].sort_values('distance_to_ball')
    defense = frame_df[frame_df['OFF'] == False].sort_values('distance_to_ball')

    features = {'gameId': gameId, 'playId': playId, 'frameId': frameId}

    for i, player in enumerate(offense.itertuples(), start=1):
        features[f'OFF_PLAYER_{i}_X'] = player.x
        features[f'OFF_PLAYER_{i}_Y'] = player.y
        # features[f'OFF_PLAYER_{i}_A'] = player.a
        # features[f'OFF_PLAYER_{i}_S'] = player.s
        features[f'OFF_PLAYER_{i}_D'] = player.distance_to_ball

    for i, player in enumerate(defense.itertuples(), start=1):
        features[f'DEF_PLAYER_{i}_X'] = player.x
        features[f'DEF_PLAYER_{i}_Y'] = player.y
        # features[f'DEF_PLAYER_{i}_A'] = player.a
        # features[f'DEF_PLAYER_{i}_S'] = player.s
        features[f'DEF_PLAYER_{i}_D'] = player.distance_to_ball

    for feature in ['x', 'y', 'a', 's', 'distance_to_ball']:
        features[f'OFF_{feature}_sum'] = offense[feature].sum()
        features[f'OFF_{feature}_avg'] = offense[feature].mean()
        features[f'DEF_{feature}_std'] = offense[feature].std()
        features[f'OFF_{feature}_min'] = offense[feature].min()
        features[f'OFF_{feature}_max'] = offense[feature].max()
        features[f'DEF_{feature}_range'] = offense[feature].max() - offense[feature].min()

    for feature in ['a', 's', 'distance_to_ball']:
        features[f'DEF_{feature}_sum'] = defense[feature].sum()
        features[f'DEF_{feature}_avg'] = defense[feature].mean()
        features[f'DEF_{feature}_std'] = defense[feature].std()
        features[f'DEF_{feature}_min'] = defense[feature].min()
        features[f'DEF_{feature}_max'] = defense[feature].max()
        features[f'DEF_{feature}_range'] = defense[feature].max() - defense[feature].min()

    return features

def create_dataset(df, game_id):
    """
    Create a dataset for model training.

    Args:
        df (pandas.DataFrame): DataFrame containing the tracking data.
        game_id (int): Game ID to filter the DataFrame.

    Returns:
        tuple: A tuple containing the final DataFrame and the list of feature columns.
    """
    # Assuming 'plays' is a DataFrame available in your scope
    plays_filtered = plays.copy()[['gameId', 'playId', 'possessionTeam', 'defensiveTeam']]
    df = df.merge(plays_filtered, on=['gameId', 'playId'], how='left')
    df['OFF'] = df['possessionTeam'] == df['club']

    df = df[df['gameId'] == game_id]

    all_feature_spaces = []
    for (gameId, playId), group in df.groupby(['gameId', 'playId']):
        ball_df = group[group['displayName'] == 'football']
        for frame in group['frameId'].unique():
            ball_position = ball_df[ball_df['frameId'] == frame].iloc
            frame_df = group[group['frameId'] == frame]
            group.loc[group['frameId'] == frame, 'distance_to_ball'] = frame_df.apply(
                lambda row: calculate_distance(row, ball_position), axis=1)

            feature_space = [create_feature_space(group[group['frameId'] == frame]) for frame in
                             group['frameId'].unique()]
            all_feature_spaces.extend(feature_space)

        feature_space_df = pd.DataFrame(all_feature_spaces)

        tackle_frames = df.loc[
            (df['displayName'] == 'football') & (df['event'] == 'tackle') | (df['event'] == 'first_contact'), ['gameId',
                                                                                                               'playId',
                                                                                                               'frameId']]
        final_df = feature_space_df.merge(tackle_frames, on=['gameId', 'playId', 'frameId'], how='left', indicator=True)
        final_df['contact_or_tackle_happened'] = np.where(final_df['_merge'] == 'both', 1, 0)
        final_df.drop(columns=['_merge'], inplace=True)

        features_list = final_df.columns[3:-1].tolist()  # Adjust the slicing based on your DataFrame

        return final_df, features_list

