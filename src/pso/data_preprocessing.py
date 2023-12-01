import pandas as pd
import os
from config import TRACKING_DATA_URL, PLAYS_URL, PLAYERS_URL

def get_preprocessed_tracking_data(
    week_num: int = 1,
    game_id: int = 2022091100,
    play_id: int = 145) -> dict:

    # check if .cache dir exists in the root
    if not os.path.exists('.cache'):
        os.mkdir('.cache')
    
    # check if the file exists in the .cache dir
    file_name = f'{week_num}_{game_id}_{play_id}.csv'
    file_path = os.path.join('.cache', file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return {
            'week_num': week_num,
            'game_id': game_id,
            'play_id': play_id,
            'ball_carrier_id': df['ball_carrier_id'].iloc[0],
            'off_abbr': df['off_abbr'].iloc[0],
            'def_abbr': df['def_abbr'].iloc[0],
            'df': df
        }
    
    # if the file does not exist, get src, preprocess, and save it
    tracking_data = pd.read_csv(TRACKING_DATA_URL.format(week=week_num))
    plays = pd.read_csv(PLAYS_URL)
    players = pd.read_csv(PLAYERS_URL)

    tracking_data = tracking_data.loc[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id)]
    tracking_data['time'] = pd.to_datetime(tracking_data['time'])
    # shift, x, y, and time
    tracking_data[['x_shifted', 'y_shifted', 'time_shifted']] = tracking_data.groupby(['nflId'], as_index=False)[['x', 'y', 'time']].shift(1)
    # calc delta columns
    tracking_data['dx'] = tracking_data['x'] - tracking_data['x_shifted']
    tracking_data['dy'] = tracking_data['y'] - tracking_data['y_shifted']
    tracking_data['dt'] = (tracking_data['time'] - tracking_data['time_shifted']).dt.microseconds / 1_000_000
    # calc velocity
    tracking_data['x_velocity'] = tracking_data['dx'] / tracking_data['dt']
    tracking_data['y_velocity'] = tracking_data['dy'] / tracking_data['dt']
    tracking_data['xy_velocity'] = tracking_data['dis'] / tracking_data['dt']

    # get the ball carrier from play dataframe
    ball_carrier_id = plays.loc[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['ballCarrierId'].iloc[0]
    off_abbr = plays.loc[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].iloc[0]
    def_abbr = plays.loc[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].iloc[0]

    tracking_data['ball_carrier_id'] = ball_carrier_id
    tracking_data['off_abbr'] = off_abbr
    tracking_data['def_abbr'] = def_abbr

    tracking_data = tracking_data.merge(players[['nflId', 'position']], on='nflId', how='left')

    # save the file
    tracking_data.to_csv(file_path, index=False)

    return {
        'week_num': week_num,
        'game_id': game_id,
        'play_id': play_id,
        'ball_carrier_id': ball_carrier_id,
        'off_abbr': off_abbr,
        'def_abbr': def_abbr,
        'df': tracking_data
    }
    