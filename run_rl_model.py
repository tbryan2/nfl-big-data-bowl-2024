from src.reinforcement_learning.preprocess_tracking_data import PreprocessTrackingData
from src.reinforcement_learning.football_play import FootballPlay
from config import (
    TRACKING_DATA_URL,
    PLAYS_URL,
    PLAYERS_URL,
    COLORS_URL
)
import pandas as pd

gameId = 2022090800
playId = 56
nflId = 38577

tracking = pd.read_csv(TRACKING_DATA_URL.format(week=1))
plays = pd.read_csv(PLAYS_URL)
players = pd.read_csv(PLAYERS_URL)
colors = pd.read_csv(COLORS_URL)

preprocessor = PreprocessTrackingData(tracking, plays, players, colors, gameId, playId)
play_data = preprocessor.get_processed_data()
max_frames = play_data['frameId'].max()

env = FootballPlay(play_data, nflId, render_mode='human', max_frames=max_frames)

for _ in range(1000):
    observation, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Sample an action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            done = True  # Mark the episode as done

env.close()
