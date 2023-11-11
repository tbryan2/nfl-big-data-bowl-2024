import pandas as pd

class PreprocessTrackingData():
    def __init__(self, tracking, plays, gameId, playId):
        self.gameId = gameId
        self.playId = playId
        self.tracking = tracking
        self.plays = plays
        self.processed_data = self.join_plays()  # Process and store the data when the instance is created
    
    def get_play(self):
        return self.tracking[(self.tracking['gameId'] == self.gameId) & (self.tracking['playId'] == self.playId)]
    
    def join_plays(self):
        play_data = self.get_play()
        return play_data.merge(self.plays, on=['gameId', 'playId'])

    def get_processed_data(self):
        return self.processed_data