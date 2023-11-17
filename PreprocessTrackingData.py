import pandas as pd

class PreprocessTrackingData():
    '''
    Filter the tracking data down to one play, join with the play data,
    and then join with player data on nflId and displayName, and colors data on club/team_abbr.
    '''
    
    def __init__(self, tracking, plays, players, colors, gameId, playId):
         # Process and store the data when the instance is created
        self.gameId = gameId
        self.playId = playId
        self.tracking = tracking
        self.plays = plays
        self.players = players
        self.colors = colors
        self.processed_data = self.join_colors() 

    def get_play(self):
        '''
        Filter the tracking data for the given gameId and playId
        '''

        return self.tracking[(self.tracking['gameId'] == self.gameId) \
                             & (self.tracking['playId'] == self.playId)]

    def join_plays(self):
        '''
        Join the filtered tracking data with the play data
        '''

        play_data = self.get_play()
        return play_data.merge(self.plays, on=['gameId', 'playId'])

    def join_plays_with_players(self):
        '''
        Join the play data with the player data on nflId and displayName
        '''

        play_data = self.join_plays()
        return play_data.merge(self.players, left_on=['nflId', 'displayName'], \
                               right_on=['nflId', 'displayName'], how='left')

    def join_colors(self):
        '''
        Join the processed data with the colors data on club/team_abbr
        '''

        play_data_with_players = self.join_plays_with_players()
        return play_data_with_players.merge\
            (self.colors, left_on='club', right_on='team_abbr', how='left')

    def get_processed_data(self):
        '''
        Return the fully processed data
        '''

        return self.processed_data
