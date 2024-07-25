from abc import ABC
import sys
from nba_api.stats.endpoints import leaguegamefinder
import os
import pandas as pd
from api_helpers.game_stats_helpers import load_past_n_games
from api_helpers.game_stats_helpers import matchup_past_n_games
import joblib

import sklearn.preprocessing as skp
import pickle

columns = ["FG_PCT","FT_PCT", "OREB", "TOV"]
all_games_df = pd.read_csv("models/data/all_games.csv")
last_20_games = load_past_n_games(all_games_df, columns, n=20)

class Last20Model(ABC):
    '''
    model that uses the last 20 games played
    from both teams to predict the outcome of the game
    '''

    def __init__(self, matchup) -> None:
        super().__init__()
        self.matchup = matchup
        self.team_a_id, self.team_b_id, self.match_date = self.get_game_info(matchup=matchup)
        self.columns = ["FG_PCT","FT_PCT", "OREB", "TOV"]
        self.all_games_df = pd.read_csv("models/data/all_games.csv")
        
        self.last_20_stats = self.get_prev_stats()
        self.input_data = self.data_preprocessing(self.last_20_stats)
        pd.DataFrame(self.input_data).to_csv("input.csv")

    def get_game_info(self, matchup):
        team_a_id = int(matchup["TEAM_ID_A"].iloc[0])
        team_b_id = int(matchup["TEAM_ID_B"].iloc[0])
        match_date = matchup["GAME_DATE"]

        return team_a_id, team_b_id, match_date
    
    def get_prev_stats(self):
        """
        Gets the previous games played by a team

        Arguments:
            team_id: integer value that represents the team's id

        Returns:
            prev_games: pd dataframe that represents the previous games played by the team
        """
        last_20_stats = matchup_past_n_games(all_games_df, self.columns, self.matchup, n=20)
        return last_20_stats
    
    def data_preprocessing(self, last_20_stats: pd.DataFrame):
        """
        Preprocesses the data for the model
        """
        #normalize x_data
        scaler = joblib.load("models/scalers/last20_8f_scaler.bin")
        last_20_games_scaled = scaler.transform(last_20_stats)
        return last_20_games_scaled
    
    def make_prediction_gnb(self):
        """
        Predicts the outcome of the game and returns it as a percentage of likelyhood
        """
        #load model
        loaded_model = pickle.load(open('models/saved_models/gnb_8f_20g_model.sav', 'rb'))
        prediction = loaded_model.predict_proba(self.input_data)[:,1]
        return prediction
        



        
    
