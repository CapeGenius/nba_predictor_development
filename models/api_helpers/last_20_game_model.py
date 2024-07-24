from abc import ABC
import sys
from nba_api.stats.endpoints import leaguegamefinder
import os
import pandas as pd
from game_stats_helpers import load_past_n_games
import sklearn.preprocessing as skp
import pickle

class Last20Model(ABC):
    '''
    model that uses the last 20 games played
    from both teams to predict the outcome of the game
    '''

    def __init__(self, matchup) -> None:
        super().__init__()
        self.columns = ["FG_PCT","FT_PCT", "OREB", "TOV"]
        self.all_games_df = pd.read_csv("models/data/all_games.csv")
        self.team_a_id, self.team_b_id, self.game_date = self.get_game_info(matchup=matchup)
        self.all_games_df = pd.read_csv("models/data/all_games.csv")
        
        self.last_20_stats = self.get_prev_stats(team_id_a=self.team_a_id, team_id_b=self.team_b_id, game_date=self.game_date)
        self.input_data = self.data_preprocessing(self.last_20_stats)

    def get_game_info(self, matchup):
        team_a_id = int(matchup["TEAM_ID_A"])
        team_b_id = int(matchup["TEAM_ID_B"])
        game_date = matchup["GAME_DATE_EST"]

        return team_a_id, team_b_id, game_date
    
    def get_prev_stats(self, team_id_a, team_id_b, game_date):
        """
        Gets the previous games played by a team

        Arguments:
            team_id: integer value that represents the team's id
            game_date: date of the game

        Returns:
            prev_games: pd dataframe that represents the previous games played by the team
        """
        last_20_stats = load_past_n_games(self.all_games_df, columns=self.columns, n=20)
        return last_20_stats
    
    def data_preprocessing(self, last_20_stats: pd.DataFrame):
        """
        Preprocesses the data for the model
        """
        #normalize x_data
        scaler = skp.StandardScaler()
        last_20_games_scaled = scaler.fit_transform(last_20_stats)
        return last_20_games_scaled
    
    def make_prediction_gnb(self):
        """
        Predicts the outcome of the game and returns it as a percentage of likelyhood
        """
        #load model
        loaded_model = pickle.load(open('models/saved_models/gnb_8f_20g_model.sav', 'rb'))
        prediction = loaded_model.predict_proba(self.last_20_stats)[:,1]
        return prediction
        
print(Last20Model)




        
    
