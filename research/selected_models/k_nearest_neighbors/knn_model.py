import pandas as pd 
import joblib
import sys

# append the path of the parent directory
sys.path.append("./research")
from api_helpers.game_stats_helpers import load_past_n_games
from api_helpers.game_stats_helpers import matchup_past_n_games

class k_nearest_neighbors():
    
    def __init__(self):
        self.knn_model = joblib.load('research/selected_models/k_nearest_neighbors/knn_model.bin')
        self.columns = ["PTS", "FG_PCT", "PLUS_MINUS", "DREB", "OREB", "TOV", "AST"]
        self.all_games_df = pd.read_csv("research/data/all_games.csv")

    def scale_data(self, data):
        scaler = joblib.load('research/selected_models/k_nearest_neighbors/knn_scaler.bin')
        scaled_data = scaler.transform(data)
        return scaled_data
        

    def predict(self, matchup):
        # get avg stats of last 20 games for both teams in matchup
        team_stats = matchup_past_n_games(self.all_games_df, self.columns, matchup, n=20)
        team_stats.fillna(0, inplace=True)
        team_stats = self.scale_data(team_stats)
        knn_model = joblib.load('research/selected_models/k_nearest_neighbors/knn_model.bin')
        prediction = knn_model.predict(team_stats)
        return prediction






