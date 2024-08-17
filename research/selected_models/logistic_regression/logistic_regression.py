import numpy as np
import pandas as pd 
import os
import sklearn.preprocessing as skp
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.utils import resample
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import sys
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# append the path of the parent directory
sys.path.append("./research")
from api_helpers.game_stats_helpers import load_past_n_games
from api_helpers.game_stats_helpers import matchup_past_n_games

class logistic_regression():
    
    def __init__(self):
        self.lr_model = joblib.load('research/selected_models/logistic_regression/logistic_model.bin')
        self.columns = ["PTS", "FG_PCT", "PLUS_MINUS", "DREB", "OREB", "TOV", "AST"]
        self.all_games_df = pd.read_csv("research/data/all_games.csv")

    def scale_data(self, data):
        scaler = joblib.load('research/selected_models/logistic_regression/lr_scaler.bin')
        scaled_data = scaler.transform(data)
        return scaled_data
        

    def predict(self, matchup):
        # get avg stats of last 20 games for both teams in matchup
        team_stats = matchup_past_n_games(self.all_games_df, self.columns, matchup, n=20)
        team_stats = self.scale_data(team_stats)
        lr_model = joblib.load('research/selected_models/logistic_regression/logistic_model.bin')
        prediction = lr_model.predict(team_stats)
        return prediction


lr = logistic_regression()
all_games_df = pd.read_csv("research/data/all_games.csv")
example_matchup = all_games_df.iloc[0]
print(lr.predict(example_matchup))






