from abc import ABC
from scipy.spatial import distance
import numpy as np
import pandas as pd
from api_helpers.team_stats_helpers import load_dataframe
from api_helpers.game_stats_helpers import home_matchups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.models import model_from_json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
import joblib
import sys


class ThreeStepPipeline(ABC):
    """
    Hybrid model of a three step pipeline, consisting of one vector
    search algorithm and two models to predict the outcome of games.
    The two models are random forest regressor and neural network
    """

    def __init__(self, matchup) -> None:
        super().__init__()

        self.team_a_id, self.team_b_id = self.get_team_ids(matchup=matchup)
        print("Team_A_ID is", self.team_a_id)

        self.matchup_rfe = Matchup_Regressor(
            team_a_id=self.team_a_id, team_b_id=self.team_b_id
        )
        self.neural_network = DNNClassifier()

    def get_team_ids(self, matchup):
        """
        Gets the NBA team ids

        Arguments:
            matchup: pd dataframe that represents the matchup between
            team a and team b

        Returns:
            team_a_id: integer value that represents team A's id
            team_b_id: integer value that represents team B's id
        """

        print("The matchup is", matchup.to_string())
        print("The matchup is", matchup["TEAM_ID_A"].to_string())

        team_a_id = int(matchup["TEAM_ID_A"])
        team_b_id = int(matchup["TEAM_ID_B"])

        return team_a_id, team_b_id

    def seasonal_team_matchup(self):
        """
        Concatenates the seasonal stats of Team A and Team B
        into one dataframe row

        Arguments:
            None

        Returns:
            seasonal_team_stats: concatenated dataframe row
            of Team A and Team B's seasonal stats
        """
        # print(self.matchup_rfe.team_a_row)
        team_a_row = self.matchup_rfe.team_a_row.drop(
            self.matchup_rfe.NON_INT_COLUMNS, axis=1
        )
        team_b_row = self.matchup_rfe.team_b_row.drop(
            self.matchup_rfe.NON_INT_COLUMNS, axis=1
        )

        team_b_row = team_b_row.add_suffix("_B")

        seasonal_team_stats = pd.concat(
            [team_a_row.reset_index(drop=True), team_b_row.reset_index(drop=True)],
            axis=1,
        )

        seasonal_team_stats = seasonal_team_stats[self.matchup_rfe.input_features]

        return seasonal_team_stats

    def make_prediction(self):
        """
        Integrates the Random Forest Regressor with the Neural Network
        to get a prediction on a specific matchup between two teams

        Arguments:
            None

        Returns:
            game_prediction: array reflecting the outcome of a game
        """
        scaler = joblib.load("models/scaler.bin")

        # declare variables
        rfe = self.matchup_rfe.rfe
        dnn = self.neural_network.network

        # get team stats
        seasonal_team_stats = np.asarray(self.seasonal_team_matchup())

        # print("The length is", seasonal_team_stats)

        # predict outcome of game
        game_stat = rfe.predict(seasonal_team_stats)

        game_stat = scaler.transform(game_stat)
        print(game_stat)

        game_prediction = dnn.predict(game_stat)

        print("\nThe outcome of this game is", game_prediction)

        return game_prediction


class DNNClassifier(ABC):
    """
    Deep Neural Network classifier to predict the outcome
    of games based on matchup statistics
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = self.initialize_network()

    def initialize_network(self):
        json_file = open("models/tuned_nn.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("models/tuned.weights.h5")
        print("Loaded model from disk")

        return loaded_model


class Matchup_Regressor(ABC):
    """
    Class to represent random forest regression that predicts
    the outcome of a game between two teams based on both team's
    seasonal statistics
    """

    def __init__(self, team_a_id, team_b_id) -> None:
        super().__init__()

        self.team_a_id = team_a_id
        self.team_b_id = team_b_id
        self.nba_dataframe, self.dataframe_2023 = self.data_preparation()
        self.NON_INT_COLUMNS = ["TEAM_ID", "SEASON_ID", "NBA_FINALS_APPEARANCE"]
        self.all_games_df = pd.read_csv("models/data/all_games.csv")
        self.rfe = self.random_forest(team_a_id=team_a_id, team_b_id=team_b_id)

    @property
    def input_features(self):
        input_a_features = ["FG_PCT", "FG3A", "FTM", "OREB", "DREB", "REB", "AST"]
        input_b_features = [str(word) + "_B" for word in input_a_features]

        input_features = input_a_features + input_b_features

        return input_features

    @property
    def output_features(self):
        output_a_features = [
            "FG_PCT_A",
            "FG3_PCT_A",
            "FTM_A",
            "OREB_A",
            "DREB_A",
            "REB_A",
            "AST_A",
        ]
        output_b_features = [word.replace("_A", "_B") for word in output_a_features]
        output_features = output_a_features + output_b_features

        return output_features

    @property
    def team_a_row(self):
        team_a_row = self.dataframe_2023[
            self.dataframe_2023["TEAM_ID"] == int(self.team_a_id)
        ]

        return team_a_row

    @property
    def team_b_row(self):
        team_b_row = self.dataframe_2023[
            self.dataframe_2023["TEAM_ID"] == self.team_b_id
        ]

        return team_b_row

    def data_preparation(self):
        """
        Cleans and processes two dataframes of NBA seasonal statistics. NBA_dataframe
        represents all seasons except the current season and the dataframe_2023 represents
        the current season. Removes data that has NAN as values

        Arguments:
            None

        Returns:
            nba_dataframe: dataframe that represents all team seasonal statistics except
                the current season
            dataframe_2023: dataframe that represents team seasonal statistics for current
                season

        """
        sys.path.append("./models")
        nba_dataframe = load_dataframe(
            ["FGM", "FGA", "FG_PCT", "FG3A", "FTM", "OREB", "DREB", "REB", "AST", "PTS"]
        )
        nba_dataframe = nba_dataframe.drop(
            nba_dataframe[nba_dataframe["FGA"] == 0].index
        )

        nba_dataframe["YEAR"] = "2" + nba_dataframe["YEAR"].str.slice(0, 4)
        pd.DataFrame.rename(nba_dataframe, columns={"YEAR": "SEASON_ID"}, inplace=True)

        nba_dataframe["NBA_FINALS_APPEARANCE"].fillna(0.0, inplace=True)
        nba_dataframe["NBA_FINALS_APPEARANCE"].replace(
            "FINALS APPEARANCE", 0, inplace=True
        )
        nba_dataframe["NBA_FINALS_APPEARANCE"].replace(
            "LEAGUE CHAMPION", 1, inplace=True
        )

        dataframe_2023 = nba_dataframe[nba_dataframe["SEASON_ID"] == "22023"]
        nba_dataframe = nba_dataframe[nba_dataframe["SEASON_ID"] != "22023"]
        nba_dataframe = nba_dataframe.reset_index(drop=True)

        return nba_dataframe, dataframe_2023

    def closest_teams(self, target, vectors_frame, k=1):
        """
        Utilizes cdist to find the closest performing teams for a given
        k value any given target team, comparing the target team seasonal
        statistics to the seasonal statistics of all nba teams.

        Arguments:
            target: a dataframe representing the seasonal statline of a
                desired nba team
            vectors_frame = a dataframe representing specified NBA teams
                to compare to
            k: integer value representing the number of k teams to compare
                data to

        Returns:
            dataframe of all the closest teams in the history of the NBA
            to the target NBA team.

        """

        vectors = np.array(vectors_frame.drop(self.NON_INT_COLUMNS, axis=1))

        distances = distance.cdist(target, vectors, "cosine")[0]
        # Sort distances (indices of closest points at the beginning)
        closest_indices = np.argsort(distances)

        # take top k closest vectors
        return vectors_frame.iloc[list(closest_indices[:k])]

    def join_teams(self, similar_rows_1, team_b_row):
        """
        Creates a dataframe that finds matchups of k teams, similar
        to team A competing against k teams, similar to team b.

        Using a dataframe of teams similar to team A, this algorithm
        then finds the corresponding teams similar to team B from
        the same season as team A. Then, it creates a concatenated
        dataframe that creates the input training set for the random forest
        regressor.

        Arguments:
            similar_rows_1: dataframe representing teams that
            are similar to team A
            team: dataframe row representing teams B

        Returns:
            final_joined: a concatenated dataframe that creates the
            training set for the random forest regressor.
        """
        joined_list = []
        for _, row in similar_rows_1.iterrows():
            k = 5
            year = row["SEASON_ID"]

            # get
            year_frame = self.nba_dataframe[self.nba_dataframe["SEASON_ID"] == year]

            similar_rows_2 = self.closest_teams(
                team_b_row.drop(self.NON_INT_COLUMNS, axis=1),
                year_frame,
                k=k,
            )
            similar_rows_2 = similar_rows_2.add_suffix("_B")

            combined_row = pd.concat([row.to_frame().T] * len(similar_rows_2), axis=0)

            # Now merge the repeated_df1 with df2
            result = pd.concat(
                [
                    combined_row.reset_index(drop=True),
                    similar_rows_2.reset_index(drop=True),
                ],
                axis=1,
            )

            joined_list.append(result)

        final_joined = pd.concat(joined_list, ignore_index=True)

        return final_joined

    def get_matchups(self, joined):
        """
        Using the concatenated table of similar teams, get matchups()
        finds all the matchups between those teams in a dataframe and
        returns them to be used a output training set

        Arguments:
            joined: concatenated dataframe representing a table of
            seasonal "matchup" statistics between similar teams
        Returns:
            final_stats_df: concatenated dataframe representing a table of
                seasonal "matchup" statistics between similar teams
                (input dataset)
            final: dataframe of all the matchups between those teams
            in a dataframe and returns them to be used a output training set
        """
        final = pd.DataFrame()
        final_stats_list = []

        # print(type(joined))

        for i in range(len(joined)):
            row = joined.iloc[i]

            matchups = home_matchups(
                all_games_df=self.all_games_df,
                team_a=int(row["TEAM_ID"]),
                team_b=int(row["TEAM_ID_B"]),
                year=int(row["SEASON_ID"]),
            )

            # output data
            final = pd.concat([final, matchups], axis=0)

            if len(matchups) == 0:
                continue
            else:
                repeated_row = pd.concat([row.to_frame().T] * len(matchups), axis=0)
                final_stats_list.append(repeated_row)

        # dataframe of team stats (input data)
        final_stats_df = pd.concat(final_stats_list, axis=0)

        return final_stats_df, final

    def regressor_preprocessing(self, team_a_id, team_b_id):
        """
        Preprocessing step to get all input and output data
        necessary for training the model

        Arguments:
            team_a_id: int representing the team A id
            team_b_id: int representing the team B id

        Returns:
            input_team_stats: dataframe representing the input
                dataframe (representing all seasonal team matchups)
            output_matchup_stats: dataframe representing the
                output dataframe (representing all individual game matchups)
        """

        team_a_row = self.dataframe_2023[self.dataframe_2023["TEAM_ID"] == team_a_id]
        team_b_row = self.dataframe_2023[self.dataframe_2023["TEAM_ID"] == team_b_id]

        similar_rows_1 = self.closest_teams(
            team_a_row.drop(self.NON_INT_COLUMNS, axis=1),
            self.nba_dataframe,
            k=10,
        )

        joined_teams = self.join_teams(
            similar_rows_1=similar_rows_1, team_b_row=team_b_row
        )

        # print(joined_teams)

        input_team_stats, output_matchup_stats = self.get_matchups(joined=joined_teams)

        # print(input_team_stats)
        # print(output_matchup_stats)

        return input_team_stats, output_matchup_stats

    def regressor_preparation(self, team_a_id, team_b_id):
        """
        Creates a test train split for the regressor, after receiving
        the input and output data. (Code can be modified to return test
        train split data)

        Arguments:
            team_a_id: int representing the team A id
            team_b_id: int representing the team B id

        Returns:
            X: datframe reprsenting all input data
            y: dataframe representing all output data
        """

        X, y = self.regressor_preprocessing(team_a_id=team_a_id, team_b_id=team_b_id)

        X = X[self.input_features]
        y = y[self.output_features]

        # print(X)
        # print(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=3
        )

        return X, y

    def random_forest(self, team_a_id, team_b_id):
        """
        Generates random forest using team A id and team
        B id to train the random forest on team seasonal
        statistics and the predicted outcome of the matchpu

        Arguments:
            team_a_id: int representing the team A id
            team_b_id: int representing the team B id

        Returns:
            rfe: fitted Random Forest Regressor object for a
            given matchup between two teams
        """
        self.X, self.y = self.regressor_preparation(
            team_a_id=team_a_id, team_b_id=team_b_id
        )

        rfe = RandomForestRegressor(random_state=10, n_estimators=1000)

        rfe.fit(self.X, self.y)

        return rfe

    def evaluate_random_forest(self):
        """
        Evaluates the random forest
        """
        # define model
        model = RandomForestRegressor()
        # define the evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
        # force the scores to be positive
        n_scores = absolute(n_scores)
        # summarize performance
        print("MAE: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))
