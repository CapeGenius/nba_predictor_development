from abc import ABC
from scipy.spatial import distance
import numpy as np
import pandas as pd
from team_stats_helpers import load_dataframe
from game_stats_helpers import home_matchups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class ThreeStepModel(ABC):

    def __init__(self, team_a_id, team_b_id) -> None:
        super().__init__()

        self.nba_dataframe, self.dataframe_2023 = self.data_preparation()
        self.NON_INT_COLUMNS = ["TEAM_ID", "SEASON_ID", "NBA_FINALS_APPEARANCE"]
        self.all_games_df = pd.read_csv("data/all_games.csv")
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

    def data_preparation(self):
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

        vectors = np.array(vectors_frame.drop(self.NON_INT_COLUMNS, axis=1))

        distances = distance.cdist(target, vectors, "cosine")[0]
        # Sort distances (indices of closest points at the beginning)
        closest_indices = np.argsort(distances)

        # take top k closest vectors
        return vectors_frame.iloc[list(closest_indices[:k])]

    def join_teams(self, similar_rows_1, team_b_row):
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
        final = pd.DataFrame()
        final_stats_list = []

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

        final_stats_df, final

    def regressor_preprocessing(self, team_a_id, team_b_id):

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

        input_team_stats, output_matchup_stats = self.get_matchups(joined=joined_teams)

        return input_team_stats, output_matchup_stats

    def regressor_preparation(self, team_a_id, team_b_id):

        X, y = self.regressor_preprocessing(team_a_id=team_a_id, team_b_id=team_b_id)

        X = X[self.input_features]
        y = y[self.input_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=3
        )

        return X_train, X_test, y_train, y_test

    def random_forest(self, team_a_id, team_b_id):

        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.regressor_preparation(team_a_id=team_a_id, team_b_id=team_b_id)
        )

        rfe = RandomForestRegressor(random_state=10, n_estimators=1000)

        rfe.fit(self.X_train, self.y_train)

        return rfe
