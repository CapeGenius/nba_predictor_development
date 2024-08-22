from datetime import datetime
from datetime import timedelta
import numpy as np

from generate_schedule import NBADaySchedule
import pandas as pd

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

# append the path of the parent directory
sys.path.append("./research")

from selected_models.k_nearest_neighbors.knn_model import k_nearest_neighbors
from selected_models.logistic_regression.logistic_regression import logistic_regression


import streamlit as st

st.title("Prototype App")

if "all_games" not in st.session_state:
    all_games_df = pd.read_csv("research/data/all_games.csv")
    st.session_state.all_games = all_games_df

if "date" not in st.session_state:
    start_date = "04/03/22"
    date_1 = datetime.strptime(start_date, "%m/%d/%y")
    st.session_state.date = date_1

if "date_delta" not in st.session_state:
    st.session_state.date_delta = timedelta(days=7)

if "current_schedule" not in st.session_state:
    st.session_state.current_schedule = NBADaySchedule(
        date_1.month, date_1.day, date_1.year
    )
    st.session_state.current_schedule_df = (
        st.session_state.current_schedule.get_dataframe()
    )


for i, row in st.session_state.current_schedule_df.iterrows():
    c = st.container(border=True)
    ht_team = str(row["htCity"]) + " " + str(row["htNickName"])
    vt_team = str(row["vtCity"]) + " " + str(row["vtNickName"])
    game_id = str(row["gameID"])

    matchup = st.session_state.all_games[
        st.session_state.all_games["GAME_ID"] == int(game_id)
    ]

    # print("\n \n The matchup is", matchup.to_string())

    knn_model = k_nearest_neighbors()
    lr_model = logistic_regression()

    c.write(ht_team + " vs. " + vt_team + " @ " + str((row["htCity"])))

    lr_pred = lr_model.predict(matchup)
    knn_pred = knn_model.predict(matchup)
    lr_winner = ht_team if lr_pred == "W" else vt_team
    knn_winner = ht_team if knn_pred == "W" else vt_team
    actual_winner = ht_team if matchup["WL_A"].values[0] == "W" else vt_team

    # c.write(type(game_prediction))

    c.write(
        "\nThe predicted winner of this match will be "
        + lr_winner
    )
    c.write(
        "\nThe second predicted winner of this match will be "
        + knn_winner
    )

    c.write("\nThe winner of this match will be " + str(actual_winner))
