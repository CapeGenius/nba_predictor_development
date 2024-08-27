from datetime import datetime
from datetime import timedelta
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

if "week_date" not in st.session_state:
    start_date = "04/03/22"
    date_1 = datetime.strptime(start_date, "%m/%d/%y")
    st.session_state.week_date = date_1

if "date" not in st.session_state:
    st.session_state.date = st.session_state.week_date

if "date_delta" not in st.session_state:
    st.session_state.date_delta = timedelta(days=7)

if "day_names" not in st.session_state:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    st.session_state.day_names = day_names

if "current_schedule" not in st.session_state:
    st.session_state.current_schedule = NBADaySchedule(
        date_1.month, date_1.day, date_1.year
    )
    st.session_state.current_schedule_df = (
        st.session_state.current_schedule.get_dataframe()
    )
    
button_col1, button_col2 = st.columns(2)
with button_col1:
    if st.button("Previous Week"):
        st.session_state.week_date -= st.session_state.date_delta
with button_col2:
    if st.button("Next Week"):
        st.session_state.week_date += st.session_state.date_delta

for i in range(0, 7):
    st.session_state.date = st.session_state.week_date + timedelta(days=i)
    st.subheader(
        st.session_state.day_names[st.session_state.date.isoweekday() - 1]
    )
    st.session_state.current_schedule = NBADaySchedule(
        st.session_state.week_date.month, st.session_state.week_date.day, st.session_state.week_date.year
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

        c.write(ht_team + " vs. " + vt_team + " @ " + str((row["htCity"])) + " on " + str(row["date"]))

        lr_pred = lr_model.predict(matchup)
        knn_pred = knn_model.predict(matchup)
        lr_winner = ht_team if lr_pred == "W" else vt_team
        knn_winner = ht_team if knn_pred == "W" else vt_team
        actual_winner = ht_team if matchup["WL_A"].values[0] == "W" else vt_team

        # c.write(type(game_prediction))

        c.write(
            "\nLogistic regression predicts the "
            + lr_winner
        )
        c.write(
            "\nK Nearest Neighbors predicts the "
            + knn_winner
        )

        c.write("\nThe winner of this match is the " + str(actual_winner))
