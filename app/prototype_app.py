from datetime import datetime
from datetime import timedelta
from generate_schedule import NBADaySchedule

import streamlit as st

st.title("Prototype App")


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

    c.write(ht_team + " vs. " + vt_team + " @ " + str((row["htCity"])))
