"""
Helper file that contains functions to use the NBA
api, process the data, and organize and plot the data
"""

from nba_api.stats.endpoints import teamyearbyyearstats
from nba_api.stats.static import teams
import pandas as pd
from numpy import histogram
import matplotlib.pyplot as plt


def find_team_id():
    """
    This helper function uses the API to get every single
    NBA team and its corresponding ID, taking values from
    a list of dictionaries and turning them into a 2D list.

    Arguments:
        No parameters

    Returns:
        team_id: 2D list with every row containing the
        team's ID and full name
    """
    # get_teams returns a list of 30 dictionaries, each an NBA team.
    nba_teams = teams.get_teams()

    # following comprehension can be rewritten as a for loop
    team_id = [[team["id"], team["full_name"]] for team in nba_teams]

    return team_id


def team_year_by_year():
    """
    This function writes to a .csv file with each team's year-by-year
    statline (ie. PTS, Blocks, Games Played). First, it calls find_team_id()
    to get the 2D list of NBA team IDs and names -- and then
    calls Team Year by Year class from the NBA API to find
    the corresponding statistics of that team on a year-by-year basis.

    Arguments:
        None:

    Returns:
        None: this function simply writes a CSV file
    """
    # gets a list of all teams
    team_id = find_team_id()

    result_dataframes = pd.DataFrame()

    # gets header data
    team_stats = teamyearbyyearstats.TeamYearByYearStats(team_id[0][0])
    result_dataframes = team_stats.get_data_frames()[0]

    # finds the year by year stats of each team in team_id list through for loop
    for i in range(1, len(team_id)):
        team_stats = teamyearbyyearstats.TeamYearByYearStats(team_id[i][0])
        team_stats_dataframes = team_stats.get_data_frames()[0]

        # concatenates temp dataframe with overall dataframe of all stats
        result_dataframes = pd.concat([result_dataframes, team_stats_dataframes])
    result_dataframes.to_csv("data/team_year_stats.csv")


def load_dataframe(columns=["CONF_RANK"]):
    """
    This function returns a loaded dictionary with each key
    representing a year. Each key maps to another dictionary -- where
    each team ID maps to that team's specified statistic for the
    season. The argument 'column' allow you to specify any
    statistic available in the CSV file.

    Argument:
        column: string representing a desired/specified
        statistic. Its default value is 'CONF_RANK' as
        for plot_statistics function later.

    Returns:
        seasons_dict: dictionary where each key represents
        a year and maps to dictionary, with a TEAM_ID as a key
        and the corresponding statistic of that team for that
        year.
    """

    # list of columns for the data frame from the CSV
    column_list = ["YEAR", "TEAM_ID", "NBA_FINALS_APPEARANCE"] + columns

    # data frame of the year, team_id, and specified statistic
    data_frame = pd.read_csv("data/team_year_stats.csv", usecols=column_list)

    return data_frame


def api_dataframe(columns=["CONF_RANK"]):
    # list of columns for the data frame from the CSV
    column_list = ["YEAR", "TEAM_ID"] + columns

    # gets a list of all teams
    team_id = find_team_id()

    result_dataframes = pd.DataFrame()

    # gets header data
    team_stats = teamyearbyyearstats.TeamYearByYearStats(team_id[0][0])
    result_dataframes = team_stats.get_data_frames()[0]

    # finds the year by year stats of each team in team_id list through for loop
    for i in range(1, len(team_id)):
        team_stats = teamyearbyyearstats.TeamYearByYearStats(team_id[i][0])
        team_stats_dataframes = team_stats.get_data_frames()[0]

        # concatenates temp dataframe with overall dataframe of all stats
        result_dataframes = pd.concat([result_dataframes, team_stats_dataframes])

    final_team_data = result_dataframes[column_list]

    return final_team_data
