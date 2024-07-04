from nba_api.stats.static import teams
from team_stats_helpers import find_team_id
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

all_games_list = []


def load_games():
    team_id = find_team_id()

    for i in range(len(team_id)):
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id[i][0])
        games = gamefinder.get_data_frames()[0]

        all_games_list.append(games)

    all_games_df = pd.concat(all_games_list)

    all_games_df.to_csv("data/all_games.csv")


def combine_team_games(df, keep_method="home"):
    """Combine a TEAM_ID-GAME_ID unique table into rows by game. Slow.

    Parameters
    ----------
    df : Input DataFrame.
    keep_method : {'home', 'away', 'winner', 'loser', ``None``}, default 'home'
        - 'home' : Keep rows where TEAM_A is the home team.
        - 'away' : Keep rows where TEAM_A is the away team.
        - 'winner' : Keep rows where TEAM_A is the losing team.
        - 'loser' : Keep rows where TEAM_A is the winning team.
        - ``None`` : Keep all rows. Will result in an output DataFrame the same
            length as the input DataFrame.

    Returns
    -------
    result : DataFrame
    """
    # Join every row to all others with the same game ID.
    joined = pd.merge(
        df, df, suffixes=["_A", "_B"], on=["SEASON_ID", "GAME_ID", "GAME_DATE"]
    )
    # Filter out any row that is joined to itself.
    result = joined[joined.TEAM_ID_A != joined.TEAM_ID_B]
    # Take action based on the keep_method flag.
    if keep_method is None:
        # Return all the rows.
        pass
    elif keep_method.lower() == "home":
        # Keep rows where TEAM_A is the home team.
        result = result[result.MATCHUP_A.str.contains(" vs. ")]
    elif keep_method.lower() == "away":
        # Keep rows where TEAM_A is the away team.
        result = result[result.MATCHUP_A.str.contains(" @ ")]
    elif keep_method.lower() == "winner":
        result = result[result.WL_A == "W"]
    elif keep_method.lower() == "loser":
        result = result[result.WL_A == "L"]
    else:
        raise ValueError(f"Invalid keep_method: {keep_method}")
    return result


def find_matchups(all_games_df, team_a, team_b):
    """
    Find all games where to specific teams go against each other
    """
    matchup_1 = all_games_df[
        (all_games_df["TEAM_ID_A"] == team_a) & (all_games_df["TEAM_ID_B"] == team_b)
    ]
    matchup_2 = all_games_df[
        (all_games_df["TEAM_ID_A"] == team_b) & (all_games_df["TEAM_ID_B"] == team_a)
    ]

    all_matchups = pd.concat([matchup_1, matchup_2])

    return all_matchups