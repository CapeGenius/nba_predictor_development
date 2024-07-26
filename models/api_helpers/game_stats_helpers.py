from nba_api.stats.static import teams
from api_helpers.team_stats_helpers import find_team_id
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


def home_matchups(all_games_df, team_a, team_b, year):

    matchup_1 = all_games_df[
        (all_games_df["TEAM_ID_A"] == int(team_a))
        & (all_games_df["TEAM_ID_B"] == int(team_b))
        & (all_games_df["SEASON_ID"] == int(year))
    ]

    return matchup_1


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


def avg_last_n_games(all_games_df: pd.DataFrame, columns: list, team="TEAM_ID_A", n=5):

    all_games_df.sort_values(by="GAME_DATE")
    avg_df = pd.DataFrame()

    team_ids = [team_id[0] for team_id in find_team_id()]

    for team_id in team_ids:
        team_df = all_games_df[all_games_df[team] == team_id]

        team_mean = (
            team_df[columns].rolling(window=5, min_periods=1).mean().shift().bfill()
        )

        avg_df = pd.concat([avg_df, team_mean])

    return avg_df


def load_past_n_games(
    all_games_df: pd.DataFrame,
    columns: list = ["FG_PCT", "FG3_PCT", "FTM", "OREB", "DREB", "REB", "AST"],
    n=5,):
    all_games_df.sort_values(by="GAME_DATE")
    string_columns = ["GAME_ID", "TEAM_ID_A", "TEAM_ID_B", "GAME_DATE", "WL_A"]
    columns_a = [column + "_A" for column in columns]
    columns_b = [column + "_B" for column in columns]

    a_avg_df = avg_last_n_games(all_games_df, columns_a, team="TEAM_ID_A", n=n)
    b_avg_df = avg_last_n_games(all_games_df, columns_b, team="TEAM_ID_B", n=n)

    last_n_df = pd.concat([all_games_df[string_columns], a_avg_df, b_avg_df], axis=1)

    last_n_df = last_n_df.dropna()

    return last_n_df

def matchup_past_n_games(all_games_df: pd.DataFrame, columns, matchup: pd.DataFrame, n=5):
    all_games_df.sort_values(by="GAME_DATE")
    # get all games before game date
    all_games_df = all_games_df[all_games_df["GAME_DATE"] < matchup["GAME_DATE"].values[0]]
    
    past_games_a = all_games_df[(all_games_df["TEAM_ID_A"] == matchup["TEAM_ID_A"].values[0])].head(n)
    past_games_b = all_games_df[(all_games_df["TEAM_ID_A"] == matchup["TEAM_ID_B"].values[0])].head(n)

    columns_a = [column + "_A" for column in columns]

    # average last n games and format column headers
    a_avg_df = pd.DataFrame(past_games_a[columns_a].mean()).T
    b_avg_df = pd.DataFrame(past_games_b[columns_a].mean()).T
    b_avg_df.columns = b_avg_df.columns.str.replace("_A", "_B")
    
    last_n_df = pd.concat([a_avg_df, b_avg_df], axis=1)
    
    return last_n_df



def load_x_y(
    all_games_df: pd.DataFrame,
    columns: list = ["FG_PCT", "FG3_PCT", "FTM", "OREB", "DREB", "REB", "AST"],
    n=5,
):

    string_columns = ["GAME_ID", "TEAM_ID_A", "TEAM_ID_B", "GAME_ID", "WL_A"]
    columns_a = [column + "_A" for column in columns]
    columns_b = [column + "_B" for column in columns]

    last_n_games = load_past_n_games(all_games_df=all_games_df, columns=columns, n=n)

    columns_x = columns_a + columns_b

    columns_x = [column + "_x" for column in columns_x]
    merged_data = pd.merge(
        all_games_df[columns_a + columns_b],
        last_n_games,
        left_index=True,
        right_index=True,
    )

    columns_x = columns_a + columns_b
    columns_x = [column + "_x" for column in columns_x]
    X = merged_data[columns_x]

    columns_y = columns_a + columns_b
    columns_y = [column + "_y" for column in columns_y]
    y = merged_data[columns_y]

    return X, y
