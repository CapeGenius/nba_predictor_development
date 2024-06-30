import pandas as pd


datasets = [
    "game_data_2010.csv",
    "game_data_2011.csv",
    "game_data_2012.csv",
    "game_data_2013.csv",
    "game_data_2014.csv",
    "game_data_2015.csv",
    "game_data_2016.csv",
    "game_data_2017.csv",
    "game_data_2018.csv",
    "game_data_2019.csv",
    "game_data_2020.csv",
    "game_data_2021.csv",
    "game_data_2022.csv",
    "game_data_2023.csv",
]

MIN = 0
MAX = 100

# Load your dataset
for dataset in datasets:
    df = pd.read_csv(dataset)
    df["TEAM_1_WIN/LOSS"] = df["TEAM_1_WIN/LOSS"].replace({"L": 0, "W": 1})

    # Columns to standardize (adjust these as per your dataset)
    columns_to_standardize = [
        "TEAM_1_WIN/LOSS",
        "TEAM_1_HOME/AWAY",
        "TEAM_1_PTS",
        "TEAM_1_FGA",
        "TEAM_1_FG_PCT",
        "TEAM_1_OREB",
        "TEAM_1_DREB",
        "TEAM_1_AST",
        "TEAM_1_TOV",
        "TEAM_1_WIN_PCT",
        "TEAM_2_HOME/AWAY",
        "TEAM_2_PTS",
        "TEAM_2_FGA",
        "TEAM_2_FG_PCT",
        "TEAM_2_OREB",
        "TEAM_2_DREB",
        "TEAM_2_AST",
        "TEAM_2_TOV",
        "TEAM_2_WIN_PCT",
    ]

    # Define the new min and max values for standardization

    # Standardize each selected column
    for col in columns_to_standardize:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = ((df[col] - min_val) / (max_val - min_val)) * (MAX - MIN) + MIN

    # Save or use the standardized DataFrame
    df.to_csv(dataset, index=False)
