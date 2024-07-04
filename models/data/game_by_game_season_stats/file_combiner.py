import pandas as pd
import glob

# Assuming all your files are CSVs and stored in a directory

# Get a list of all CSV files
file_list = [
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

# Initialize an empty list to store DataFrames
dfs = []

# Load each CSV file into a DataFrame and append to the list
for file in file_list:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames in the list into one
combined_df = pd.concat(dfs, ignore_index=True)

# Now combined_df contains all rows from all original DataFrames

combined_df.to_csv(
    "/home/swisnoski/nba_predictor_development/models/data/combined_data_2010-2023.csv",
    index=False,
)
