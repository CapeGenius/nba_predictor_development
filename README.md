# Quickstart

To access our repository (data and models), you can clone the repository with the following command:
`git clone https://github.com/CapeGenius/nba_predictor_development.git` 

Once cloned locally, you can download the packages in the repository using `pip install -r requirements.txt`. We highly recommend using a virtual environment to use this repository

Finally, if you wish to open the app on your local host, you can run the following command:
`streamlit run app/prototype_app.py`

# Project Overview
This is our NBA game predictor, a collection of predictors and an accompanying app for a convenient user interface. This repository contains various machine learning models designed to predict the outcomes of NBA and WNBA games. Throughout the research and development process for this project, over 20 models were created using various data manipulation and AI techniques. To illustrate the progression of our approach, we have selected five models of increasing complexity. The highest accuracy achieved was 71% with a hybrid xgBoost and DNN network model.

This project was inspired by our previous work at the Olin College of Engineering on predicting NBA Champions using PCA and SVMs. Check it out here [Link text](https://github.com/CapeGenius/NBAChampionPredictor)!

# Data Aggregation

The data for this project was collected from the NBA_API python package using a combination of provided and custom endpoints. We aggregated data from three categories for this project: seasonal statistics, game-by-game statistics, and player statistics. 

To obtain game-by-game data, we created helpers in `game_stats_helpers.py` that enabled quick, programmatic access to all games since 1984 for the NBA and 1997 for the WNBA models. To get our game-by-game training data, we utilized a rolling window function that averaged game-by-game stats for the last n games. For player game-by-game data, we followed the same methodology for specified columns. 

The initial scope of this project was predicting NBA champions, so we also developed helpers in team_stats_helpers.py to access seasonal NBA data. 

You can access our data in CSV files in the data folder. Examples on how to use these helpers can be found in the data_preparation folder. 


### More Coming Soon...