import pandas as np
from abc import ABC
from nba_api.stats.endpoints import boxscoretraditionalv2


class StatScore(ABC):

    def __init__(self, game_ID, all_games):
        super().__init__()
        self.game_ID = game_ID
        self.statline = all_games[all_games["GAME_ID"] == self.game_ID]
        self.box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_ID)
