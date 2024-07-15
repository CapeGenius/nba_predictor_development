from nba_api.library import http
import pandas as pd
import requests
from abc import ABC


class NBADaySchedule(ABC):

    def __init__(self, month, date, year) -> None:
        super().__init__()
        self._url = self.get_url(month, date, year)
        self._response = self.get_response()
        self._data = self.get_json()

    def get_url(self, month, day, year):
        season = 2023
        date = f"{month}/{day}/{year}"
        url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={season}&RegionID=1&Date={date}&EST=Y"
        return url

    def get_response(self):
        return requests.get(self._url)

    def get_json(self):
        self._response.json()

    def get_dataframe(self):
        games_list = self._data["resultSets"][1]["CompleteGameList"]
        df = pd.DataFrame.from_records(games_list, index=range(len(games_list)))
        return df
