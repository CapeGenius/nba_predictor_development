from nba_api.library import http
import pandas as pd
import requests
from abc import ABC


class NBADaySchedule(ABC):

    def __init__(self, month, day, year) -> None:
        super().__init__()
        self._url = self.get_url(month, day, year)
        self._response = self.get_response()
        self._data = self.get_json()

    def get_url(self, month, day, year):
        date = f"{month}/{day}/{year}"

        if month < 8:
            season = year - 1
        else:
            season = year
        url = str(
            f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={season}&RegionID=1&Date={date}&EST=Y"
        )
        return url

    def get_response(self):
        return requests.get(self._url)

    def get_json(self):
        return self._response.json()

    def get_dataframe(self):
        games_list = self._data["resultSets"][1]["CompleteGameList"]

        try:
            df = pd.DataFrame.from_records(games_list, index=range(len(games_list)))
            return df
        except ValueError:
            return "No Matches Today"

    def __repr__(self) -> str:
        return (
            "The url is "
            + self._url
            + ". The response is "
            + str(self._response)
            + ". The data is "
            + str(self._data)
        )
