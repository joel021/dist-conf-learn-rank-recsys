import pandas as pd
import os
import numpy as np

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class JesterJokeReader:

    def __init__(self, dataset_info: DatasetInfo, export_file_name: str) -> None:

        self.info = dataset_info
        self.export_file_name = export_file_name

    def read(self):

        df_uri = f"{self.info.root_uri}/data/{self.info.database_name}/{self.export_file_name}"
        if os.path.isfile(df_uri):
            return pd.read_csv(df_uri)
        else:
            return self.__save_and_load_ratings()

    def __save_and_load_ratings(self) -> pd.DataFrame:

        user_ratings_matrix = pd.read_csv(f"{self.info.root_uri}/data/{self.info.database_name}/{self.info.interactions_file}",
                                          header=None, low_memory=False)
        jokes_ids = np.arange(1, len(user_ratings_matrix.loc[0]))
        user_ratings = []

        for user_id in user_ratings_matrix.index:

            jokes_ratings = user_ratings_matrix.loc[user_id].values

            for joke_id in jokes_ids:
                if "99" == jokes_ratings[joke_id]:
                    continue
                user_ratings.append({self.info.user_col: user_id,
                                     self.info.item_col: joke_id - 1,
                                     self.info.relevance_col: float(jokes_ratings[joke_id])})

        user_ratings_df = pd.DataFrame(user_ratings)

        user_ratings_df.replace(99, None, inplace=True)
        user_ratings_df.dropna(inplace=True)
        user_ratings_df.to_csv(f"{self.info.root_uri}/data/{self.info.database_name}/{self.export_file_name}", index=False)

        return user_ratings_df
