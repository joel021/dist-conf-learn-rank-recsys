import pandas as pd
import os

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class GoodReadsReader:

    def __init__(self, dataset_info: DatasetInfo, rebuild: bool = False, export_filename: str = "ratings.csv"):
        super().__init__()

        self.info = dataset_info
        self.rebuild = rebuild
        self.export_filename = export_filename

    def read(self):

        file_uri = f"./data/{self.info.database_name}/{self.info.interactions_file}"

        if not self.rebuild and os.path.isfile(file_uri):
            ratings_df = pd.read_csv(file_uri)
            return ratings_df
        else:
            return self.__build_ratings_df(file_uri)

    def __build_ratings_df(self, file_uri: str) -> pd.DataFrame:

        chunks = pd.read_json(file_uri, lines=True, chunksize=128)
        dataset_list = []
        for chunk_ratings_df in chunks:
            chunk_ratings_df = chunk_ratings_df[self.info.columns]
            dataset_list.append(chunk_ratings_df[chunk_ratings_df[self.info.relevance_col] > 0])

        chunks.close()
        dataset = pd.concat(dataset_list)
        dataset.to_csv(f"./data/{self.info.database_name}/{self.export_filename}", index=False)
        return dataset
