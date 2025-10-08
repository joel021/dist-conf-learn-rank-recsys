import pandas as pd

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class CsvReader:

    def __init__(self, dataset_info: DatasetInfo):
        self.info = dataset_info

    def read(self):

        file_uri = f"{self.info.root_uri}/data/{self.info.database_name}/{self.info.interactions_file}"
        if not ".csv" in file_uri or not self.info.has_head:
            return self.read_with_dynamic_cols(file_uri)
        else:
            return pd.read_csv(file_uri)

    def read_with_dynamic_cols(self, ratings_uri: str):

        df = pd.read_csv(ratings_uri, header=None, sep=self.info.sep)
        df.columns = self.info.columns
        return df
