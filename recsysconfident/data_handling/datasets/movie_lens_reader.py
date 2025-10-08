from recsysconfident.data_handling.datasets.csv_reader import CsvReader
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class MovieLensReader:

    def __init__(self, dataset_info: DatasetInfo):
        self.csv_reader = CsvReader(dataset_info)

    def read(self):
        return self.csv_reader.read()
