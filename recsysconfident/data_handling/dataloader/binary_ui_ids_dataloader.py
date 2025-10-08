import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from recsysconfident.utils.binary_encoding import binary_encoding


class BinaryUIIdsDataSetBuilder:

    def __init__(self, icol: str, ucol: str, rcol: str, ccol,
                 user_dim: int, item_dim: int):
        self.ucol = ucol
        self.icol = icol
        self.rcol = rcol
        self.ccol = ccol
        self.user_dim = user_dim
        self.item_dim = item_dim

    def get_ui_matrix(self, ratings_df: pd.DataFrame) -> (torch.Tensor, torch.Tensor):

        user_matrix = binary_encoding(ratings_df[self.ucol].values, self.user_dim)
        item_matrix = binary_encoding(ratings_df[self.icol].values, self.item_dim)
        return user_matrix.float(), item_matrix.float()

    def get_data_loader(self, ratings_df: pd.DataFrame, batch_size: int = 128) -> DataLoader:

        u_matrix, i_matrix, labels = self.get_entire_dataset(ratings_df)

        dataloader = DataLoader(TensorDataset(u_matrix, i_matrix, labels),
                                batch_size=batch_size,
                                shuffle=True)
        return dataloader

    def get_entire_dataset(self, ratings_df):

        ui_matrix = self.get_ui_matrix(ratings_df)
        labels = torch.tensor(ratings_df[self.rcol].values).unsqueeze(-1).float()

        return ui_matrix[0], ui_matrix[1], labels
