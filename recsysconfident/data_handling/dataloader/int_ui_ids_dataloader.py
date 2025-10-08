"""
pakcage: recsysconfident.data_handling.dataloader.int_ui_ids_dataloader.py
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def ui_ids_label(info: DatasetInfo):
    fit_df, val_df, test_df = info.fit_df, info.val_df, info.test_df

    fit_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(fit_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(fit_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(fit_df[info.relevance_col].values.astype(float)).float()),
        batch_size=info.batch_size,
        shuffle=True)

    eval_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(val_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(val_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(val_df[info.relevance_col].values.astype(float)).float()),
        batch_size=info.batch_size,
        shuffle=False)

    test_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(test_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(test_df[info.item_col].values.astype(float)).int(),
                      torch.from_numpy(test_df[info.relevance_col].values.astype(float)).float()),
        batch_size=info.batch_size,
        shuffle=False)

    return fit_dataloader, eval_dataloader, test_dataloader
