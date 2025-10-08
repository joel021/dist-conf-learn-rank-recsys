"""
package: recsysconfident.ranking.elementwise_error
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.environment import Environment
from recsysconfident.ml.ranking.rank_helper import get_low_rank_items
from recsysconfident.ml.eval.predict_helper import predict
from recsysconfident.ml.models.torchmodel import TorchModel


def elementwise_pos_neg_scores(model, split_df: pd.DataFrame, environ: Environment, device) -> (pd.DataFrame, pd.DataFrame):

    if environ.learn_to_rank:
        pos_indices = split_df[environ.dataset_info.relevance_col] == 1
    else:
        pos_indices = split_df[environ.dataset_info.relevance_col] >= environ.dataset_info.ratio_t * \
                      environ.dataset_info.rate_range[1]

    positive_split_df = split_df[pos_indices]

    neg_true, neg_pred, neg_conf = obtain_neg_scores(model,
                                                     torch.from_numpy(
                                                         positive_split_df[environ.dataset_info.user_col].values),
                                                     environ.dataset_info,
                                                     device)
    positive_split_df.loc[:, environ.dataset_info.relevance_col] = 1
    positive_split_df.loc[:, "neg_pred"], positive_split_df.loc[:, "neg_conf"] = neg_pred, neg_conf

    neg_split_df = split_df[~pos_indices]
    neg_true, neg_pred, neg_conf = obtain_neg_scores(model,
                                                     torch.from_numpy(
                                                         neg_split_df[environ.dataset_info.user_col].values),
                                                     environ.dataset_info,
                                                     device)
    neg_split_df.loc[:, "neg_pred"], neg_split_df.loc[:, "neg_conf"] = neg_pred, neg_conf
    neg_split_df.loc[:, environ.dataset_info.relevance_col] = 0

    split_df = pd.concat([neg_split_df, positive_split_df], axis=0, ignore_index=True)

    return split_df

def set_bpr_error(df: pd.DataFrame):
    diff = df['r_pred'] - df['neg_pred']
    df.loc[:, "bpr_error"] = -torch.log(torch.sigmoid(torch.from_numpy(diff.values)) + 1e-8)
    return df

def obtain_neg_scores(model: TorchModel, users_ids: torch.Tensor, data_info:DatasetInfo, device):
    """
    Obtain negative items scores for evaluation during training only.: For each user, sample only one negative item.
    """
    test_low_rank_items = get_low_rank_items(users_ids,
                                             data_info.items_per_user,
                                             data_info.n_items)

    neg_items_dataloader = DataLoader(
        TensorDataset(users_ids,
                      test_low_rank_items,
                      torch.zeros_like(users_ids)),
        batch_size=1024,
        shuffle=False)
    return predict(model, neg_items_dataloader, device)
