"""
package: recsysconfident.ml.fit_eval.inference_loss
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from recsysconfident.constants import RANK_SCORES_COL, NEG_FLAG_COL, ABS_ERROR_COL, LEARN_RANK, RELEVANCE_RATIO
from recsysconfident.data_handling.miscellaneous import keep_users_any_r_higher_than, filter_out_users_less_than_k_inter
from recsysconfident.environment import Environment
from recsysconfident.ml.eval.predict_helper import predict
from recsysconfident.ml.models.torchmodel import TorchModel
from recsysconfident.ml.ranking.elementwise_error import set_bpr_error, elementwise_pos_neg_scores
from recsysconfident.ml.ranking.sample_pred_negative import SamplePredNegatives


def set_elementwise_metrics(model, split_df, environ, device):

    if environ.learn_to_rank:
        split_df = elementwise_pos_neg_scores(model, split_df, environ, device)
        set_bpr_error(split_df)

    split_df.loc[:, ABS_ERROR_COL] = abs(split_df[environ.dataset_info.relevance_col] - split_df[environ.dataset_info.r_pred_col])

    return split_df

def export_elementwise_error(model, environ: Environment, device) -> (pd.DataFrame, pd.DataFrame):

    fit_df, eval_df, test_df = environ.dataset_info.get_splits()

    eval_df = inference(model, eval_df, environ, device)
    test_df = inference(model, test_df, environ, device)

    eval_df = set_elementwise_metrics(model, eval_df, environ, device)
    test_df = set_elementwise_metrics(model, test_df, environ, device)

    print("Exporting data splits predictions")
    eval_df.to_csv(f"{environ.instance_dir}/eval_error_conf-{environ.split_position}.csv", index=False)
    test_df.to_csv(f"{environ.instance_dir}/test_error_conf-{environ.split_position}.csv", index=False)

    return eval_df.copy(), test_df.copy()

def append_neg_samples(split_df: pd.DataFrame, environ: Environment):

    split_df.loc[:, NEG_FLAG_COL] = 0
    if environ.min_inter_per_user == 0:
        return split_df

    sample_pred_neg = SamplePredNegatives(data_info=environ.dataset_info, num_negatives=environ.min_inter_per_user)
    users_ids = set(split_df[environ.dataset_info.user_col].unique().tolist())

    if environ.learn_to_rank:
        rmin = 0
    else:
        rmin = environ.dataset_info.rate_range[0]

    neg_df = sample_pred_neg.get_neg_candidates(users_ids, rmin)
    neg_df.loc[:, NEG_FLAG_COL] = 1

    split_df = pd.concat([split_df, neg_df], ignore_index=True)

    return split_df


def inference(model: TorchModel, split_df: pd.DataFrame, environ: Environment, device):

    if hasattr(model, 'switch_to_rating'):
        model.switch_to_rating()

    user_col = environ.dataset_info.user_col
    relevance_col = environ.dataset_info.relevance_col
    r_pred_col = environ.dataset_info.r_pred_col

    split_df = filter_out_users_less_than_k_inter(split_df, user_col, 10)
    split_df = append_neg_samples(split_df, environ)
    split_df[environ.dataset_info.relevance_col] = split_df[environ.dataset_info.relevance_col].astype(float)

    dataloader = DataLoader(
        TensorDataset(torch.from_numpy(split_df[user_col].values),
                      torch.from_numpy(split_df[environ.dataset_info.item_col].values),
                      torch.from_numpy(split_df[relevance_col].values),),
        batch_size=environ.dataset_info.batch_size,
        shuffle=False)

    y_true, y_pred, conf_pred = predict(model, dataloader, device)
    split_df.loc[:, r_pred_col] = y_pred
    split_df.loc[:, environ.dataset_info.conf_pred_col] = conf_pred

    if hasattr(model, 'switch_to_ranking'):
        model.switch_to_ranking()
        y_true, y_pred, conf_pred = predict(model, dataloader, device)
        split_df.loc[:, RANK_SCORES_COL] = y_pred

    return split_df

