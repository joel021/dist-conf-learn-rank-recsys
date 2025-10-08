"""
package: recsysconfident.ml.fit_eval.conf_threshold_searcher
"""
from typing import Callable
import numpy as np
from pandas import DataFrame


def find_best_conf_threshold(df_true_pred: DataFrame, rank_metric_calculator: Callable, k: int):
    """"
    ndcd_calculator: Any function that returns the mean of NDCG over all users along with the STD.: ConfAwareRanking
    """
    b_quantile, b_conf_mean_ndcg, b_conf_std_ndcg, conf_threshold = -1, [-1,-1,-1,-1], [-1,-1,-1,-1], -1
    df_true_pred = df_true_pred.sort_values(by=["conf_pred"])
    start_ct = df_true_pred['conf_pred'].quantile(0.65)
    end_ct = df_true_pred['conf_pred'].quantile(0.99)

    for c_t in np.arange(start_ct, end_ct, 0.04):

        conf_mean_ndcg, conf_std_ndcg = rank_metric_calculator(df_true_pred.copy(), k, c_t)
        if conf_mean_ndcg[0] > b_conf_mean_ndcg[0]:
            quantile = np.sum(df_true_pred['conf_pred'] <= c_t) / len(df_true_pred['conf_pred'])
            b_quantile, b_conf_mean_ndcg, b_conf_std_ndcg, conf_threshold = quantile, conf_mean_ndcg, conf_std_ndcg, c_t

    return b_quantile, b_conf_mean_ndcg, b_conf_std_ndcg, conf_threshold
