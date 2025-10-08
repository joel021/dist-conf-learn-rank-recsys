"""
package: recsysconfident.ml.ranking.conf_aware_rank_metrics
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score, recall_score

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo

warnings.filterwarnings("ignore")

class ConfAwareRankingMetrics:

    def __init__(self, data_info: DatasetInfo, r_ratio: float=0.75, alpha: float = 5):
        self.data_info = data_info
        self.r_ratio = r_ratio
        self.alpha = alpha

    def calibrate_ratings(self, df: pd.DataFrame, c_t: float):
        def calibrate(row):
            r, c = row[self.data_info.r_pred_col], row[self.data_info.conf_pred_col]
            if r >= self.r_ratio * self.data_info.rate_range[1] and c >= c_t:
                return r + self.alpha * c
            else:
                return r

        df[self.data_info.r_pred_col] = df.apply(calibrate, axis=1)
        return df

    def conf_filter(self, df: pd.DataFrame, c_t: float):
        df = df.copy()
        return self.calibrate_ratings(df, c_t)

    def binarize(self, true_relevances):

        return (true_relevances >= self.r_ratio).astype(int)

    def reciprocal_rank_at_k(self, pred_relevances, true_relevance, k):
        top_k_indices = np.argsort(-pred_relevances)[:k]
        for rank, idx in enumerate(top_k_indices, start=1):
            if true_relevance[idx] >= 1:
                return 1.0 / rank
        return 0.0

    def _get_true_pred_scores(self, df: pd.DataFrame, c_t: float) -> dict:
        if c_t >= 0:
            df = self.conf_filter(df, c_t)

        user_true_pred_scores = df.groupby(self.data_info.user_col).apply(
            lambda x: (
                x[self.data_info.relevance_col].values,
                x[self.data_info.r_pred_col].values
            )
        ).to_dict()

        return user_true_pred_scores

    def rank_metrics(self, candidates_df: pd.DataFrame, k: int, c_t: float) -> list:

        user_true_pred_scores = self._get_true_pred_scores(candidates_df, c_t)
        metrics = []
        for user_key in user_true_pred_scores.keys():
            true_ratings, pred_ratings = user_true_pred_scores[user_key]

            sorted_indices = np.argsort(pred_ratings)[::-1]
            pred_ratings_sorted = np.array(pred_ratings)[sorted_indices]
            true_ratings_sorted = np.array(true_ratings)[sorted_indices]

            # Truncate to top-k
            pred_top_k = pred_ratings_sorted[:k]
            true_top_k = true_ratings_sorted[:k]

            # Binarize after sorting
            binary_pred = self.binarize(pred_top_k)
            binary_true = self.binarize(true_top_k)

            # Compute metrics
            metrics.append([
                ndcg_score([true_ratings_sorted], [pred_ratings_sorted], k=k),
                average_precision_score(binary_true, binary_pred),
                recall_score(binary_true, binary_pred, average="micro"),
                self.reciprocal_rank_at_k(binary_pred, binary_true, k)
            ])
        return metrics

    def users_mean_std_rank_metrics(self, candidates_df: pd.DataFrame, k: int, c_t: float) -> tuple:
        users_scores = self.rank_metrics(candidates_df, k, c_t)
        scores = np.array(users_scores)

        mean_metrics = np.mean(scores, axis=0)
        std_metrics = np.std(scores, axis=0)

        return mean_metrics, std_metrics
