"""
package: recsysconfident.ml.ranking.sample_pred_negative.py
"""
import numpy as np
import pandas as pd

from collections import defaultdict
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class SamplePredNegatives:

    def __init__(self, data_info: DatasetInfo, num_negatives: int=100):
        self.data_info = data_info
        self.num_negatives = num_negatives

    def _get_items_per_user(self, users_set: set):

        return {k: v for k, v in self.data_info.items_per_user.items() if k in users_set}

    def get_neg_candidates(self, users_set: set, rmin: float) -> pd.DataFrame:

        neg_candidate_sets = self._sample_negative_candidates_sets(self._get_items_per_user(users_set),
                                                                   self.data_info.n_items,
                                                                   self.num_negatives)
        users_ids, items_ids = self._candidate_dict_to_lists(neg_candidate_sets)
        neg_df = pd.DataFrame({
            self.data_info.user_col: users_ids,
            self.data_info.item_col: items_ids
        })
        neg_df.loc[:, self.data_info.relevance_col] = rmin
        return neg_df

    def _candidate_dict_to_lists(self, candidates_per_user: dict) -> (list, list):
        user_ids = []
        item_ids = []

        for user, items in candidates_per_user.items():
            user_ids.extend([user] * len(items))
            item_ids.extend(items)

        return user_ids, item_ids

    def _sample_negative_candidates_sets(self, items_per_users: dict, n_items: int, num_negatives=100) -> dict:

        """
        Generate a list of candidate items for each user: positives + negatives
        """
        user_candidate_sets = defaultdict(list)

        all_items = set(range(0, n_items))

        for user, pos_items_labels in items_per_users.items():
            pos_items, labels = pos_items_labels
            neg_items = list(all_items - pos_items)
            sampled_neg_items = np.random.choice(neg_items, size=min(num_negatives, len(neg_items)), replace=False)

            user_candidate_sets[user] = list(sampled_neg_items)

        return user_candidate_sets
