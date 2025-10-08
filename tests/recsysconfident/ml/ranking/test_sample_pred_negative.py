import unittest
import pandas as pd

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.ranking.sample_pred_negative import SamplePredNegatives


class TestSamplePredNegatives(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        data = {
            'userId': [1, 1, 2, 2, 3, 3],
            'itemId': [1, 2, 3, 4, 5, 6],
            'rating': [1, 1, 2, 3, 4, 4]
        }
        self.df = pd.DataFrame(data)
        self.datainfo = DatasetInfo('userId', 'itemId', 'rating', "./interactions.csv",
                              ['userId', 'itemId'], [1, 5], 'ml-1m')
        self.items_per_user = self.datainfo.get_user_item_sets(self.df)

        n_items = 10  # Total number of items
        num_negatives = 4  # Number of negative samples.  Changed to 2 for easier verification

        self.rank_generator = SamplePredNegatives(self.datainfo, n_items, num_negatives)

    def test_sample_negative_candidates_sets(self):

        user_candidate_sets = self.rank_generator._sample_negative_candidates_sets(self.items_per_user, 100, 4)
        assert len(user_candidate_sets.keys()) == 3, "A sample for each user."
        assert len(user_candidate_sets[1]) == 4, "There are only 4 negatives."

    def test_all_negative_are_not_in_positive(self):

        user_candidate_sets = self.rank_generator._sample_negative_candidates_sets(self.items_per_user, 4,
                                                                                   4)
        user_1_df = self.df[self.df['userId'] == 1]
        user1_items = set(user_1_df['itemId'].unique())
        diff = user1_items - set(user_candidate_sets[1])

        assert len(diff) == len(user1_items), "The negative user set has no positive items in it, then, the difference keeps all positive items."

    def test_sample_negative_candidates_sets_expected_length(self):

        user_candidate_sets = self.rank_generator._sample_negative_candidates_sets(self.items_per_user, 4,
                                                                                   4)
        assert len(user_candidate_sets[1]) == 2, "User 1 has items ids 1 and 2, remaining only 0 and 3 as negative possible items."

    def test_get_items_per_user(self):
        self.rank_generator.data_info.items_per_user = {
            1: (),
            2: (),
            3: (),
            4: (),
            5: ()
        }
        items_per_user = self.rank_generator._get_items_per_user({1,2,3})
        assert items_per_user.keys() == {1,2,3}


