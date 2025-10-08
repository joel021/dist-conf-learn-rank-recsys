import unittest
import pandas as pd
import os

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class TestPositiveOnlyInteractions(unittest.TestCase):

    def setUp(self):

        script_dir = str(os.path.dirname(os.path.abspath(__file__)))
        ratings_uri = f'{str(script_dir[0:script_dir.index("tests")])}/tests/data/ratings-1m.csv'
        self.dataset = DatasetInfo(
            user_col='userId',
            item_col='itemId',
            rating_col='rating',
            interactions_file='ratings-1m.csv',
            columns=['user_id', 'item_id', 'rating'],
            rate_range=(1, 5),
            database_name='test_db',
            root_uri=f'{str(script_dir[0:script_dir.index("tests")])}/tests/data',
            split_run_uri=f'{str(script_dir[0:script_dir.index("tests")])}/tests/runs/data_splits/0/ml-1m'
        )
        self.ratings = pd.read_csv(ratings_uri)
        self.dataset.build(self.ratings, True)

    def test_all_in_fit_belong_to_items_per_users(self):
        users = self.dataset.items_per_user.keys()
        fit_users = set(self.dataset.fit_df['userId'].unique())

        assert len(fit_users - users) == 0

    def test_all_in_val_belong_to_items_per_users(self):
        users = self.dataset.items_per_user.keys()
        val_users = set(self.dataset.fit_df['userId'].unique())

        assert len(val_users - users) == 0

    def test_all_in_test_belong_to_items_per_users(self):
        users = self.dataset.items_per_user.keys()
        test_users = set(self.dataset.fit_df['userId'].unique())

        assert len(test_users - users) == 0

