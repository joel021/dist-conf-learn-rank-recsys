import os

import numpy as np
import pandas as pd
import unittest

from recsysconfident.data_handling.splitting import split_ratings


class TestRatingsLoader(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        script_dir = str(os.path.dirname(os.path.abspath(__file__)))
        ratings_uri = f'{str(script_dir[0:script_dir.index("tests")])}/tests/data/ratings-1m.csv'

        self.ratings_df = pd.read_csv(ratings_uri)
        fit_df, test_df = split_ratings(self.ratings_df,
                                        user_col="userId",
                                        item_col="itemId",
                                        timestamp_col="timestamp",
                                        fit_ratio= 0.75)
        self.fit_df = fit_df
        self.test_df = test_df

    def test_split_users_in_fit(self):
        assert np.isin(self.test_df['userId'].unique(), self.fit_df['userId'].unique()).all()

    def test_split_items_in_fit(self):
        assert np.isin(self.test_df['itemId'].unique(), self.fit_df['itemId'].unique()).all()

    def test_split_ratings_length(self):
        assert len(self.fit_df) < len(self.ratings_df)

    def test_split_ratings(self):
        assert len(self.test_df) < len(self.fit_df)

    def test_min_interactions_u(self):
        assert (self.fit_df["userId"].value_counts() < 2).sum() == 0

    def test_min_interactions_i(self):
        assert len(set(self.test_df["itemId"].unique()) - set(self.fit_df["itemId"].unique())) == 0

    def test_len_test_set(self):
        assert len(self.test_df) > 0

    def test_len_fit_set(self):
        assert len(self.fit_df) >= int(0.6 * len(self.ratings_df)), "The fit ratio is at least 0.6"

    def test_with_shuffle(self):
        fit_df, test_df = split_ratings(self.ratings_df,
                                        user_col="userId",
                                        item_col="itemId",
                                        timestamp_col="timestamp",
                                        fit_ratio=0.75,
                                        shuffle=True)
        assert len(fit_df) > len(test_df) > 0
