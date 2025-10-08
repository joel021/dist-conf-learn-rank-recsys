import unittest

import pandas as pd

from recsysconfident.ml.eval.eval import filter_out_users_less_than_k_inter


class TestEval(unittest.TestCase):


    def test_filter_out_users(self):

        df = pd.DataFrame({
            "userId": [1,1,1,1,2,2],
            "itemId": [1,1,1,1,2,2]
        })
        filtered_df = filter_out_users_less_than_k_inter(df, "userId", 3)
        assert len(filtered_df) == 4, "Only user with id 1 is kept."

