import unittest
import os
import pandas as pd

from recsysconfident.utils.datasets import map_ids


class TestDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        script_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.run_folder = f'{str(script_dir[0:script_dir.index("tests")])}/tests/static'
        self.empty_folder = f"{self.run_folder}/sub_empty_folder"

    def test_map_ids_all_items_in_items_df(self):
        # Create a small ratings DataFrame
        ratings_data = {
            "userId": ["U1", "U2", "U1", "U3"],
            "itemId": ["I1", "I2", "I3", "I2"],
            "rating": [5, 3, 4, 2],
        }
        ratings_df = pd.DataFrame(ratings_data)

        # Create an items DataFrame with all itemIds from ratings_df
        items_data = {
            "itemId": ["I1", "I2", "I3"],
            "title": ["Item 1", "Item 2", "Item 3"],
        }
        items_df = pd.DataFrame(items_data)

        mapped_ratings_df, mapped_items_df = map_ids(
            ratings_df.copy(), items_df.copy(), user_col="userId", item_col="itemId"
        )

        self.assertSetEqual(set(mapped_ratings_df["itemId"]), set(mapped_items_df["itemId"]), "All ids in ratings_df are also in items_df.")
