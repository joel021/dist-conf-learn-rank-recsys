"""
package: recsysconfident.data_handling.datasets.datasets.py
"""
from pandas import DataFrame, read_csv, concat
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from recsysconfident.data_handling.splitting import split_ratings
from recsysconfident.utils.datasets import filter_positives


class DatasetInfo:

    def __init__(self, user_col: str, item_col: str, rating_col: str, interactions_file: str, columns: list,
                 rate_range: list[float], database_name: str, split_run_uri: str, sep: str= ",", has_head: bool=False, timestamp_col: str = None,
                 batch_size: int=1024, root_uri: str= "./"):

        self.fit_df = None
        self.val_df = None
        self.test_df = None
        self.ratings_df = None
        self.n_users = 0
        self.n_items = 0
        self.rate_range = rate_range

        self.user_col = user_col
        self.item_col = item_col
        self.conf_pred_col = "conf_pred"
        self.r_pred_col = "r_pred"

        self.relevance_col = rating_col
        self.interactions_file = interactions_file
        self.timestamp_col = timestamp_col
        self.sep = sep
        self.has_head = has_head
        self.database_name = database_name
        self.columns = columns
        self.batch_size = batch_size
        self.root_uri = root_uri
        self.split_run_uri = split_run_uri
        self.ratio_t = 0.78

    def build(self, ratings_df: DataFrame, shuffle: bool):
        self.ratings_df = ratings_df
        self.split_interactions(shuffle)
        self.items_per_user = self.get_user_item_sets(self.ratings_df)

        print(f"{len(list(self.items_per_user.keys()))} mapped users sequentially!")

    def split_interactions(self, shuffle: bool):

        fit_path = f"{self.split_run_uri}/ratings.fit.csv"
        val_path = f"{self.split_run_uri}/ratings.val.csv"
        test_path = f"{self.split_run_uri}/ratings.test.csv"
        os.makedirs(self.split_run_uri, exist_ok=True)

        if os.path.exists(fit_path) and os.path.exists(val_path) and os.path.exists(test_path):  # double check
            # Load existing splits
            fit_df = read_csv(fit_path)
            eval_df = read_csv(val_path)
            test_df = read_csv(test_path)

            self.fit_df, self.val_df, self.test_df = fit_df, eval_df, test_df
            self.update_n_users()
        else:
            # Load and split the data
            self.ratings_df = filter_positives(self.ratings_df, self.relevance_col, self.ratio_t)
            fit_df, test_df = split_ratings(self.ratings_df, self.user_col, self.item_col, self.timestamp_col,
                                            0.75, shuffle)

            test_df, eval_df = train_test_split(test_df, test_size=0.5, random_state=42)

            self.fit_df, self.val_df, self.test_df = fit_df, eval_df, test_df
            self.update_n_users()
            self.map_ids(self.ratings_df[self.user_col].unique(), self.ratings_df[self.item_col].unique())

            # Save the splits
            self.fit_df.to_csv(fit_path, index=False)
            self.val_df.to_csv(val_path, index=False)
            self.test_df.to_csv(test_path, index=False)

        return self

    def map_ids(self, unique_users, unique_items):

        user_encoder = LabelEncoder().fit(unique_users)
        item_encoder = LabelEncoder().fit(unique_items)

        self.fit_df.loc[:, self.user_col] = user_encoder.transform(self.fit_df[self.user_col])
        self.fit_df.loc[:, self.item_col] = item_encoder.transform(self.fit_df[self.item_col])

        self.val_df.loc[:, self.user_col] = user_encoder.transform(self.val_df[self.user_col])
        self.val_df.loc[:, self.item_col] = item_encoder.transform(self.val_df[self.item_col])

        self.test_df.loc[:, self.user_col] = user_encoder.transform(self.test_df[self.user_col])
        self.test_df.loc[:, self.item_col] = item_encoder.transform(self.test_df[self.item_col])

    def update_n_users(self):

        self.ratings_df = concat([self.fit_df, self.val_df, self.test_df], ignore_index=True)
        self.n_users = len(self.ratings_df[self.user_col].unique())
        self.n_items = len(self.ratings_df[self.item_col].unique())

    def get_splits(self):

        return self.fit_df, self.val_df, self.test_df

    def is_loaded(self):
        return self.fit_df is not None and self.val_df is not None and self.test_df is not None

    def get_user_item_sets(self, df: DataFrame) -> dict:
        user_item_dict = (
            df.groupby(self.user_col)
            .apply(lambda x: (set(x[self.item_col].tolist()), x[self.relevance_col].tolist()))
            .to_dict()
        )
        return user_item_dict
