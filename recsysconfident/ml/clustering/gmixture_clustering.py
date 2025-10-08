import os

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.mixture import GaussianMixture
import joblib

from recsysconfident.data_handling.pivot_data_generator import PivotDataGenerator


class GaussianMixtureClustering:

    def __init__(self,
                 user_col: str,
                 item_col: str,
                 rating_col: str,
                 encoder_uri: str | None,
                 replace=False):

        self.clusters_grouping = None
        self.user_cluster_df = None
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.encoder_uri = encoder_uri
        self.replace = replace

    def get_encoder(self, latent_dim: int) -> tf.keras.Model:

        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim,
                                  activation='linear',
                                  name='output-encoder')
        ])
        encoder.compile(optimizer='adam', loss='mse')
        if not os.path.isfile(self.encoder_uri) or self.replace:
            encoder.save_weights(self.encoder_uri)
        else:
            encoder.load_weights(self.encoder_uri)

        return encoder

    def get_model(self, users_encoding: list):

        cluster_model_uri = f"{self.encoder_uri}.cluster.joblib"
        if os.path.isdir(cluster_model_uri) and not self.replace:
            return joblib.load(cluster_model_uri)
        else:
            cluster_model = GaussianMixture(n_components=10, random_state=0)
            prediction = cluster_model.fit_predict(users_encoding)
            joblib.dump(cluster_model, cluster_model_uri)
            return prediction

    def get_users_encoding(self, ratings_df: pd.DataFrame) -> np.ndarray:

        if self.encoder_uri:
            print("encoding with encoder (random linear mapping)")
            encoder = self.get_encoder(256)
            data_generator = PivotDataGenerator(ratings_df,
                                                len_unique_columns=len(ratings_df[self.item_col].unique()),
                                                rows_col=self.user_col,
                                                cols_col=self.item_col,
                                                values_col=self.rating_col,
                                                batch_size=64)

            return encoder.predict(data_generator)
        else:
            print("encoding with pivot table (sparse data)")
            return ratings_df.pivot_table(index=self.user_col,
                                          columns=self.item_col,
                                          values=self.rating_col).fillna(0).values

    def build(self, ratings_df: pd.DataFrame):

        users_encoding = self.get_users_encoding(ratings_df)
        pred_clusters = self.get_model(users_encoding)

        clusters = pd.DataFrame({
            "clusterId": pred_clusters,
            self.user_col: ratings_df[self.user_col].unique()
            })
        self.user_cluster_df = clusters.set_index(self.user_col)

        return self

    def get_user_group(self, user_id: int) -> list:

        cluster_id = int(self.user_cluster_df.loc[user_id, "clusterId"])
        return self.user_cluster_df[self.user_cluster_df["clusterId"] == cluster_id].index

    def set_clusters(self, ratings_df: pd.DataFrame):

        for user_id in ratings_df[self.user_col].unique():
            ratings_df.loc[ratings_df[self.user_col] == user_id, 'cluster'] = self.user_cluster_df.loc[user_id, 'clusterId']

        return ratings_df
