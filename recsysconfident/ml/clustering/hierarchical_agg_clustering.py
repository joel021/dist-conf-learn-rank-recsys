import tensorflow as tf
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import ClusterNode
from sklearn.cluster import AgglomerativeClustering
from pandas import DataFrame
from typing import List
import time

import numpy as np
import os.path

from recsysconfident.ml.autoencoder import Autoencoder
from recsysconfident.data_handling.pivot_data_generator import PivotDataGenerator
from recsysconfident.ml.losses import MaskedIntensiveMeanSquaredError


class HierarchicalHandler:

    def __init__(self, ratings_df: DataFrame,
                 item_col: str = 'itemId',
                 user_col: str = 'userId',
                 rating_col: str = 'rating',
                 encoder_uri: str = None
                 ):
        self.unique_items = ratings_df[item_col].unique()
        self.unique_users = ratings_df[user_col].unique()
        self.item_col = item_col
        self.user_col = user_col
        self.rating_col = rating_col
        self.encoder_uri = encoder_uri
        self.groups_containing_map: dict = dict()

        t = time.time()
        self.__map_nodes(self.__build_model(ratings_df))
        print(f"tree hierarchical handler built in {time.time() - t} s")

    def __fit_with_pivot(self, cluster_model: AgglomerativeClustering, ratings_df: DataFrame):

        t = time.time()
        user_item_table = ratings_df.pivot_table(index=self.user_col,
                                                 columns=self.item_col,
                                                 values=self.rating_col,
                                                 fill_value=0).reindex(columns=self.unique_items,
                                                                       index=self.unique_users,
                                                                       fill_value=0)
        print(f"Pivot table built in {time.time() - t} s")
        t = time.time()

        cluster_model = cluster_model.fit(user_item_table)
        print(f"Model Built in {time.time() - t} s")

        del user_item_table
        return cluster_model

    def __fit_with_user_encoder(self, cluster_model: AgglomerativeClustering, ratings_df: DataFrame):

        encoder = tf.keras.models.load_model(self.encoder_uri,
                                             custom_objects={'Autoencoder': Autoencoder,
                                                             'MaskedIntensiveMeanSquaredError': MaskedIntensiveMeanSquaredError}
                                             )
        unique_columns = ratings_df[self.item_col].unique()

        data_generator = PivotDataGenerator(ratings_df=ratings_df,
                                            len_unique_columns=unique_columns,
                                            rows_col=self.user_col,
                                            cols_col=self.item_col,
                                            values_col=self.rating_col,
                                            batch_size=32)

        users_encoding = encoder.predict(data_generator)

        t = time.time()
        cluster_model = cluster_model.fit(users_encoding)
        print(f"Model Built in {time.time() - t} s")

        return cluster_model

    def __build_model(self, ratings_df: DataFrame) -> ClusterNode:

        t = time.time()

        cluster_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')

        if self.encoder_uri is not None and os.path.isfile(self.encoder_uri):
            cluster_model = self.__fit_with_user_encoder(cluster_model, ratings_df)
        else:
            cluster_model = self.__fit_with_pivot(cluster_model, ratings_df)
        print(f"Cluster model in {time.time() - t} s")

        t = time.time()
        linkage_matrix = self.build_linkage_matrix(cluster_model)
        print(f"Linkage matrix built in {time.time() - t} s")

        t = time.time()
        rootnode = hierarchy.to_tree(linkage_matrix)
        print(f"tree build in {time.time() - t} s")

        return rootnode

    def get_groups_containing(self, id: int) -> List[List[int]]:
        return self.groups_containing_map[id]

    def __map_nodes(self, root_node: ClusterNode):

        if root_node is None:
            return

        root_node.parent = None
        stack: List[ClusterNode] = []
        stack.append(root_node)

        self.leafs: List[ClusterNode] = []
        while stack:

            node = stack.pop(0)

            if node.left is not None:
                node.left.parent = node
                stack.append(node.left)

            if node.right is not None:
                node.right.parent = node
                stack.append(node.right)

            if node.is_leaf():
                self.leafs.append(node)

        for leaf_node in self.leafs:

            self.groups_containing_map[leaf_node.id] = []
            parent_node = leaf_node.parent

            while parent_node:
                self.groups_containing_map.setdefault(parent_node.id, []).append(leaf_node.id)
                self.groups_containing_map[leaf_node.id].append(self.groups_containing_map[parent_node.id])
                parent_node = parent_node.parent

    def build_linkage_matrix(self, model: AgglomerativeClustering):

        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        return np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)
