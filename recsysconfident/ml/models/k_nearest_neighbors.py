import torch
import torch.nn as nn
import pandas as pd

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def get_knn_cosine_basic(info: DatasetInfo):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)
    
    model = SparseKNNRecommender(
        train_df=info.fit_df,
        user_col=info.user_col,
        item_col=info.item_col,
        rating_col=info.relevance_col,
        n_users=info.n_users,
        n_items=info.n_items,
        metric='cosine',
        estimator='basic',
        device=device
    )

    return model, [], eval_dataloader, test_dataloader

def get_knn_pearson_baseline_basic(info: DatasetInfo, fold):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = SparseKNNRecommender(
        train_df=info.fit_df,
        user_col=info.user_col,
        item_col=info.item_col,
        rating_col=info.relevance_col,
        n_users=info.n_users,
        n_items=info.n_items,
        metric='pearson_baseline',
        estimator='baseline',
        device=device
    )

    return model, [], eval_dataloader, test_dataloader


class SparseKNNRecommender():

    def __init__(self, train_df,
        user_col,
        item_col,
        rating_col,
        n_users,
        n_items,
        k=40,
        metric="cosine",
        estimator="baseline",
        shrinkage=100,
        device="cpu",
        chunk_size=512):

        self.estimator = estimator
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.metric = metric
        self.shrinkage = shrinkage
        self.device = device
        self.chunk_size = chunk_size

        u_ids = torch.from_numpy(train_df[user_col].values).int()
        i_ids = torch.from_numpy(train_df[item_col].values).int()
        vals = torch.from_numpy(train_df[rating_col].values).float()

        indices = torch.stack([u_ids, i_ids])
        self.R = torch.sparse_coo_tensor(
            indices, vals, (n_users, n_items)
        ).coalesce().to(device)

        self.R_dense = self.R.to_dense()

        self.global_mean = vals.mean()

        user_counts = (self.R_dense > 0).sum(dim=1)
        user_sums = self.R_dense.sum(dim=1)
        self.user_means = user_sums / (user_counts + 1e-9)

        user_var = ((self.R_dense - self.user_means.unsqueeze(1))**2 * (self.R_dense > 0)).sum(dim=1) / (user_counts + 1e-9)
        self.user_stds = torch.sqrt(user_var) + 1e-9

        item_counts = (self.R_dense > 0).sum(dim=0)
        item_sums = self.R_dense.sum(dim=0)
        self.item_means = item_sums / (item_counts + 1e-9)

        self.sim = self._compute_similarity_matrix()

        self.sim.fill_diagonal_(0)

        self.topk_sim, self.topk_idx = torch.topk(self.sim, k=self.k, dim=1)

    def _compute_similarity_matrix(self):

        if self.metric == "pearson":
            X = self.R_dense
            mask = X > 0
            X_centered = (X - self.user_means.unsqueeze(1)) * mask

            norms = torch.norm(X_centered, dim=1, keepdim=True)
            Xn = X_centered / (norms + 1e-9)

            return Xn @ Xn.T

        elif self.metric == "msd":
            return self._pairwise_msd()

        elif self.metric == "pearson_baseline":
            return self._pairwise_pearson_baseline()

        else: #cosine
            X = self.R_dense
            norms = torch.norm(X, dim=1, keepdim=True)
            Xn = X / (norms + 1e-9)
            return Xn @ Xn.T

    def _pairwise_msd(self):
        X = self.R_dense
        mask = (X > 0).float()

        n = X.size(0)
        sim = torch.zeros((n, n), device=self.device)

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)

            Xi = X[start:end]              # (C, I)
            Mi = mask[start:end]

            diff = Xi.unsqueeze(1) - X.unsqueeze(0)      # (C, N, I)
            common = Mi.unsqueeze(1) * mask.unsqueeze(0)

            msd = (diff**2 * common).sum(dim=2) / (common.sum(dim=2) + 1e-9)

            sim[start:end] = 1 / (msd + 1)

        return sim

    def _pairwise_pearson_baseline(self):
        X = self.R_dense
        mask = (X > 0).float()
        mu = self.global_mean

        bu = self.user_means - mu
        bi = self.item_means - mu

        baseline = mu + bu.unsqueeze(1) + bi.unsqueeze(0)

        Xc = (X - baseline) * mask

        n = X.size(0)
        sim = torch.zeros((n, n), device=self.device)

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)

            Xi = Xc[start:end]
            Mi = mask[start:end]

            num = (Xi.unsqueeze(1) * Xc.unsqueeze(0)).sum(dim=2)

            den = torch.sqrt((Xi**2).sum(dim=1, keepdim=True)) * \
                  torch.sqrt((Xc**2).sum(dim=1)).unsqueeze(0)

            rho = num / (den + 1e-9)

            n_common = (Mi.unsqueeze(1) * mask.unsqueeze(0)).sum(dim=2)

            shrink = (n_common - 1) / (n_common - 1 + self.shrinkage)

            sim[start:end] = shrink * rho

        return sim

    def predict(self, u_ids, i_ids):
        u_ids = u_ids.to(self.device)
        i_ids = i_ids.to(self.device)

        neigh_idx = self.topk_idx[u_ids]        # (B, K)
        neigh_sim = self.topk_sim[u_ids]        # (B, K)

        neigh_ratings = self.R_dense[neigh_idx, i_ids.unsqueeze(1)]  # (B, K)

        mask = neigh_ratings > 0

        sims = neigh_sim * mask
        ratings = neigh_ratings * mask

        sim_sum = sims.abs().sum(dim=1) + 1e-9

        if self.estimator == "basic":
            pred = (sims * ratings).sum(dim=1) / sim_sum

        elif self.estimator == "means":
            mu_u = self.user_means[u_ids]  # (B,)
            mu_v = self.user_means[neigh_idx]  # (B, K)

            pred = mu_u + (sims * (ratings - mu_v)).sum(dim=1) / sim_sum

        elif self.estimator == "zscore":
            mu_u = self.user_means[u_ids]
            std_u = self.user_stds[u_ids]

            mu_v = self.user_means[neigh_idx]
            std_v = self.user_stds[neigh_idx]

            normalized = (ratings - mu_v) / std_v

            pred = mu_u + std_u * (sims * normalized).sum(dim=1) / sim_sum

        else: #baseline
            mu = self.global_mean

            b_u = self.user_means[u_ids] - mu
            b_i = self.item_means[i_ids] - mu

            b_ui = mu + b_u + b_i  # (B,)

            b_v = self.user_means[neigh_idx] - mu
            b_vi = mu + b_v + b_i.unsqueeze(1)

            pred = b_ui + (sims * (ratings - b_vi)).sum(dim=1) / sim_sum

        user_mean = self.user_means[u_ids]
        pred = torch.where(sim_sum < 1e-8, user_mean, pred)

        std = torch.std(ratings)
        certainty = 1.0 / (std + 1.0)

        return pred, certainty

    def loss(self, *args):
        return nn.MSELoss()(torch.tensor([0]), torch.tensor([0]))

    def eval_loss(self, *args):
        return self.loss()

    def eval(self):
        return self
    
    def train(self, mode):
        return self
    
    def to(self, device):
        return self

    def train_method(self, **args):

        return {}

