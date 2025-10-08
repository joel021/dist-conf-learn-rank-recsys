import torch
import torch.nn as nn
import torch.distributions as d
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel
from recsysconfident.ml.ranking.rank_helper import get_low_rank_items


def get_learn_rank_cpgatbpr_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = CPGATBPR(
        items_per_user=info.items_per_user,
        n_users=info.n_users,
        n_items=info.n_items,
        emb_dim=64,
        rmin=info.rate_range[0],
        rmax=info.rate_range[1]
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class CPGATBPR(TorchModel):

    def __init__(self, items_per_user, n_users, n_items, emb_dim, rmin: float, rmax: float):
        super().__init__(items_per_user)

        self.n_users = n_users
        self.n_items = n_items
        self.rmin = rmin
        self.rmax = rmax
        n_bins = int(2 * (rmax - rmin))
        self.delta_r = 1 / n_bins

        self.ui_lookup = nn.Embedding(n_users + n_items, emb_dim)

        self.ui_gat_layer = GATConv(in_channels=emb_dim,
                                   out_channels=emb_dim,
                                   heads=1,
                                   concat=False
                                   )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)

        # Variance parameters (γ_u, γ_v), initialized to 1.0
        self.user_gamma = nn.Embedding(n_users, 1)
        self.item_gamma = nn.Embedding(n_items, 1)
        nn.init.ones_(self.user_gamma.weight)
        nn.init.ones_(self.item_gamma.weight)

        self.alpha = nn.Parameter(torch.tensor(1.))
        nn.init.xavier_uniform(self.ui_lookup.weight)

    def forward(self, users_ids, items_ids):

        ui_edges = torch.stack([users_ids, items_ids + self.n_users]) #(batch,),(batch,) -> (2, batch)

        ui_x = self.ui_lookup.weight
        ui_graph_emb = self.ui_gat_layer(x=ui_x, edge_index=ui_edges)  # (max_u_id+1, emb_dim)

        u_graph_emb = ui_graph_emb[ui_edges[0]]
        i_graph_emb = ui_graph_emb[ui_edges[1]]

        x = F.leaky_relu(self.fc1(torch.concat([u_graph_emb, i_graph_emb], dim=1)))
        x = self.dropout(x)
        mean = self.fc2(x).squeeze()

        # Softplus ensures γ > 0
        gamma_u = torch.clamp(self.user_gamma(users_ids), min=0.00001) #the article does not mention, but it does not work without.
        gamma_v = torch.clamp(self.item_gamma(items_ids), min=0.00001)
        alpha = torch.exp(self.alpha)

        precision = alpha * gamma_u * gamma_v
        variance = 1.0 / precision
        std = torch.sqrt(variance).squeeze() #=> precision = 1/(std * std) = 1/var

        return torch.stack([mean, std], dim=1)

    def nll_bpr_loss(self, user_ids, item_ids):
        # Positive (high-ranked) item scores: each has mu and sigma
        high_scores = self.forward(user_ids, item_ids)
        high_mu = high_scores[:, 0]
        high_sigma = high_scores[:, 1].clamp(min=1e-6)  # prevent zero variance

        # Negative (low-ranked) item scores
        low_items = get_low_rank_items(user_ids, self.items_per_user, self.n_items)
        low_scores = self.forward(user_ids, low_items.to(user_ids.device))
        low_mu = low_scores[:, 0]
        low_sigma = low_scores[:, 1].clamp(min=1e-6)

        # Compute mean and std of the difference distribution
        diff_mu = high_mu - low_mu
        diff_sigma = torch.sqrt(high_sigma ** 2 + low_sigma ** 2)

        # CDF of difference being > 0 is 1 - CDF(0)
        standard_normal = d.Normal(0, 1)
        prob = standard_normal.cdf(diff_mu / diff_sigma)

        nll = -torch.log(prob + 1e-8).mean()  # BPR loss = -log(P(s_h > s_l))

        return nll

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()

        nll = self.nll_bpr_loss(user_ids, item_ids)
        nll.backward()
        optimizer.step()
        return nll

    def eval_loss(self, user_ids, item_ids, labels):
        nll = self.nll_bpr_loss(user_ids, item_ids)
        return nll

    def regularization(self):
        return 0

    def predict(self, user_ids, item_ids):
        scores = self.forward(user_ids, item_ids)

        mu = scores[:, 0]
        sigma = scores[:, 1]
        standard_normal = d.Normal(0, 1)
        confidence = standard_normal.cdf(mu / sigma)

        return mu, confidence
