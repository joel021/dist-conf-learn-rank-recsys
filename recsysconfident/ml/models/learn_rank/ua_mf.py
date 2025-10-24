import torch
import torch.nn as nn
import torch.distributions as d

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel
from recsysconfident.ml.ranking.rank_helper import get_low_rank_items


def get_uamf_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = UAMF(
        items_per_user=info.items_per_user,
        n_users=info.n_users,
        n_items=info.n_items,
        emb_dim=64,
        rmin=info.rate_range[0],
        rmax=info.rate_range[1]
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class UAMF(TorchModel):

    def __init__(self, items_per_user, n_users, n_items, emb_dim, rmin: float, rmax: float):
        super().__init__(items_per_user, None, n_items)

        self.n_users = n_users
        self.n_items = n_items
        self.rmin = rmin
        self.rmax = rmax
        n_bins = int(2 * (rmax - rmin))
        self.delta_r = 1 / n_bins

        self.user_factors = nn.Embedding(n_users, emb_dim)  # User Latent Factors (stack multiple in channels)
        self.item_factors = nn.Embedding(n_items + 1, emb_dim)  # Item Latent Factors
        self.user_bias = nn.Embedding(n_users, 1)  # User Bias
        self.item_bias = nn.Embedding(n_items + 1, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias

        # Initialize embeddings
        nn.init.xavier_uniform(self.user_factors.weight)
        nn.init.xavier_uniform(self.item_factors.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # Variance parameters (γ_u, γ_v), initialized to 1.0
        self.user_gamma = nn.Embedding(n_users, 1)
        self.item_gamma = nn.Embedding(n_items, 1)
        nn.init.ones_(self.user_gamma.weight)
        nn.init.ones_(self.item_gamma.weight)

        self.alpha = nn.Parameter(torch.tensor(1.))

    def forward(self, users_ids, items_ids):

        user_embedding = self.user_factors(users_ids)
        item_embedding = self.item_factors(items_ids)
        user_bias = self.user_bias(users_ids).squeeze()
        item_bias = self.item_bias(items_ids).squeeze()

        dot_product = (user_embedding * item_embedding).sum(dim=1)  # Element-wise product, summed over latent factors
        mean = dot_product + user_bias + item_bias + self.global_bias

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
