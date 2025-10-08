import torch
import torch.nn as nn

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.ml.ranking.rank_helper import bpr_loss
from recsysconfident.ml.models.torchmodel import TorchModel


def get_learn_rank_mf_not_reg_and_dl(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = MFNonRegularizedModel(
        num_users=info.n_users,
        num_items=info.n_items,
        num_factors=64,
        items_per_user=info.items_per_user
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class MFNonRegularizedModel(TorchModel):

    def __init__(self, num_users, num_items, num_factors, items_per_user):
        super(MFNonRegularizedModel, self).__init__(items_per_user)
        self.n_items = num_items
        self.n_users = num_users

        # User and Item Embeddings
        self.user_factors = nn.Embedding(num_users, num_factors)  # User Latent Factors (stack multiple in channels)
        self.item_factors = nn.Embedding(num_items+1, num_factors)  # Item Latent Factors
        self.user_bias = nn.Embedding(num_users, 1)  # User Bias
        self.item_bias = nn.Embedding(num_items+1, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias

        # Initialize embeddings
        nn.init.xavier_uniform(self.user_factors.weight)
        nn.init.xavier_uniform(self.item_factors.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):

        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        user_bias = self.user_bias(user).squeeze()
        item_bias = self.item_bias(item).squeeze()

        dot_product = (user_embedding * item_embedding).sum(dim=1)  # Element-wise product, summed over latent factors
        prediction = dot_product + user_bias + item_bias + self.global_bias

        return torch.stack([prediction.squeeze(), torch.zeros_like(prediction)], dim=1)

    def regularization(self):
        return 0

    def predict(self, user_ids, item_ids):

        scores = self.forward(user_ids, item_ids)
        return scores[:,0], scores[:,1]

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()
        loss = bpr_loss(self, user_ids, item_ids)
        loss.backward()
        optimizer.step()
        return loss

    def eval_loss(self, user_ids, item_ids, labels):

        loss = bpr_loss(self, user_ids, item_ids)
        return loss
