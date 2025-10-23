import torch
import torch.nn as nn
from recsysconfident.utils.binary_encoding import get_n_bits

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.ranking.rank_helper import bpr_loss
from recsysconfident.ml.models.torchmodel import TorchModel


def get_dnn_and_dl(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = Dnn(n_users = info.n_users,
                n_items = info.n_items,
                emb_dim = 512,
                items_per_user = info.items_per_user)
    return model, fit_dataloader, eval_dataloader, test_dataloader


class Dnn(TorchModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, items_per_user):
        super(Dnn, self).__init__(items_per_user, None, n_items)

        self.n_users = n_users
        self.emb_dim = emb_dim

        u_cols = get_n_bits(n_users+1)
        i_cols = get_n_bits(n_items+1)
        u_lookup = ((torch.arange(n_users).unsqueeze(1) & (1 << torch.arange(u_cols))) > 0).float()
        i_lookup = ((torch.arange(n_items).unsqueeze(1) & (1 << torch.arange(i_cols))) > 0).float()
        self.register_buffer('u_lookup', u_lookup)
        self.register_buffer('i_lookup', i_lookup)

        self.f_ui = nn.Linear(u_cols + i_cols, emb_dim)
        self.f_2 = nn.Linear(self.emb_dim, emb_dim)
        self.f_r = nn.Linear(emb_dim, 1)

        # Initialize embeddings
        nn.init.xavier_uniform(self.f_ui.weight)
        nn.init.xavier_uniform(self.f_2.weight)
        nn.init.xavier_uniform(self.f_r.weight)

    def forward(self, u_idx, i_idx):
        u_binary_encoded = self.u_lookup[u_idx]
        i_binary_encoded = self.i_lookup[i_idx]

        ui_binary_encoded = torch.concat([u_binary_encoded, i_binary_encoded], dim=1)
        f_ui = torch.relu(self.f_ui(ui_binary_encoded))
        f_2 = torch.relu(self.f_2(f_ui))
        pred = self.f_r(f_2)

        return torch.stack([pred, torch.zeros_like(pred)], dim=1)

    def l2(self, layer):
        l2_loss = torch.norm(layer.weight, p=2) ** 2  # L2 norm squared for weights
        return l2_loss

    def l2_bias(self, layer):
        l2_loss = self.l2(layer)
        l2_loss += torch.norm(layer.bias, p=2) ** 2
        return l2_loss

    def l1(self, layer):
        l_loss = torch.sum(torch.abs(layer.weight))  # L1 norm (sum of absolute values)
        return l_loss

    def l1_bias(self, layer):

        l1 = self.l1(layer)
        l1 += torch.sum(torch.abs(layer.bias))
        return l1

    def predict(self, users_ids, items_ids):
        scores = self.forward(users_ids, items_ids)
        return scores[:,0], scores[:,0]

    def loss(self, user_ids, item_ids, optimizer):
        optimizer.zero_grad()
        loss = bpr_loss(self, user_ids, item_ids) + self.regularization() * 0.0001
        loss.backward()
        optimizer.step()
        return loss

    def eval_loss(self, user_ids, item_ids):
        loss = bpr_loss(self, user_ids, item_ids)
        return loss

    def regularization(self):
        return 0
