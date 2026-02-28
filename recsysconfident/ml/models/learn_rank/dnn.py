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
        super(Dnn, self).__init__(items_per_user, None, n_users, n_items, emb_dim)

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        self.f_ui = nn.Linear(2 * emb_dim, emb_dim)
        self.f_2 = nn.Linear(emb_dim, emb_dim)
        self.f_r = nn.Linear(emb_dim, 1)

        nn.init.xavier_uniform_(self.f_ui.weight)
        nn.init.xavier_uniform_(self.f_2.weight)
        nn.init.xavier_uniform_(self.f_r.weight)

    def forward(self, u_idx, i_idx):
        u_feat = self.user_emb(u_idx)
        i_feat = self.item_emb(i_idx)

        ui_feat = torch.cat([u_feat, i_feat], dim=1)
        f_ui = torch.relu(self.f_ui(ui_feat))
        f_2 = torch.relu(self.f_2(f_ui))
        pred = self.f_r(f_2)

        return torch.stack([pred, torch.zeros_like(pred)], dim=1)

    def l2(self, layer):
        return torch.norm(layer.weight, p=2) ** 2

    def l2_bias(self, layer):
        return self.l2(layer) + torch.norm(layer.bias, p=2) ** 2

    def l1(self, layer):
        return torch.sum(torch.abs(layer.weight))

    def l1_bias(self, layer):
        return self.l1(layer) + torch.sum(torch.abs(layer.bias))

    def predict(self, users_ids, items_ids):
        scores = self.forward(users_ids, items_ids)
        return scores[:, 0], self.embedding_instability(users_ids, items_ids)

    def loss(self, user_ids, item_ids, optimizer):
        optimizer.zero_grad()
        loss = bpr_loss(self, user_ids, item_ids) + self.regularization() * 0.0001
        loss.backward()
        optimizer.step()
        return loss

    def eval_loss(self, user_ids, item_ids):
        return bpr_loss(self, user_ids, item_ids)

    def regularization(self):
        return 0