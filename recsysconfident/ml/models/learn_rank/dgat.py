import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel
from recsysconfident.ml.ranking.rank_helper import get_low_rank_items, bpr_loss


def get_dgat_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = DGAT(
        items_per_user=info.items_per_user,
        n_users=info.n_users,
        n_items=info.n_items,
        emb_dim=64,
        rmin=info.rate_range[0],
        rmax=info.rate_range[1]
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class DGAT(TorchModel):

    def __init__(self, items_per_user, n_users, n_items, emb_dim, rmin: float, rmax: float):
        super().__init__(items_per_user, None, n_users, n_items, emb_dim)

        self.n_users = n_users
        self.n_items = n_items
        self.rmin = rmin
        self.rmax = rmax

        self.ui_gat_layer = GATConv(in_channels=emb_dim,
                                    out_channels=emb_dim,
                                    heads=1,
                                    concat=False
                                    )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)


    def forward(self, users_ids, items_ids):
        ui_edges = torch.stack([users_ids, items_ids + self.n_users])

        ui_x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        ui_graph_emb = self.ui_gat_layer(x=ui_x, edge_index=ui_edges)

        u_graph_emb = ui_graph_emb[users_ids]
        i_graph_emb = ui_graph_emb[items_ids + self.n_users]

        x = F.leaky_relu(self.fc1(torch.cat([u_graph_emb, i_graph_emb], dim=1)))
        x = self.dropout(x)
        mu = self.fc2(x)

        return mu

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

    def predict(self, user_ids, item_ids):
        scores = self.forward(user_ids, item_ids)
        mu = scores[:, 0]
        return mu, self.embedding_instability(user_ids, item_ids)
