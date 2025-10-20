import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.ranking.rank_helper import bpr_loss
from recsysconfident.ml.models.torchmodel import TorchModel


def get_learn_rank_att_cluster_and_dl(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    if info.metadata_columns:
        x = torch.from_numpy(info.items_df[info.metadata_columns].values).float()
        x_size = len(info.metadata_columns)
    else:
        x = None
        x_size = 0

    model = ATTCluster(n_users = info.n_users,
                       n_items = info.n_items,
                       emb_dim = 64,
                       items = x,
                       x_size = x_size,
                       items_per_user = info.items_per_user)
    return model, fit_dataloader, eval_dataloader, test_dataloader


class ATTCluster(TorchModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, items, x_size: int, items_per_user):
        super(ATTCluster, self).__init__(items_per_user, items, n_items)

        self.emb_dim = emb_dim

        # User and Item Embeddings
        self.u_emb = nn.Embedding(n_users+1, emb_dim)  # User Latent Factors (stack multiple in channels)
        self.i_emb = nn.Embedding(n_items+1, emb_dim)  # Item Latent Factors
        self.u_bias = nn.Embedding(n_users+1, 1)  # User Bias
        self.i_bias = nn.Embedding(n_items+1, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias
        self.w_x = nn.Linear(x_size, emb_dim)
        self.w_combined = nn.Linear(emb_dim, 1)

        # Initialize embeddings
        nn.init.xavier_uniform(self.u_emb.weight)
        nn.init.xavier_uniform(self.i_emb.weight)
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.i_bias.weight)

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

    def self_cluster(self, x, idx):

        x_norm = F.normalize(x, p=2, dim=1)
        #sim_emb: (batch_size, emb_dim)
        sim_matrix = torch.matmul(x_norm[idx], x_norm.T)  # Shape: (batch_size, n_entities)
        sim_matrix[sim_matrix == 1] = 0 #remove self weights
        similarity = F.sigmoid(sim_matrix)  # Shape: (batch_size, n_entities)
        sim_emb = torch.matmul(similarity, x) # Shape: (batch_size, emb_dim)
        return sim_emb

    def cross_cluster(self, e_embedding, x_meta, idx):

        x_meta_norm = F.normalize(x_meta, p=2, dim=1)
        e_embedding_norm = F.normalize(e_embedding, p=2, dim=1)

        sim_matrix = torch.matmul(e_embedding_norm[idx], x_meta_norm.T) #(batch_size, n_entities)
        sim_matrix[sim_matrix == 1] = 0
        attn_weights = torch.softmax(sim_matrix, dim=1)
        sim_emb = torch.matmul(attn_weights, x_meta)  # Shape: (batch_size, emb_dim)

        return sim_emb

    def forward(self, u_idx, i_idx):

        device = self.u_emb.weight.device
        x = self.items.to(device)

        user_embedding = self.u_emb(u_idx)
        item_embedding = self.i_emb(i_idx)

        emb_product = user_embedding * item_embedding

        x = torch.relu(self.w_x(x))
        u_cluster = self.self_cluster(self.i_emb.weight, i_idx)
        i_cluster = self.self_cluster(self.u_emb.weight, u_idx)

        combined = u_cluster * i_cluster
        pred = self.w_combined(combined)

        return torch.stack([pred, torch.zeros_like(pred)], dim=1)

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
        return self.l2_bias(self.w_x)
