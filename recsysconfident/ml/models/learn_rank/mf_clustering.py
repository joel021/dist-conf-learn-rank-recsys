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
                       emb_dim = 128,
                       items = x,
                       x_size = x_size,
                       items_per_user = info.items_per_user)
    return model, fit_dataloader, eval_dataloader, test_dataloader


class ATTCluster(TorchModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, items, x_size: int, items_per_user):
        super(ATTCluster, self).__init__(items_per_user, items, n_items)

        self.emb_dim = emb_dim
        self.n_items = n_items

        # User and Item Embeddings
        self.u_emb = nn.Embedding(n_users, emb_dim)  # User Latent Factors (stack multiple in channels)
        self.i_emb = nn.Embedding(n_items + 1, emb_dim)  # Item Latent Factors
        self.u_bias = nn.Embedding(n_users, 1)  # User Bias
        self.i_bias = nn.Embedding(n_items + 1, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias
        self.w_u = nn.Linear(emb_dim, emb_dim)
        self.w_i = nn.Linear(emb_dim, emb_dim)
        self.w_r = nn.Linear(emb_dim, 1)

        self.dropout1 = nn.Dropout(p=0.25)

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

    def conf_cluster(self, emb_weight, W_emb, idx):
        #W_emb, shape: (emb_dim, emb_dim)
        emb_weight = W_emb(emb_weight)
        norm_embeddings = F.normalize(emb_weight, p=2, dim=1)  # Shape: (n_entities, emb_dim)
        sim_matrix = torch.matmul(norm_embeddings[idx], norm_embeddings.T)  # Shape: (batch_size, n_entities)
        similarity = F.softmax(sim_matrix, dim=1)  # Shape: (batch_size, n_entities)
        att_embeddings = torch.matmul(similarity, emb_weight)  # Shape: (batch_size, emb_dim)

        entropy = -(similarity * torch.log(similarity + 1e-8)).sum(dim=1)
        confidence = 1 - entropy / math.log(similarity.size(1))

        return att_embeddings, confidence

    def cross_cluster(self, e1_emb, e2_emb, idx1):

        x_meta_norm = F.normalize(e2_emb, p=2, dim=1)
        e_embedding_norm = F.normalize(e1_emb, p=2, dim=1)

        sim_matrix = torch.matmul(e_embedding_norm[idx1], x_meta_norm.T) #(batch_size, n_entities)
        attn_weights = torch.softmax(sim_matrix, dim=1)
        sim_emb = torch.matmul(attn_weights, e2_emb)  # Shape: (batch_size, emb_dim)

        return sim_emb

    def forward(self, users, items):

        user_embedding = self.u_emb(users)
        item_embedding = self.i_emb(items)
        user_bias = self.u_bias(users)
        item_bias = self.i_bias(items)

        emb_product = user_embedding * item_embedding

        u_x, c_u = self.learned_cluster(self.u_emb.weight, self.w_u, users)
        i_x, c_i = self.learned_cluster(self.i_emb.weight, self.w_i, items)

        c_ui = torch.sqrt(c_u * c_i).squeeze()
        x = self.w_r(u_x + i_x + emb_product)

        pred = (x.squeeze() + user_bias.squeeze() + item_bias.squeeze() + self.global_bias).squeeze()

        return torch.stack([pred, c_ui], dim=1)

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
