from torch import nn
import torch
import torch.nn.functional as F


class TorchModel(nn.Module):

    def __init__(self, items_per_user: dict|None, items, n_users, n_items: int, emb_size: int):
        super(TorchModel, self).__init__()
        self.items = items
        self.items_per_user = items_per_user
        self.n_items = n_items
        self.n_users = n_users

        self.user_emb = nn.Embedding(n_users, emb_size)  # User Latent Factors (stack multiple in channels)
        self.item_emb = nn.Embedding(n_items + 1, emb_size)

        self.register_buffer("u_emb_ema", torch.zeros_like(self.user_emb.weight))
        self.register_buffer("i_emb_ema", torch.zeros_like(self.item_emb.weight))

        nn.init.xavier_uniform(self.user_emb.weight)
        nn.init.xavier_uniform(self.item_emb.weight)

    def regularization(self):
        raise NotImplementedError("This method is not implemented yet")

    def predict(self, user_ids, item_ids):
        raise NotImplementedError("This method is not implemented yet")

    def eval_loss(self, user_ids, item_ids):
        raise NotImplementedError("This method is not implemented yet")

    def loss(self, user_ids, item_ids, optimizer):
        raise NotImplementedError("This method is not implemented yet")
