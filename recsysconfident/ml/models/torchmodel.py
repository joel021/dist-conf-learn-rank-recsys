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
        self.ema_initialized = False

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

    @torch.no_grad()
    def update_ema(self, beta=0.99):
        if not self.ema_initialized:
            self.initialize_ema()
            return

        self.u_emb_ema.mul_(beta).add_((1 - beta) * self.user_emb.weight.data)
        self.i_emb_ema.mul_(beta).add_((1 - beta) * self.item_emb.weight.data)

    def embedding_instability(self, u, i):

        u_ref = self.u_emb_ema[u]
        i_ref = self.i_emb_ema[i]

        v_u = (F.cosine_similarity(self.user_emb(u), u_ref, dim=1) + 1)/2
        v_i = (F.cosine_similarity(self.item_emb(i), i_ref, dim=1) + 1)/2

        return v_u + v_i

    @torch.no_grad()
    def initialize_ema(self):
        self.u_emb_ema.copy_(self.user_emb.weight.data)
        self.i_emb_ema.copy_(self.item_emb.weight.data)
        self.ema_initialized = True
