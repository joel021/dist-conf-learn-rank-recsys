import torch
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.ranking.rank_helper import bpr_loss


def get_MCDropoutRecModel_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = MCDropoutRecModel(
        items_per_user=info.items_per_user,
        n_users=info.n_users,
        n_items=info.n_items,
        emb_dim=64,
        hidden_dim=64
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader

class MCDropoutRecModel(nn.Module):

    def __init__(
        self,
        n_users,
        n_items,
        items_per_user=None,
        emb_dim=64,
        hidden_dim=64,
        dropout=0.2,
        l2_reg=1e-6,
        mc_samples=50,
    ):
        super(MCDropoutRecModel, self).__init__()
        self.items_per_user = items_per_user
        self.n_users = n_users
        self.n_items = n_items

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.fc1 = nn.Linear(2 * emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

        self.l2_reg = l2_reg
        self.mc_samples = mc_samples

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)

        x = torch.cat([u, i], dim=-1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        pred = self.out(x).squeeze(-1)
        return torch.stack([pred, torch.zeros_like(pred)], dim=1)
 
    def regularization(self):
        reg = 0.0
        for param in self.parameters():
            reg += torch.sum(param ** 2)
        return self.l2_reg * reg

    def _enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict(self, user_ids, item_ids):
        """
        Returns:
            mean: [B] (denormalized)
            certainty: [B]
        """
        self._enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self.forward(user_ids, item_ids)[:, 0])

        preds = torch.stack(preds, dim=0)

        mean = preds.mean(dim=0)
        std = preds.std(dim=0)

        certainty = 1.0 / (std + 1.0)

        return mean, certainty
    
    def eval_loss(self, user_ids, item_ids):
        """
        Deterministic evaluation (normalized space)
        """

        bprloss = bpr_loss(self, user_ids, item_ids)

        return bprloss
    
    def loss(self, user_ids, item_ids, optimizer):

        optimizer.zero_grad()
        bprloss = bpr_loss(self, user_ids, item_ids)
        reg = self.regularization()

        loss = bprloss + reg

        loss.backward()
        optimizer.step()

        return loss
