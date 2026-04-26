import torch
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_multivae_m_dl(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = MultVAERecModel(
        items_per_user=info.items_per_user,
        n_users=info.n_users,
        n_items=info.n_items,
        hidden_dim=64,
        latent_dim=64
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class MultVAERecModel(TorchModel):

    def __init__(
        self,
        items_per_user,
        n_users,
        n_items,
        hidden_dim=600,
        latent_dim=200,
        dropout=0.5,
        beta=0.2,
        mc_samples=50
    ):
        super().__init__(items_per_user=items_per_user, items=None, n_users=n_users, n_items=n_items, emb_size=hidden_dim)

        self.n_items = n_items
        self.beta = beta
        self.mc_samples = mc_samples

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_items),
        )

    def encode(self, x):

        x = x / (x.sum(dim=1, keepdim=True) + 1e-8)

        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def forward(self, x):
        """
        x: [B, n_items] multi-hot user interaction vector
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def _kl_loss(self, mu, logvar):
        # KL(q(z|x) || p(z)) with p(z)=N(0,1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def regularization(self, mu=None, logvar=None):
        if mu is None or logvar is None:
            return 0.0
        return torch.mean(self._kl_loss(mu, logvar))
    
    def _build_x(self, users):
        
        device = next(self.parameters()).device
        batch_size = len(users)

        x = torch.zeros(batch_size, self.n_items, device=device)

        for i, u in enumerate(users):
            user_items = self.items_per_user.get(int(u), ([], None))[0]
            if len(user_items) > 0:
                x[i, list(user_items)] = 1.0
        
        return x

    def predict(self, users, items):

        x = self._build_x(users)

        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits, _, _ = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                preds.append(probs.unsqueeze(0))

        preds = torch.cat(preds, dim=0)  # [S, B, n_items]

        mean = preds.mean(dim=0)
        std = preds.std(dim=0)

        idx = torch.arange(len(items))
        mean_ = mean[idx, items]
        std_ = std[idx, items]

        certainty = 1 / (1.0 + std_)
        return mean_, certainty

    def eval_loss(self, user_ids, item_ids):
        """
        labels: same shape as x (multi-hot or counts)
        Deterministic (use mean of posterior)
        """
        x = self._build_x(user_ids)

        with torch.no_grad():
            mu, logvar = self.encode(x)
            logits = self.decode(mu)

            log_softmax = F.log_softmax(logits, dim=-1)

            recon_loss = -torch.sum(log_softmax * x, dim=1)

            kl = self._kl_loss(mu, logvar)

            loss = torch.mean(recon_loss + self.beta * kl)

        return loss

    def loss(self, user_ids, item_ids, optimizer):

        self.train()
        x = self._build_x(user_ids)

        logits, mu, logvar = self.forward(x)

        log_softmax = F.log_softmax(logits, dim=-1)

        recon_loss = -torch.sum(log_softmax * x, dim=1)

        kl = self._kl_loss(mu, logvar)

        loss = torch.mean(recon_loss + self.beta * kl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
