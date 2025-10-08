import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import ndcg_score, mean_squared_error

from recsysconfident.environment import Environment
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class MatrixFactorizationModel(nn.Module):

    def __init__(self, num_users, num_items, num_factors):
        super(MatrixFactorizationModel, self).__init__()
        # User and Item Embeddings
        self.user_factors = nn.Embedding(num_users, num_factors)  # User Latent Factors (stack multiple in channels)
        self.item_factors = nn.Embedding(num_items+1, num_factors)  # Item Latent Factors
        self.ff1 = nn.Linear(2*num_factors, num_factors)
        self.ff2 = nn.Linear(num_factors, num_factors)
        self.ff3 = nn.Linear(num_factors, 2)

        # Initialize embeddings
        nn.init.xavier_uniform(self.user_factors.weight)
        nn.init.xavier_uniform(self.item_factors.weight)

    def forward(self, user, item):

        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)

        x = torch.concat([user_embedding, item_embedding], axis=1)
        x = torch.relu(self.ff1(x))
        x = torch.relu(self.ff2(x))
        x = torch.sigmoid(self.ff3(x))

        return x

    def l2(self, layer):
        l2_loss = torch.norm(layer.weight, p=2) ** 2  # L2 norm squared for weights
        return l2_loss

    def regularization(self):
        return self.l2(self.user_factors) + self.l2(self.item_factors)


def evaluate_rmse(model, eval_dataloader, device):
    model.eval()
    total_rmse_score = 0.0

    with torch.no_grad():
        for batch in eval_dataloader:
            u_ids, i_ids, labels = batch
            u_ids, i_ids, labels = u_ids.to(device), i_ids.to(device), labels.to(device)

            outputs = model(u_ids, i_ids)
            mu, sigma = outputs[:, 0], outputs[:, 1]
            total_rmse_score += math.sqrt(mean_squared_error(labels.cpu().numpy(), (mu * 4 + 1).cpu().numpy()))

    avg_rmse = total_rmse_score / len(eval_dataloader)
    print(f"Evaluation RMSE: {avg_rmse:.4f}")
    return avg_rmse

def evaluate_ndcg(model, data_info: DatasetInfo, eval_dataloader, device):
    model.eval()
    total_ndcg_score = 0.0

    outputs_list = []

    with torch.no_grad():
        for batch in eval_dataloader:
            u_ids, i_ids, labels = batch
            u_ids, i_ids, labels = u_ids.to(device), i_ids.to(device), labels.to(device)
            m_ndcg_score10 = 0
            n_users = 0

            for uid in torch.unique(u_ids):
                items, ilabels = data_info.items_per_user[int(uid)]
                i_ids_batch, labels_batch = torch.tensor(items).to(device), np.array(ilabels)

                u_ids_batch = (torch.ones_like(i_ids_batch) * uid).to(device)
                outputs = model(u_ids_batch, i_ids_batch)

                mu = outputs[:, 0]
                pred_labels = (mu * 4 + 1).cpu().numpy()

                m_ndcg_score10 += ndcg_score([labels_batch], [pred_labels], k=10)
                n_users += 1

            outputs = model(u_ids, i_ids)

            outputs_list.append(torch.concat([outputs, labels.unsqueeze(1)], axis=1).cpu().numpy())

            m_ndcg_score10 = m_ndcg_score10 / n_users
            total_ndcg_score += m_ndcg_score10

    avg_ndcg = total_ndcg_score / len(eval_dataloader)

    models_outputs = np.concatenate(outputs_list)
    df = pd.DataFrame.from_records(models_outputs, columns=["mu", "sigma", "label"])
    df.to_csv("./results.csv", index=False)
    print(f"Evaluation NDCG: {avg_ndcg:.4f}")
    return avg_ndcg

def periodic_bining(x):
    return x - x.int()

def negative_samples_loss(labels_norm, bce_fn, dist):
    labels_norm_shifted = periodic_bining(labels_norm)  # shifted in 3 * bin_size
    p_labels_shifted = p(labels_norm_shifted, dist)
    negative_loss = bce_fn(p_labels_shifted, torch.zeros_like(p_labels_shifted))
    return negative_loss

def p(x, dist):
    return torch.abs(dist.cdf(x+0.12) - dist.cdf(x-0.12))

def train(model, data_info, dataloader, eval_dl, optimizer, device, epochs):
    model.to(device)
    model.train()

    bce_fn = torch.nn.BCELoss()
    mse_fn = torch.nn.MSELoss()

    b_model, b_score = None, None

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            u_ids, i_ids, labels = batch
            u_ids, i_ids, labels = u_ids.to(device), i_ids.to(device), labels.to(device)

            labels_norm = (labels - 1) / 4

            optimizer.zero_grad()

            outputs = model(u_ids, i_ids) # shape: (batch, 2)
            mu, sigma = outputs[:, 0], outputs[:, 1]
            sigma = torch.clamp(sigma, min=1e-3)

            dist = torch.distributions.Normal(mu, sigma)

            p_labels = p(labels_norm, dist)
            positive_loss = bce_fn(p_labels, torch.ones_like(p_labels))

            negative_loss = negative_samples_loss(labels_norm + 0.25, bce_fn, dist)
            negative_loss += negative_samples_loss(labels_norm + 0.5, bce_fn, dist)
            negative_loss += negative_samples_loss(labels_norm + 0.75, bce_fn, dist)
            bce_loss = positive_loss + negative_loss / 3

            pred_labels = (mu * 4 + 1)
            mse_loss = mse_fn(labels, pred_labels)

            print(f"BCE loss {bce_loss.item()}, mse loss {mse_loss.item()}", end="\r")
            loss = bce_loss + mse_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        eval_avg_loss = evaluate_rmse(model, eval_dl, device)
        if not b_model or b_score < eval_avg_loss:
            b_model, b_score = model, eval_avg_loss
    print(f"Best RMSE: {b_score}")
    eval_ndcg = evaluate_ndcg(b_model, data_info, eval_dl, device)


if __name__ == "__main__":


    environ = Environment(
    model_name="mf",
    database_name="ml-1m",
    instance_dir="../runs/ml-1m-mf-2025-05-02-17-38-43",
        root_path="../"
    )

    _, fit_dl, val_dl, test_dl = environ.get_model_dataloaders(False)
    fit_df, eval_df, test_df = environ.dataset_info.get_splits()

    model = MatrixFactorizationModel(num_users=environ.dataset_info.n_users,
                                 num_items=environ.dataset_info.n_items,
                                 num_factors=256)
    optimizer = Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, environ.dataset_info, fit_dl, val_dl, optimizer, device, epochs=10)

