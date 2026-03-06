import scipy.sparse as sp
import numpy as np
import torch

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def get_adj_matrix(df, info: DatasetInfo):

    rows_u = df[info.user_col].values
    cols_i = df[info.item_col].values

    # user -> item edges
    rows = np.concatenate([rows_u, cols_i + info.n_users])
    cols = np.concatenate([cols_i + info.n_users, rows_u])

    data = np.ones(len(rows), dtype=np.float32)

    adj = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(info.n_users + info.n_items, info.n_users + info.n_items)
    )

    return adj

def normalize_adj(adj):
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum!=0)
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def scipy_to_torch_sparse(mat):
    mat = mat.tocoo()
    indices = torch.from_numpy(
        np.vstack((mat.row, mat.col)).astype(np.int64)
    )
    values = torch.from_numpy(mat.data)
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

