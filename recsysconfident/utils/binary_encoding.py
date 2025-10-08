import torch

def get_n_bits(max_value: float):
    return int(torch.ceil(torch.log2(torch.tensor(max_value) + 1.0)))

def binary_encoding(int_arr, n_cols: int):

    labels_tensor = torch.tensor(int_arr, dtype=torch.int)
    binary_encoded = ((labels_tensor.unsqueeze(1) & (1 << torch.arange(n_cols))) > 0).float()
    return binary_encoded