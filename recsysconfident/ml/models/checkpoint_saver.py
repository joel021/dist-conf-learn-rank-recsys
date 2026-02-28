"""
ema_checkpoint_saver
"""

import torch
import os


def save_checkpoint(u_emb_ema, i_emb_ema, epoch, path):

    if os.path.isfile(path):
        path = os.path.dirname(path)

    checkpoint = {
        "epoch": epoch,
        "u_emb_ema": u_emb_ema.cpu(),
        "i_emb_ema": i_emb_ema.cpu(),
    }

    torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch.pt"))

def load_checkpoint(model, model_uri, device):

    path = os.path.dirname(model_uri)
    checkpoint_uri = os.path.join(path, f"checkpoint_epoch.pt")

    checkpoint = torch.load(checkpoint_uri, map_location=device)
    model.u_emb_ema.copy_(checkpoint["u_emb_ema"].to(device))
    model.i_emb_ema.copy_(checkpoint["i_emb_ema"].to(device))

    return checkpoint["epoch"]



