from threading import Thread

import torch
import numpy as np


class EarlyStopping:
    """Early stopping to stop training when the validation loss doesn't improve after a certain patience."""
    def __init__(self, patience=8, verbose=True, path='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.path = path

    def stop(self, val_loss, model) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.counter >= self.patience

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

