import torch

from recsysconfident.environment import Environment
from recsysconfident.ml.fit.early_stopping import EarlyStopping
from recsysconfident.ml.models.torchmodel import TorchModel


def train_model(model: TorchModel, training_loader, validation_loader, environ: Environment, optimizer,
                epochs: int, device, patience=12) -> list:

    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, path=environ.model_uri)

    history = []
    for epoch in range(epochs):

        print('EPOCH {}:'.format(epoch))
        model.train(True)
        avg_loss = train_one_epoch(model, training_loader, optimizer, device)
        avg_vloss = run_val(model, validation_loader, device)

        print('LOSS train loss {} valid loss {}'.format(avg_loss, avg_vloss))
        history.append({
            "epoch": epoch + 1,
            "loss_fit": avg_loss,
            "loss_val": avg_vloss,
        })
        if early_stopping.stop(avg_vloss, model):
            print("Early stopping triggered.")
            break
    model.load_state_dict(torch.load(environ.model_uri, weights_only=True, map_location=device))

    return history

def train_one_epoch(model, training_loader, optimizer, device):
    running_loss = 0.
    model = model.to(device)
    for i, data in enumerate(training_loader):

        users_ids, items_ids, labels = data
        user_ids, item_ids, labels = users_ids.to(device), items_ids.to(device), labels.to(device)
        loss = model.loss(user_ids, item_ids, optimizer)

        running_loss += loss.item()

    return running_loss / len(training_loader)

def run_val(model, validation_loader, device):

    model = model.to(device)
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for data in validation_loader:
            users_ids, items_ids, labels = data
            user_ids, item_ids, labels = users_ids.to(device), items_ids.to(device), labels.to(device)
            vloss = model.eval_loss(user_ids, item_ids)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(validation_loader)
    return avg_vloss

def get_running_and_last_loss(running_loss: int, last_loss: float, loss: float, i: int):

    running_loss += loss
    if i % 1000 == 999:
        last_loss = running_loss / 1000  # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        running_loss = 0.

    if last_loss == .0:
        last_loss = running_loss

    return running_loss, last_loss
