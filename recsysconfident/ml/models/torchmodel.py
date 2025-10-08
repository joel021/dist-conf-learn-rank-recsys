from torch import nn

class TorchModel(nn.Module):

    def __init__(self, items_per_user: dict|None):
        super(TorchModel, self).__init__()
        self.items_per_user = items_per_user

    def regularization(self):
        raise NotImplementedError("This method is not implemented yet")

    def predict(self, data):
        raise NotImplementedError("This method is not implemented yet")

    def eval_loss(self, data):
        raise NotImplementedError("This method is not implemented yet")

    def loss(self, data, optimizer):
        raise NotImplementedError("This method is not implemented yet")
