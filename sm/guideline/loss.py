import torch.nn as nn

class Loss():
    """
    __init__(self, loss):
    loss = name of loss func, this variable can determine loss function by input string
    """
    
    def __init__(self, loss):
        self.loss =loss

    def loss_function(self):
        if self.loss == '':
            pass

        else:
            return nn.CrossEntropyLoss()