import torch.nn as nn
import torch 

class ExampleNet(nn.Module):
    """
    This Neural Network does nothing! Woohoo!!!!
    """
    def __init__(self):
        super(ExampleNet, self).__init__()

    def forward(self, x):
        return x