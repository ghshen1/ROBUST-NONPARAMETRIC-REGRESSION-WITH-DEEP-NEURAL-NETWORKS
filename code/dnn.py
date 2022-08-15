import torch
from torch import nn

class DNN(torch.nn.Module):
    """
    width_vec: Please input a list object specifying the width of each layer. \n
    """
    def __init__(self, width_vec: list = None):
        super(DNN, self).__init__()
        self.width_vec= width_vec

        modules = []
        if width_vec is None:
            width_vec = [1,256, 256, 256,1] # Default network shape

        # Network
        for i in range(len(width_vec) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(width_vec[i],width_vec[i+1]),
                    nn.ReLU()))

        self.net = nn.Sequential(*modules,
                                 nn.Linear(width_vec[-2],width_vec[-1]))

    def forward(self, input):
        output = self.net(input)
        return  output

