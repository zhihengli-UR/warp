import torch.nn as nn
from WarpFunction import WarpFunction


class Warp(nn.Module):
    def __init__(self):
        super(Warp, self).__init__()
        self.f = WarpFunction()

    def forward(self, input, flow):
        return self.f(input, flow)
