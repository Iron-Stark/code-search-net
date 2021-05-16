import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, inp_embedding_size, out_embedding_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(inp_embedding_size, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, out_embedding_size))
        self.decoder = nn.Sequential(nn.Linear(out_embedding_size, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, inp_embedding_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
