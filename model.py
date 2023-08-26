import torch
import torch.nn as nn


class NET(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,3) # TODO: --> skip gram to high Dim
        )

    def foward(self,x):
        return self.net(x)
