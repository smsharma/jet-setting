import torch.nn as nn
from models.coordconv import CoordConv2d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, channels=1, kernel_size=3, patch_size=8, n_out=2):
        super(ConvMixer, self).__init__()

        self.net = nn.Sequential(
            CoordConv2d(channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
            # nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, n_out)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)
