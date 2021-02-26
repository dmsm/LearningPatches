import torch as th
import torch.nn as nn
from torchvision.models import resnet18


class ReconstructionModel(nn.Module):
    """Sketch to patch model that encodes an image and outputs parameters."""

    def __init__(self, output_dim, init=None):
        """Construct the model according to the necessary output dimension."""
        super(ReconstructionModel, self).__init__()

        self.encode = resnet18(num_classes=1024)
        self.decode = nn.Sequential(
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(256, output_dim)

        if init is not None:
            nn.init.zeros_(self.out.weight)
            with th.no_grad():
                self.out.bias.data = init.clone()

    def forward(self, ims):
        """Process one our more images, corresponding to different views."""
        codes = th.stack([self.encode(im.squeeze(1))
                          for im in th.split(ims, 1, dim=1)], dim=-1)
        return self.out(self.decode(codes.max(-1)[0]))
