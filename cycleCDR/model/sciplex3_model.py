import torch
from torch import nn
from typing import Dict

from .utils import MLP
from .utils import ResidualBlock


class Encode(nn.Module):
    def __init__(self, config: Dict, dtype=torch.float32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2000, 1024, dtype=dtype),
            nn.BatchNorm1d(1024) if config["encode_batch_norm"] else nn.Identity(),
            nn.Dropout(config["encode_dropout"]),
            nn.ReLU(),
            nn.Linear(1024, 512, dtype=dtype),
            nn.BatchNorm1d(512) if config["encode_batch_norm"] else nn.Identity(),
            nn.Dropout(config["encode_dropout"]),
            nn.ReLU(),
            ResidualBlock(
                512, config["encode_batch_norm"], config["encode_dropout"], dtype
            ),
            nn.Linear(512, 256, dtype=dtype),
            nn.BatchNorm1d(256) if config["encode_batch_norm"] else nn.Identity(),
            nn.Dropout(config["encode_dropout"]),
            nn.ReLU(),
            nn.Linear(256, 128, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decode(nn.Module):
    def __init__(self, config: dict, dtype=torch.float32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(128, 256, dtype=dtype),
            nn.BatchNorm1d(256) if config["decode_batch_norm"] else nn.Identity(),
            nn.Dropout(config["decode_dropout"]),
            nn.ReLU(),
            nn.Linear(256, 512, dtype=dtype),
            nn.BatchNorm1d(512) if config["decode_batch_norm"] else nn.Identity(),
            nn.Dropout(config["decode_dropout"]),
            nn.ReLU(),
            ResidualBlock(
                512, config["decode_batch_norm"], config["decode_dropout"], dtype
            ),
            nn.Linear(512, 1024, dtype=dtype),
            nn.BatchNorm1d(1024) if config["decode_batch_norm"] else nn.Identity(),
            nn.Dropout(config["decode_dropout"]),
            nn.ReLU(),
            nn.Linear(1024, 2000, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def sciplex3_model(config, dtype=torch.float32):
    encoderG_A = Encode(config, dtype)

    decoderG_A = Decode(config, dtype)

    encoderG_B = Encode(config, dtype)

    decoderG_B = Decode(config, dtype)

    discriminator_A = MLP(
        [2000, 1024, 512, 128, 1],
        append_layer_width=None,
        append_layer_position=None,
        batch_norm=config["discriminator_batch_norm"],
        last_activate="sigmoid",
        dropout=config["discriminator_dropout"],
        dtype=dtype,
    )

    discriminator_B = MLP(
        [2000, 1024, 512, 128, 1],
        append_layer_width=None,
        append_layer_position=None,
        batch_norm=config["discriminator_batch_norm"],
        last_activate="sigmoid",
        dropout=config["discriminator_dropout"],
        dtype=dtype,
    )
    return (
        encoderG_A,
        decoderG_A,
        encoderG_B,
        decoderG_B,
        discriminator_A,
        discriminator_B,
    )
