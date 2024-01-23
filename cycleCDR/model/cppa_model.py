import torch
from .utils import MLP


def cppa_model(config, dtype=torch.float32):
    encoderG_A = MLP(
        [538, 512, 512, 128],
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    decoderG_A = MLP(
        [128, 512, 512, 538],
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    encoderG_B = MLP(
        [538, 512, 512, 128],
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    decoderG_B = MLP(
        [128, 512, 512, 538],
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    discriminator_A = MLP(
        [538, 512, 128, 1],
        batch_norm=config["batch_norm"],
        last_activate="sigmoid",
        dtype=dtype,
    )

    discriminator_B = MLP(
        [538, 512, 128, 1],
        batch_norm=config["batch_norm"],
        last_activate="sigmoid",
        dtype=dtype,
    )
    # for idx, m in enumerate(self.discriminator_B.modules()):
    #     print(idx, '->', m)
    # exit()

    return (
        encoderG_A,
        decoderG_A,
        encoderG_B,
        decoderG_B,
        discriminator_A,
        discriminator_B,
    )
