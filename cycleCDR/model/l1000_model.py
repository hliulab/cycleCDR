import torch
from .utils import MLP


def l1000_model(config, dtype=torch.float32):
    encoderG_A = MLP(
        [978, 512, 512, 128],
        append_layer_width=config["append_layer_width"],
        append_layer_position="first",
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    decoderG_A = MLP(
        [128, 512, 512, 978],
        append_layer_width=config["append_layer_width"],
        append_layer_position="last",
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    encoderG_B = MLP(
        [978, 512, 512, 128],
        append_layer_width=config["append_layer_width"],
        append_layer_position="first",
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    decoderG_B = MLP(
        [128, 512, 512, 978],
        append_layer_width=config["append_layer_width"],
        append_layer_position="last",
        batch_norm=config["batch_norm"],
        last_activate=None,
        dtype=dtype,
    )

    discriminator_A = MLP(
        [978, 512, 128, 1],
        append_layer_width=config["append_layer_width"],
        append_layer_position="first",
        batch_norm=config["batch_norm"],
        last_activate="sigmoid",
        dtype=dtype,
    )

    discriminator_B = MLP(
        [978, 512, 128, 1],
        append_layer_width=config["append_layer_width"],
        append_layer_position="first",
        batch_norm=config["batch_norm"],
        last_activate="sigmoid",
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
