import torch
import torch.nn as nn
import torch.autograd as autograd
from collections import OrderedDict


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    critic_interpolates = critic(interpolates)
    # fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        inputs=interpolates,
        outputs=critic_interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class MLP(torch.nn.Module):
    def __init__(
        self,
        sizes,
        append_layer_width=None,
        append_layer_position=None,
        batch_norm=False,
        dropout=0.2,
        last_activate="relu",
        dtype=torch.float32,
    ):
        super().__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1], dtype=dtype),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.Dropout(dropout) if s < len(sizes) - 2 else None,
                torch.nn.ReLU() if s < len(sizes) - 2 else None,
            ]

        if append_layer_width is None or append_layer_position == "first":
            if last_activate == "relu":
                layers += [torch.nn.ReLU()]
            elif last_activate == "sigmoid":
                layers += [torch.nn.Sigmoid()]
            elif last_activate == "softmax":
                layers += [torch.nn.Softmax(dim=1)]
        elif append_layer_width is not None and append_layer_position == "last":
            layers += [
                torch.nn.BatchNorm1d(sizes[-1]) if batch_norm else None,
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
            ]

        layers = [layer for layer in layers if layer is not None]

        if append_layer_width:
            assert append_layer_position in ("first", "last")
            if append_layer_position == "first":
                layers_dict = OrderedDict()
                layers_dict["append_linear"] = torch.nn.Linear(
                    append_layer_width, sizes[0]
                )
                if batch_norm:
                    layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[0])

                layers_dict["append_dropout"] = torch.nn.Dropout(dropout)
                layers_dict["append_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                layers_dict = OrderedDict(
                    {str(i): module for i, module in enumerate(layers)}
                )

                layers_dict["append_linear"] = torch.nn.Linear(
                    sizes[-1], append_layer_width
                )
                if last_activate == "relu":
                    layers_dict["append_activate"] = torch.nn.ReLU()
                elif last_activate == "sigmoid":
                    layers_dict["append_activate"] = torch.nn.Sigmoid()
                elif last_activate == "softmax":
                    layers_dict["append_activate"] = torch.nn.Softmax()
        else:
            layers_dict = OrderedDict({str(i): module for i, module in enumerate(layers)})

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, dim: int, batch_norm: bool, dropout=0.2, dtype=torch.float32
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.dropout1(x1)
        x1 = self.relu1(x1)
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.dropout2(x2)
        x2 = self.relu2(x2)
        return x2 + x
