"""Model definitions used during LoRA fine-tuning of the DeepSet encoder."""

import math
from typing import Tuple

import torch
import torch.nn as nn


class DeepSetEncoder(nn.Module):
    """Same architecture as the pretrained encoder."""

    def __init__(self, input_size, encoding_size, n_encoder=2, max_encoder=1024, dropout_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, encoding_size)

    def forward(self, x):  # x: [num_items, input_size] (variable-length set)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        encoded = torch.sum(x, dim=0)  # DeepSets sum aggregation
        return encoded


class NeuralNetwork(nn.Module):
    """Original regression head trained on top of the encoder."""

    def __init__(self, input_size, n_main=2, max_main=1024, dropout_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 1)  # regression head

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class LoRALinear(nn.Module):
    """Low-rank update on a Linear layer: y = Wx + (alpha/r) * A(Bx).

    Only A and B are trained; the wrapped base Linear is frozen.
    """

    def __init__(self, base_linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(self.base.out_features, r))
        self.B = nn.Parameter(torch.zeros(r, self.base.in_features))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.B.t() @ self.A.t())


class DeepSetEncoderWithLoRAAndCls(nn.Module):
    """DeepSet encoder with LoRA adapters on every linear layer plus a per-reaction classification head.

      - Input:        x ∈ [L, in_dim]            (a route with L reactions)
      - Backbone:     fc1..fc5 wrapped with LoRALinear (same shapes as DeepSetEncoder)
      - Aggregation:  encoded = sum(h5, dim=0) ∈ [encoding_size]
      - Classifier:   cls_head(h5) -> [L, num_classes] (per-reaction class logits)

    Only the LoRA parameters (A, B) and cls_head are trained; base Linear weights are frozen.
    """

    def __init__(self, input_size: int, encoding_size: int, num_classes: int = 3,
                 lora_r: int = 8, lora_alpha: int = 16):
        super().__init__()
        # Linear shapes aligned with DeepSetEncoder
        base_fc1 = nn.Linear(input_size, 64)
        base_fc2 = nn.Linear(64, 128)
        base_fc3 = nn.Linear(128, 512)
        base_fc4 = nn.Linear(512, 512)
        base_fc5 = nn.Linear(512, encoding_size)

        # LoRA wrappers (each freezes its base internally)
        self.fc1 = LoRALinear(base_fc1, r=lora_r, alpha=lora_alpha)
        self.fc2 = LoRALinear(base_fc2, r=lora_r, alpha=lora_alpha)
        self.fc3 = LoRALinear(base_fc3, r=lora_r, alpha=lora_alpha)
        self.fc4 = LoRALinear(base_fc4, r=lora_r, alpha=lora_alpha)
        self.fc5 = LoRALinear(base_fc5, r=lora_r, alpha=lora_alpha)

        # Per-reaction classification head: encoding_size -> 128 -> num_classes
        self.cls_head = nn.Sequential(
            nn.Linear(encoding_size, 128),
            nn.Linear(128, num_classes),
        )

    @torch.no_grad()
    def load_from_pretrained(self, pretrained_encoder: DeepSetEncoder):
        """Copy pretrained DeepSetEncoder weights into the LoRA model's frozen base layers."""
        mapping = [
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("fc3", self.fc3),
            ("fc4", self.fc4),
            ("fc5", self.fc5),
        ]
        for name, lora_layer in mapping:
            src = getattr(pretrained_encoder, name)
            lora_layer.base.weight.copy_(src.weight.data)
            if src.bias is not None:
                lora_layer.base.bias.copy_(src.bias.data)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [L, in_dim]
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        h5 = self.fc5(h)                     # [L, encoding_size]
        encoded = torch.sum(h5, dim=0)       # [encoding_size]
        logits_per_step = self.cls_head(h5)  # [L, num_classes]
        return encoded, logits_per_step
