"""Sparse Autoencoder architectures for protein model interpretability.

Implements vanilla, TopK, and gated SAE variants. Based on published
methodologies from Anthropic (Bricken et al.), OpenAI (Gao et al.),
and DeepMind (Rajamanoharan et al.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SAEConfig:
    """Configuration for sparse autoencoder training."""
    input_dim: int              # Model hidden dimension
    expansion_factor: int = 8   # Dictionary size = input_dim * expansion_factor
    k: int = 64                 # TopK: number of active features per input
    architecture: str = "topk"  # "vanilla", "topk", or "gated"
    l1_coeff: float = 1e-3      # L1 sparsity penalty (vanilla only)
    tied_weights: bool = False  # Tie encoder/decoder weights

    @property
    def dict_size(self) -> int:
        return self.input_dim * self.expansion_factor


class VanillaSAE(nn.Module):
    """Standard sparse autoencoder with L1 penalty."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.dict_size)
        self.decoder = nn.Linear(config.dict_size, config.input_dim, bias=True)
        self.encoder_bias = nn.Parameter(torch.zeros(config.dict_size))
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations."""
        x_centered = x - self.pre_bias
        return F.relu(self.encoder(x_centered) + self.encoder_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        return self.decoder(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        l1_loss = z.abs().sum(dim=-1).mean()
        loss = recon_loss + self.config.l1_coeff * l1_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "z": z,
            "x_hat": x_hat,
            "l0": (z > 0).float().sum(dim=-1).mean(),
        }


class TopKSAE(nn.Module):
    """TopK sparse autoencoder (Gao et al., 2024).

    Enforces exactly K active features per input instead of using L1 penalty.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.dict_size, bias=True)
        self.decoder = nn.Linear(config.dict_size, config.input_dim, bias=True)
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with TopK sparsity."""
        x_centered = x - self.pre_bias
        pre_activations = self.encoder(x_centered)

        # Keep only top-K activations
        topk_values, topk_indices = torch.topk(pre_activations, self.config.k, dim=-1)
        z = torch.zeros_like(pre_activations)
        z.scatter_(-1, topk_indices, F.relu(topk_values))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)

        return {
            "loss": recon_loss,
            "recon_loss": recon_loss,
            "l1_loss": torch.tensor(0.0, device=x.device),
            "z": z,
            "x_hat": x_hat,
            "l0": (z > 0).float().sum(dim=-1).mean(),
        }


class GatedSAE(nn.Module):
    """Gated sparse autoencoder (Rajamanoharan et al., 2024).

    Uses separate gating and magnitude pathways for feature activation.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        # Gating pathway
        self.gate_encoder = nn.Linear(config.input_dim, config.dict_size, bias=True)
        # Magnitude pathway
        self.mag_encoder = nn.Linear(config.input_dim, config.dict_size, bias=True)
        # Decoder
        self.decoder = nn.Linear(config.dict_size, config.input_dim, bias=True)
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.gate_encoder.weight)
        nn.init.kaiming_uniform_(self.mag_encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.pre_bias
        gate = torch.sigmoid(self.gate_encoder(x_centered))
        magnitude = F.relu(self.mag_encoder(x_centered))
        return gate * magnitude

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> dict:
        x_centered = x - self.pre_bias
        gate_logits = self.gate_encoder(x_centered)
        gate = torch.sigmoid(gate_logits)
        magnitude = F.relu(self.mag_encoder(x_centered))
        z = gate * magnitude

        x_hat = self.decode(z)
        recon_loss = F.mse_loss(x_hat, x)
        # Sparsity via L1 on gate probabilities
        sparsity_loss = gate.sum(dim=-1).mean()
        loss = recon_loss + self.config.l1_coeff * sparsity_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "l1_loss": sparsity_loss,
            "z": z,
            "x_hat": x_hat,
            "l0": (z > 1e-6).float().sum(dim=-1).mean(),
        }


def build_sae(config: SAEConfig) -> nn.Module:
    """Factory function to build SAE from config."""
    if config.architecture == "vanilla":
        return VanillaSAE(config)
    elif config.architecture == "topk":
        return TopKSAE(config)
    elif config.architecture == "gated":
        return GatedSAE(config)
    else:
        raise ValueError(f"Unknown SAE architecture: {config.architecture}")
