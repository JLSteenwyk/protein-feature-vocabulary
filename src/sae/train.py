"""SAE training loop.

Trains sparse autoencoders on pre-extracted model activations.
Supports all architectures from model.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from dataclasses import dataclass

from .model import SAEConfig, build_sae


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    lr: float = 3e-4
    batch_size: int = 4096
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    log_every: int = 100
    save_every: int = 1000
    device: str = "cuda"
    output_dir: str = "models/sae"
    normalize_decoder: bool = True  # Renormalize decoder columns each step


class SAETrainer:
    """Trains a sparse autoencoder on activation data."""

    def __init__(self, sae_config: SAEConfig, train_config: TrainConfig):
        self.sae_config = sae_config
        self.train_config = train_config
        self.sae = build_sae(sae_config).to(train_config.device)
        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        self.step = 0
        self.log_history: list[dict] = []

    def _get_lr_multiplier(self) -> float:
        """Linear warmup then constant."""
        if self.step < self.train_config.warmup_steps:
            return self.step / max(self.train_config.warmup_steps, 1)
        return 1.0

    def train_on_tensor(
        self,
        activations: torch.Tensor,
        val_activations: torch.Tensor | None = None,
    ) -> dict:
        """Train SAE on a tensor of activations.

        Args:
            activations: (N, input_dim) tensor of model activations
            val_activations: optional validation set

        Returns:
            Training summary dict
        """
        dataset = TensorDataset(activations)
        loader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.sae.train()
        best_val_loss = float("inf")

        for epoch in range(self.train_config.num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l0 = 0.0
            num_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.train_config.device)

                # LR warmup
                lr_mult = self._get_lr_multiplier()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.train_config.lr * lr_mult

                output = self.sae(batch)
                loss = output["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Normalize decoder weights to unit norm
                if self.train_config.normalize_decoder:
                    with torch.no_grad():
                        self.sae.decoder.weight.data = nn.functional.normalize(
                            self.sae.decoder.weight.data, dim=0
                        )

                epoch_loss += output["loss"].item()
                epoch_recon += output["recon_loss"].item()
                epoch_l0 += output["l0"].item()
                num_batches += 1
                self.step += 1

                if self.step % self.train_config.log_every == 0:
                    log_entry = {
                        "step": self.step,
                        "epoch": epoch,
                        "loss": output["loss"].item(),
                        "recon_loss": output["recon_loss"].item(),
                        "l1_loss": output["l1_loss"].item(),
                        "l0": output["l0"].item(),
                        "lr": self.train_config.lr * lr_mult,
                    }
                    self.log_history.append(log_entry)

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_recon = epoch_recon / max(num_batches, 1)
            avg_l0 = epoch_l0 / max(num_batches, 1)

            # Compute dead features
            dead_frac = self._compute_dead_features(loader)

            print(
                f"Epoch {epoch+1}/{self.train_config.num_epochs} | "
                f"Loss: {avg_loss:.6f} | Recon: {avg_recon:.6f} | "
                f"L0: {avg_l0:.1f} | Dead: {dead_frac:.3f}"
            )

            # Validation
            if val_activations is not None:
                val_loss = self._validate(val_activations)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save(Path(self.train_config.output_dir) / "best.pt")

        return {
            "final_loss": avg_loss,
            "final_recon": avg_recon,
            "final_l0": avg_l0,
            "dead_frac": dead_frac,
            "best_val_loss": best_val_loss if val_activations is not None else None,
            "total_steps": self.step,
        }

    def _compute_dead_features(self, loader: DataLoader, max_batches: int = 10) -> float:
        """Compute fraction of features that never activate."""
        self.sae.eval()
        ever_active = torch.zeros(self.sae_config.dict_size, device=self.train_config.device)

        with torch.no_grad():
            for i, (batch,) in enumerate(loader):
                if i >= max_batches:
                    break
                batch = batch.to(self.train_config.device)
                z = self.sae.encode(batch)
                ever_active += (z > 0).any(dim=0).float()

        dead_frac = (ever_active == 0).float().mean().item()
        self.sae.train()
        return dead_frac

    def _validate(self, val_activations: torch.Tensor) -> float:
        """Compute validation loss."""
        self.sae.eval()
        with torch.no_grad():
            # Process in chunks to avoid OOM
            val_losses = []
            for i in range(0, len(val_activations), self.train_config.batch_size):
                batch = val_activations[i : i + self.train_config.batch_size].to(
                    self.train_config.device
                )
                output = self.sae(batch)
                val_losses.append(output["recon_loss"].item())
        self.sae.train()
        val_loss = sum(val_losses) / len(val_losses)
        print(f"  Val recon loss: {val_loss:.6f}")
        return val_loss

    def save(self, path: str | Path):
        """Save SAE model and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sae_config": self.sae_config,
                "train_config": self.train_config,
                "model_state_dict": self.sae.state_dict(),
                "step": self.step,
                "log_history": self.log_history,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "SAETrainer":
        """Load a saved SAE trainer."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        train_config = checkpoint["train_config"]
        train_config.device = device
        trainer = cls(checkpoint["sae_config"], train_config)
        trainer.sae.load_state_dict(checkpoint["model_state_dict"])
        trainer.step = checkpoint["step"]
        trainer.log_history = checkpoint["log_history"]
        return trainer
