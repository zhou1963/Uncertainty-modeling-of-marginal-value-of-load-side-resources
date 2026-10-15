"""Conditional Variational Autoencoder for marginal value distribution learning.

The encoder maps the marginal value together with the operating condition into
a latent Gaussian; the decoder reconstructs the marginal value from a latent
sample and the same condition.  The reparameterisation trick keeps the sampling
operation differentiable so that the evidence lower bound can be optimised
end-to-end.  The conditional feature vector supplies load, price, weather and
time information, letting the model learn the MV distribution *conditioned on*
the current operating state.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (COND_DIM, LATENT_DIM, HIDDEN_DIM, BATCH_SIZE,
                    LEARNING_RATE, EPOCHS, BETA_KL, SEED)


class Encoder(nn.Module):
    def __init__(self, cond_dim=COND_DIM, latent_dim=LATENT_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, cond):
        h = self.net(torch.cat([x, cond], dim=-1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, cond_dim=COND_DIM, latent_dim=LATENT_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], dim=-1))


class CVAE(nn.Module):
    def __init__(self, cond_dim=COND_DIM, latent_dim=LATENT_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = Encoder(cond_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(cond_dim, latent_dim, hidden_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)
        return recon, mu, logvar


def _standardise_mv(mv: np.ndarray):
    mean = float(mv.mean())
    std = float(mv.std() + 1e-9)
    return mean, std


def train_cvae(mv: np.ndarray, cond: np.ndarray, train_idx, val_idx,
               epochs: int = EPOCHS, seed: int = SEED, verbose: bool = True):
    """Train the CVAE and return the model plus the training history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mv_mean, mv_std = _standardise_mv(mv)
    mv_n = (mv - mv_mean) / mv_std

    x_tr = torch.tensor(mv_n[train_idx], dtype=torch.float32).unsqueeze(-1)
    c_tr = torch.tensor(cond[train_idx], dtype=torch.float32)
    x_va = torch.tensor(mv_n[val_idx], dtype=torch.float32).unsqueeze(-1)
    c_va = torch.tensor(cond[val_idx], dtype=torch.float32)

    model = CVAE()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n = x_tr.shape[0]

    history = {"recon": [], "kl": [], "total": [], "val_recon": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_recon = ep_kl = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, cb = x_tr[idx], c_tr[idx]
            recon, mu, logvar = model(xb, cb)
            recon_loss = F.mse_loss(recon, xb, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + BETA_KL * kl
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_recon += recon_loss.item() * len(idx)
            ep_kl += kl.item() * len(idx)
        ep_recon /= n
        ep_kl /= n

        model.eval()
        with torch.no_grad():
            recon_v, _, _ = model(x_va, c_va)
            val_recon = F.mse_loss(recon_v, x_va).item()

        history["recon"].append(ep_recon)
        history["kl"].append(BETA_KL * ep_kl)
        history["total"].append(ep_recon + BETA_KL * ep_kl)
        history["val_recon"].append(val_recon)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"[cvae] epoch {epoch + 1:3d}/{epochs}  "
                  f"recon={ep_recon:.4f}  kl={BETA_KL * ep_kl:.4f}  "
                  f"total={ep_recon + BETA_KL * ep_kl:.4f}")

    for k in history:
        history[k] = np.array(history[k])
    return model, history, mv_mean, mv_std


def reconstruct(model: CVAE, mv: np.ndarray, cond: np.ndarray,
                mv_mean: float, mv_std: float, seed: int = SEED):
    """Conditional reconstruction of the marginal value.

    For every operating condition a latent code is drawn from the prior and
    decoded together with the condition, giving the model-implied marginal
    value.  The gap between this reconstruction and the observed value reflects
    how much of the MV variability the conditional generator can explain.
    """
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        c = torch.tensor(cond, dtype=torch.float32)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        z = torch.randn(c.shape[0], model.latent_dim, dtype=torch.float32)
        out = model.decoder(z, c).squeeze(-1).numpy()
    return out * mv_std + mv_mean


def generate_scenarios(model: CVAE, cond: np.ndarray, mv_mean: float,
                       mv_std: float, n_samples: int = 1, seed: int = SEED):
    """Sample MV scenarios conditioned on the operating state."""
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        c = torch.tensor(cond, dtype=torch.float32)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        z = torch.tensor(rng.standard_normal((n_samples, c.shape[0],
                                              LATENT_DIM)), dtype=torch.float32)
        out = []
        for s in range(n_samples):
            sample = model.decoder(z[s], c).squeeze(-1).numpy()
            out.append(sample * mv_std + mv_mean)
    return np.array(out)
