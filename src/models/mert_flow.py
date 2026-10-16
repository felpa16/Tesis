"""MERT-only detection baseline: layer mixes of time-pooled MERT straight into a flow.

The ablation of the whole representation-learning stage. There are no encoders,
no bottlenecks and no content/style objectives: each window's 25 MERT hidden
states are mean-pooled over time, mixed by one softmax layer-weight vector
(25 weights; scripts/train_mert_flow.py --mix picks which), standardized, and
modelled by one unconditional neural spline flow, log p(x) of human music. A full-covariance
Gaussian on the same features is the closed-form baseline.

Time pooling commutes with the layer mix (both are linear), so the per-layer
pooled features (B, 25, 1024) are cached once and any weight vector is applied
afterwards, exactly. That is what lets the layer weights train against a
cache instead of running MERT online.

Why the features are standardized by the mix's *exact* moments. Maximum
likelihood over a learnable, non-invertible feature map rewards whichever map
has the lowest entropy: a smaller scale is worth d * log(scale) nats, and
redundant (collinear) dimensions are worth -0.5 * log det(correlation). Neither
has anything to do with music. `LayerCovariance` holds the training-set mean
and covariance of all 25 pooled layers jointly, which gives the mean and
covariance of any mix in closed form. Three things follow:

  * fixed per-layer normalization (unfrozen arm): every layer enters the mix at
    unit per-dimension scale, so no layer wins by being small
  * per-dimension standardization by the mix's exact std, as a differentiable
    function of the weights: averaging layers can no longer shrink the variance
    (the standardization's Jacobian is deliberately NOT added to log_prob)
  * the correlation matrix R(w) of the standardized features, whose
    -0.5 * log det R / F is precisely the Gaussian part of the likelihood's
    reward for redundancy. Adding it as a penalty cancels that reward, so the
    weights can only gain likelihood from structure beyond second order.

log_prob values are therefore densities of the standardized features, not of
the raw mixes. Ranking songs is all detection needs, and the frozen arm's
standardization is a constant affine map, so this costs nothing there.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager

import torch
from torch import nn

from src.config import NsfConfig

STREAMS = ("content", "style")


@torch.no_grad()
def pool_layers(
    mert: nn.Module,
    waves: torch.Tensor,
    micro_batch: int,
    autocast: AbstractContextManager,
) -> torch.Tensor:
    """Waveforms (B, L) -> time-mean-pooled hidden states (B, 25, 1024), fp32.

    The one function both the feature cache and MoM scoring call, so cached
    training windows and scored windows are pooled identically.
    """
    chunks = []
    for chunk in waves.split(micro_batch):
        with autocast:
            hidden = mert(chunk)  # (b, 25, N, 1024)
        chunks.append(hidden.mean(dim=2, dtype=torch.float32))
    return torch.cat(chunks)


def load_mix_logits(path) -> dict[str, torch.Tensor]:
    """The two layer-weight logit vectors from a phase-1 training checkpoint
    (`model.{content,style}_mix.weights`) or a phase1_layer_weights.pt file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state = checkpoint["model"]
        return {s: state[f"{s}_mix.weights"].detach().float().clone() for s in STREAMS}
    if "content_logits" in checkpoint:
        return {s: checkpoint[f"{s}_logits"].detach().float().clone() for s in STREAMS}
    raise SystemExit(f"{path}: holds neither a model state dict nor layer-weight logits")


class LayerCovariance:
    """Training-set moments of the pooled layers: mean (L, D), covariance (L, D, L, D).

    Training-only: at full size the covariance is 25*1024 squared fp32, 2.6 GB,
    so it never goes into a checkpoint. The model keeps only the moments of
    the final mix. With `layer_norm`, the covariance is converted in place to
    that of the per-layer standardized features, whose mean is zero.
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, layer_norm: bool) -> None:
        n_layers, dim = mean.shape
        self.layer_mean = mean
        self.layer_std = (
            cov.reshape(n_layers * dim, n_layers * dim).diagonal().reshape(n_layers, dim)
        ).clamp_min(1e-12).sqrt()
        self.layer_norm = layer_norm
        if layer_norm:
            cov.div_(self.layer_std[:, :, None, None]).div_(self.layer_std[None, None])
        self.cov = cov

    def mix_mean(self, weights: torch.Tensor) -> torch.Tensor:
        if self.layer_norm:
            return torch.zeros_like(self.layer_mean[0])
        return weights @ self.layer_mean

    def moments(self, weights: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Weights per stream -> mean (F,) and covariance (F, F) of the
        concatenated mixes, differentiable in the weights."""
        partial = [torch.einsum("ldke,k->lde", self.cov, w) for w in weights]
        rows = [
            torch.cat([torch.einsum("l,lde->de", wi, pj) for pj in partial], dim=1)
            for wi in weights
        ]
        cov = torch.cat(rows, dim=0)
        cov = 0.5 * (cov + cov.T)
        mean = torch.cat([self.mix_mean(w) for w in weights])
        return mean, cov


def correlation(cov: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    std = cov.diagonal().clamp_min(eps).sqrt()
    return cov / std[:, None] / std[None, :], std


def redundancy_penalty(corr: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """-0.5 * log det R / F >= 0: the likelihood's Gaussian-level reward for
    correlated (redundant) standardized features, in nats per dimension."""
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    chol = torch.linalg.cholesky(corr + eps * eye)
    return -chol.diagonal().log().sum() / corr.shape[0]


def kl_to_uniform(weights: torch.Tensor) -> torch.Tensor:
    """KL(w || uniform) in nats; 0 at uniform, log(L) for a one-hot vector."""
    return (weights * (weights.clamp_min(1e-12) * weights.numel()).log()).sum()


class MertFlow(nn.Module):
    """Layer mix(es) of pooled MERT -> standardized features -> NSF + Gaussian.

    The baseline uses a single stream; several streams are concatenated, which
    the module still supports but the training script no longer exposes.

    The logits are always nn.Parameters (so the 50 weights travel in the
    state dict); the frozen arm simply builds them with requires_grad=False.
    feat_mean / feat_std / gauss_chol are the moments at the weights the
    checkpoint was saved with; during unfrozen training the live, differentiable
    moments are passed to `features` instead.
    """

    def __init__(
        self,
        n_layers: int,
        dim: int,
        streams: tuple[str, ...],
        nsf: NsfConfig,
        layer_norm: bool,
        trainable: bool,
    ) -> None:
        super().__init__()
        from src.models.flow import build_nsf  # zuko, imported lazily

        self.streams = tuple(streams)
        self.layer_norm = layer_norm
        self.n_features = dim * len(self.streams)
        self.logits = nn.ParameterDict(
            {s: nn.Parameter(torch.zeros(n_layers), requires_grad=trainable) for s in self.streams}
        )
        self.register_buffer("layer_mean", torch.zeros(n_layers, dim))
        self.register_buffer("layer_std", torch.ones(n_layers, dim))
        self.register_buffer("feat_mean", torch.zeros(self.n_features))
        self.register_buffer("feat_std", torch.ones(self.n_features))
        self.register_buffer("gauss_chol", torch.eye(self.n_features))
        self.register_buffer("ready", torch.tensor(False))
        self.flow = build_nsf(self.n_features, 0, nsf)

    def weights(self) -> dict[str, torch.Tensor]:
        return {s: torch.softmax(self.logits[s], dim=0) for s in self.streams}

    def mixes(self, pooled: torch.Tensor) -> torch.Tensor:
        """(B, L, D) pooled layers -> (B, F) concatenated mixes, unstandardized."""
        if self.layer_norm:
            pooled = (pooled - self.layer_mean) / self.layer_std
        weights = self.weights()
        return torch.cat(
            [torch.einsum("bld,l->bd", pooled, weights[s]) for s in self.streams], dim=1
        )

    def features(
        self,
        pooled: torch.Tensor,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = self.feat_mean if mean is None else mean
        std = self.feat_std if std is None else std
        return (self.mixes(pooled) - mean) / std

    def flow_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow().log_prob(z)

    def gaussian_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        y = torch.linalg.solve_triangular(self.gauss_chol, z.T, upper=False)
        log_det = 2.0 * self.gauss_chol.diagonal().log().sum()
        return -0.5 * (self.n_features * math.log(2 * math.pi) + log_det + (y**2).sum(dim=0))

    @torch.no_grad()
    def set_moments(self, mean: torch.Tensor, cov: torch.Tensor, shrinkage: float) -> None:
        """Freeze the standardization and the Gaussian at these mix moments.

        The Gaussian's covariance is the correlation R of the standardized
        features, shrunk toward the identity: (1 - shrinkage) R + shrinkage I.
        """
        corr, std = correlation(cov.detach().float())
        eye = torch.eye(corr.shape[0], device=corr.device)
        self.feat_mean.copy_(mean.detach())
        self.feat_std.copy_(std)
        self.gauss_chol.copy_(torch.linalg.cholesky((1 - shrinkage) * corr + shrinkage * eye))
        self.ready.fill_(True)

    @torch.no_grad()
    def log_probs(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, L, D) pooled layers -> per-window log-densities, each (B,)."""
        if not bool(self.ready):
            raise RuntimeError("MertFlow.set_moments() must run before scoring")
        z = self.features(pooled.float())
        return {"flow": self.flow_log_prob(z), "gaussian": self.gaussian_log_prob(z)}


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def spectrum_stats(eigvals: torch.Tensor) -> dict[str, float]:
    """Eigenvalues of a correlation/covariance matrix -> collapse indicators.

    effective rank = exp(entropy of the normalized spectrum) (Roy & Vetterli);
    participation ratio = (sum l)^2 / sum l^2. Both equal F for a flat spectrum
    and fall toward 1 as variance concentrates in few directions.
    """
    lam = eigvals.double().clamp_min(0).flip(0)  # descending
    p = lam / lam.sum()
    entropy = -(p * p.clamp_min(1e-300).log()).sum()
    return {
        "effective_rank": float(entropy.exp()),
        "participation_ratio": float(lam.sum() ** 2 / (lam**2).sum()),
        "eig_max": float(lam[0]),
        "eig_min": float(lam[-1]),
        "condition": float(lam[0] / lam[-1].clamp_min(1e-12)),
        "top5_share": float(p[:5].sum()),
    }


def weight_stats(w: torch.Tensor, w_init: torch.Tensor) -> dict[str, float]:
    w, w_init = w.detach().double().cpu(), w_init.double().cpu()
    uniform = torch.full_like(w, 1.0 / w.numel())
    top = torch.argsort(w, descending=True)[:3]
    return {
        "kl_uniform": float(kl_to_uniform(w)),
        "max_min_ratio": float(w.max() / w.min()),
        "cos_uniform": float(nn.functional.cosine_similarity(w, uniform, dim=0)),
        "cos_init": float(nn.functional.cosine_similarity(w, w_init, dim=0)),
        "argmax": int(top[0]),
        "top3": [int(i) for i in top],
    }


@torch.no_grad()
def diagnostics(
    model: MertFlow,
    mean: torch.Tensor,
    cov: torch.Tensor,
    init_weights: dict[str, torch.Tensor],
    probe: torch.Tensor | None,
) -> dict:
    """Everything the collapse watch needs, at the model's current weights.

    `mean`, `cov`: exact training-set moments of the unstandardized mixes.
    `probe`: (M, L, D) pooled held-out windows, to see whether the training
    statistics still describe unseen data (M should exceed F).
    Returns {"scalars": {name: float}, "weights": ..., "spectra": ...}.
    """
    dim = model.n_features // len(model.streams)
    corr, std = correlation(cov.detach().float())
    scalars: dict[str, float] = {}
    spectra: dict[str, list[float]] = {}
    weights = {s: w.detach().cpu() for s, w in model.weights().items()}

    eig = torch.linalg.eigvalsh(corr.double())
    spectra["train/joint"] = eig.flip(0).tolist()
    scalars.update({f"train/joint/{k}": v for k, v in spectrum_stats(eig).items()})
    var = std**2
    for i, s in enumerate(model.streams):
        block = slice(i * dim, (i + 1) * dim)
        for k, v in weight_stats(weights[s], init_weights[s]).items():
            if not isinstance(v, list):
                scalars[f"weights/{s}/{k}"] = v
        sub = torch.linalg.eigvalsh(corr[block, block].double())
        spectra[f"train/{s}"] = sub.flip(0).tolist()
        scalars.update({f"train/{s}/{k}": v for k, v in spectrum_stats(sub).items()})
        v = var[block]
        scalars[f"train/{s}/mix_mean_abs"] = float(mean[block].abs().mean())
        scalars[f"train/{s}/mix_var_min"] = float(v.min())
        scalars[f"train/{s}/mix_var_median"] = float(v.median())
        scalars[f"train/{s}/mix_var_max"] = float(v.max())
        scalars[f"train/{s}/mix_expected_norm"] = float((v.sum() + (mean[block] ** 2).sum()).sqrt())
    if len(model.streams) == 2:
        cross = corr[:dim, dim:].diagonal()
        scalars["train/cross_stream_corr_mean"] = float(cross.mean())
        scalars["train/cross_stream_corr_max"] = float(cross.max())
    scalars["train/redundancy_nats_per_dim"] = float(redundancy_penalty(corr))

    if probe is not None and len(probe):
        raw = model.mixes(probe)
        z = (raw - mean) / std
        scalars["probe/mix_norm_mean"] = float(raw.norm(dim=1).mean())
        scalars["probe/z_norm_mean"] = float(z.norm(dim=1).mean())
        scalars["probe/z_norm_expected"] = math.sqrt(model.n_features)
        scalars["probe/z_mean_abs"] = float(z.mean(dim=0).abs().mean())
        z_var = z.var(dim=0)
        scalars["probe/z_var_min"] = float(z_var.min())
        scalars["probe/z_var_median"] = float(z_var.median())
        scalars["probe/z_var_max"] = float(z_var.max())
        centered = z - z.mean(dim=0)
        emp = (centered.T @ centered).double() / max(len(z) - 1, 1)
        eig_p = torch.linalg.eigvalsh(emp)
        spectra["probe/joint"] = eig_p.flip(0).tolist()
        scalars.update({f"probe/joint/{k}": v for k, v in spectrum_stats(eig_p).items()})
    return {
        "scalars": scalars,
        "weights": {s: w.tolist() for s, w in weights.items()},
        "spectra": spectra,
    }
