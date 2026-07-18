"""Variational autoencoder for the asymmetric multivariate t mixture model.

Model
-----
For observation i and coordinate j,

    G_ij ~ Gamma(shape=2 / nu_j, rate=2 / nu_j)
    tau_ij = 1 / G_ij
    Y_i | tau_i ~ N(mu + eta * nu * tau_i,
                      D(sqrt(tau_i)) Theta^{-1} D(sqrt(tau_i))).

The exact posterior p(tau_i | Y_i) is not assumed to be available.  The
encoder approximates it with a full-covariance Gaussian in log(tau)-space,
optionally followed by conditional RealNVP coupling layers.  The decoder is
the exact Gaussian likelihood above; it is not a neural network.

Install PyTorch before running this file:

    python -m pip install torch

Example:

    python aat_vae.py --epochs 500 --flow-layers 4
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# DESIGN CHOICE: float64 is slower than float32, but it is safer for Cholesky
# factors, log determinants, and heavy-tailed posterior samples.
torch.set_default_dtype(torch.float64)


LOG_2PI = math.log(2.0 * math.pi)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_softplus(x: Tensor) -> Tensor:
    """Numerically stable inverse of softplus for a strictly positive x."""

    return x + torch.log(-torch.expm1(-x))


def inverse_sigmoid(x: Tensor) -> Tensor:
    """Logit transform for values strictly between zero and one."""

    return torch.log(x) - torch.log1p(-x)


@dataclass
class VAEConfig:
    """All important modeling and optimization choices in one place."""

    hidden_dim: int = 64
    hidden_layers: int = 2
    flow_layers: int = 4
    flow_hidden_dim: int = 64
    max_flow_log_scale: float = 2.0
    min_encoder_scale: float = 1.0e-4
    min_theta_cholesky: float = 1.0e-4

    # DESIGN CHOICE: nu is bounded below for numerical safety and below 2 by
    # default so E[tau_j] exists. Increase nu_max only if the intended model
    # genuinely permits an infinite latent mean.
    nu_min: float = 1.0e-3
    nu_max: float = 1.95
    initial_nu: float = 0.5

    batch_size: int = 128
    epochs: int = 500
    posterior_samples: int = 4
    encoder_steps: int = 2
    model_steps: int = 1
    encoder_lr: float = 1.0e-3
    model_lr: float = 3.0e-3
    weight_decay: float = 0.0
    gradient_clip: float = 10.0

    # ADDED: discard this fraction of early epochs before averaging parameters.
    # A value of 0.5 averages the final half of the training trajectory.
    parameter_average_burn_in: float = 0.5

    # DESIGN CHOICE: set theta_l1 > 0 to estimate a sparse precision matrix.
    # The penalty is applied to off-diagonal entries only.
    theta_l1: float = 0.0
    print_every: int = 25
    seed: int = 42


class MLP(nn.Module):
    """A small MLP used by the encoder and the conditional flow."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            # DESIGN CHOICE: SiLU is smooth, which is useful when differentiating
            # the ELBO through sampled latent variables.
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ConditionalAffineCoupling(nn.Module):
    """One conditional RealNVP affine coupling transformation."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        mask: Tensor,
        max_log_scale: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.max_log_scale = max_log_scale
        self.scale_shift_network = MLP(
            input_dim=latent_dim + context_dim,
            output_dim=2 * latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
        )

        # DESIGN CHOICE: initialize each flow near the identity. This prevents
        # unstable transformations before the base encoder has learned anything.
        final_layer = self.scale_shift_network.network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, u: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform u and return both transformed u and log |det Jacobian|."""

        mask = self.mask
        masked_u = mask * u
        raw_scale, shift = self.scale_shift_network(
            torch.cat((masked_u, context), dim=-1)
        ).chunk(2, dim=-1)

        # DESIGN CHOICE: bound the log scale so exp(log_scale) cannot overflow.
        log_scale = self.max_log_scale * torch.tanh(raw_scale)
        changed = 1.0 - mask
        log_scale = changed * log_scale
        shift = changed * shift

        transformed = masked_u + changed * (u * torch.exp(log_scale) + shift)
        log_abs_det = log_scale.sum(dim=-1)
        return transformed, log_abs_det


class PosteriorEncoder(nn.Module):
    """Approximate q_phi(tau | y) with a reparameterizable conditional flow."""

    def __init__(
        self,
        y_mean: Tensor,
        y_scale: Tensor,
        config: VAEConfig,
    ) -> None:
        super().__init__()
        self.dimension = int(y_mean.numel())
        self.config = config
        self.register_buffer("y_mean", y_mean.clone())
        self.register_buffer("y_scale", y_scale.clone())

        # A p-dimensional full Cholesky factor has p(p+1)/2 free entries.
        self.num_tril_entries = self.dimension * (self.dimension + 1) // 2
        self.context_network = MLP(
            input_dim=self.dimension,
            output_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        )
        self.base_parameter_head = nn.Linear(
            config.hidden_dim,
            self.dimension + self.num_tril_entries,
        )

        rows, cols = torch.tril_indices(self.dimension, self.dimension)
        self.register_buffer("tril_rows", rows)
        self.register_buffer("tril_cols", cols)

        flows: List[nn.Module] = []
        for layer_index in range(config.flow_layers):
            # DESIGN CHOICE: alternating masks ensure that every latent coordinate
            # is transformed by some coupling layers.
            mask = (
                (torch.arange(self.dimension) + layer_index) % 2
            ).to(dtype=torch.get_default_dtype())
            flows.append(
                ConditionalAffineCoupling(
                    latent_dim=self.dimension,
                    context_dim=config.hidden_dim,
                    hidden_dim=config.flow_hidden_dim,
                    mask=mask,
                    max_log_scale=config.max_flow_log_scale,
                )
            )
        self.flows = nn.ModuleList(flows)

    def _base_parameters(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return context, Gaussian mean, and valid full Cholesky factor."""

        standardized_y = (y - self.y_mean) / self.y_scale
        context = self.context_network(standardized_y)
        output = self.base_parameter_head(context)
        mean = output[..., : self.dimension]
        raw_tril = output[..., self.dimension :]

        batch_shape = raw_tril.shape[:-1]
        scale_tril = raw_tril.new_zeros(
            (*batch_shape, self.dimension, self.dimension)
        )
        scale_tril[..., self.tril_rows, self.tril_cols] = raw_tril

        diagonal_index = torch.arange(self.dimension, device=y.device)
        raw_diagonal = scale_tril[..., diagonal_index, diagonal_index]
        positive_diagonal = (
            F.softplus(raw_diagonal) + self.config.min_encoder_scale
        )
        scale_tril = scale_tril.clone()
        scale_tril[..., diagonal_index, diagonal_index] = positive_diagonal
        return context, mean, scale_tril

    def sample_and_log_prob(
        self,
        y: Tensor,
        num_samples: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return tau samples and their log q_phi(tau | y) values.

        Shapes:
            y:             (batch, p)
            tau:           (samples, batch, p)
            log_q_tau:     (samples, batch)
        """

        context, mean, scale_tril = self._base_parameters(y)
        base_distribution = MultivariateNormal(mean, scale_tril=scale_tril)
        u = base_distribution.rsample((num_samples,))
        log_q_u = base_distribution.log_prob(u)

        expanded_context = context.unsqueeze(0).expand(num_samples, -1, -1)
        for flow in self.flows:
            u, log_abs_det = flow(u, expanded_context)
            log_q_u = log_q_u - log_abs_det

        # DESIGN CHOICE: exponentiating the unconstrained sample guarantees tau>0.
        tau = torch.exp(u)

        # Change of variables for tau=exp(u): log q(tau)=log q(u)-sum(log tau).
        log_q_tau = log_q_u - u.sum(dim=-1)
        return tau, log_q_tau

    @torch.no_grad()
    def posterior_summary(
        self,
        y: Tensor,
        num_samples: int = 2_000,
    ) -> Dict[str, Tensor]:
        """Monte Carlo posterior mean, standard deviation, and central intervals."""

        tau, _ = self.sample_and_log_prob(y, num_samples=num_samples)
        return {
            "mean": tau.mean(dim=0),
            "std": tau.std(dim=0),
            "q025": torch.quantile(tau, 0.025, dim=0),
            "median": torch.quantile(tau, 0.5, dim=0),
            "q975": torch.quantile(tau, 0.975, dim=0),
        }


class ExactAATDecoder(nn.Module):
    """Exact p_theta(y | tau) and p_nu(tau), with trainable model parameters."""

    def __init__(
        self,
        initial_mu: Tensor,
        initial_theta: Tensor,
        config: VAEConfig,
    ) -> None:
        super().__init__()
        self.dimension = int(initial_mu.numel())
        self.config = config

        self.mu = nn.Parameter(initial_mu.clone())

        # DESIGN CHOICE: optimize gamma=eta*nu directly because it is the
        # combination appearing in E[Y | tau]. Recover eta as gamma/nu.
        self.gamma = nn.Parameter(torch.zeros_like(initial_mu))

        initial_fraction = (
            (config.initial_nu - config.nu_min)
            / (config.nu_max - config.nu_min)
        )
        if not 0.0 < initial_fraction < 1.0:
            raise ValueError("initial_nu must lie strictly between nu_min and nu_max")
        initial_raw_nu = inverse_sigmoid(
            torch.full_like(initial_mu, initial_fraction)
        )
        self.raw_nu = nn.Parameter(initial_raw_nu)

        # DESIGN CHOICE: Theta=R R^T with positive diagonal R. This guarantees
        # positive definiteness throughout optimization without projection.
        initial_theta_cholesky = torch.linalg.cholesky(initial_theta)
        rows, cols = torch.tril_indices(self.dimension, self.dimension)
        self.register_buffer("tril_rows", rows)
        self.register_buffer("tril_cols", cols)
        raw_theta_cholesky = initial_theta_cholesky[rows, cols].clone()
        diagonal_mask = rows == cols
        adjusted_diagonal = (
            raw_theta_cholesky[diagonal_mask] - config.min_theta_cholesky
        ).clamp_min(1.0e-8)
        raw_theta_cholesky[diagonal_mask] = inverse_softplus(adjusted_diagonal)
        self.raw_theta_cholesky = nn.Parameter(raw_theta_cholesky)

        # ADDED: persistent buffers store post-burn-in parameter averages so the
        # averages move with the model across devices and survive state_dict saves.
        self.register_buffer("average_mu", torch.full_like(initial_mu, torch.nan))
        self.register_buffer("average_gamma", torch.full_like(initial_mu, torch.nan))
        self.register_buffer("average_eta", torch.full_like(initial_mu, torch.nan))
        self.register_buffer("average_nu", torch.full_like(initial_mu, torch.nan))
        self.register_buffer(
            "average_theta",
            torch.full_like(initial_theta, torch.nan),
        )
        self.register_buffer(
            "parameter_average_count",
            torch.zeros((), dtype=torch.long, device=initial_mu.device),
        )

    @property
    def nu(self) -> Tensor:
        return self.config.nu_min + (
            self.config.nu_max - self.config.nu_min
        ) * torch.sigmoid(self.raw_nu)

    @property
    def eta(self) -> Tensor:
        return self.gamma / self.nu

    def theta_cholesky(self) -> Tensor:
        r = self.raw_theta_cholesky.new_zeros(
            (self.dimension, self.dimension)
        )
        r[self.tril_rows, self.tril_cols] = self.raw_theta_cholesky
        diagonal_index = torch.arange(
            self.dimension,
            device=self.raw_theta_cholesky.device,
        )
        raw_diagonal = r[diagonal_index, diagonal_index]
        r = r.clone()
        r[diagonal_index, diagonal_index] = (
            F.softplus(raw_diagonal) + self.config.min_theta_cholesky
        )
        return r

    @property
    def theta(self) -> Tensor:
        r = self.theta_cholesky()
        return r @ r.transpose(-1, -2)

    def log_prior(self, tau: Tensor) -> Tensor:
        """Exact log density of independent inverse-gamma latent priors."""

        nu = self.nu
        concentration = 2.0 / nu
        scale = 2.0 / nu
        log_tau = torch.log(tau)
        coordinate_log_density = (
            concentration * torch.log(scale)
            - torch.lgamma(concentration)
            - (concentration + 1.0) * log_tau
            - scale / tau
        )
        return coordinate_log_density.sum(dim=-1)

    def log_likelihood(self, y: Tensor, tau: Tensor) -> Tensor:
        """Exact log p_theta(y | tau), allowing leading sample dimensions."""

        theta = self.theta
        r = self.theta_cholesky()
        log_det_theta = 2.0 * torch.log(torch.diagonal(r)).sum()

        # The gamma parameter is exactly eta*nu, so this mean is unchanged from
        # mu + eta*nu*tau but is much better conditioned for optimization.
        conditional_mean = self.mu + self.gamma * tau
        residual = y.unsqueeze(0) - conditional_mean
        standardized_residual = residual / torch.sqrt(tau)
        quadratic_form = torch.einsum(
            "...i,ij,...j->...",
            standardized_residual,
            theta,
            standardized_residual,
        )

        log_det_covariance = -log_det_theta + torch.log(tau).sum(dim=-1)
        return -0.5 * (
            self.dimension * LOG_2PI
            + log_det_covariance
            + quadratic_form
        )

    def log_joint(self, y: Tensor, tau: Tensor) -> Tensor:
        return self.log_likelihood(y, tau) + self.log_prior(tau)

    def theta_penalty(self) -> Tensor:
        if self.config.theta_l1 == 0.0:
            return self.raw_theta_cholesky.new_zeros(())
        theta = self.theta
        identity = torch.eye(
            self.dimension,
            dtype=theta.dtype,
            device=theta.device,
        )
        off_diagonal = theta * (1.0 - identity)
        return self.config.theta_l1 * off_diagonal.abs().sum()

    @torch.no_grad()
    def tensor_estimates(self) -> Dict[str, Tensor]:
        """ADDED: return detached parameter tensors for trajectory averaging."""

        return {
            "mu": self.mu.detach().clone(),
            "gamma": self.gamma.detach().clone(),
            "eta": self.eta.detach().clone(),
            "nu": self.nu.detach().clone(),
            "Theta": self.theta.detach().clone(),
        }

    @torch.no_grad()
    def set_parameter_averages(
        self,
        averages: Dict[str, Tensor],
        count: int,
    ) -> None:
        """ADDED: save completed post-burn-in trajectory averages."""

        self.average_mu.copy_(averages["mu"])
        self.average_gamma.copy_(averages["gamma"])
        self.average_eta.copy_(averages["eta"])
        self.average_nu.copy_(averages["nu"])
        self.average_theta.copy_(averages["Theta"])
        self.parameter_average_count.fill_(count)

    @torch.no_grad()
    def estimates(self) -> Dict[str, np.ndarray | int]:
        """Return final estimates plus post-burn-in trajectory averages."""

        # UNCHANGED KEYS: these remain the final parameter values, preserving all
        # existing code that accesses result["mu"], result["eta"], and so on.
        result: Dict[str, np.ndarray | int] = {
            "mu": self.mu.detach().cpu().numpy().copy(),
            "gamma": self.gamma.detach().cpu().numpy().copy(),
            "eta": self.eta.detach().cpu().numpy().copy(),
            "nu": self.nu.detach().cpu().numpy().copy(),
            "Theta": self.theta.detach().cpu().numpy().copy(),
        }

        # ADDED KEYS: these are available after fit_aat_vae completes.
        if int(self.parameter_average_count.item()) > 0:
            result.update(
                {
                    "mu_average": self.average_mu.detach().cpu().numpy().copy(),
                    "gamma_average": self.average_gamma.detach().cpu().numpy().copy(),
                    "eta_average": self.average_eta.detach().cpu().numpy().copy(),
                    "nu_average": self.average_nu.detach().cpu().numpy().copy(),
                    "Theta_average": self.average_theta.detach().cpu().numpy().copy(),
                    "parameter_average_count": int(
                        self.parameter_average_count.item()
                    ),
                }
            )
        return result


class AATVAE(nn.Module):
    """Encoder plus exact decoder, grouped for saving and posterior inference."""

    def __init__(self, encoder: PosteriorEncoder, decoder: ExactAATDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, y: Tensor, num_samples: int) -> Tensor:
        tau, log_q_tau = self.encoder.sample_and_log_prob(y, num_samples)
        return (self.decoder.log_joint(y, tau) - log_q_tau).mean()


def robust_initial_values(y: Tensor) -> Tuple[Tensor, Tensor]:
    """Construct conservative initial mu and positive-definite Theta."""

    dimension = y.shape[1]
    initial_mu = torch.median(y, dim=0).values
    centered = y - initial_mu
    covariance = centered.transpose(0, 1) @ centered / max(y.shape[0] - 1, 1)
    ridge = 0.1 * torch.trace(covariance) / dimension
    regularized_covariance = covariance + ridge.clamp_min(1.0e-3) * torch.eye(
        dimension,
        dtype=y.dtype,
        device=y.device,
    )
    initial_theta = torch.linalg.inv(regularized_covariance)
    return initial_mu, initial_theta


def build_aat_vae(y: Tensor, config: VAEConfig) -> AATVAE:
    """Build the encoder and exact decoder from a data tensor of shape (n,p)."""

    if y.ndim != 2:
        raise ValueError("y must have shape (number_of_observations, dimension)")
    if not torch.isfinite(y).all():
        raise ValueError("y contains NaN or infinite values")

    y_mean = y.mean(dim=0)
    y_scale = y.std(dim=0).clamp_min(1.0e-6)
    initial_mu, initial_theta = robust_initial_values(y)

    encoder = PosteriorEncoder(y_mean=y_mean, y_scale=y_scale, config=config)
    decoder = ExactAATDecoder(
        initial_mu=initial_mu,
        initial_theta=initial_theta,
        config=config,
    )
    return AATVAE(encoder=encoder, decoder=decoder)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


@torch.no_grad()
def estimate_dataset_elbo(
    model: AATVAE,
    y: Tensor,
    batch_size: int,
    num_samples: int,
) -> float:
    """Estimate the mean ELBO per observation for training diagnostics."""

    loader = DataLoader(TensorDataset(y), batch_size=batch_size, shuffle=False)
    total = 0.0
    count = 0
    model.eval()
    for (batch_y,) in loader:
        batch_elbo = model.elbo(batch_y, num_samples=num_samples)
        total += float(batch_elbo) * batch_y.shape[0]
        count += batch_y.shape[0]
    model.train()
    return total / count


def fit_aat_vae(
    y: np.ndarray | Tensor,
    config: Optional[VAEConfig] = None,
    device: Optional[str] = None,
) -> Tuple[AATVAE, List[Dict[str, float]]]:
    """Fit the VAE by alternating approximate E-steps and stochastic M-steps."""

    config = config or VAEConfig()
    set_seed(config.seed)

    # ADDED: validate the fraction of training discarded before averaging.
    if not 0.0 <= config.parameter_average_burn_in < 1.0:
        raise ValueError("parameter_average_burn_in must lie in [0, 1)")

    if isinstance(y, np.ndarray):
        y_tensor = torch.as_tensor(y, dtype=torch.get_default_dtype())
    else:
        y_tensor = y.to(dtype=torch.get_default_dtype())

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    y_tensor = y_tensor.to(selected_device)
    model = build_aat_vae(y_tensor, config).to(selected_device)

    encoder_optimizer = torch.optim.Adam(
        model.encoder.parameters(),
        lr=config.encoder_lr,
        weight_decay=config.weight_decay,
    )
    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=config.model_lr,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: List[Dict[str, float]] = []

    # ADDED: average actual transformed parameters, rather than raw Cholesky or
    # sigmoid variables, beginning immediately after the configured burn-in.
    average_start_epoch = (
        int(config.epochs * config.parameter_average_burn_in) + 1
    )
    parameter_sums: Optional[Dict[str, Tensor]] = None
    parameter_average_count = 0

    model.train()
    for epoch in range(1, config.epochs + 1):
        for (batch_y,) in loader:
            # Approximate E-step: theta is fixed while q_phi learns the current
            # posterior by maximizing the ELBO.
            set_requires_grad(model.decoder, False)
            set_requires_grad(model.encoder, True)
            for _ in range(config.encoder_steps):
                encoder_optimizer.zero_grad(set_to_none=True)
                encoder_loss = -model.elbo(
                    batch_y,
                    num_samples=config.posterior_samples,
                )
                if not torch.isfinite(encoder_loss):
                    raise FloatingPointError(
                        "Non-finite encoder loss. Reduce learning rates or flow scale."
                    )
                encoder_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(),
                    config.gradient_clip,
                )
                encoder_optimizer.step()

            # Stochastic M-step: q_phi is fixed and samples from it are treated
            # as Monte Carlo draws when updating mu, gamma, nu, and Theta.
            set_requires_grad(model.encoder, False)
            set_requires_grad(model.decoder, True)
            for _ in range(config.model_steps):
                decoder_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    tau, _ = model.encoder.sample_and_log_prob(
                        batch_y,
                        num_samples=config.posterior_samples,
                    )
                expected_log_joint = model.decoder.log_joint(batch_y, tau).mean()
                decoder_loss = -expected_log_joint + model.decoder.theta_penalty()
                if not torch.isfinite(decoder_loss):
                    raise FloatingPointError(
                        "Non-finite decoder loss. Check nu bounds and data scaling."
                    )
                decoder_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(),
                    config.gradient_clip,
                )
                decoder_optimizer.step()

        set_requires_grad(model.encoder, True)
        set_requires_grad(model.decoder, True)

        # ADDED: record one equally weighted parameter snapshot per completed
        # post-burn-in epoch. Averaging Theta directly preserves positive definiteness.
        if epoch >= average_start_epoch:
            current_parameters = model.decoder.tensor_estimates()
            if parameter_sums is None:
                parameter_sums = {
                    name: value.clone()
                    for name, value in current_parameters.items()
                }
            else:
                for name, value in current_parameters.items():
                    parameter_sums[name].add_(value)
            parameter_average_count += 1

        should_report = (
            epoch == 1
            or epoch % config.print_every == 0
            or epoch == config.epochs
        )
        if should_report:
            mean_elbo = estimate_dataset_elbo(
                model,
                y_tensor,
                batch_size=config.batch_size,
                num_samples=max(config.posterior_samples, 8),
            )
            record = {
                "epoch": float(epoch),
                "mean_elbo": mean_elbo,
            }
            history.append(record)
            print(f"epoch={epoch:5d}  mean_ELBO={mean_elbo: .6f}")

    # ADDED: finalize and attach the averages without changing the original
    # two-object return signature: model, history = fit_aat_vae(...).
    if parameter_sums is None or parameter_average_count == 0:
        raise RuntimeError("No epochs were available for parameter averaging")
    parameter_averages = {
        name: total / parameter_average_count
        for name, total in parameter_sums.items()
    }
    model.decoder.set_parameter_averages(
        parameter_averages,
        parameter_average_count,
    )

    model.eval()
    return model, history


@torch.no_grad()
def simulate_aat_data(
    n: int,
    mu: np.ndarray,
    eta: np.ndarray,
    nu: np.ndarray,
    theta: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate exactly from the model used by the VAE decoder."""

    set_seed(seed)
    mu_t = torch.as_tensor(mu)
    eta_t = torch.as_tensor(eta)
    nu_t = torch.as_tensor(nu)
    theta_t = torch.as_tensor(theta)
    psi_t = torch.linalg.inv(theta_t)

    concentration = 2.0 / nu_t
    rate = 2.0 / nu_t
    g = torch.distributions.Gamma(concentration, rate).sample((n,))
    tau = 1.0 / g
    x = MultivariateNormal(
        torch.zeros_like(mu_t),
        covariance_matrix=psi_t,
    ).sample((n,))
    y = mu_t + eta_t * nu_t * tau + torch.sqrt(tau) * x
    return y.cpu().numpy(), tau.cpu().numpy()
