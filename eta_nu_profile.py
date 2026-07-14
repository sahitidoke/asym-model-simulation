"""Profile-likelihood diagnostic for identifiability of eta_j and nu_j.

This file must be placed in the same directory as ``aat_vae.py``.

The diagnostic fixes one pair (eta_j, nu_j), re-optimizes the encoder and all
remaining model parameters, and estimates the observed-data log likelihood
using importance sampling.  A sharply bounded profile region supports local
practical identifiability; a long or boundary-touching valley indicates weak
identifiability (or a grid that must be enlarged).

Use from an existing Python script after fitting the VAE:

    from eta_nu_profile import profile_eta_nu

    diagnostic = profile_eta_nu(
        y=Y,
        coordinate=0,
        fitted_model=model,
        grid_size=7,
        profile_epochs=150,
        importance_samples=512,
        output_prefix="eta_nu_coordinate_0",
    )

Or run as a standalone command:

    python eta_nu_profile.py --data Y.npy --coordinate 0
    python eta_nu_profile.py --data Y.csv --coordinate 0

The command-line route first fits the unconstrained VAE and can therefore be
slow. Passing an already fitted model from Python avoids that extra fit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from aat_vae import (
    AATVAE,
    VAEConfig,
    fit_aat_vae,
    inverse_sigmoid,
    set_requires_grad,
    set_seed,
    simulate_aat_data,
)


# DESIGN CHOICE: 5.991 is the usual 95% likelihood-ratio cutoff for two fixed
# parameters. Because the likelihood is estimated variationally, this cutoff is
# a diagnostic reference rather than an exact finite-sample hypothesis test.
CHI_SQUARE_2_95 = 5.991


def _as_model_tensor(y: np.ndarray | Tensor, model: AATVAE) -> Tensor:
    """Move data to the fitted model's device and floating-point type."""

    reference_parameter = next(model.parameters())
    return torch.as_tensor(
        y,
        dtype=reference_parameter.dtype,
        device=reference_parameter.device,
    )


def _raw_nu_from_value(nu_value: float, config: VAEConfig, like: Tensor) -> Tensor:
    """Convert a constrained nu value to the decoder's unconstrained value."""

    if not config.nu_min < nu_value < config.nu_max:
        raise ValueError(
            f"nu={nu_value} must lie strictly between "
            f"nu_min={config.nu_min} and nu_max={config.nu_max}."
        )
    fraction = (nu_value - config.nu_min) / (config.nu_max - config.nu_min)
    return inverse_sigmoid(like.new_tensor(fraction))


@torch.no_grad()
def _restore_fixed_pair(
    model: AATVAE,
    coordinate: int,
    eta_value: float,
    nu_value: float,
) -> None:
    """Restore the fixed profile parameters after every optimizer step."""

    model.decoder.gamma[coordinate] = eta_value * nu_value
    model.decoder.raw_nu[coordinate] = _raw_nu_from_value(
        nu_value,
        model.decoder.config,
        model.decoder.raw_nu,
    )


def _zero_fixed_pair_gradients(model: AATVAE, coordinate: int) -> None:
    """Prevent Adam from using the fixed eta_j and nu_j gradients."""

    if model.decoder.gamma.grad is not None:
        model.decoder.gamma.grad[coordinate] = 0.0
    if model.decoder.raw_nu.grad is not None:
        model.decoder.raw_nu.grad[coordinate] = 0.0


def _fit_one_profile_point(
    y: Tensor,
    baseline_model: AATVAE,
    coordinate: int,
    eta_value: float,
    nu_value: float,
    profile_epochs: int,
    profile_config: VAEConfig,
) -> AATVAE:
    """Refit all nuisance parameters with eta_j and nu_j held fixed."""

    # DESIGN CHOICE: warm-start from the unconstrained fit. This makes an entire
    # profile grid feasible while allowing the posterior encoder to readapt.
    model = copy.deepcopy(baseline_model)
    model.train()
    _restore_fixed_pair(model, coordinate, eta_value, nu_value)

    encoder_optimizer = torch.optim.Adam(
        model.encoder.parameters(),
        lr=profile_config.encoder_lr,
        weight_decay=profile_config.weight_decay,
    )
    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=profile_config.model_lr,
        weight_decay=profile_config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(y),
        batch_size=profile_config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # DESIGN CHOICE: use the same random seed and mini-batch order at each grid
    # point. Common randomness reduces noise when comparing nearby profile values.
    set_seed(profile_config.seed)
    for _ in range(profile_epochs):
        for (batch_y,) in loader:
            # Variational E-step: the full decoder, including the fixed pair, is
            # held constant while q_phi(tau | Y) adapts to this profile point.
            set_requires_grad(model.decoder, False)
            set_requires_grad(model.encoder, True)
            for _ in range(profile_config.encoder_steps):
                encoder_optimizer.zero_grad(set_to_none=True)
                encoder_loss = -model.elbo(
                    batch_y,
                    num_samples=profile_config.posterior_samples,
                )
                if not torch.isfinite(encoder_loss):
                    raise FloatingPointError(
                        "Non-finite encoder loss while profiling "
                        f"eta={eta_value}, nu={nu_value}."
                    )
                encoder_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(),
                    profile_config.gradient_clip,
                )
                encoder_optimizer.step()

            # Stochastic M-step: optimize mu, Theta, and every non-profiled
            # coordinate of eta and nu.
            set_requires_grad(model.encoder, False)
            set_requires_grad(model.decoder, True)
            for _ in range(profile_config.model_steps):
                decoder_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    tau, _ = model.encoder.sample_and_log_prob(
                        batch_y,
                        num_samples=profile_config.posterior_samples,
                    )
                expected_log_joint = model.decoder.log_joint(batch_y, tau).mean()
                decoder_loss = (
                    -expected_log_joint + model.decoder.theta_penalty()
                )
                if not torch.isfinite(decoder_loss):
                    raise FloatingPointError(
                        "Non-finite decoder loss while profiling "
                        f"eta={eta_value}, nu={nu_value}."
                    )
                decoder_loss.backward()
                _zero_fixed_pair_gradients(model, coordinate)
                torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(),
                    profile_config.gradient_clip,
                )
                decoder_optimizer.step()
                # Restoring after the step also protects against weight decay or
                # future optimizer changes that could move the fixed entries.
                _restore_fixed_pair(model, coordinate, eta_value, nu_value)

    set_requires_grad(model.encoder, True)
    set_requires_grad(model.decoder, True)
    model.eval()
    return model


@torch.no_grad()
def importance_log_likelihood(
    model: AATVAE,
    y: Tensor,
    importance_samples: int = 512,
    batch_size: int = 64,
    sample_chunk_size: int = 64,
    seed: int = 12_345,
) -> float:
    """Estimate sum_i log p_theta(Y_i) with encoder-based importance sampling."""

    if importance_samples < 1:
        raise ValueError("importance_samples must be positive")
    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be positive")

    # DESIGN CHOICE: importance sampling evaluates the observed-data likelihood,
    # not the complete-data likelihood. The latter can overstate identifiability
    # because tau is not observed.
    set_seed(seed)
    model.eval()
    loader = DataLoader(TensorDataset(y), batch_size=batch_size, shuffle=False)
    total_log_likelihood = 0.0

    for (batch_y,) in loader:
        accumulated_log_sum: Optional[Tensor] = None
        samples_drawn = 0
        while samples_drawn < importance_samples:
            current_samples = min(
                sample_chunk_size,
                importance_samples - samples_drawn,
            )
            tau, log_q_tau = model.encoder.sample_and_log_prob(
                batch_y,
                num_samples=current_samples,
            )
            log_weights = model.decoder.log_joint(batch_y, tau) - log_q_tau
            chunk_log_sum = torch.logsumexp(log_weights, dim=0)
            accumulated_log_sum = (
                chunk_log_sum
                if accumulated_log_sum is None
                else torch.logaddexp(accumulated_log_sum, chunk_log_sum)
            )
            samples_drawn += current_samples

        assert accumulated_log_sum is not None
        batch_log_probability = accumulated_log_sum - math.log(importance_samples)
        total_log_likelihood += float(batch_log_probability.sum())

    return total_log_likelihood


def _automatic_grids(
    eta_hat: float,
    nu_hat: float,
    config: VAEConfig,
    grid_size: int,
    eta_relative_width: float,
    nu_relative_width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Make centered eta and nu grids that include the fitted values exactly."""

    if grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least 3")

    # Near eta=0, a purely relative grid would collapse, so use a minimum scale.
    eta_scale = max(abs(eta_hat), 0.25)
    eta_half_width = eta_relative_width * eta_scale
    eta_grid = np.linspace(
        eta_hat - eta_half_width,
        eta_hat + eta_half_width,
        grid_size,
    )

    lower_nu = max(
        config.nu_min + 1.0e-6,
        nu_hat * (1.0 - nu_relative_width),
    )
    upper_nu = min(
        config.nu_max - 1.0e-6,
        nu_hat * (1.0 + nu_relative_width),
    )
    nu_grid = np.linspace(lower_nu, upper_nu, grid_size)

    # Preserve the fitted point as the center even when a nu bound truncated one
    # side of the automatic grid.
    center = grid_size // 2
    eta_grid[center] = eta_hat
    nu_grid[center] = nu_hat
    eta_grid.sort()
    nu_grid.sort()
    return eta_grid, nu_grid


def _save_profile_csv(
    path: Path,
    eta_grid: np.ndarray,
    nu_grid: np.ndarray,
    log_likelihood: np.ndarray,
    likelihood_ratio: np.ndarray,
) -> None:
    """Save every profile point for later analysis or publication figures."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["eta", "nu", "profile_log_likelihood", "lr_statistic", "inside_95"]
        )
        for nu_index, nu_value in enumerate(nu_grid):
            for eta_index, eta_value in enumerate(eta_grid):
                lr_value = likelihood_ratio[nu_index, eta_index]
                writer.writerow(
                    [
                        eta_value,
                        nu_value,
                        log_likelihood[nu_index, eta_index],
                        lr_value,
                        int(lr_value <= CHI_SQUARE_2_95),
                    ]
                )


def _save_profile_plot(
    path: Path,
    eta_grid: np.ndarray,
    nu_grid: np.ndarray,
    likelihood_ratio: np.ndarray,
    eta_hat: float,
    nu_hat: float,
    coordinate: int,
) -> None:
    """Save the profile likelihood-ratio surface and 95% reference contour."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to create the profile plot. "
            "Install it with: python -m pip install matplotlib"
        ) from error

    eta_mesh, nu_mesh = np.meshgrid(eta_grid, nu_grid)
    figure, axis = plt.subplots(figsize=(7.0, 5.5))
    filled = axis.contourf(
        eta_mesh,
        nu_mesh,
        likelihood_ratio,
        levels=20,
        cmap="viridis_r",
    )
    colorbar = figure.colorbar(filled, ax=axis)
    colorbar.set_label(r"$2(\ell_{\max}-\ell_p)$")

    if np.nanmin(likelihood_ratio) <= CHI_SQUARE_2_95 <= np.nanmax(
        likelihood_ratio
    ):
        contour = axis.contour(
            eta_mesh,
            nu_mesh,
            likelihood_ratio,
            levels=[CHI_SQUARE_2_95],
            colors="red",
            linewidths=2.0,
        )
        axis.clabel(contour, fmt={CHI_SQUARE_2_95: "95% reference"})

    axis.scatter(
        [eta_hat],
        [nu_hat],
        marker="*",
        s=170,
        color="white",
        edgecolor="black",
        label="Unconstrained fit",
        zorder=5,
    )
    axis.set_xlabel(rf"$\eta_{{{coordinate}}}$")
    axis.set_ylabel(rf"$\nu_{{{coordinate}}}$")
    axis.set_title(f"Profile likelihood for coordinate {coordinate}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _summarize_profile(
    eta_grid: np.ndarray,
    nu_grid: np.ndarray,
    likelihood_ratio: np.ndarray,
) -> Dict[str, object]:
    """Summarize whether the approximate 95% profile region is bounded."""

    inside = likelihood_ratio <= CHI_SQUARE_2_95
    accepted_indices = np.argwhere(inside)
    if accepted_indices.size == 0:
        return {
            "bounded_95_region": False,
            "touches_grid_boundary": True,
            "eta_95_grid_range": (float("nan"), float("nan")),
            "nu_95_grid_range": (float("nan"), float("nan")),
            "decision": "INCONCLUSIVE: no grid point entered the 95% region.",
        }

    nu_indices = accepted_indices[:, 0]
    eta_indices = accepted_indices[:, 1]
    touches_boundary = bool(
        (nu_indices == 0).any()
        or (nu_indices == len(nu_grid) - 1).any()
        or (eta_indices == 0).any()
        or (eta_indices == len(eta_grid) - 1).any()
    )
    eta_range = (
        float(eta_grid[eta_indices.min()]),
        float(eta_grid[eta_indices.max()]),
    )
    nu_range = (
        float(nu_grid[nu_indices.min()]),
        float(nu_grid[nu_indices.max()]),
    )

    if touches_boundary:
        decision = (
            "WEAK OR INCONCLUSIVE: the 95% profile region reaches the grid "
            "boundary. Enlarge the grid; if it remains open, eta and nu are "
            "not practically identifiable at this sample size."
        )
    else:
        decision = (
            "LOCALLY PRACTICALLY IDENTIFIABLE: the approximate 95% profile "
            "region is closed inside the tested grid. Its width quantifies "
            "the remaining uncertainty."
        )

    return {
        "bounded_95_region": not touches_boundary,
        "touches_grid_boundary": touches_boundary,
        "eta_95_grid_range": eta_range,
        "nu_95_grid_range": nu_range,
        "decision": decision,
    }


def profile_eta_nu(
    y: np.ndarray | Tensor,
    coordinate: int,
    fitted_model: Optional[AATVAE] = None,
    fit_config: Optional[VAEConfig] = None,
    eta_grid: Optional[Sequence[float]] = None,
    nu_grid: Optional[Sequence[float]] = None,
    grid_size: int = 7,
    eta_relative_width: float = 1.0,
    nu_relative_width: float = 0.5,
    profile_epochs: int = 150,
    importance_samples: int = 512,
    evaluation_batch_size: int = 64,
    output_prefix: str = "eta_nu_profile",
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Profile eta_j and nu_j while re-optimizing all nuisance parameters.

    Returns a dictionary containing the grids, log likelihoods, LR statistics,
    the bounded-region diagnostic, the fitted model, and output file paths.
    """

    fit_config = fit_config or VAEConfig()
    if fitted_model is None:
        # A caller with an existing fit should pass it to avoid repeating this.
        fitted_model, _ = fit_aat_vae(y, config=fit_config, device=device)

    y_tensor = _as_model_tensor(y, fitted_model)
    dimension = y_tensor.shape[1]
    if not 0 <= coordinate < dimension:
        raise IndexError(f"coordinate must be between 0 and {dimension - 1}")

    estimates = fitted_model.decoder.estimates()
    eta_hat = float(estimates["eta"][coordinate])
    nu_hat = float(estimates["nu"][coordinate])
    profile_config = replace(
        fitted_model.decoder.config,
        epochs=profile_epochs,
        print_every=max(profile_epochs, 1),
    )

    if eta_grid is None or nu_grid is None:
        automatic_eta, automatic_nu = _automatic_grids(
            eta_hat=eta_hat,
            nu_hat=nu_hat,
            config=profile_config,
            grid_size=grid_size,
            eta_relative_width=eta_relative_width,
            nu_relative_width=nu_relative_width,
        )
        eta_values = automatic_eta if eta_grid is None else np.asarray(eta_grid)
        nu_values = automatic_nu if nu_grid is None else np.asarray(nu_grid)
    else:
        eta_values = np.asarray(eta_grid, dtype=float)
        nu_values = np.asarray(nu_grid, dtype=float)

    eta_values = np.asarray(eta_values, dtype=float)
    nu_values = np.asarray(nu_values, dtype=float)
    if eta_values.ndim != 1 or nu_values.ndim != 1:
        raise ValueError("eta_grid and nu_grid must be one-dimensional")

    print(
        f"Unconstrained coordinate {coordinate}: "
        f"eta_hat={eta_hat:.8g}, nu_hat={nu_hat:.8g}"
    )
    print(
        f"Profiling {len(eta_values)} x {len(nu_values)} = "
        f"{len(eta_values) * len(nu_values)} constrained fits."
    )

    baseline_log_likelihood = importance_log_likelihood(
        fitted_model,
        y_tensor,
        importance_samples=importance_samples,
        batch_size=evaluation_batch_size,
        seed=fit_config.seed + 10_000,
    )
    profile_log_likelihood = np.empty(
        (len(nu_values), len(eta_values)),
        dtype=float,
    )

    total_points = len(eta_values) * len(nu_values)
    point = 0
    for nu_index, nu_value in enumerate(nu_values):
        for eta_index, eta_value in enumerate(eta_values):
            point += 1
            print(
                f"[{point:3d}/{total_points}] "
                f"eta={eta_value:.7g}, nu={nu_value:.7g}",
                flush=True,
            )
            constrained_model = _fit_one_profile_point(
                y=y_tensor,
                baseline_model=fitted_model,
                coordinate=coordinate,
                eta_value=float(eta_value),
                nu_value=float(nu_value),
                profile_epochs=profile_epochs,
                profile_config=profile_config,
            )
            profile_log_likelihood[nu_index, eta_index] = (
                importance_log_likelihood(
                    constrained_model,
                    y_tensor,
                    importance_samples=importance_samples,
                    batch_size=evaluation_batch_size,
                    seed=fit_config.seed + 10_000,
                )
            )
            del constrained_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    maximum_log_likelihood = max(
        baseline_log_likelihood,
        float(np.max(profile_log_likelihood)),
    )
    likelihood_ratio = 2.0 * (
        maximum_log_likelihood - profile_log_likelihood
    )
    likelihood_ratio = np.maximum(likelihood_ratio, 0.0)
    summary = _summarize_profile(
        eta_grid=eta_values,
        nu_grid=nu_values,
        likelihood_ratio=likelihood_ratio,
    )

    prefix = Path(output_prefix)
    csv_path = prefix.with_suffix(".csv")
    plot_path = prefix.with_suffix(".png")
    _save_profile_csv(
        csv_path,
        eta_grid=eta_values,
        nu_grid=nu_values,
        log_likelihood=profile_log_likelihood,
        likelihood_ratio=likelihood_ratio,
    )
    _save_profile_plot(
        plot_path,
        eta_grid=eta_values,
        nu_grid=nu_values,
        likelihood_ratio=likelihood_ratio,
        eta_hat=eta_hat,
        nu_hat=nu_hat,
        coordinate=coordinate,
    )

    print("\nProfile diagnostic")
    print("------------------")
    print(summary["decision"])
    print(f"Approximate eta 95% grid range: {summary['eta_95_grid_range']}")
    print(f"Approximate nu 95% grid range:  {summary['nu_95_grid_range']}")
    print(f"CSV results: {csv_path}")
    print(f"Profile plot: {plot_path}")

    return {
        "coordinate": coordinate,
        "eta_hat": eta_hat,
        "nu_hat": nu_hat,
        "eta_grid": eta_values,
        "nu_grid": nu_values,
        "baseline_log_likelihood": baseline_log_likelihood,
        "profile_log_likelihood": profile_log_likelihood,
        "likelihood_ratio": likelihood_ratio,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "fitted_model": fitted_model,
        **summary,
    }


def _load_data(path: Path) -> np.ndarray:
    """Load an n-by-p matrix from .npy or comma-separated text."""

    if path.suffix.lower() == ".npy":
        y = np.load(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        y = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("Data must be a .npy, .csv, or comma-separated .txt file")
    if y.ndim != 2:
        raise ValueError("Loaded data must have shape (n, p)")
    return y


def _demonstration_data(n: int, seed: int) -> np.ndarray:
    """Create a small default dataset when no command-line data file is given."""

    mu = np.array([-0.5, 0.25, 1.0])
    eta = np.array([0.8, -0.6, 0.5])
    nu = np.array([0.45, 0.60, 0.35])
    theta = np.array(
        [
            [1.30, -0.25, 0.15],
            [-0.25, 1.10, -0.20],
            [0.15, -0.20, 0.95],
        ]
    )
    y, _ = simulate_aat_data(n, mu, eta, nu, theta, seed=seed)
    return y


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--coordinate", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--profile-epochs", type=int, default=150)
    parser.add_argument("--importance-samples", type=int, default=512)
    parser.add_argument("--fit-epochs", type=int, default=500)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="eta_nu_profile")
    parser.add_argument("--demo-n", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    y = (
        _load_data(arguments.data)
        if arguments.data is not None
        else _demonstration_data(arguments.demo_n, arguments.seed)
    )
    config = VAEConfig(
        epochs=arguments.fit_epochs,
        flow_layers=arguments.flow_layers,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
    )
    profile_eta_nu(
        y=y,
        coordinate=arguments.coordinate,
        fitted_model=None,
        fit_config=config,
        grid_size=arguments.grid_size,
        profile_epochs=arguments.profile_epochs,
        importance_samples=arguments.importance_samples,
        output_prefix=arguments.output_prefix,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
