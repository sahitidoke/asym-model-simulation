import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import norm, skew

def simulate_aat_data(n, p, Theta_true, mu, eta, nu, rng):
    Psi_true = np.linalg.inv(Theta_true)
    alpha = 2.0 / nu
    beta = 2.0 / nu
    G = rng.gamma(shape=alpha, scale=1.0 / beta, size=(n, p))
    tau = 1.0 / G
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    Y = mu[None, :] + eta[None, :] * nu[None, :] * tau + np.sqrt(tau) * X
    return Y, tau

def simulate_gaussian_data(n,p, Theta_true):
    Psi_true = np.linalg.inv(Theta_true)
    Y = np.random.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    return Y


def simulate_noisy_gaussian_data(n, p, Theta_true,
                  skewness=0.0, 
                  outlier_frac=0.0, 
                  outlier_scale=4.0,
                  noise_scale=0.0,
                  seed=None):
    """
    
    Parameters:
    -----------
    n : int
    p : int
    Theta_true : (p, p) array
        true precision matrix
    skewness : float, default=0.0
        skewness strength. 0=symmetric, 0.3=light, 0.5=moderate, 0.8=heavy
    outlier_frac : float, default=0.0
        outlier fraction. 0.02=2%, 0.05=5%
    outlier_scale : float, default=4.0
        outlier magnitude (multiple of data standard deviation)
    noise_scale : float, default=0.0
        additional Gaussian noise standard deviation (multiple of data standard deviation)
    seed : int, optional
        random seed
    
    Returns:
    --------
    Y : (n, p) array
        generated observed data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Generate multivariate Gaussian data
    Psi_true = np.linalg.inv(Theta_true)
    Y = np.random.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    
    # 2. Add skewness
    if skewness != 0:
        Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
        Y = np.sinh(np.arcsinh(Y_std) + skewness)
        Y = Y * Y.std(axis=0, ddof=1) + Y.mean(axis=0)
    
    # 3. Add outliers
    n_outliers = int(n * outlier_frac)
    if n_outliers > 0:
        idx = np.random.choice(n, size=n_outliers, replace=False)
        scale = outlier_scale * Y.std(axis=0, ddof=1)
        Y[idx] = np.random.normal(loc=0, scale=scale, size=(n_outliers, p))
    
    # 4. 
    if noise_scale > 0:
        noise = np.random.normal(loc=0, 
                                 scale=noise_scale * Y.std(axis=0, ddof=1), 
                                 size=(n, p))
        Y = Y + noise
    
    return Y

if __name__ == "__main__":
    # imported here, not at module scope: simulation.simulation imports this
    # module, so a top-level import would be circular when this file is __main__
    from simulation.simulation import make_true_theta

    n, p = 2000, 20
    true_theta = make_true_theta(p)
    Y = simulate_noisy_gaussian_data(n, p, true_theta, skewness=0.6, outlier_frac=0.001)

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    BAR, REF = "#2a78d6", "#52514e"

    ncol = 5
    nrow = int(np.ceil(p / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow),
                            sharex=False, sharey=False)
    for j, ax in enumerate(axs.ravel()):
        if j >= p:
            ax.set_axis_off()
            continue
        yj = Y[:, j]
        ax.hist(yj, bins=50, density=True, color=BAR,
                edgecolor="white", linewidth=0.3)
        # Gaussian with the same mean/sd: the gap between this and the bars is
        # exactly the skew + outlier contamination the generator introduces.
        xs = np.linspace(yj.min(), yj.max(), 200)
        ax.plot(xs, norm.pdf(xs, yj.mean(), yj.std(ddof=1)),
                color=REF, linewidth=1.5, linestyle="--")
        ax.set_title(f"$Y_{{{j + 1}}}$   skew {skew(yj):+.2f}",
                     fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED, length=3)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    fig.suptitle(
        f"Simulated Y: n={n}, p={p} (skewness=0.6, outlier_frac=0.001)\n"
        "dashed = Gaussian with matched mean/sd",
        fontsize=11, color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs("results/simulations", exist_ok=True)
    out = "results/simulations/data_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"pooled skewness across all {p} dims: {skew(Y.ravel()):+.3f}")
    print(f"saved {out}")
