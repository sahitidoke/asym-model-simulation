import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import norm, skew
from scipy.optimize import brentq

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

"""
Contaminated-normal data generator.

Finegold & Drton (2011), "Robust graphical modeling of gene networks using
classical and alternative t-distributions", AOAS 5(2), Section 6.1.
"""

def make_true_theta(p, prob=0.01, min_eig=0.6, rng=None):
    """Random sparse precision matrix.

    (a) lower-triangular entries iid in {-1, 0, 1} w.p. {1%, 98%, 1%}
    (b) symmetrize
    (c) theta_kk = 1 + h_k  (h_k = # nonzeros in row k)
    then shrink the diagonal by the largest common factor keeping
    lambda_min(Theta) == min_eig.
    """
    rng = np.random.default_rng(rng)

    off = np.zeros((p, p))
    il = np.tril_indices(p, -1)
    off[il] = rng.choice([-1.0, 0.0, 1.0], size=il[0].size,
                         p=[prob, 1 - 2 * prob, prob])
    off = off + off.T

    h = (off != 0).sum(axis=1)
    d = 1.0 + h

    def lmin(c):
        return np.linalg.eigvalsh(off + c * np.diag(d)).min()

    # c=1 -> lambda_min >= 1 (diagonal dominance); c=0 -> lambda_min <= 0
    c = brentq(lambda c: lmin(c) - min_eig, 1e-10, 1.0, xtol=1e-12)
    return off + c * np.diag(d)


def simulate_contaminated_normal_data(n, p, Theta_true, eps=0.02,
                            contam_var=0.2, mult=2.5, random_sign=False, rng=None):
    """N_p(0, theta^-1) sample with a fraction `eps` of the individual
    ENTRIES replaced by N(mu_star, contam_var) draws, where
    mu_star = mult * max(diag(theta^-1))."""
    rng = np.random.default_rng(rng)
    p = Theta_true.shape[0]

    sigma = np.linalg.inv(Theta_true)
    sigma = (sigma + sigma.T) / 2
    L = np.linalg.cholesky(sigma)

    Y = rng.standard_normal((n, p)) @ L.T

    mu_star = mult * np.max(np.diag(sigma))
    k = int(round(eps * n * p))
    idx = rng.choice(n * p, size=k, replace=False)
    vals = rng.normal(mu_star, np.sqrt(contam_var), size=k)
    if random_sign:
        vals *= rng.choice([-1.0, 1.0], size=k)
    Y.flat[idx] = vals

    return Y

if __name__ == "__main__":
    n, p = 2000, 20
    
    rng = np.random.default_rng()
    mu_true  = rng.uniform(low=-5, high=5, size=p)
    eta_true = rng.uniform(low=-5, high=5, size=p)
    nu_true = rng.uniform(low=0.15, high=0.9, size=p)
    Theta_true = make_true_theta(p)

    # Swap these two lines for whichever generator you want to look at.
    # name = "contaminated_normal"
    # Y = simulate_contaminated_normal_data(n, p, true_theta)
    name = "aat"
    Y,_ = simulate_aat_data(n, p, Theta_true, mu_true, eta_true, nu_true, rng)

    skews = skew(Y, axis=0)

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    BAR, REF = "#2a78d6", "#52514e"

    ncol = 5
    nrow = int(np.ceil(p / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow),
                            squeeze=False)
    for j, ax in enumerate(axs.ravel()):
        if j >= p:
            ax.set_axis_off()
            continue
        yj = Y[:, j]
        ax.hist(yj, bins=50, density=True, color=BAR,
                edgecolor="white", linewidth=0.3)
        # Normal with this column's own mean/sd -- the null, not a fit. Any gap
        # between it and the bars is that marginal's non-normality, whatever
        # generator produced Y.
        xs = np.linspace(yj.min(), yj.max(), 300)
        ax.plot(xs, norm.pdf(xs, yj.mean(), yj.std(ddof=1)),
                color=REF, linewidth=1.5, linestyle="--")
        ax.set_title(f"$Y_{{{j + 1}}}$   skew {skews[j]:+.2f}",
                     fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED, length=3)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    fig.suptitle(f"Marginals of Y from {name}: n={n}, p={p}\n"
                 "dashed = Gaussian with matched mean/sd",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs("results/simulations", exist_ok=True)
    out = f"results/simulations/marginals_{name}.pdf"
    fig.savefig(out, bbox_inches="tight")

    print(f"skewness of each marginal ({name}):")
    for j in range(p):
        print(f"  Y_{j + 1:<3} {skews[j]:+.3f}")
    print(f"min {skews.min():+.3f}   median {np.median(skews):+.3f}   "
          f"max {skews.max():+.3f}   mean |skew| {np.abs(skews).mean():.3f}")
    print(f"saved {out}")
