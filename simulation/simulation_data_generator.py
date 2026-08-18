import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import norm, skew
from scipy.optimize import brentq

def simulate_aat_data(n, p, Theta_true, mu, eta, nu, rng = None):
    Psi_true = np.linalg.inv(Theta_true)
    alpha = 2.0 / nu
    beta = 2.0 / nu
    G = rng.gamma(shape=alpha, scale=1.0 / beta, size=(n, p))
    tau = 1.0 / G
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    Y = mu[None, :] + eta[None, :] * nu[None, :] * tau + np.sqrt(tau) * X
    return Y, tau

# --------------------------------------------------------------------------
# Finegold & Drton (2011), "Robust graphical modeling of gene networks using
# classical and alternative t-distributions", AOAS 5(2)  --  docs/tlasso.pdf.
#
# Both of their models, plus the per-coordinate-nu modification of the second.
# Both take nu in THIS PROJECT'S convention -- the one simulate_aat_data and
# EM_algorithm use, not F&D's:
#
#     tau ~ Inv-Gamma(2/nu, 2/nu),    df = 4/nu,    Var(Y_j) = Psi_jj 2/(2 - nu)
#
# so nu -> 0 is the Gaussian limit and LARGE nu is the heavy tail. An aat design
# therefore hands its nu_true straight to these and gets the same tails back,
# with only the eta term gone.
#
# F&D write their own tau ~ Inv-Gamma(nu/2, nu/2), with nu the df itself, and so
# do run_tlasso and run_tstar_varlasso. That runs the OTHER WAY. Convert on the
# way out to those fits -- median(4/nu) summarizes per coordinate and only then
# reduces -- and never hand one of them a nu from here unconverted. The two
# conventions agree at exactly one point, nu = 2, which is also where Var(Y)
# stops existing; anything above it in aat units has no variance at all.
#
# None of these carry an eta, so every marginal is symmetric about mu_j. And as
# everywhere else in this project, theta_jk = 0 is NOT conditional independence
# of Y_j and Y_k given the rest -- the latent tau's sit between Theta and Y, in
# the classical case coupling all p coordinates through one draw. The support
# of Theta_true is the estimation target, and nothing more than that.
# --------------------------------------------------------------------------

def simulate_classical_t_data(n, p, Theta_true, mu, nu, rng = None):
    """Classical multivariate t, F&D Sec. 2: ONE divisor per OBSERVATION.

        Y_i = mu + sqrt(tau_i) * X_i,
        tau_i ~ Inv-Gamma(2/nu, 2/nu)  independent across i,
        X_i ~ N_p(0, Psi),  Psi = Theta^-1.

    Y_i is then exactly multivariate t with 4/nu degrees of freedom. One tau_i
    scales a whole row, so an outlying observation is outlying in every
    coordinate at once -- this is the data run_tlasso is the ML fit for, and the
    contrast run_tstar_varlasso is meant to lose on.

    `nu` is a scalar in aat units (see the module section comment): df = 4/nu,
    and Var(Y) = Psi * 2/(2 - nu) exists only for nu < 2. run_tlasso wants the
    df, so it gets 4/nu, never this.

    Returns (Y, tau) with tau of shape (n,) -- one per observation, NOT the
    (n, p) the t* generators return.
    """
    Psi_true = np.linalg.inv(Theta_true)
    nu = float(nu)
    if nu <= 0:
        raise ValueError("nu must be positive; nu -> 0 is the Gaussian limit")
    # The same draw simulate_aat_data makes, one per row instead of per cell.
    # numpy's gamma takes the SCALE, so the rate beta goes in as 1/beta.
    alpha = 2.0 / nu
    beta = 2.0 / nu
    tau = 1.0 / rng.gamma(shape=alpha, scale=1.0 / beta, size=n)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    Y = mu[None, :] + np.sqrt(tau)[:, None] * X
    return Y, tau

def simulate_alternative_t_data(n, p, Theta_true, mu, nu, rng = None):
    """Modified alternative t: Finegold & Drton's t* with a per-coordinate nu_j.

        Y_j = mu_j + sqrt(tau_j) * X_j,
        tau_j ~ Inv-Gamma(2/nu_j, 2/nu_j)  independent across j,
        X ~ N_p(0, Psi),  Psi = Theta^-1.

    F&D's alternative t (Sec. 5) draws one tau_ij per coordinate but ties every
    coordinate to a single df; the modification here is that each tau_j carries
    its own nu_j, so the tails vary across coordinates while the marginals stay
    symmetric. That is the point of having it: it separates "the df is not one
    number" from "the marginals are skewed" as reasons a scalar-nu method loses
    to ours, which simulate_aat_data confounds.

    `nu` is a length-p vector in aat units (see the module section comment), so
    it is the SAME vector an aat design already has: pass nu_true straight
    through and the tails match simulate_aat_data coordinate for coordinate,
    with only the eta term gone. A scalar broadcasts, which is F&D's t* exactly.

    Marginally Y_j = mu_j + sqrt(Psi_jj) * t_{4/nu_j}, so Var(Y_j) =
    Psi_jj * 2/(2 - nu_j) exists only for nu_j < 2. A design mixing nu_j across
    that threshold has some coordinates with no variance at all, which is
    legitimate but worth doing on purpose rather than by accident.

    Whatever the t-methods get as their one scalar df is wrong for every
    coordinate unless the nu_j happen to agree; median(4/nu) is the summary that
    converts per coordinate and only then reduces.

    Returns (Y, tau), both (n, p), matching simulate_aat_data.
    """
    Psi_true = np.linalg.inv(Theta_true)
    nu = np.broadcast_to(np.asarray(nu, dtype=float), (p,))
    if np.any(nu <= 0):
        raise ValueError("nu must be positive; nu -> 0 is the Gaussian limit")
    # The same draw simulate_aat_data makes. numpy's gamma takes the SCALE, so
    # the rate beta goes in as 1/beta.
    alpha = 2.0 / nu
    beta = 2.0 / nu
    G = rng.gamma(shape=alpha, scale=1.0 / beta, size=(n, p))
    tau = 1.0 / G
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    Y = mu[None, :] + np.sqrt(tau) * X
    return Y, tau


def simulate_gaussian_data(n,p, Theta_true, rng = None):
    Psi_true = np.linalg.inv(Theta_true)
    Y = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    return Y


def simulate_noisy_gaussian_data(n, p, Theta_true,
                  skewness=0.0, 
                  outlier_frac=0.0, 
                  outlier_scale=4.0,
                  noise_scale=0.0,
                  rng = None):
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
    
    # 1. Generate multivariate Gaussian data
    Psi_true = np.linalg.inv(Theta_true)
    Y = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    
    # 2. Add skewness
    if skewness != 0:
        Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
        Y = np.sinh(np.arcsinh(Y_std) + skewness)
        Y = Y * Y.std(axis=0, ddof=1) + Y.mean(axis=0)
    
    # 3. Add outliers
    n_outliers = int(n * outlier_frac)
    if n_outliers > 0:
        idx = rng.choice(n, size=n_outliers, replace=False)
        scale = outlier_scale * Y.std(axis=0, ddof=1)
        Y[idx] = rng.normal(loc=0, scale=scale, size=(n_outliers, p))
    
    # 4. 
    if noise_scale > 0:
        noise = rng.normal(loc=0, 
                                 scale=noise_scale * Y.std(axis=0, ddof=1), 
                                 size=(n, p))
        Y = Y + noise
    
    return Y

"""
Contaminated-normal data generator.

Finegold & Drton (2011), "Robust graphical modeling of gene networks using
classical and alternative t-distributions", AOAS 5(2), Section 6.1.
"""

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

def make_true_theta(p, rng, prob=0.01, min_eig=0.6):
    """Random sparse precision matrix.

    (a) lower-triangular entries iid in {-1, 0, 1} w.p. {1%, 98%, 1%}
    (b) symmetrize
    (c) theta_kk = 1 + h_k  (h_k = # nonzeros in row k)
    then shrink the diagonal by the largest common factor keeping
    lambda_min(Theta) == min_eig.
    """


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

if __name__ == "__main__":
    n, p = 2000, 20
    
    rng = np.random.default_rng()
    mu_true  = rng.uniform(low=-5, high=5, size=p)
    eta_true = rng.uniform(low=-5, high=5, size=p)
    nu_true = rng.uniform(low=0.15, high=0.9, size=p)
    Theta_true = make_true_theta(p)

    # Swap these two lines for whichever generator you want to look at. The t
    # generators share the aat convention, so nu_true goes to them unchanged and
    # the tails are the same as the aat draw below -- only the eta term differs.
    # Their panels should come back with skew ~ 0, which is the whole point.
    # name = "contaminated_normal"
    # Y = simulate_contaminated_normal_data(n, p, true_theta)
    # name = "classical_t"
    # Y, _ = simulate_classical_t_data(n, p, Theta_true, mu_true,
    #                                  float(np.median(nu_true)), rng)
    # name = "alternative_t"
    # Y, _ = simulate_alternative_t_data(n, p, Theta_true, mu_true, nu_true, rng)
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
