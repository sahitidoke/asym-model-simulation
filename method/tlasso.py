import numpy as np
from scipy.special import gammaln
from sklearn.covariance import graphical_lasso


def run_tlasso(Y, nu=3.0, n_iter=60, rho=0.05, init = None, verbose=True):
    """Finegold & Drton (2011), Sec. 4. Classical multivariate t: one divisor
    tau_i per observation, so whole observations get downweighted."""
    n, p = Y.shape

    if init is None:
        mu = Y.mean(axis=0)
        theta_bar = 1.0 / Y.var(axis=0)
        Theta = np.diag(theta_bar)
    else:
        mu = init["mu"]
        Theta = init["Theta"]

    hist = {"mu": [], "theta_diag": [], "tau": []}
    it = 0
    for it in range(n_iter):
        # E-step: tau_i = (nu + p) / (nu + delta_i)
        Z = Y - mu[None, :]
        delta = np.einsum("ij,jk,ik->i", Z, Theta, Z)
        tau = (nu + p) / (nu + delta)

        # Update parameter mu
        mu_new = (tau @ Y) / tau.sum()

        # Compute the expected S given mu
        z_mean = np.sqrt(tau)[:, None] * (Y - mu_new[None, :])
        S_tau = (z_mean.T @ z_mean) / n

        S_tau = (S_tau + S_tau.T) / 2.0
        S_tau += 1e-10 * np.eye(p)
        
        # Glasso step to estimate Theta. sklearn penalizes off-diagonals only;
        # adding rho*I recovers the paper's fully penalized objective, since
        # tr(S Theta) + rho * sum_j theta_jj = tr((S + rho I) Theta).
        try:
            cov_glasso, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p), alpha=rho, max_iter=200, tol=1e-3
            )
        except Exception as e:
            if verbose:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            Theta_new = Theta

        theta_bar_new = np.clip(np.diag(Theta_new).copy(), 1e-10, None)
        diff = np.abs(mu_new - mu).sum() + np.abs(Theta_new - Theta).sum()

        mu = mu_new
        theta_bar, Theta = theta_bar_new, Theta_new

        hist["mu"].append(mu.copy())
        hist["theta_diag"].append(theta_bar.copy())
        hist["tau"].append(tau.copy())

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | param-change {diff:.10f}")

    return {"mu": mu, "nu": nu, "Theta": Theta, "tau": tau, "history": hist,
            "S_tau": S_tau}


def run_tstar_varlasso(Y, nu=3.0, n_iter=60, rho=0.05, init=None, verbose=True):
    """Finegold & Drton (2011), Sec. 5.3. Alternative t: one divisor tau_ij per
    coordinate, mean-field E-step replacing Theta by its diagonal. Same shape as
    the skewed version, but with eta = 0 the GIG posterior collapses to
    Gamma(alpha, beta) and the moments are closed form."""
    n, p = Y.shape
    if init is None:
        mu = Y.mean(axis=0)
        theta_bar = 1.0 / Y.var(axis=0)
        Theta = np.diag(theta_bar)
    else:
        mu = init["mu"]
        Theta = init["Theta"]
        theta_bar = np.clip(np.diag(Theta).copy(), 1e-10, None)

    alpha = (nu + 1.0) / 2.0
    log_ratio = gammaln(alpha + 0.5) - gammaln(alpha)

    hist = {"mu": [], "theta_diag": [], "tau": []}
    it = 0
    for it in range(n_iter):
        # Compute Gamma posterior parameters
        beta = (nu + theta_bar[None, :] * (Y - mu[None, :]) ** 2) / 2.0

        # Compute expectations
        M_pos1 = alpha / beta
        M_pos_half = np.exp(log_ratio) / np.sqrt(beta)

        # Update parameter mu
        mu_new = (M_pos1 * Y).sum(axis=0) / M_pos1.sum(axis=0)

        # Compute the expected S given mu, reusing the moments from the E-step
        z_mean = M_pos_half * (Y - mu_new[None, :])
        S_tau = (z_mean.T @ z_mean) / n

        S_diag = (M_pos1 * (Y - mu_new[None, :]) ** 2).mean(axis=0)

        np.fill_diagonal(S_tau, S_diag)
        S_tau = (S_tau + S_tau.T) / 2.0
        S_tau += 1e-10 * np.eye(p)

        # Glasso step to estimate Theta. sklearn penalizes off-diagonals only;
        # adding rho*I recovers the paper's fully penalized objective, since
        # tr(S Theta) + rho * sum_j theta_jj = tr((S + rho I) Theta).
        try:
            cov_glasso, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p), alpha=rho, max_iter=200, tol=1e-2
            )
        except Exception as e:
            if verbose:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            Theta_new = Theta

        theta_bar_new = np.clip(np.diag(Theta_new).copy(), 1e-10, None)
        diff = np.abs(mu_new - mu).sum() + np.abs(Theta_new - Theta).sum()

        mu = mu_new
        theta_bar, Theta = theta_bar_new, Theta_new

        hist["mu"].append(mu.copy())
        hist["theta_diag"].append(theta_bar.copy())
        hist["tau"].append(M_pos1.copy())

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | param-change {diff:.10f}")

    return {"mu": mu, "nu": nu, "Theta": Theta, "tau": M_pos1, "history": hist,
            "S_tau": S_tau}
