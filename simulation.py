import numpy as np
from scipy.special import kve, digamma, gammaln
from scipy.optimize import brentq
from scipy.stats import skew
from sklearn.covariance import graphical_lasso


def gig_moment(r, lam, chi, psi):

    x = np.sqrt(chi * psi)

    log_ratio = (
        np.log(kve(lam + r, x))
        -
        np.log(kve(lam, x))
    )

    log_scale = (r / 2.0) * np.log(chi / psi)

    out = np.exp(log_scale + log_ratio)

    return np.clip(out, 1e-12, 1e12)


def gig_log_moment_fd(lam, chi, psi, h=1e-4):
    x = np.sqrt(chi * psi)
    log_num = np.log(kve(lam + h, x))
    log_den = np.log(kve(lam - h, x))
    return 0.5 * np.log(chi / psi) + (log_num - log_den) / (2 * h)

simulation_settings = {
    "baseline": {
        "eta": 0.5,
        "nu": 0.5,
    },
    "eta_small": {
        "eta": 1e-4,
        "nu": 0.5,
    },
    "eta_large": {
        "eta": 5.0,
        "nu": 0.5,
    },
    "nu_small": {
        "eta": 0.5,
        "nu": 0.05,
    },
    "nu_large": {
        "eta": 0.5,
        "nu": 20.0,
    },
    "eta_small_nu_small": {
        "eta": 1e-4,
        "nu": 0.05,
    },
    "eta_small_nu_large": {
        "eta": 1e-4,
        "nu": 20.0,
    },
    "eta_large_nu_small": {
        "eta": 5.0,
        "nu": 0.05,
    },
    "eta_large_nu_large": {
        "eta": 5.0,
        "nu": 20.0,
    },
}


def Q_function(Y, nu, Theta, M_neg1, L_log, S_tau):
    """
    Expected complete-data log likelihood (up to an additive constant).
    Used only for monitoring EM convergence.
    """
    n, p = Y.shape

    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return -np.inf

    Q = 0.5 * n * logdet
    Q -= 0.5 * n * np.trace(Theta @ S_tau)

    a = 2.0 / nu

    Q += np.sum(
        n * (a * np.log(a) - gammaln(a))
        - a * L_log.sum(axis=0)
        - a * M_neg1.sum(axis=0)
    )

    return Q

def _solve_nu(S_j, n, p):

    def stationarity(nu_j, S):
        a = 2.0 / nu_j
        return n * (np.log(a) + 1.0 - digamma(a)) - S

    nu_new = np.empty(p)

    for j in range(p):
        f = lambda x: stationarity(x, S_j[j])

        try:
            nu_new[j] = brentq(
                f,
                1e-4,
                100.0,
                xtol=1e-6
            )

        except ValueError:
            # fallback instead of crashing
            nu_new[j] = np.clip(
                2.0 / max(S_j[j] / n, 1e-8),
                1e-3,
                100.0
            )

    return nu_new


def run_em_diagonal(Y, n_iter=60, rho=0.05, verbose=True, err=1e-3, run_until_convergence=False):
    n, p = Y.shape
    mu = Y.mean(axis=0)
    gamma = np.full(p, 0.5)
    nu = np.full(p, 0.5)
    eta = gamma / nu
    theta_bar = 1.0 / Y.var(axis=0)
    Theta = np.diag(theta_bar)

    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": []}
    it = 0
    diff = np.inf

    while True:
     
        lam = -2.0 / nu - 0.5                                
        chi = 4.0 / nu[None, :] + theta_bar[None, :] * (Y - mu[None, :]) ** 2  
        psi = np.clip(theta_bar * eta ** 2 * nu ** 2, 1e-12, None)      

        M_neg1 = gig_moment(-1.0, lam[None, :], chi, psi[None, :])
        M_pos1 = gig_moment(1.0, lam[None, :], chi, psi[None, :])
        M_neg_half = gig_moment(-0.5, lam[None, :], chi, psi[None, :])
        M_pos_half = gig_moment(0.5, lam[None, :], chi, psi[None, :])
        L_log = gig_log_moment_fd(lam[None, :], chi, psi[None, :])

        Aj = M_neg1.sum(axis=0)          
        Bj = M_pos1.sum(axis=0)          
        Rj = (M_neg1 * Y).sum(axis=0)    
        Tj = Y.sum(axis=0)               

        denom = Aj * Bj - n ** 2
        denom = np.maximum(denom, 1e-4 * n ** 2)
        mu_new = (Bj * Rj - n * Tj) / denom
        gamma_new = (Aj * Tj - n * Rj) / denom
        gamma_new = np.clip(gamma_new, -20, 20)

        # Update parameters nu, eta
        S_j = (L_log + M_neg1).sum(axis=0)
        nu_new = _solve_nu(S_j, n, p)
        nu_new = np.clip(nu_new, 1e-3, 100)
        eta_new = gamma_new / nu_new
        eta_new = np.clip(
    eta_new,
    -20,
    20
)

       
        z_mean = (
            M_neg_half * (Y - mu_new[None, :])
            - M_pos_half * gamma_new[None, :]
        )
        S_tau = (z_mean.T @ z_mean) / n

        S_diag = (
            M_neg1 * (Y - mu_new[None, :]) ** 2
            + M_pos1 * gamma_new[None, :] ** 2
            - 2.0 * (Y - mu_new[None, :]) * gamma_new[None, :]
        ).mean(axis=0)

        np.fill_diagonal(S_tau, S_diag)
        S_tau = (S_tau + S_tau.T) / 2.0
        S_tau += 1e-10 * np.eye(p)


        try:
            cov_glasso, Theta_new = graphical_lasso(S_tau, alpha=rho, max_iter=200)
        except Exception as e:
            if verbose:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            Theta_new = Theta

        theta_bar_new = np.diag(Theta_new).copy()
        theta_bar_new = np.clip(theta_bar_new, 1e-10, None)

       
        diff = max(
            np.max(np.abs(mu_new - mu)),
            np.max(np.abs(gamma_new - gamma)),
            np.max(np.abs(nu_new - nu)),
            np.max(np.abs(theta_bar_new - theta_bar)),
        )

        mu, gamma, nu, eta = mu_new, gamma_new, nu_new, eta_new
        theta_bar, Theta = theta_bar_new, Theta_new

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(theta_bar.copy())

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | param-change {diff:.10f}")

        if (diff < err and it > 5) or (not run_until_convergence and it >= n_iter):
            if verbose:
                print(f"Converged at iteration {it}.")
            break
        it += 1

    return {"mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "history": hist}


def run_em_exact(
    Y,
    n_iter=100,
    rho=0.05,
    verbose=True,
    err=1e-4,
    run_until_convergence=True,
    max_iter=300
):
    n, p = Y.shape
    mu = Y.mean(axis=0)
    nu = np.full(p, 0.5) 
    scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
    sample_skew = skew(Y, axis=0, bias=False)
    gamma = 0.1 * scale * np.tanh(sample_skew)
    eta = gamma / nu
    Theta = np.diag(1.0 / Y.var(axis=0))

    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": []}
    it = 0
    prev_Q = -np.inf
    diff = np.inf

    while it < max_iter:
        theta_diag = np.diag(Theta)


        lam = -2.0 / nu - 0.5
        chi = 4.0 / nu[None, :] + theta_diag[None, :] * (Y - mu[None, :]) ** 2
        psi = np.clip(theta_diag * eta ** 2 * nu ** 2, 1e-12, None)

        # Compute expectations
        M_neg1 = gig_moment(-1.0, lam[None, :], chi, psi[None, :])
        M_pos1 = gig_moment(1.0, lam[None, :], chi, psi[None, :])
        M_neg_half = gig_moment(-0.5, lam[None, :], chi, psi[None, :])
        M_pos_half = gig_moment(0.5, lam[None, :], chi, psi[None, :])
        L_log = gig_log_moment_fd(lam[None, :], chi, psi[None, :])

        Theta_off = Theta.copy()
        np.fill_diagonal(Theta_off, 0.0)

        H_mu_mu = Theta * (M_neg_half.T @ M_neg_half)
        np.fill_diagonal(H_mu_mu, theta_diag * M_neg1.sum(axis=0))

        H_mu_gamma = Theta * (M_neg_half.T @ M_pos_half)
        np.fill_diagonal(H_mu_gamma, n * theta_diag)

        H_gamma_mu = Theta * (M_pos_half.T @ M_neg_half)
        np.fill_diagonal(H_gamma_mu, n * theta_diag)

        H_gamma_gamma = Theta * (M_pos_half.T @ M_pos_half)
        np.fill_diagonal(H_gamma_gamma, theta_diag * M_pos1.sum(axis=0))

        weighted_Y = M_neg_half * Y
        b_mu = (
            theta_diag * (M_neg1 * Y).sum(axis=0)
            + (M_neg_half * (weighted_Y @ Theta_off.T)).sum(axis=0)
        )
        b_gamma = (
            theta_diag * Y.sum(axis=0)
            + (M_pos_half * (weighted_Y @ Theta_off.T)).sum(axis=0)
        )

        H = np.block([
            [H_mu_mu, H_mu_gamma],
            [H_gamma_mu, H_gamma_gamma],
        ])
        b = np.concatenate([b_mu, b_gamma])
        mu_gamma_new = np.linalg.solve(H, b)
        mu_new = mu_gamma_new[:p]
        gamma_new = mu_gamma_new[p:]

  
        S_j = (L_log + M_neg1).sum(axis=0)
        nu_new = _solve_nu(S_j, n, p)
        eta_new = gamma_new / nu_new


        z_mean = (
            M_neg_half * (Y - mu_new[None, :])
            - M_pos_half * gamma_new[None, :]
        )
        S_tau = (z_mean.T @ z_mean) / n

        S_diag = (
            M_neg1 * (Y - mu_new[None, :]) ** 2
            + M_pos1 * gamma_new[None, :] ** 2
            - 2.0 * (Y - mu_new[None, :]) * gamma_new[None, :]
        ).mean(axis=0)

        np.fill_diagonal(S_tau, S_diag)
        S_tau = (S_tau + S_tau.T) / 2.0
        S_tau += 1e-10 * np.eye(p)


        try:
            _, Theta_new = graphical_lasso(S_tau, alpha=rho, max_iter=1000)
        except Exception as e:
            if verbose:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            Theta_new = Theta

        Q = Q_function(Y, nu_new, Theta_new, M_neg1, L_log, S_tau)

        diff = max(
        np.max(np.abs(mu_new - mu)),
        np.max(np.abs(gamma_new - gamma)),
        np.max(np.abs(nu_new - nu)),
        np.max(np.abs(eta_new - eta)),
        np.max(np.abs(Theta_new - Theta)),
    )       
       
        mu, gamma, nu, eta = mu_new, gamma_new, nu_new, eta_new
        Theta = Theta_new

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(np.diag(Theta).copy())

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | Q = {Q:.8f} | \u0394Q = {diff:.3e}")

        if diff < err and it > 5:
            if verbose:
                print(f"Converged at iteration {it}")
            break
        it += 1

    return {"mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "history": hist}
