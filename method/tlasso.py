import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln
from sklearn.covariance import graphical_lasso

# Bracket for the degrees-of-freedom root, shared by both fits in this module.
# Finegold & Drton's nu runs the other way from EM_algorithm's: here
# tau ~ Gamma(nu/2, nu/2), so SMALL nu is the heavy tail and nu -> infinity is
# the Gaussian limit. NU_MAX is therefore "indistinguishable from Gaussian at
# this sample size", not a heavy-tail ceiling, and NU_MIN is the tail floor.
NU_MIN, NU_MAX = 0.1, 100.0


def _solve_nu_ecm(gap):
    """The CM step for nu, given gap = mean(E[tau] - E[log tau]) over the
    latent tau's: np of them for t*, n of them for the classical t.

    Differentiating the nu-block of the t* complete-data log-likelihood,

        np * (nu/2 log(nu/2) - lgamma(nu/2))
        + (nu - 1)/2 sum_ij E[log tau_ij] - nu/2 sum_ij E[tau_ij],

    and dividing by np/2 gives the stationarity condition

        g(nu) = log(nu/2) + 1 - digamma(nu/2) - gap = 0.

    The classical-t nu-block differs only in carrying n terms and a log tau
    coefficient of (nu + p - 2)/2 -- the determinant contributes p/2 log tau_i
    on n latents instead of 1/2 log tau_ij on np of them, the same total weight
    np/2 -- and the p drops out on differentiating, landing on the same g with
    the mean over i alone. So both fits share this solver.

    log(a) - digamma(a) is positive and strictly decreasing to 0, so g falls
    monotonically from +inf at nu = 0 to 1 - gap as nu -> infinity: at most one
    root, and the two ways out of the bracket are opposite limits.

      * g(NU_MAX) >= 0, i.e. gap at its low end. Since
        gap = mean(E[tau] - E[log tau]) >= E[tau] - log E[tau] >= 1 by Jensen,
        with equality only at tau == 1, this end is the Gaussian limit and the
        root is at nu = +inf. Clamp to NU_MAX rather than reporting a number
        the data cannot support.
      * g(NU_MIN) <= 0: tails heavier than nu = NU_MIN. Clamp to the other end.

    Unlike EM_algorithm._solve_nu_eta there is no nu = 0 escape hatch: nu here
    is one scalar for the whole fit, and 0 is the heavy end, so the Gaussian
    limit is NU_MAX.
    """
    def g(nu):
        a = nu / 2.0
        return np.log(a) + 1.0 - digamma(a) - gap

    if g(NU_MAX) >= 0.0:            # Gaussian limit: tau == 1
        return NU_MAX
    if g(NU_MIN) <= 0.0:            # heavier than the bracket
        return NU_MIN
    return brentq(g, NU_MIN, NU_MAX, xtol=1e-6)


def run_tlasso(Y, nu=3.0, n_iter=60, rho=0.05, init = None, verbose=True,
               tol=None, warning=True):
    """Finegold & Drton (2011), Sec. 4. Classical multivariate t: one divisor
    tau_i per observation, so whole observations get downweighted.

    A numeric `nu` is held for the whole fit, which is (4.2): nu then enters
    only through the posterior, so the parts of the complete-data likelihood
    that depend on it are constant and get dropped. `nu=None` estimates it
    instead, keeping those parts -- the p/2 sum_i log tau_i from the Gaussian
    determinant, plus the whole Gamma density -- and adding one ECM step, the
    same one run_tstar_varlasso takes with mean_ij replaced by mean_i. That
    block involves nu alone, no mu and no Theta, so (3.5) and the glasso call
    on S_tauYY stand exactly as in Sec. 4; the only extra E-step quantity is
    E[log tau_i] = digamma((nu + p)/2) - log((nu + delta_i)/2). The E-step here
    is exact, with no variational step, so unlike run_tstar_varlasso the
    estimate carries no bias from an approximate posterior. The estimation path
    starts from init["nu"] if a warm start carries one, else from the same 3.0
    the fixed path defaults to.

    Stops once the relative L1 change of (mu, Theta) in one EM step drops
    below `tol`; n_iter is then a cap, not a fixed cost. The 1e-5 default is
    the loosest power of ten at which the returned edge set was identical to
    running all 200 iterations, on every fit of a warm-started rho sweep over
    contaminated-normal data (p=100, n=50); it typically fires within ~10-30
    iterations. tol=0 restores the old fixed-n_iter behavior. Estimating nu
    adds iterations, and adds most of them at heavy tails, where nu and Theta
    take longest to settle on each other: on simulated t data (p=15, n=500,
    rho=0.05) 51 -> 95 iterations at nu = 1, 36 -> 42 at nu = 3, and no cost at
    nu = 10. nu is deliberately excluded from the stopping criterion, as in
    run_em_diagonal: it is a tail-shape nuisance parameter and the target is
    the edge set, so a nu still drifting inside the bracket should not keep a
    settled edge set iterating."""
    n, p = Y.shape
    estimate_nu = nu is None

    if init is None:
        mu = Y.mean(axis=0)
        theta_bar = 1.0 / Y.var(axis=0)
        Theta = np.diag(theta_bar)
    else:
        mu = init["mu"]
        Theta = init["Theta"]

    if estimate_nu:
        nu = 3.0 if init is None else float(init.get("nu", 3.0))

    hist = {"mu": [], "theta_diag": [], "tau": [], "nu": []}
    # Counts iterations whose Theta is STALE because the glasso call raised:
    # what gets reported at this rho is then not the fit that was requested.
    n_glasso_fail = 0
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
        #
        # enet_tol (inner coordinate-descent accuracy) has to sit well below
        # tol (outer dual-gap target). sklearn checks |dual gap| < tol, but the
        # gap is computed from a precision matrix that the inner solver only
        # resolved to enet_tol, which puts a floor of a few times enet_tol
        # under |gap|. At the 1e-4 default that floor straddles tol=1e-3: the
        # gap stalls (often negative, e.g. -2.7e-3) and every call burns all
        # max_iter sweeps at the optimum, which is what produced the
        # ConvergenceWarning storm. 1e-6 puts the floor ~1e-5 and the outer
        # loop exits in a handful of sweeps; 1e-8 is no better.
        try:
            cov_glasso, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p), alpha=rho, max_iter=200, tol=1e-3,
                enet_tol=1e-6,
            )
        except Exception as e:
            # `warning`, not `verbose` -- see run_em_diagonal. The ROC sweeps
            # run verbose=False, and a stale Theta being recorded as a fit is
            # exactly what a sweep needs to be told about.
            if warning:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            n_glasso_fail += 1
            Theta_new = Theta

        # CM step for nu, off the same E-step moments the glasso call used.
        # The nu-block is additively separable from (mu, Theta), so this runs
        # after the glasso call without feeding back into it this iteration.
        if estimate_nu:
            # E[log tau_i], with tau_i | Y_i ~ Gamma((nu + p)/2, (nu + d_i)/2)
            L_log = digamma((nu + p) / 2.0) - np.log((nu + delta) / 2.0)
            nu_new = _solve_nu_ecm(float(np.mean(tau - L_log)))
        else:
            nu_new = nu

        theta_bar_new = np.clip(np.diag(Theta_new).copy(), 1e-10, None)
        diff = np.abs(mu_new - mu).sum() + np.abs(Theta_new - Theta).sum()
        rel_change = diff / (np.abs(mu).sum() + np.abs(Theta).sum())

        mu = mu_new
        nu = nu_new
        theta_bar, Theta = theta_bar_new, Theta_new

        hist["mu"].append(mu.copy())
        hist["theta_diag"].append(theta_bar.copy())
        hist["tau"].append(tau.copy())
        hist["nu"].append(nu)

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | param-change {diff:.10f} | nu {nu:.4f}")

        if tol is not None and rel_change < tol:
            if verbose:
                print(f"converged at iter {it}: "
                      f"rel change {rel_change:.2e} < tol {tol:.0e}")
            break

    return {"mu": mu, "nu": nu, "Theta": Theta, "tau": tau, "history": hist,
            "S_tau": S_tau, "n_glasso_fail": n_glasso_fail,
            "n_iter_run": it + 1}


def run_tstar_varlasso(Y, nu=3.0, n_iter=60, rho=0.05, init=None, verbose=True,
                       tol=None, warning=True):
    """Finegold & Drton (2011), Sec. 5.3. Alternative t: one divisor tau_ij per
    coordinate, mean-field E-step replacing Theta by its diagonal. Same shape as
    the skewed version, but with eta = 0 the GIG posterior collapses to
    Gamma(alpha, beta) and the moments are closed form.

    A numeric `nu` is held for the whole fit, which is (5.1): nu then enters
    only through the posterior, so the parts of the complete-data likelihood
    that depend on it are constant and get dropped. `nu=None` estimates it
    instead, keeping those parts -- the 1/2 sum_ij log tau_ij from the Gaussian
    determinant, plus the whole Gamma density -- and adding one ECM step. That
    block involves nu alone, no mu and no Theta, so the mu-update and the
    glasso call are untouched; the only extra E-step quantity is
    E[log tau_ij] = digamma(alpha) - log(beta_ij). nu is a single scalar shared
    across coordinates here, unlike EM_algorithm's per-coordinate nu_j. The
    estimation path starts from init["nu"] if a warm start carries one, else
    from the same 3.0 the fixed path defaults to.

    Read the estimated nu as the mean-field model's tail parameter, not the
    data's. The variational posterior scores residuals against theta_jj, but
    Y_ij actually has conditional variance Sigma_jj / tau_ij, so beta_ij is
    inflated by theta_jj Sigma_jj >= 1 -- exactly 1 only when Theta is
    diagonal. That inflation pushes E[tau] down and the CM step's gap up, i.e.
    nu down. On simulated t* data (p=15, n=2000, rho=0.01) nu came back at
    1.00/3.11/10.8 for a true 1/3/10 with Theta diagonal, and at
    0.79/1.55/2.16 with a 0.4 chain (theta_jj Sigma_jj = 1.59). The bias grows
    with both the correlation and nu itself, so a large nu_hat is evidence of
    light tails but its value is not calibrated.

    `tol` stops the EM on relative (mu, Theta) change, same rule and
    calibration as run_tlasso; with nu held this method converges even faster
    than run_tlasso (~5-10 iterations), while estimating nu costs roughly 3x
    the iterations, since nu and Theta move each other through beta. nu is
    deliberately excluded from that criterion, as in run_em_diagonal: it is a
    tail-shape nuisance parameter and the target is the edge set, so a nu still
    drifting inside the bracket should not keep a settled edge set iterating."""
    n, p = Y.shape
    estimate_nu = nu is None

    if init is None:
        mu = Y.mean(axis=0)
        theta_bar = 1.0 / Y.var(axis=0)
        Theta = np.diag(theta_bar)
    else:
        mu = init["mu"]
        Theta = init["Theta"]
        theta_bar = np.clip(np.diag(Theta).copy(), 1e-10, None)

    if estimate_nu:
        nu = 3.0 if init is None else float(init.get("nu", 3.0))

    hist = {"mu": [], "theta_diag": [], "tau": [], "nu": []}
    # Counts iterations whose Theta is STALE because the glasso call raised:
    # what gets reported at this rho is then not the fit that was requested.
    n_glasso_fail = 0
    it = 0
    for it in range(n_iter):
        # alpha/beta move with nu, so both are rebuilt every iteration; with nu
        # held they come out the same numbers each time.
        alpha = (nu + 1.0) / 2.0
        log_ratio = gammaln(alpha + 0.5) - gammaln(alpha)

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
            # See run_tlasso for why enet_tol is pinned below tol.
            cov_glasso, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p), alpha=rho, max_iter=200, tol=1e-3,
                enet_tol=1e-6,
            )
        except Exception as e:
            # `warning`, not `verbose` -- see run_em_diagonal. The ROC sweeps
            # run verbose=False, and a stale Theta being recorded as a fit is
            # exactly what a sweep needs to be told about.
            if warning:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            n_glasso_fail += 1
            Theta_new = Theta

        # CM step for nu, off the same E-step moments the glasso call used.
        # The nu-block is additively separable from (mu, Theta), so this runs
        # after the glasso call without feeding back into it this iteration.
        if estimate_nu:
            L_log = digamma(alpha) - np.log(beta)     # E[log tau_ij]
            nu_new = _solve_nu_ecm(float(np.mean(M_pos1 - L_log)))
        else:
            nu_new = nu

        theta_bar_new = np.clip(np.diag(Theta_new).copy(), 1e-10, None)
        diff = np.abs(mu_new - mu).sum() + np.abs(Theta_new - Theta).sum()
        rel_change = diff / (np.abs(mu).sum() + np.abs(Theta).sum())

        mu = mu_new
        nu = nu_new
        theta_bar, Theta = theta_bar_new, Theta_new

        hist["mu"].append(mu.copy())
        hist["theta_diag"].append(theta_bar.copy())
        hist["tau"].append(M_pos1.copy())
        hist["nu"].append(nu)

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | param-change {diff:.10f} | nu {nu:.4f}")

        if tol is not None and rel_change < tol:
            if verbose:
                print(f"converged at iter {it}: "
                      f"rel change {rel_change:.2e} < tol {tol:.0e}")
            break

    return {"mu": mu, "nu": nu, "Theta": Theta, "tau": M_pos1, "history": hist,
            "S_tau": S_tau, "n_glasso_fail": n_glasso_fail,
            "n_iter_run": it + 1}
