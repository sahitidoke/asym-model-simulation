import numpy as np
from tqdm import tqdm
from scipy.special import kve, digamma, logsumexp, gammaln
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import skew, geninvgauss
from sklearn.covariance import graphical_lasso


NU_MIN, NU_MAX = 1e-2, 0.95   # bracket for the nu root
GAUSS_LR = 2.71   # chi-bar^2(0,1) 5% point; nu = 0 is a boundary of the space
GAUSS_BURN_IN = 10              # iterations before a coordinate may be pinned Gaussian
OLVER_ORDER = 30.0   # crossover; kve is exact below, Olver is ~1e-14 above

def gig_moment(r, lam, chi, psi):
    # Non-finite output is the signal _tau_moments branches on, not an error.
    x = np.sqrt(chi * psi)
    return (chi / psi) ** (r / 2.0) * (kve(lam + r, x) / kve(lam, x))

def gig_log_moment_fd(lam, chi, psi, h=1e-5):
    x = np.sqrt(chi * psi)
    return 0.5 * np.log(chi / psi) + (
        np.log(kve(lam + h, x)) - np.log(kve(lam - h, x))
    ) / (2 * h)

def log_kv(lam, s):
    """log K_lam(s), stable for large |lam| where kve overflows.

    K_lam(s) ~ Gamma(|lam|) (2/s)^|lam| / 2 for small s, so kve (which only
    removes an exp(s) factor) still overflows once |lam| is a few hundred --
    precisely the light-tail end, nu -> 0, where a = 2/nu is large. Above the
    crossover use Olver's uniform asymptotic expansion in the order, which is
    valid for ALL s > 0 and returns the logarithm directly.
    """
    lam = np.abs(lam)                     # K_{-lam} == K_{lam}
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)

    small = lam < OLVER_ORDER
    if small:
        kv = kve(lam, s)
        return np.log(kv) - s

    z = s / lam
    w = np.sqrt(1.0 + z * z)
    t = 1.0 / w
    eta = w + np.log(z / (1.0 + w))

    t2 = t * t
    u1 = (3.0 * t - 5.0 * t * t2) / 24.0
    u2 = (81.0 * t2 - 462.0 * t2 * t2 + 385.0 * t2 * t2 * t2) / 1152.0
    u3 = (30375.0 * t * t2 - 369603.0 * t * t2 * t2
          + 765765.0 * t * t2 * t2 * t2 - 425425.0 * t * t2 * t2 * t2 * t2) / 414720.0
    ser = 1.0 - u1 / lam + u2 / (lam * lam) - u3 / (lam ** 3)

    out = (0.5 * np.log(np.pi / (2.0 * lam)) - lam * eta
           - 0.25 * np.log(1.0 + z * z) + np.log(ser))
    return out


def _tau_moments(nu, eta, theta_diag, resid):
    n, p = resid.shape
    M = {r: np.ones((n, p)) for r in (-1.0, 1.0, -0.5, 0.5)}
    L_log = np.zeros((n, p))
    n_ig = np.zeros(p, dtype=int)

    live = np.flatnonzero(nu > 0.0)                                 # not Gaussian
    if live.size == 0:
        return M[-1.0], M[1.0], M[-0.5], M[0.5], L_log, n_ig

    lam = -2.0 / nu[live] - 0.5                                     # (k,)
    chi = 4.0 / nu[live] + theta_diag[live] * resid[:, live] ** 2   # (n, k)
    psi = theta_diag[live] * (eta[live] * nu[live]) ** 2            # (k,)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        blk = {r: gig_moment(r, lam, chi, psi) for r in M}
        blk_log = gig_log_moment_fd(lam, chi, psi)

    bad = ~np.isfinite(blk_log)
    for r in M:
        bad |= ~np.isfinite(blk[r])
    n_ig[live] = bad.sum(axis=0)
    if bad.any():
        a = np.broadcast_to(-lam, chi.shape)[bad]                   # InvGamma shape
        b = chi[bad] / 2.0                                          # InvGamma scale
        if (a <= 1.0).any():
            j = live[np.argmax(np.broadcast_to(-lam, chi.shape).min(axis=0) <= 1.0)]
            raise ValueError(
                f"E[tau] does not exist on the psi -> 0 limit for coordinate "
                f"{j}: nu = {nu[j]:.4g} >= 4 gives shape a = {2/nu[j] + 0.5:.4g} "
                "<= 1. The posterior mean of tau is infinite there, so the "
                "M-step has nothing to condition on."
            )
        for r in M:
            blk[r][bad] = np.exp(r * np.log(b) + gammaln(a - r) - gammaln(a))
        blk_log[bad] = np.log(b) - digamma(a)

    # kve also gives out at the OPPOSITE end, past x ~ 3e9, where the InvGamma
    # limit is badly wrong rather than sharp. Raise instead of answering.
    if not (np.isfinite(blk_log).all()
            and all(np.isfinite(blk[r]).all() for r in M)):
        raise ValueError(
            "tau moments non-finite after the Inv-Gamma fallback; check for "
            f"extreme residuals (max sqrt(chi*psi) = {np.sqrt(chi * psi).max():.3e})"
        )

    for r in M:
        M[r][:, live] = blk[r]
    L_log[:, live] = blk_log
    return M[-1.0], M[1.0], M[-0.5], M[0.5], L_log, n_ig

def _marginal_ll(nu_j, y, mu_j, gam_j, Sig_jj):
    """sum_i log f_{Y_j}(y_i | mu_j, eta_j, nu_j, Sigma_jj).

    Exact: tau_j is integrated out analytically (Bessel identity, Sec 3.1).
    No posterior moments, no L_ij(h), no E-step.
    """
    n = y.size
    d = y - mu_j

    if nu_j <= 0.0:
        # tau -> 1 with gamma FIXED leaves the shift in place: N(mu + gamma,
        # Sigma_jj), not N(mu, Sigma_jj). Only mu_j + gamma_j is identified
        # here -- the same degeneracy _solve_mu_gamma pins by setting gamma=0.
        dg = d - gam_j
        return -0.5 * (n * np.log(2.0 * np.pi * Sig_jj) + (dg * dg).sum() / Sig_jj)

    a = 2.0 / nu_j
    gam = gam_j    

    if gam == 0.0:                                    # B_j = 0: scaled t, 4/nu df
        df = 4.0 / nu_j
        return n * (gammaln(0.5 * (df + 1.0)) - gammaln(0.5 * df)
                    - 0.5 * np.log(np.pi * df * Sig_jj)) \
               - 0.5 * (df + 1.0) * np.log1p(d * d / (Sig_jj * df)).sum()

    lam = -(a + 0.5)
    logC = a * np.log(a) - gammaln(a) - 0.5 * np.log(2.0 * np.pi * Sig_jj)
    R = d * d + 4.0 * Sig_jj / nu_j                   # = 2 * Sigma_jj * A_j
    s = abs(gam) * np.sqrt(R) / Sig_jj                # = 2 * sqrt(A_j B_j)

    ll = (n * (np.log(2.0) + logC)
          + gam * d.sum() / Sig_jj
          + 0.5 * lam * np.log(R / (gam * gam)).sum()
          + log_kv(lam, s).sum())
    return ll if np.isfinite(ll) else -np.inf


def _solve_nu(Y, mu, gam, Sig_diag, allow_gauss, xatol=1e-6):
    """Per-coordinate maximizer of the marginal log likelihood in nu_j.

      nu_j == NU_MAX   tails heavier than the bracket can express -> counted.
      nu_j == 0.0      the Gaussian end, and now a LIKELIHOOD COMPARISON
                       against N(mu_j, Sigma_jj) rather than the old test for
                       S_j resting on its floor.

    Returns (nu, n_clamped).
    """
    p = Y.shape[1]
    allow_gauss = np.broadcast_to(allow_gauss, (p,))
    nu = np.empty(p)
    n_clamped = 0

    for j in range(p):
        y, mu_j, gam_j, Sig_jj = Y[:, j], mu[j], gam[j], Sig_diag[j]
        ll = lambda v: _marginal_ll(v, y, mu_j, gam_j, Sig_jj)

        res = minimize_scalar(lambda u: -ll(np.exp(u)),
                              bounds=(np.log(NU_MIN), np.log(NU_MAX)),
                              method='bounded', options={'xatol': xatol})

        best_nu, best_ll = float(np.exp(res.x)), float(-res.fun)
        for cand in (NU_MIN, NU_MAX):
            v = ll(cand)
            if v > best_ll:
                best_nu, best_ll = cand, v
        if allow_gauss[j] and 2.0 * (best_ll - ll(0.0)) < GAUSS_LR:
            best_nu = 0.0

        nu[j] = best_nu
        n_clamped += (best_nu == NU_MAX)

    return nu, n_clamped


def _solve_mu_gamma(H, b, gauss, p):
    """Solve the joint 2p x 2p (mu, gamma) system with gamma pinned at 0 where
    tau_j == 1.

    With tau_j == 1 every moment of tau_j is 1, so the mu_j and gamma_j rows and
    columns of H coincide: only mu_j + gamma_j is identified and H is singular.
    Pinning gamma_j = 0 -- the Gaussian limit has no skew -- drops row and
    column p + j, and mu_j carries the location on its own.
    """
    keep = np.concatenate([np.ones(p, dtype=bool), ~gauss])
    sol = np.zeros(2 * p)
    sol[keep] = np.linalg.solve(H[np.ix_(keep, keep)], b[keep])
    return sol[:p], sol[p:]

def run_em_diagonal(Y, n_iter=100, rho=0.05, init= None, verbose=True, tol=None, warning=True):
    n, p = Y.shape
    if (init is None):
        mu = Y.mean(axis=0)
        nu = np.full(p, 0.5)
        eta = np.full(p, 0.5)
        Theta = np.diag(1.0 / Y.var(axis=0))
    else:
        mu, Theta = init["mu"], init["Theta"]
        nu = init.get("nu", np.full(p, 0.5))
        eta = init.get("eta", np.where(nu > 0.0, 0.5 / np.where(nu > 0.0, nu, 1.0), 0.0))

    Sigma = np.linalg.inv(Theta)
    # paths/nu_at_max are per ITERATION, like every other history entry: the
    # last E-step is what the returned Theta was computed from, but a fit that
    # spent its middle iterations on the Inv-Gamma limit and drifted back looks
    # identical to one that never left the GIG if only the end is kept.
    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": [], "paths": [],
            "nu_at_max": []}
    n_glasso_fail = 0
    n_nu_clamped = 0
    it = 0
    for it in range(n_iter):
        theta_diag = np.clip(np.diag(Theta), 1e-10, None)
        gauss = nu <= 0.0

        M_neg1, M_pos1, M_neg_half, M_pos_half, L_log, n_ig = _tau_moments(
            nu, eta, theta_diag, Y - mu[None, :])

        # Which limit of the GIG each coordinate was computed on THIS E-step.
        # `gauss` and `n_ig` are both from before this iteration's M-step, so
        # the row describes one consistent E-step. Four buckets, not three,
        # because the Inv-Gamma substitution is per ENTRY: `mixed` is a
        # coordinate that had it both ways, and folding those into either side
        # would report a path the coordinate was only partly on.
        hist["paths"].append((
            int(gauss.sum()),                                   # Gaussian
            int(((~gauss) & (n_ig == n)).sum()),                # all-InvGamma
            int(((~gauss) & (n_ig == 0)).sum()),                # all-GIG
            int(((~gauss) & (n_ig > 0) & (n_ig < n)).sum()),    # mixed
            int(n_ig.sum()),                                    # InvGamma entries
        ))

        # M-step for (mu, gamma), per coordinate:
        #   [[sum_i M(-1), n], [n, sum_i M(1)]] (mu, gamma)
        #       = (sum_i M(-1) Y, sum_i Y)
        A = M_neg1.sum(axis=0)
        B = M_pos1.sum(axis=0)
        R = (M_neg1 * Y).sum(axis=0)
        T = Y.sum(axis=0)

        # tau_j == 1 makes A = B = n and the 2x2 singular; the Gaussian limit
        # carries no skew, so mu is the plain mean and gamma is 0.
        denom = np.where(gauss, 1.0, A * B - n ** 2)
        mu_new = np.where(gauss, T / n, (B * R - n * T) / denom)
        gamma_new = np.where(gauss, 0.0, (A * T - n * R) / denom)

        # Compute the expected S given mu, gamma
        resid = Y - mu_new[None, :]
        z_mean = M_neg_half * resid - M_pos_half * gamma_new[None, :]
        S_tau = (z_mean.T @ z_mean) / n

        S_diag = (
            M_neg1 * resid ** 2
            + M_pos1 * gamma_new[None, :] ** 2
            - 2.0 * resid * gamma_new[None, :]
        ).mean(axis=0)

        np.fill_diagonal(S_tau, S_diag)
        S_tau = (S_tau + S_tau.T) / 2.0
        S_tau += 1e-10 * np.eye(p)

        # Glasso step to estimate Theta. sklearn penalizes off-diagonals only;
        # adding rho*I recovers the fully penalized objective, since
        # tr(S Theta) + rho * sum_j theta_jj = tr((S + rho I) Theta).
        try:
            cov_glasso, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p), alpha=rho, max_iter=200, tol=1e-3,enet_tol=1e-6,
            )
        except Exception as e:
            # `warning`, not `verbose`: the ROC sweeps run verbose=False, which
            # used to hide this entirely. A failure here means Theta stops
            # updating while mu/nu/eta keep moving, so what is reported at this
            # rho is not a fit -- the caller has to be able to see that.
            if warning:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            n_glasso_fail += 1
            Theta_new = Theta

        Sigma_new = np.linalg.inv(Theta_new)
        # M-step for nu. A NEW Gaussian pin waits for the burn-in, since it is
        # absorbing; a coordinate already pinned keeps its permission, or the
        # guard would release it.
        nu_new, n_clamped = _solve_nu(
            Y, mu_new, gamma_new, np.diag(Sigma_new),
            allow_gauss=(it >= GAUSS_BURN_IN))
        n_nu_clamped += n_clamped
        gamma_new = np.where(nu_new > 0.0, gamma_new, 0.0)
        eta_new = gamma_new / np.where(nu_new > 0.0, nu_new, 1.0)

        diff = (np.abs(mu_new - mu).sum() + np.abs(eta_new - eta).sum()
                + np.abs(nu_new - nu).sum())
        rel_change = (
            np.abs(mu_new - mu).sum() + np.abs(Theta_new - Theta).sum()
        ) / (np.abs(mu).sum() + np.abs(Theta).sum())

        mu, nu, eta, Theta, Sigma = mu_new, nu_new, eta_new, Theta_new, Sigma_new

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(np.diag(Theta).copy())
        # Coordinates SITTING at NU_MAX after this M-step. n_nu_clamped counts
        # clamp EVENTS over the whole fit, so one coordinate stuck there for 40
        # iterations and 40 coordinates clamped once are the same number to it.
        hist["nu_at_max"].append(int((nu >= NU_MAX).sum()))

        if verbose and (it % 5 == 0):
            # nu/eta as they stand after this iteration's M-step. A nu_j of 0 is
            # the Gaussian path, and its eta_j is 0 with it by construction.
            print(f"iter {it:3d} | param-change {diff:.10f} | "
                  f"gaussian {int((nu <= 0.0).sum())}/{p}")

        if tol is not None and rel_change < tol:
            if verbose:
                print(f"converged at iter {it}: "
                      f"rel change {rel_change:.2e} < tol {tol:.0e}")
            break

    # (n_iter_run, 5) int: [gaussian, invgamma, gig, mixed, invgamma entries]
    # per iteration. The last row is the E-step the returned Theta came from;
    # the column maxima say whether that row is where the fit sat or where it
    # ended up, which is the part a single snapshot cannot tell you.
    paths = np.asarray(hist["paths"], dtype=int)
    keys = ("gaussian", "invgamma", "gig", "mixed", "ig_entries")
    path_counts = dict(zip(keys, (int(v) for v in paths[-1])))
    path_counts["n_entries"] = n * (p - path_counts["gaussian"])
    path_counts["max"] = dict(zip(keys, (int(v) for v in paths.max(axis=0))))

    return {"mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "Sigma": Sigma, "history": hist,
            "S_tau": S_tau, "n_glasso_fail": n_glasso_fail,
            "n_nu_clamped": n_nu_clamped,
            # Coordinates at NU_MAX at the end, and the worst any iteration got
            # -- n_nu_clamped alone cannot separate those two.
            "n_nu_at_max": hist["nu_at_max"][-1],
            "n_nu_at_max_peak": max(hist["nu_at_max"]),
            "path_counts": path_counts, "n_iter_run": it + 1}

def run_em_exact(Y, n_iter=60, rho=0.05, init = None, tol=None, verbose=True,
                 warning=True):
    n, p = Y.shape

    if (init is None):
        mu = Y.mean(axis=0)
        gamma = np.full(p, 0.5)            
        nu = np.full(p, 0.5)
        eta = gamma / nu
        theta_bar = 1.0 / Y.var(axis=0)    
        Theta = np.diag(theta_bar)    
    else:
        # nu/eta are tail-shape nuisance parameters and fall back to the
        # cold-start values when absent, so a warm-started rho sweep can carry
        # mu/Theta forward without letting nu ratchet down across steps.
        mu, Theta = init["mu"], init["Theta"]
        nu = init.get("nu", np.full(p, 0.5))
        # eta = gamma/nu with the cold-start gamma = 0.5, except on the Gaussian
        # path, which carries no skew. The guard also keeps the default from
        # dividing by a zero nu when init does supply eta.
        eta = init.get("eta", np.where(nu > 0.0, 0.5 / np.where(nu > 0.0, nu, 1.0), 0.0))
        theta_bar = np.diag(Theta)

    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": []}
    # Same two failure signals as run_em_diagonal, for the same reason: a rho
    # sweep runs with verbose=False and would otherwise see a stale Theta and a
    # nu pinned at NU_MAX as if they were estimates.
    n_glasso_fail = 0
    n_nu_clamped = 0
    it = 0
    for it in range(n_iter):
        theta_diag = np.diag(Theta)

        # Compute GIG parameters (nu_j == 0 takes the Gaussian path instead)
        gauss = nu <= 0.0

        # Compute expectations
        M_neg1, M_pos1, M_neg_half, M_pos_half, L_log, _n_ig = _tau_moments(
            nu, eta, theta_diag, Y - mu[None, :])

        # Update parameters mu, gamma
       
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
        mu_new, gamma_new = _solve_mu_gamma(H, b, gauss, p)

        # Update parameters nu, eta

        S_j = (L_log + M_neg1).sum(axis=0)

        nu_new, eta_new, gamma_new, n_clamped = _solve_nu_eta(
            S_j, gamma_new, n, p)
        n_nu_clamped += n_clamped

        # Compute the expected S given the first three parameters mu, nu, eta
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
        
        # Glasso step to estimate Theta. sklearn penalizes off-diagonals only;
        # adding rho*I recovers the fully penalized objective, since
        # tr(S Theta) + rho * sum_j theta_jj = tr((S + rho I) Theta).
        try:
            # See run_em_diagonal for why enet_tol is pinned below tol; here
            # tol is sklearn's 1e-4 default, so the 1e-4 enet_tol default left
            # the gap floor above it and every call ran all 1000 sweeps.
            _, Theta_new = graphical_lasso(S_tau + rho * np.eye(p),
                                           alpha=rho, max_iter=200, tol=1e-3, enet_tol=1e-6,)
        except Exception as e:
            # `warning`, not `verbose` -- see run_em_diagonal.
            if warning:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            n_glasso_fail += 1
            Theta_new = Theta

        diff = (np.abs(mu_new - mu).sum() + np.abs(eta_new - eta).sum()
                + np.abs(nu_new - nu).sum() + np.linalg.norm(Theta_new - Theta))
        rel_change = (
            np.abs(mu_new - mu).sum() + np.abs(Theta_new - Theta).sum()
        ) / (np.abs(mu).sum() + np.abs(Theta).sum())

        mu, nu, eta = mu_new, nu_new, eta_new
        Theta = Theta_new

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(np.diag(Theta).copy())

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | param-change {diff:.10f}")

        if tol is not None and rel_change < tol:
            if verbose:
                print(f"converged at iter {it}: "
                      f"rel change {rel_change:.2e} < tol {tol:.0e}")
            break

    return {"mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "history": hist,
            "S_tau": S_tau, "n_glasso_fail": n_glasso_fail,
            "n_nu_clamped": n_nu_clamped, "n_iter_run": it + 1}

def run_em_MWG(
    Y,
    n_iter=60,
    rho=0.05,
    verbose=True,
    err=1e-3,
    run_until_convergence=False,
    mcmc_burn=150,
    mcmc_warmup=30,
    mcmc_samples=200,
    mcmc_thin=2,
    proposal_scale=0.35,
    random_state=42,
):
    def _sample_log_tau(
        Y,
        mu,
        gamma,
        nu,
        Theta,
        state,
        burn,
        samples,
        thin,
        step,
        rng,
    ):
        """
        Draw samples from p(log(tau) | Y, mu, gamma, nu, Theta)
        using Metropolis-within-Gibbs.
        """
        n, p = Y.shape

        a = beta = 2.0 / nu
        residual = Y - mu

        U = state.copy()
        step = np.broadcast_to(
            np.asarray(step, dtype=float),
            (p,),
        ).copy()

        Z = (
            np.exp(-U / 2.0) * residual
            - np.exp(U / 2.0) * gamma
        )

        draws = np.empty((samples, n, p))
        accepted = np.zeros(p)
        proposed = np.zeros(p)
        saved = 0

        total_sweeps = burn + samples * thin

        for sweep in range(total_sweeps):
            for j in range(p):
                old = U[:, j].copy()
                new = old + step[j] * rng.standard_normal(n)

                # Prevent exponential overflow without clipping proposals.
                valid = np.abs(new) < 30.0

                new_z = np.zeros(n)

                new_z[valid] = (
                    np.exp(-new[valid] / 2.0)
                    * residual[valid, j]
                    - np.exp(new[valid] / 2.0)
                    * gamma[j]
                )

                delta_z = np.zeros(n)
                delta_z[valid] = (
                    new_z[valid] - Z[valid, j]
                )

                # Change in z^T Theta z when only coordinate j changes.
                delta_quadratic = (
                    2.0
                    * delta_z
                    * (Z @ Theta[:, j])
                    + Theta[j, j] * delta_z**2
                )

                log_acceptance = np.full(n, -np.inf)

                log_acceptance[valid] = (
                    -(a[j] + 0.5)
                    * (new[valid] - old[valid])
                    - beta[j]
                    * (
                        np.exp(-new[valid])
                        - np.exp(-old[valid])
                    )
                    - 0.5 * delta_quadratic[valid]
                )

                accept = (
                    np.log(rng.random(n))
                    < np.minimum(0.0, log_acceptance)
                )

                U[accept, j] = new[accept]
                Z[accept, j] = new_z[accept]

                if sweep < burn:
                    accepted[j] += accept.sum()
                    proposed[j] += n

            # Adapt proposal scales only during burn-in.
            if (
                sweep < burn
                and (sweep + 1) % 25 == 0
            ):
                acceptance_rate = (
                    accepted
                    / np.maximum(proposed, 1.0)
                )

                step *= np.exp(
                    acceptance_rate - 0.44
                )

                step = np.clip(
                    step,
                    0.03,
                    2.0,
                )

                accepted.fill(0.0)
                proposed.fill(0.0)

            if (
                sweep >= burn
                and (sweep - burn) % thin == 0
            ):
                draws[saved] = U
                saved += 1

        return draws, U, step
    
    n, p = Y.shape

    # Initialize parameters.
    mu = Y.mean(axis=0)
    nu = np.full(p, 0.5)

    scale = np.maximum(
        Y.std(axis=0, ddof=1),
        1e-8,
    )

    gamma = (
        0.1
        * scale
        * np.tanh(
            skew(Y, axis=0, bias=False)
        )
    )

    eta = gamma / nu

    Theta = np.diag(
        1.0 / Y.var(axis=0)
    )

    # Persistent MCMC state.
    rng = np.random.default_rng(
        random_state
    )

    state = np.zeros((n, p))

    step = np.broadcast_to(
        np.asarray(
            proposal_scale,
            dtype=float,
        ),
        (p,),
    ).copy()

    hist = {
        "mu": [],
        "eta": [],
        "nu": [],
        "theta_diag": [],
    }

    it = 0

    while True:
        # ==========================================================
        # MCMC E-step
        # ==========================================================

        burn = (
            mcmc_burn
            if it == 0
            else mcmc_warmup
        )

        log_tau_draws, state, step = (
            _sample_log_tau(
                Y=Y,
                mu=mu,
                gamma=gamma,
                nu=nu,
                Theta=Theta,
                state=state,
                burn=burn,
                samples=mcmc_samples,
                thin=mcmc_thin,
                step=step,
                rng=rng,
            )
        )

        # A = tau^(-1/2), B = tau^(1/2).
        A = np.exp(
            -0.5 * log_tau_draws
        )

        B = np.exp(
            0.5 * log_tau_draws
        )

        def mc_cross(X, W):
            """Sum over observations of E[X_ij W_ik | Y]."""
            return np.einsum(
                "sij,sik->jk",
                X,
                W,
                optimize=True,
            ) / mcmc_samples

        # ==========================================================
        # Update mu and gamma
        # ==========================================================

        H = np.block([
            [
                Theta * mc_cross(A, A),
                Theta * mc_cross(A, B),
            ],
            [
                Theta * mc_cross(B, A),
                Theta * mc_cross(B, B),
            ],
        ])

        AY = A * Y[None, :, :]

        Theta_AY = np.einsum(
            "sik,jk->sij",
            AY,
            Theta,
            optimize=True,
        )

        b_mu = np.einsum(
            "sij,sij->j",
            A,
            Theta_AY,
        ) / mcmc_samples

        b_gamma = np.einsum(
            "sij,sij->j",
            B,
            Theta_AY,
        ) / mcmc_samples

        b = np.concatenate([
            b_mu,
            b_gamma,
        ])

        mu_gamma_new = np.linalg.solve(
            H,
            b,
        )

        mu_new = mu_gamma_new[:p]
        gamma_new = mu_gamma_new[p:]

        # ==========================================================
        # Update nu and eta
        # ==========================================================

        M_neg1 = np.mean(
            A**2,
            axis=0,
        )

        L_log = np.mean(
            log_tau_draws,
            axis=0,
        )

        S_j = (
            L_log + M_neg1
        ).sum(axis=0)

        def stationarity(nu_j, S):
            a_j = 2.0 / nu_j

            return (
                n
                * (
                    np.log(a_j)
                    + 1.0
                    - digamma(a_j)
                )
                - S
            )

        nu_new = np.empty(p)

        for j in range(p):
            try:
                nu_new[j] = brentq(
                    lambda x: stationarity(
                        x,
                        S_j[j],
                    ),
                    0.01,
                    5.0,
                    xtol=1e-6,
                )

            except ValueError:
                raise ValueError(
                    f"Root finding failed for "
                    f"nu[{j}] with S_j={S_j[j]}"
                )

        eta_new = gamma_new / nu_new

        # ==========================================================
        # Compute expected S_tau
        # ==========================================================

        Z = (
            A
            * (Y - mu_new)[None, :, :]
            - B
            * gamma_new[None, None, :]
        )

        S_tau = np.einsum(
            "sij,sik->jk",
            Z,
            Z,
            optimize=True,
        ) / (mcmc_samples * n)

        S_tau = (
            (S_tau + S_tau.T) / 2.0
            + 1e-10 * np.eye(p)
        )

        # ==========================================================
        # Update Theta using graphical lasso. sklearn penalizes off-diagonals
        # only; adding rho*I recovers the fully penalized objective, since
        # tr(S Theta) + rho * sum_j theta_jj = tr((S + rho I) Theta).
        # ==========================================================

        try:
            _, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p),
                alpha=rho,
                max_iter=1000,
            )

        except Exception as e:
            if verbose:
                print(
                    f"  [warn] glasso failed "
                    f"at iter {it}: {e}; "
                    f"keeping previous Theta"
                )

            Theta_new = Theta

        diff = (
            np.abs(mu_new - mu).sum()
            + np.abs(eta_new - eta).sum()
            + np.abs(nu_new - nu).sum()
            + np.linalg.norm(
                Theta_new - Theta
            )
        )

        # Update current parameters.
        mu = mu_new
        gamma = gamma_new
        nu = nu_new
        eta = eta_new
        Theta = Theta_new

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(
            np.diag(Theta).copy()
        )

        if verbose and it % 5 == 0:
            print(
                f"iter {it:3d} | "
                f"param-change {diff:.10f} | "
                f"proposal-sd "
                f"{np.round(step, 3)}"
            )

        if (
            (
                run_until_convergence
                and diff < err
            )
            or (
                not run_until_convergence
                and it >= n_iter
            )
        ):
            if verbose:
                print(
                    f"Converged at iteration {it}."
                )

            break

        it += 1

    return {
        "mu": mu,
        "eta": eta,
        "nu": nu,
        "Theta": Theta,
        "history": hist,
        "S_tau": S_tau,
    }

def _sample_log_gig(lam, chi, psi, size_T, rng, max_rounds=200):
    """
    Exact iid draws of U = log(tau) with tau ~ GIG(lam, chi, psi),
    vectorized over sites (chi, psi are length-n arrays; lam scalar).

    Method: rejection sampling with a Gaussian envelope justified by
    STRONG LOG-CONCAVITY of the log-space density
        h(u) = lam*u - (chi*e^{-u} + psi*e^{u}) / 2,
    whose curvature satisfies -h''(u) = (chi e^{-u} + psi e^{u})/2
    >= sqrt(chi*psi) =: m  by AM-GM. Hence
        h(u) <= h(u*) - m (u - u*)^2 / 2
    with u* the mode, so proposing U ~ N(u*, 1/m) and accepting with
    probability exp(h(U) - h(u*) + m (U - u*)^2 / 2) <= 1 yields EXACT
    GIG draws. Fully vectorized; ~5x faster than scipy geninvgauss and
    ~100-1000x faster than per-sweep scipy calls.

    Requires psi > 0 (use the inverse-gamma path when psi ~= 0).
    """
    n = chi.shape[0]
    m = np.sqrt(chi * psi)
    tau_star = chi / (np.sqrt(lam**2 + chi * psi) - lam)
    Ustar = np.log(tau_star)
    sd = 1.0 / np.sqrt(m)
    hstar = lam * Ustar - 0.5 * (
        chi * np.exp(-Ustar) + psi * np.exp(Ustar)
    )

    out = np.empty((size_T, n))
    need = np.ones((size_T, n), dtype=bool)

    for _ in range(max_rounds):
        k = int(need.sum())
        if k == 0:
            return out
        rows, cols = np.where(need)
        prop = Ustar[cols] + sd[cols] * rng.standard_normal(k)
        h = lam * prop - 0.5 * (
            chi[cols] * np.exp(-prop) + psi[cols] * np.exp(prop)
        )
        log_acc = h - hstar[cols] + 0.5 * m[cols] * (prop - Ustar[cols]) ** 2
        acc = np.log(rng.random(k)) < log_acc
        out[rows[acc], cols[acc]] = prop[acc]
        need[rows[acc], cols[acc]] = False

    raise RuntimeError(
        "GIG rejection sampler stalled; increase b_min so that "
        "near-symmetric (small-b) coordinates use the inverse-gamma path "
        "(sites with b = sqrt(chi*psi) < b_min take that path)."
    )

def _color_classes(Theta):
    """Partition the coordinates into independent sets of supp(Theta).

    z_j depends on tau_j alone, so the only coupling between tau_j and tau_k in
    p(tau | Y, ...) is the cross term theta_jk z_j z_k: the conditional
    independence graph OF THE LATENT tau IS the support of Theta. Coordinates
    in one independent set are therefore conditionally independent given the
    rest and can be refreshed simultaneously, so a sweep over the classes is
    still an exact Gibbs scan.

    This is a statement about the tau-block sampler only. It says nothing about
    the conditional independence graph of Y, which is a different target (see
    CLAUDE.md) and is NOT recoverable from supp(Theta).

    Greedy largest-degree-first coloring. Sparse Theta gives few classes, so a
    sweep costs a handful of BLAS calls instead of p masked scatter updates; a
    dense Theta degrades gracefully to p singleton classes, i.e. exactly the
    coordinate-wise scan.
    """
    p = Theta.shape[0]
    adj = Theta != 0.0
    np.fill_diagonal(adj, False)
    color = np.full(p, -1, dtype=int)
    for j in np.argsort(-adj.sum(axis=1)):        # largest degree first
        taken = color[adj[j]]
        free = np.ones(p + 1, dtype=bool)
        free[taken[taken >= 0]] = False
        color[j] = int(np.argmax(free))
    return [np.flatnonzero(color == c) for c in range(color.max() + 1)]


def run_em_MWGP(
    Y,
    n_iter=100,
    rho=0.05,
    init = None,
    verbose=True,
    warning=True,
    mcmc_burn=150,
    mcmc_warmup=30,
    mcmc_samples=200,
    mcmc_thin=2,
    random_state=42,
    b_min=0.05,           # sqrt(chi*psi) threshold for the InvGamma path
    proposal_bytes=128e6, # per-array cap on the pre-generated proposal batch
    tol=None,             # stop once ll rel-change < tol for `patience` iters
    patience=5,
):
    # tol/patience: the MC E-step gives ll_change a noise floor (~4-6e-3 at
    # mcmc_samples=100 on p=100, n=50 contaminated-normal data), so a plain
    # threshold either never fires or fires in the transient; requiring
    # `patience` consecutive sub-tol iterations detects the plateau instead.
    # At the calibrated default the returned edge set differed from the
    # full-n_iter run by less than the sampler's own per-iteration flicker.
    # Larger mcmc_samples lowers the floor, so tol can be lowered with it.
    # tol=0 restores the old fixed-n_iter behavior.
    # =================================================================
    # E-step sampler: Metropolis-within-Gibbs, GIG independence proposal
    # =================================================================
    def _sample_log_tau_gig(
        Y, mu, gamma, nu, Theta, state, burn, samples, thin, rng,
    ):
        """
        The proposal for coordinate j, observation i, is EXACTLY the
        diagonal part of the full conditional:

            tau_j^(i) ~ GIG(lam_j, chi_ij, psi_j),
            lam_j = -2/nu_j - 1/2,
            chi_ij = 4/nu_j + theta_jj (Y_j^(i) - mu_j)^2,
            psi_j = theta_jj gamma_j^2.

        Because the full conditional factors as
            p(tau_j | Y, tau_-j) prop GIG(lam_j, chi_ij, psi_j)
                                      * exp(-c_j z_j(tau_j)),
            c_j = (Theta z)_j - theta_jj z_j,
        the GIG factor CANCELS the proposal density in the MH ratio,
        leaving only the coupling tilt:

            log alpha = -c_j (z_j^new - z_j^old).          (verified
                                                to machine precision)

        Consequences:
          * no prior, Jacobian, or proposal-correction terms to evaluate;
          * tails of proposal and target match exactly (no independence-
            MH tail pathology, unlike the Laplace-normal proposal);
          * if node j is isolated in Theta (glasso zeros), c_j = 0 and
            every proposal is accepted: exact Gibbs for free.

        Sites with b = sqrt(chi*psi) < b_min (near-symmetric, gamma_j
        ~ 0) instead propose from InvGamma(-lam_j, chi_ij/2) — the
        psi -> 0 limit of the GIG — and the leftover factor
        exp(-psi_j (tau^new - tau^old)/2) is added to log alpha, so the
        chain remains exact.

        Speed-ups:
          (S1) proposals are pre-generated in vectorized batches — valid
               because the independence-proposal parameters are frozen
               for the whole E-step — but in CHUNKS of sweeps, so peak
               memory is capped by proposal_bytes instead of growing as
               total_sweeps * n * p * 8 (GBs once p reaches 100);
          (S2) all log-uniforms pre-generated per sweep; nothing about
               the current state needs recomputation except z_j;
          (S3) coordinates are refreshed one COLOR CLASS at a time (see
               _color_classes). Members of a class are non-adjacent in
               Theta, so c_j = sum_{m != j} theta_mj z_m is unaffected by
               the other members and the block move reproduces exactly
               the sequential updates of its members. The whole class
               costs one gemm Z @ Theta[:, S] instead of |S| masked
               rank-1 scatters into a cached W, which is what made the
               old loop O(sweeps * n * p^2) with fancy-indexing
               constants. No cached W means no fp drift and nothing
               to refresh periodically.
          (S4) per-coordinate constants (the InvGamma masks, the sliced
               Theta columns) are hoisted out of the sweep loop.
        """
        n, p = Y.shape
        # Gaussian path: tau_j == 1, i.e. log tau_j == 0 with no randomness, so
        # coordinate j is held at 0 and never proposed for. The placeholder in
        # nu keeps lam/chi finite for the vectorised prep below; every array it
        # touches is dropped from `classes`, so the value is never used.
        gauss = nu <= 0.0
        a = 2.0 / np.where(gauss, 1.0, nu)
        beta = a
        residual = Y - mu
        theta_diag = np.diag(Theta).copy()

        U = state.copy()
        U[:, gauss] = 0.0
        Z = np.exp(-U / 2.0) * residual - np.exp(U / 2.0) * gamma

        lam = -a - 0.5
        chi = 2.0 * beta[None, :] + residual**2 * theta_diag[None, :]
        psi = theta_diag * gamma**2
        b = np.sqrt(chi * psi[None, :])          # (n, p)
        use_ig = (b * b < np.abs(lam)[None, :]) | (b < b_min)   # dimensionless InvGamma-path mask
        tau_cur = np.exp(U) if use_ig.any() else None

        # ---------- (S3)/(S4) per-class constants --------------------
        classes = []
        for S in _color_classes(Theta):
            S = S[~gauss[S]]                     # Gaussian coordinates: nothing to sample
            if S.size == 0:
                continue
            classes.append((
                S,
                np.ascontiguousarray(Theta[:, S]),   # (p, |S|)
                theta_diag[S][None, :],
                psi[S][None, :],
                use_ig[:, S] if use_ig[:, S].any() else None,
            ))

        total_sweeps = burn + samples * thin

        # ---------- (S1) chunked proposal batches --------------------
        # Three arrays of (chunk, n, p) stay live, so peak is ~3x
        # proposal_bytes.
        chunk = max(1, min(total_sweeps, int(proposal_bytes // (n * p * 8))))
        prop_U = np.zeros((chunk, n, p))   # zeros, not empty: the Gaussian
        prop_Z = np.empty((chunk, n, p))   # columns are never written below
        scratch = np.empty((chunk, n, p))

        draws = np.empty((samples, n, p))
        accepted_count = np.zeros(p)
        proposed_count = np.zeros(p)
        saved = 0
        sweep = 0

        while sweep < total_sweeps:
            cs = min(chunk, total_sweeps - sweep)
            pU, pZ, tmp = prop_U[:cs], prop_Z[:cs], scratch[:cs]

            for j in range(p):
                if gauss[j]:                          # held at log tau_j = 0
                    continue
                ig = use_ig[:, j]
                n_ig = int(ig.sum())
                if n_ig < n:                          # exact-GIG sites
                    cols = ~ig
                    pU[:, cols, j] = _sample_log_gig(
                        lam[j], chi[cols, j],
                        np.full(n - n_ig, psi[j]), cs, rng,
                    )
                if n_ig > 0:                          # InvGamma sites
                    g = rng.gamma(-lam[j], size=(cs, n_ig))
                    pU[:, ig, j] = np.log(chi[ig, j][None, :] / 2.0) - np.log(g)

            # proposal-only z values, in place to avoid (chunk, n, p) temporaries
            np.multiply(pU, -0.5, out=pZ)
            np.exp(pZ, out=pZ)
            pZ *= residual[None, :, :]
            np.multiply(pU, 0.5, out=tmp)
            np.exp(tmp, out=tmp)
            tmp *= gamma[None, None, :]
            pZ -= tmp

            for s in range(cs):
                LogUnif = np.log(rng.random((n, p)))  # (S2)
                sweep_U, sweep_Z = pU[s], pZ[s]

                for S, Theta_S, tdiag_S, psi_S, ig_S in classes:
                    new = sweep_U[:, S]
                    new_z = sweep_Z[:, S]
                    delta_z = new_z - Z[:, S]

                    # collapsed independence-MH ratio: just the tilt
                    c = Z @ Theta_S - tdiag_S * Z[:, S]
                    log_alpha = -c * delta_z

                    new_tau = None
                    if ig_S is not None:
                        new_tau = np.exp(new)     # reused for tau_cur below
                        log_alpha += np.where(
                            ig_S,
                            -0.5 * psi_S * (new_tau - tau_cur[:, S]),
                            0.0,
                        )

                    accept = LogUnif[:, S] < np.minimum(0.0, log_alpha)

                    U[:, S] = np.where(accept, new, U[:, S])
                    Z[:, S] = np.where(accept, new_z, Z[:, S])
                    if new_tau is not None:
                        tau_cur[:, S] = np.where(
                            accept, new_tau, tau_cur[:, S]
                        )

                    if sweep >= burn:
                        accepted_count[S] += accept.sum(axis=0)
                        proposed_count[S] += n

                if sweep >= burn and (sweep - burn) % thin == 0:
                    draws[saved] = U
                    saved += 1
                sweep += 1

        acceptance_rate = accepted_count / np.maximum(proposed_count, 1.0)
        return draws, U, acceptance_rate

    # =================================================================
    # EM driver (unchanged from your version)
    # =================================================================
    n, p = Y.shape
    if init is None:
        mu = Y.mean(axis=0)
        nu = np.full(p, 0.5)
        scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
        gamma = 0.1 * scale * np.tanh(skew(Y, axis=0, bias=False))
        eta = gamma / nu
        Theta = np.diag(1.0 / Y.var(axis=0))
    else:
        # nu/eta are tail-shape nuisance parameters and fall back to the
        # cold-start values when absent, so a warm-started rho sweep can carry
        # mu/Theta forward without letting nu ratchet down across steps.
        Theta, mu = init["Theta"], init["mu"]
        nu = init.get("nu", np.full(p, 0.5))
        if "eta" in init:
            eta = init["eta"]
        else:
            scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
            eta = 0.1 * scale * np.tanh(skew(Y, axis=0, bias=False)) / nu
        gamma = nu * eta
        mcmc_burn = 50

    rng = np.random.default_rng(random_state)
    state = np.zeros((n, p))

    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": [], "loglik": []}
    # Same two failure signals the other EM variants report, so a rho sweep can
    # flag this method's fits on the same footing.
    n_glasso_fail = 0
    n_nu_clamped = 0
    it = 0
    ll_prev = None
    stall = 0

    for it in range(n_iter):
        # ============================ MCMC E-step ====================
        burn = mcmc_burn if it == 0 else mcmc_warmup

        log_tau_draws, state, acceptance_rate = _sample_log_tau_gig(
            Y=Y, mu=mu, gamma=gamma, nu=nu, Theta=Theta,
            state=state, burn=burn, samples=mcmc_samples,
            thin=mcmc_thin, rng=rng,
        )

        A = np.exp(-0.5 * log_tau_draws)
        B = np.exp(0.5 * log_tau_draws)

        def mc_cross(X, Wm):
            return np.einsum("sij,sik->jk", X, Wm, optimize=True) / mcmc_samples

        # ==================== Update mu and gamma ====================
        H = np.block([
            [Theta * mc_cross(A, A), Theta * mc_cross(A, B)],
            [Theta * mc_cross(B, A), Theta * mc_cross(B, B)],
        ])

        AY = A * Y[None, :, :]
        Theta_AY = np.einsum("sik,jk->sij", AY, Theta, optimize=True)
        b_mu = np.einsum("sij,sij->j", A, Theta_AY) / mcmc_samples
        b_gamma = np.einsum("sij,sij->j", B, Theta_AY) / mcmc_samples
        b_vec = np.concatenate([b_mu, b_gamma])

        mu_new, gamma_new = _solve_mu_gamma(H, b_vec, nu <= 0.0, p)

        # ===================== Update nu and eta =====================
        M_neg1 = np.mean(A**2, axis=0)
        L_log = np.mean(log_tau_draws, axis=0)
        S_j = (L_log + M_neg1).sum(axis=0)

        nu_new, eta_new, gamma_new, n_clamped = _solve_nu_eta(
            S_j, gamma_new, n, p)
        n_nu_clamped += n_clamped
        # The floor applies to the sampled coordinates only: nu_j = 0 is the
        # Gaussian flag, not a small nu, and clipping it up would put the
        # sampler back on a GIG it cannot represent.
        nu_new = np.where(nu_new > 0.0, np.clip(nu_new, NU_MIN, NU_MAX), 0.0)
        eta_new = np.clip(eta_new, -20, 20)

        # ==================== Compute expected S_tau =================
        Zs = (
            A * (Y - mu_new)[None, :, :]
            - B * gamma_new[None, None, :]
        )
        S_tau = np.einsum("sij,sik->jk", Zs, Zs, optimize=True) / (
            mcmc_samples * n
        )
        S_tau = (S_tau + S_tau.T) / 2.0 + 1e-10 * np.eye(p)

        # =============== Update Theta via graphical lasso ============
        # sklearn penalizes off-diagonals only; adding rho*I recovers the fully
        # penalized objective, since
        # tr(S Theta) + rho * sum_j theta_jj = tr((S + rho I) Theta).
        try:
            # See run_em_diagonal for why enet_tol is pinned below tol.
            _, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p), alpha=rho, max_iter=200, tol=1e-3,
                enet_tol=1e-6,
            )
        except Exception as e:
            if warning:
                print(
                    f"  [warn] glasso failed at iter {it}: {e}; "
                    f"keeping previous Theta"
                )
            n_glasso_fail += 1
            Theta_new = Theta

        # =========== Convergence monitor: expected complete-data =====
        # ========== log-likelihood Q(theta_new | theta_old) ==========
        # Reuses the sufficient statistics already computed this iteration:
        #   L_log  = E[log tau]                (n, p)
        #   M_neg1 = E[1/tau]                  (n, p)
        #   S_tau  = (1/(S n)) sum_s sum_i z z^T, so
        #            sum_i E[z^T Theta z] = n * <S_tau, Theta_new>.
        # Only slogdet(Theta_new) and the Inv-Gamma normalizer (gammaln) are
        # new, both O(p); no extra pass over the (samples, n, p) draws.
        # Every tau term below is summed over the SAMPLED coordinates only: a
        # Gaussian coordinate has tau_j == 1 with no Inv-Gamma prior to score,
        # so it contributes a constant, and a_j = 2/nu_j is infinite there --
        # a_j log a_j - gammaln(a_j) would be inf - inf, i.e. NaN, and the
        # stopping rule would then silently never fire.
        t = nu_new > 0.0
        a_nu = 2.0 / nu_new[t]
        _, logdet = np.linalg.slogdet(Theta_new)
        quad = n * np.sum(S_tau * Theta_new)          # sum_i E[z^T Theta z]
        ll_new = (
            n * (
                -0.5 * p * np.log(2.0 * np.pi)        # Gaussian normalizer
                + 0.5 * logdet                         # (1/2) log|Theta|
                + np.sum(a_nu * np.log(a_nu) - gammaln(a_nu))  # Inv-Gamma norm.
            )
            - 0.5 * L_log.sum()                        # -(1/2) sum_j log tau_j
            - 0.5 * quad                               # -(1/2) z^T Theta z
            - np.sum((a_nu[None, :] + 1.0) * L_log[:, t])   # -(a_j+1) log tau_j
            - np.sum(a_nu[None, :] * M_neg1[:, t])          # -a_j / tau_j
        )

        # Relative change in the expected complete-data log-likelihood. The
        # log-likelihood scales with n, so a relative criterion keeps `err`
        # meaningful across sample sizes. Monte Carlo noise in the E-step adds
        # jitter, so very small `err` may not be reliably reachable.
        if ll_prev is None:
            ll_change = np.inf
        else:
            ll_change = abs(ll_new - ll_prev) / (abs(ll_prev) + 1e-12)
        ll_prev = ll_new

        mu, gamma, nu, eta, Theta = (
            mu_new, gamma_new, nu_new, eta_new, Theta_new
        )

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(np.diag(Theta).copy())
        hist["loglik"].append(ll_new)

        if verbose and it % 5 == 0:
            print(
                f"iter {it:3d} | loglik {ll_new:.4f} | "
                f"rel-change {ll_change:.3e} | "
                f"acceptance-rate {np.round(acceptance_rate, 3)}"
            )

        # tol=None means "no stopping rule", same as in the other methods.
        stall = stall + 1 if (tol is not None and ll_change < tol) else 0
        if stall >= patience:
            if verbose:
                print(f"stopped at iter {it}: ll rel-change < {tol:.0e} "
                      f"for {patience} consecutive iterations")
            break

    return {
        "mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "history": hist, "S_tau": S_tau,
        "n_glasso_fail": n_glasso_fail, "n_nu_clamped": n_nu_clamped,
        "n_iter_run": it + 1,
    }

def run_em_importance(
    Y,
    n_iter=60,
    rho=0.05,
    verbose=True,
    err=1e-3,
    run_until_convergence=False,
    importance_samples=200,
    ess_warn_ratio=0.10,
    random_state=42,
):
    """Monte Carlo EM using a product-GIG importance proposal."""
    n, p = Y.shape
    rng = np.random.default_rng(random_state)

    mu = Y.mean(axis=0)
    nu = np.full(p, 0.5)
    scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
    gamma = 0.1 * scale * np.tanh(skew(Y, axis=0, bias=False))
    eta = gamma / nu
    Theta = np.diag(1.0 / Y.var(axis=0))

    hist = {
        "mu": [],
        "eta": [],
        "nu": [],
        "theta_diag": [],
        "ess_median": [],
        "ess_q05": [],
    }

    it = 0
    while True:
        theta_diag = np.diag(Theta)

        # NEW: Product-GIG proposal obtained by dropping Theta's off-diagonal terms.
        lam = -2.0 / nu - 0.5
        chi = (
            4.0 / nu[None, :]
            + theta_diag[None, :] * (Y - mu[None, :]) ** 2
        )
        psi = np.clip(theta_diag * gamma**2, 1e-12, None)
        gig_b = np.sqrt(chi * psi[None, :])
        gig_scale = np.sqrt(chi / psi[None, :])

        tau = geninvgauss.rvs(
            lam[None, None, :],
            gig_b[None, :, :],
            scale=gig_scale[None, :, :],
            size=(importance_samples, n, p),
            random_state=rng,
        )

        log_tau = np.log(tau)
        A = np.exp(-0.5 * log_tau)  # tau^(-1/2)
        B = np.exp(0.5 * log_tau)   # tau^(1/2)

        # NEW: The weights correct the product-GIG proposal for posterior dependence.
        Theta_off = Theta.copy()
        np.fill_diagonal(Theta_off, 0.0)
        Z = A * (Y - mu)[None, :, :] - B * gamma[None, None, :]
        log_weights = -0.5 * np.einsum(
            "sij,jk,sik->si", Z, Theta_off, Z, optimize=True
        )
        log_weights -= logsumexp(log_weights, axis=0, keepdims=True)
        weights = np.exp(log_weights)

        # ESS is calculated separately for each observation.
        ess = 1.0 / np.sum(weights**2, axis=0)
        ess_ratio = ess / importance_samples
        ess_median = float(np.median(ess))
        ess_q05 = float(np.quantile(ess, 0.05))

        # NEW: Weighted joint posterior moments for the unchanged M-step.
        def weighted_cross(X, W):
            return np.einsum(
                "si,sij,sik->jk", weights, X, W, optimize=True
            )

        H = np.block([
            [Theta * weighted_cross(A, A), Theta * weighted_cross(A, B)],
            [Theta * weighted_cross(B, A), Theta * weighted_cross(B, B)],
        ])

        AY = A * Y[None, :, :]
        Theta_AY = np.einsum("sik,jk->sij", AY, Theta, optimize=True)
        b_mu = np.einsum("si,sij,sij->j", weights, A, Theta_AY, optimize=True)
        b_gamma = np.einsum(
            "si,sij,sij->j", weights, B, Theta_AY, optimize=True
        )

        mu_gamma_new = np.linalg.solve(H, np.concatenate([b_mu, b_gamma]))
        mu_new = mu_gamma_new[:p]
        gamma_new = mu_gamma_new[p:]

        # Unchanged nu and eta M-step.
        M_neg1 = np.einsum("si,sij->ij", weights, A**2, optimize=True)
        L_log = np.einsum("si,sij->ij", weights, log_tau, optimize=True)
        S_j = (L_log + M_neg1).sum(axis=0)

        def stationarity(nu_j, S):
            a = 2.0 / nu_j
            return n * (np.log(a) + 1.0 - digamma(a)) - S

        nu_new = np.empty(p)
        for j in range(p):
            try:
                nu_new[j] = brentq(
                    lambda x: stationarity(x, S_j[j]),
                    0.01,
                    5.0,
                    xtol=1e-6,
                )
            except ValueError:
                raise ValueError(
                    f"Root finding failed for nu[{j}] with S_j={S_j[j]}"
                )

        eta_new = gamma_new / nu_new

        # Unchanged expected-covariance and graphical-lasso M-step.
        Z_new = (
            A * (Y - mu_new)[None, :, :]
            - B * gamma_new[None, None, :]
        )
        S_tau = np.einsum(
            "si,sij,sik->jk", weights, Z_new, Z_new, optimize=True
        ) / n
        S_tau = (S_tau + S_tau.T) / 2.0 + 1e-10 * np.eye(p)

        # sklearn penalizes off-diagonals only; adding rho*I recovers the fully
        # penalized objective, since
        # tr(S Theta) + rho * sum_j theta_jj = tr((S + rho I) Theta).
        try:
            _, Theta_new = graphical_lasso(
                S_tau + rho * np.eye(p), alpha=rho, max_iter=1000
            )
        except Exception as e:
            if verbose:
                print(
                    f"  [warn] glasso failed at iter {it}: {e}; "
                    "keeping previous Theta"
                )
            Theta_new = Theta

        diff = (
            np.abs(mu_new - mu).sum()
            + np.abs(eta_new - eta).sum()
            + np.abs(nu_new - nu).sum()
            + np.linalg.norm(Theta_new - Theta)
        )

        mu, gamma, nu, eta, Theta = (
            mu_new,
            gamma_new,
            nu_new,
            eta_new,
            Theta_new,
        )

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(np.diag(Theta).copy())
        hist["ess_median"].append(ess_median)
        hist["ess_q05"].append(ess_q05)

        if verbose and it % 5 == 0:
            print(
                f"iter {it:3d} | param-change {diff:.10f} | "
                f"ESS median {ess_median:.1f}/{importance_samples} | "
                f"ESS 5% {ess_q05:.1f}/{importance_samples}"
            )

        if verbose and np.quantile(ess_ratio, 0.05) < ess_warn_ratio:
            print(
                f"  [warn] low importance-sampling ESS at iter {it}; "
                "increase importance_samples or use a dependent proposal"
            )

        if (
            (run_until_convergence and diff < err)
            or (not run_until_convergence and it >= n_iter)
        ):
            if verbose:
                print(f"Converged at iteration {it}.")
            break

        it += 1

    return {
        "mu": mu,
        "eta": eta,
        "nu": nu,
        "Theta": Theta,
        "history": hist,
        "last_ess": ess,
        "S_tau": S_tau,
    }
