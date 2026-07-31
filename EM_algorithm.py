import numpy as np
from tqdm import tqdm
from scipy.special import kve, digamma, logsumexp
from scipy.optimize import brentq
from scipy.stats import skew, geninvgauss
from sklearn.covariance import graphical_lasso

def gig_moment(r, lam, chi, psi):
    x = np.sqrt(chi * psi)
    num = kve(lam + r, x)
    den = kve(lam, x)
    return (chi / psi) ** (r / 2.0) * (num / den)

def gig_log_moment_fd(lam, chi, psi, h=1e-4):
    x = np.sqrt(chi * psi)
    log_num = np.log(kve(lam + h, x))
    log_den = np.log(kve(lam - h, x))
    return 0.5 * np.log(chi / psi) + (log_num - log_den) / (2 * h)


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
    while True:
        # Compute GIG parameters 
        lam = -2.0 / nu - 0.5                                   # (p,)
        chi = 4.0 / nu[None, :] + theta_bar[None, :] * (Y - mu[None, :]) ** 2   # (n,p)
        psi = np.clip(theta_bar * eta ** 2 * nu ** 2, 1e-12, None)               # (p,)\
            
        # Compute expectations

        M_neg1   = gig_moment(-1.0, lam[None, :], chi, psi[None, :])
        M_pos1   = gig_moment(1.0,  lam[None, :], chi, psi[None, :])
        M_neg_half = gig_moment(-0.5, lam[None, :], chi, psi[None, :])
        M_pos_half = gig_moment(0.5,  lam[None, :], chi, psi[None, :])
        L_log = gig_log_moment_fd(lam[None, :], chi, psi[None, :])
        
        # Update parameters mu, gamma  
       
        Aj = M_neg1.sum(axis=0)                       # sum_i M_ij(-1)
        Bj = M_pos1.sum(axis=0)                        # sum_i M_ij(1)
        Rj = (M_neg1 * Y).sum(axis=0)                   # sum_i M_ij(-1) Y_ij
        Tj = Y.sum(axis=0)                              # sum_i Y_ij

        denom = Aj * Bj - n ** 2
        denom = np.maximum(denom, 1e-4 * n ** 2)
        mu_new = (Bj * Rj - n * Tj) / denom
        gamma_new = (Aj * Tj - n * Rj) / denom
        
        # Update parameters nu, eta

        S_j = (L_log + M_neg1).sum(axis=0)

        def stationarity(nu_j, S):
            a = 2.0 / nu_j
            return n * (np.log(a) + 1.0 - digamma(a)) - S

        nu_new = np.empty(p)

        for j in range(p):
            f = lambda x: stationarity(x, S_j[j])

            try:
                nu_new[j] = brentq(f, 0.01, 5.0, xtol=1e-6)

            except ValueError:
                raise ValueError(f"Root finding failed for nu[{j}] with S_j={S_j[j]}")
                # res = minimize_scalar(
                #     lambda x: -(
                #         n * (
                #             (2.0 / x) * np.log(2.0 / x)
                #             - gammaln(2.0 / x)
                #         )
                #         - (2.0 / x) * S_j[j]
                #     ),
                #     bounds=(0.01, 5.0),
                #     method="bounded"
                # )
                # nu_new[j] = res.x

        eta_new = gamma_new / nu_new

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
        
        # Glasso step to estimate Theta
        try:
            cov_glasso, Theta_new = graphical_lasso(S_tau, alpha=rho, max_iter=200)
        except Exception as e:
            if verbose:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            Theta_new = Theta

        theta_bar_new = np.diag(Theta_new).copy()
        theta_bar_new = np.clip(theta_bar_new, 1e-10, None)
        diff = (np.abs(mu_new - mu).sum() + np.abs(eta_new - eta).sum()
                + np.abs(nu_new - nu).sum())

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

def run_em_exact(Y, n_iter=60, rho=0.05, verbose=True, err=1e-3, run_until_convergence=False):
    n, p = Y.shape
    mu = Y.mean(axis=0)           
    nu = np.full(p, 0.5)  # moderate tail initialization
    scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
    sample_skew = skew(Y, axis=0, bias=False)
    gamma = 0.1 * scale * np.tanh(sample_skew)
    eta = gamma / nu
    Theta = np.diag(1.0 / Y.var(axis=0))

    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": []}
    it = 0
    while True:
        theta_diag = np.diag(Theta)

        # Compute GIG parameters 
        lam = -2.0 / nu - 0.5
        chi = 4.0 / nu[None, :] + theta_diag[None, :] * (Y - mu[None, :]) ** 2
        psi = np.clip(theta_diag * eta ** 2 * nu ** 2, 1e-12, None)
        
        # Compute expectations
        M_neg1   = gig_moment(-1.0, lam[None, :], chi, psi[None, :])
        M_pos1   = gig_moment(1.0,  lam[None, :], chi, psi[None, :])
        M_neg_half = gig_moment(-0.5, lam[None, :], chi, psi[None, :])
        M_pos_half = gig_moment(0.5,  lam[None, :], chi, psi[None, :])
        L_log = gig_log_moment_fd(lam[None, :], chi, psi[None, :])
        
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
        mu_gamma_new = np.linalg.solve(H, b)
        mu_new = mu_gamma_new[:p]
        gamma_new = mu_gamma_new[p:]
        
        # Update parameters nu, eta

        S_j = (L_log + M_neg1).sum(axis=0)

        def stationarity(nu_j, S):
            a = 2.0 / nu_j
            return n * (np.log(a) + 1.0 - digamma(a)) - S

        nu_new = np.empty(p)

        for j in range(p):
            f = lambda x: stationarity(x, S_j[j])

            try:
                nu_new[j] = brentq(f, 0.01, 5.0, xtol=1e-6)

            except ValueError:
                raise ValueError(f"Root finding failed for nu[{j}] with S_j={S_j[j]}")
                # res = minimize_scalar(
                #     lambda x: -(
                #         n * (
                #             (2.0 / x) * np.log(2.0 / x)
                #             - gammaln(2.0 / x)
                #         )
                #         - (2.0 / x) * S_j[j]
                #     ),
                #     bounds=(0.01, 5.0),
                #     method="bounded"
                # )
                # nu_new[j] = res.x

        eta_new = gamma_new / nu_new

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
        
        # Glasso step to estimate Theta
        try:
            _, Theta_new = graphical_lasso(S_tau, 
                                           alpha=rho, 
                                           # alpha=2 * rho / n, 
                                           max_iter=1000)
        except Exception as e:
            if verbose:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            Theta_new = Theta

        diff = (np.abs(mu_new - mu).sum() + np.abs(eta_new - eta).sum()
                + np.abs(nu_new - nu).sum() + np.linalg.norm(Theta_new - Theta))

        mu, nu, eta = mu_new, nu_new, eta_new
        Theta = Theta_new

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(np.diag(Theta).copy())

        if verbose and (it % 5 == 0):
            print(f"iter {it:3d} | param-change {diff:.10f}")

        if (run_until_convergence and diff < err) or (not run_until_convergence and it >= n_iter):
            if verbose:
                print(f"Converged at iteration {it}.")
            break
        it += 1

    return {"mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "history": hist}

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
        # Update Theta using graphical lasso
        # ==========================================================

        try:
            _, Theta_new = graphical_lasso(
                S_tau,
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
    verbose=True,
    err=1e-3,
    run_until_convergence=False,
    mcmc_burn=150,
    mcmc_warmup=30,
    mcmc_samples=200,
    mcmc_thin=2,
    random_state=42,
    proposal="gig",       # "gig" (recommended) or "laplace"
    b_min=0.05,           # sqrt(chi*psi) threshold for the InvGamma path
    refresh_every=250,    # periodic refresh of cached W = Z @ Theta (laplace)
    proposal_bytes=128e6, # per-array cap on the pre-generated proposal batch
):
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
               constants. No cached W means no fp drift and no
               refresh_every.
          (S4) per-coordinate constants (the InvGamma masks, the sliced
               Theta columns) are hoisted out of the sweep loop.
        """
        n, p = Y.shape
        a = 2.0 / nu
        beta = a
        residual = Y - mu
        theta_diag = np.diag(Theta).copy()

        U = state.copy()
        Z = np.exp(-U / 2.0) * residual - np.exp(U / 2.0) * gamma

        lam = -a - 0.5
        chi = 2.0 * beta[None, :] + residual**2 * theta_diag[None, :]
        psi = theta_diag * gamma**2
        b = np.sqrt(chi * psi[None, :])          # (n, p)
        use_ig = b < b_min                        # InvGamma-path mask
        tau_cur = np.exp(U) if use_ig.any() else None

        # ---------- (S3)/(S4) per-class constants --------------------
        classes = [
            (
                S,
                np.ascontiguousarray(Theta[:, S]),   # (p, |S|)
                theta_diag[S][None, :],
                psi[S][None, :],
                use_ig[:, S] if use_ig[:, S].any() else None,
            )
            for S in _color_classes(Theta)
        ]

        total_sweeps = burn + samples * thin

        # ---------- (S1) chunked proposal batches --------------------
        # Three arrays of (chunk, n, p) stay live, so peak is ~3x
        # proposal_bytes.
        chunk = max(1, min(total_sweeps, int(proposal_bytes // (n * p * 8))))
        prop_U = np.empty((chunk, n, p))
        prop_Z = np.empty((chunk, n, p))
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
    # E-step sampler: Laplace-normal independence proposal (fallback)
    # =================================================================
    def _sample_log_tau_laplace(
        Y, mu, gamma, nu, Theta, state, burn, samples, thin, rng,
    ):
        """Batched/cached Laplace-normal version (see em_mwgp_fast.py
        for the fully commented derivation of S1-S3)."""
        n, p = Y.shape
        a = 2.0 / nu
        beta = a
        residual = Y - mu

        U = state.copy()
        Z = np.exp(-U / 2.0) * residual - np.exp(U / 2.0) * gamma

        lam = -a - 0.5
        chi = 2.0 * beta[None, :] + residual**2 * np.diag(Theta)[None, :]
        psi = np.diag(Theta) * gamma**2
        sqrt_term = np.sqrt(lam[None, :] ** 2 + chi * psi[None, :])
        tau_mode = chi / (sqrt_term - lam[None, :])
        proposal_mean = np.log(tau_mode)
        proposal_sd = np.sqrt(
            2.0 / (chi / tau_mode + psi[None, :] * tau_mode)
        )

        g_old = -(a + 0.5)[None, :] * U - beta[None, :] * np.exp(-U)
        logq_old = -0.5 * ((U - proposal_mean) / proposal_sd) ** 2
        W = Z @ Theta

        draws = np.empty((samples, n, p))
        accepted_count = np.zeros(p)
        proposed_count = np.zeros(p)
        saved = 0
        total_sweeps = burn + samples * thin

        for sweep in range(total_sweeps):
            if sweep > 0 and sweep % refresh_every == 0:
                W = Z @ Theta

            New = proposal_mean + proposal_sd * rng.standard_normal((n, p))
            LogUnif = np.log(rng.random((n, p)))
            Valid = np.isfinite(New) & (np.abs(New) < 30.0)
            NewC = np.clip(New, -30.0, 30.0)

            New_z = (
                np.exp(-NewC / 2.0) * residual
                - np.exp(NewC / 2.0) * gamma[None, :]
            )
            g_new = -(a + 0.5)[None, :] * NewC - beta[None, :] * np.exp(-NewC)
            logq_new = -0.5 * ((New - proposal_mean) / proposal_sd) ** 2

            for j in range(p):
                delta_z = New_z[:, j] - Z[:, j]
                delta_quad = 2.0 * delta_z * W[:, j] + Theta[j, j] * delta_z**2
                log_alpha = (
                    (g_new[:, j] - g_old[:, j])
                    - 0.5 * delta_quad
                    + (logq_old[:, j] - logq_new[:, j])
                )
                accept = Valid[:, j] & (
                    LogUnif[:, j] < np.minimum(0.0, log_alpha)
                )
                if accept.any():
                    W[accept] += delta_z[accept, None] * Theta[j][None, :]
                    U[accept, j] = New[accept, j]
                    Z[accept, j] = New_z[accept, j]
                    g_old[accept, j] = g_new[accept, j]
                    logq_old[accept, j] = logq_new[accept, j]
                if sweep >= burn:
                    accepted_count[j] += accept.sum()
                    proposed_count[j] += n

            if sweep >= burn and (sweep - burn) % thin == 0:
                draws[saved] = U
                saved += 1

        acceptance_rate = accepted_count / np.maximum(proposed_count, 1.0)
        return draws, U, acceptance_rate

    _sampler = (
        _sample_log_tau_gig if proposal == "gig" else _sample_log_tau_laplace
    )

    # =================================================================
    # EM driver (unchanged from your version)
    # =================================================================
    n, p = Y.shape

    mu = Y.mean(axis=0)
    nu = np.full(p, 0.5)
    scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
    gamma = 0.1 * scale * np.tanh(skew(Y, axis=0, bias=False))
    eta = gamma / nu
    Theta = np.diag(1.0 / Y.var(axis=0))

    rng = np.random.default_rng(random_state)
    state = np.zeros((n, p))

    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": []}
    it = 0

    while True:
        # ============================ MCMC E-step ====================
        burn = mcmc_burn if it == 0 else mcmc_warmup

        log_tau_draws, state, acceptance_rate = _sampler(
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

        mu_gamma_new = np.linalg.solve(H, b_vec)
        mu_new = mu_gamma_new[:p]
        gamma_new = mu_gamma_new[p:]

        # ===================== Update nu and eta =====================
        M_neg1 = np.mean(A**2, axis=0)
        L_log = np.mean(log_tau_draws, axis=0)
        S_j = (L_log + M_neg1).sum(axis=0)

        def stationarity(nu_j, S):
            a_j = 2.0 / nu_j
            return n * (np.log(a_j) + 1.0 - digamma(a_j)) - S

        nu_new = np.empty(p)
        for j in range(p):
            try:
                nu_new[j] = brentq(
                    lambda x: stationarity(x, S_j[j]), 0.01, 5.0, xtol=1e-6,
                )
            except ValueError:
                raise ValueError(
                    f"Root finding failed for nu[{j}] with S_j={S_j[j]}"
                )

        eta_new = gamma_new / nu_new

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
        try:
            _, Theta_new = graphical_lasso(S_tau, alpha=rho, max_iter=2000)
        except Exception as e:
            if verbose:
                print(
                    f"  [warn] glasso failed at iter {it}: {e}; "
                    f"keeping previous Theta"
                )
            Theta_new = Theta

        diff = (
            np.abs(mu_new - mu).sum()
            + np.abs(eta_new - eta).sum()
            + np.abs(nu_new - nu).sum()
            + np.linalg.norm(Theta_new - Theta)
        )

        mu, gamma, nu, eta, Theta = (
            mu_new, gamma_new, nu_new, eta_new, Theta_new
        )

        hist["mu"].append(mu.copy())
        hist["eta"].append(eta.copy())
        hist["nu"].append(nu.copy())
        hist["theta_diag"].append(np.diag(Theta).copy())

        if verbose and it % 5 == 0:
            print(
                f"iter {it:3d} | param-change {diff:.10f} | "
                f"acceptance-rate {np.round(acceptance_rate, 3)}"
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
        "mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "history": hist
    }, S_tau

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

        try:
            _, Theta_new = graphical_lasso(S_tau, alpha=rho, max_iter=1000)
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
    }
