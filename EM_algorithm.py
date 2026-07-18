import numpy as np
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
