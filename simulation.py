import numpy as np
from scipy.special import kve, digamma, gammaln
from scipy.optimize import brentq, minimize_scalar
from sklearn.covariance import graphical_lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def simulate_aat_data(n, p, mu, eta, nu, Theta_true, rng):
    Psi_true = np.linalg.inv(Theta_true)
    alpha = 2.0 / nu  
    beta = 2.0 / nu 
    G = rng.gamma(shape=alpha, scale=1.0 / beta, size=(n, p))
    tau = 1.0 / G                                 
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)  
    Y = mu[None, :] + eta[None, :] * nu[None, :] * tau + np.sqrt(tau) * X
    return Y, tau

p = 5
n = 2000
mu_true  = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
eta_true = np.array([1.5, -1.0, 0.0, 2.0, -1.5])    
nu_true  = np.array([0.15, 0.25, 0.35, 0.10, 0.30]) 
Theta_true = np.eye(p) * 2.0
for j in range(p - 1):
    Theta_true[j, j + 1] = Theta_true[j + 1, j] = 0.5
assert np.all(np.linalg.eigvalsh(Theta_true) > 0)

Y, tau_true = simulate_aat_data(n, p, mu_true, eta_true, nu_true, Theta_true, rng)
print(f"Simulated data: Y shape = {Y.shape}")


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

def run_em(Y, n_iter=60, rho=0.05, verbose=True, seed=1, err=1e-3):
    n, p = Y.shape
    mu = Y.mean(axis=0)
    gamma = np.zeros(p)             
    nu = np.full(p, 0.5)
    eta = gamma / nu
    theta_bar = 1.0 / Y.var(axis=0)    
    Theta = np.diag(theta_bar)        

    hist = {"mu": [], "eta": [], "nu": [], "theta_diag": []}
    it = 0
    while True:
        lam = -2.0 / nu - 0.5                                   # (p,)
        chi = 4.0 / nu[None, :] + theta_bar[None, :] * (Y - mu[None, :]) ** 2   # (n,p)
        psi = np.clip(theta_bar * eta ** 2 * nu ** 2, 1e-2, None)               # (p,)

        M_neg1   = gig_moment(-1.0, lam[None, :], chi, psi[None, :])
        M_pos1   = gig_moment(1.0,  lam[None, :], chi, psi[None, :])
        M_neg_half = gig_moment(-0.5, lam[None, :], chi, psi[None, :])
        M_pos_half = gig_moment(0.5,  lam[None, :], chi, psi[None, :])
        L_log = gig_log_moment_fd(lam[None, :], chi, psi[None, :])  

       
        Aj = M_neg1.sum(axis=0)                       # sum_i M_ij(-1)
        Bj = M_pos1.sum(axis=0)                        # sum_i M_ij(1)
        Rj = (M_neg1 * Y).sum(axis=0)                   # sum_i M_ij(-1) Y_ij
        Tj = Y.sum(axis=0)                              # sum_i Y_ij

        denom = Aj * Bj - n ** 2
        denom = np.maximum(denom, 1e-4 * n ** 2)
        mu_new = (Bj * Rj - n * Tj) / denom
        gamma_new = (Aj * Tj - n * Rj) / denom

        S_j = (L_log + M_neg1).sum(axis=0)  
        def stationarity(nu_j, S):
            a = 2.0 / nu_j
            return n * (np.log(a) + 1.0 - digamma(a)) - S

        nu_new = np.empty(p)
        for j in range(p):
            f = lambda x: stationarity(x, S_j[j])
            try:
                nu_new[j] = brentq(f, 0.01, 5.0 , xtol=1e-6)
            except ValueError:
                res = minimize_scalar(lambda x: f(x) ** 2, bounds=(0.01, 3.0),
                                       method="bounded")
                nu_new[j] = res.x
        eta_new = gamma_new / nu_new
        z_j = (M_neg_half * (Y - mu_new[None, :]) - M_pos_half * gamma_new[None, :])
        S_tau = (z_j.T @ z_j) / n          
        S_tau = (S_tau + S_tau.T) / 2.0

        diag_cap = np.median(np.diag(S_tau)) * 20.0
        np.fill_diagonal(S_tau, np.minimum(np.diag(S_tau), diag_cap))
        S_tau += 1e-6 * np.eye(p) 
        try:
            cov_glasso, Theta_new = graphical_lasso(S_tau, alpha=rho, max_iter=200)
        except Exception as e:
            if verbose:
                print(f"  [warn] glasso failed at iter {it}: {e}; keeping previous Theta")
            Theta_new = Theta

        theta_bar_new = np.diag(Theta_new).copy()
        theta_bar_new = np.clip(theta_bar_new, 1e-6, None)
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

        if diff < err and it > 5:
            if verbose:
                print(f"Converged at iteration {it}.")
            break
        it += 1

    return {"mu": mu, "eta": eta, "nu": nu, "Theta": Theta, "history": hist}

result = run_em(Y, n_iter=150, rho=0.05, err=1e-5)

print(f"{'':>10} {'true':>30} {'estimated':>30}")
print(f"{'mu':>10} {np.round(mu_true, 3)} {np.round(result['mu'], 3)}")
print(f"{'eta':>10} {np.round(eta_true, 3)} {np.round(result['eta'], 3)}")
print(f"{'nu':>10} {np.round(nu_true, 3)} {np.round(result['nu'], 3)}")
print("\nTrue Theta:\n", np.round(Theta_true, 3))
print("\nEstimated Theta (glasso):\n", np.round(result["Theta"], 3))


print("Mean squared error for mu:" , mean_squared_error(np.round(mu_true, 3), np.round(result['mu'], 3)))
print("Mean squared error for nu:" , mean_squared_error(np.round(nu_true, 3), np.round(result['nu'], 3)))
print("Mean squared error for eta:" , mean_squared_error(np.round(eta_true, 3), np.round(result['eta'], 3)))
print("Mean squared error for Theta:" , mean_squared_error(np.round(Theta_true, 3), np.round(result['Theta'], 3)))

print("Relative error for mu:", mean_absolute_percentage_error(np.round(mu_true, 3), np.round(result['mu'], 3)))
print("Relative error for nu:", mean_absolute_percentage_error(np.round(nu_true, 3), np.round(result['nu'], 3)))
print("Relative error for eta:", mean_absolute_percentage_error(np.round(eta_true, 3), np.round(result['eta'], 3)))
print("Relative error for Theta:", mean_absolute_percentage_error(np.round(Theta_true, 3), np.round(result['Theta'], 3)))




