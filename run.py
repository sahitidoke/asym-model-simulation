import numpy as np
import simulation as em
from sklearn.metrics import mean_squared_error

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

results = {}

for name, pars in em.simulation_settings.items():

    eta_true = np.full(p, pars["eta"])
    nu_true  = np.full(p, pars["nu"])

    Y, tau = simulate_aat_data(
    n=n,
    p=p,
    mu=mu_true,
    eta=eta_true,
    nu=nu_true,
    Theta_true=Theta_true,
    rng=rng,
    )

    fit = em.run_em_exact(
    Y,
    rho=0.05,
    run_until_convergence=True,
    verbose=True,
    max_iter=200,
    )
    def relative_error(true, estimated):
        return np.linalg.norm(estimated - true) / np.linalg.norm(true)

    results[name] = {
        "mu_true": mu_true.copy(),
        "eta_true": eta_true.copy(),
        "nu_true": nu_true.copy(),
        "Theta_true": Theta_true.copy(),

        "mu": fit["mu"],
        "eta": fit["eta"],
        "nu": fit["nu"],
        "Theta": fit["Theta"],

        "mse_mu": mean_squared_error(mu_true, fit["mu"]),
        "mse_eta": mean_squared_error(eta_true, fit["eta"]),
        "mse_nu": mean_squared_error(nu_true, fit["nu"]),
        "mse_theta": mean_squared_error(Theta_true, fit["Theta"]),

        "rel_mu": relative_error(mu_true, fit["mu"]),
        "rel_eta": relative_error(eta_true, fit["eta"]),
        "rel_nu": relative_error(nu_true, fit["nu"]),
        "rel_theta": relative_error(Theta_true, fit["Theta"]),
    }
print(f"{'Scenario':<20} {'Mu RelErr':>12} {'Eta RelErr':>12} {'Nu RelErr':>12} {'Theta RelErr':>15}")

for name, r in results.items():
    print(f"{name:<20}"
          f"{r['rel_mu']:12.4e}"
          f"{r['rel_eta']:12.4e}"
          f"{r['rel_nu']:12.4e}"
          f"{r['rel_theta']:15.4e}")






