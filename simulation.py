"""
Running instructions (run in terminal):
# VAE without diagnostics
python simulation.py --method VAE

# VAE with identifiability diagnostics
python simulation.py --method VAE --diagnostics

# EM
python simulation.py --method EM
"""

import argparse
import numpy as np
import EM_algorithm as em
from sklearn.metrics import mean_squared_error
from aat_vae import VAEConfig, fit_aat_vae
from eta_nu_profile import profile_eta_nu

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

"""
True distribution parameters
"""
p = 5
n = 5000
mu_true  = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
eta_true = np.array([1.5, -1.0, 0.3, 2.0, -1.5])    
nu_true  = np.array([0.15, 0.25, 0.35, 0.10, 0.30]) 
# nu_true  = np.array([0.5, 0.5, 0.7, 0.45, 0.30]) 
Theta_true = np.eye(p) * 2.0
for j in range(p - 1):
    Theta_true[j, j + 1] = Theta_true[j + 1, j] = 0.5
assert np.all(np.linalg.eigvalsh(Theta_true) > 0)

Y, tau_true = simulate_aat_data(n, p, mu_true, eta_true, nu_true, Theta_true, rng)
print(f"Simulated data: Y shape = {Y.shape}")

"""
Command-line argument parsing
"""

parser = argparse.ArgumentParser(
    description="Run the asymmetric t-distribution simulation."
)

parser.add_argument(
    "--method",
    type=str.upper,
    choices=["VAE", "EM_EXACT", "EM_DIAGONAL"],
    default="VAE",
    help="Parameter estimation method. Default: VAE.",
)

parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="Run eta-nu identifiability diagnostics after fitting the VAE.",
)

args = parser.parse_args()

"""
Run the specified method (VAE or EM) to estimate parameters from the simulated data.
"""
if args.method == "VAE":
    config = VAEConfig(
        epochs=250,
        posterior_samples=16,
        theta_l1=0.0025,
        encoder_steps=5,
        flow_layers=6
    )

    model, history = fit_aat_vae(Y, config=config)
    result = model.decoder.estimates()

    if args.diagnostics:
        diagnostic = profile_eta_nu(
            y=Y,
            coordinate=0,
            fitted_model=model,
            grid_size=7,
            profile_epochs=150,
            importance_samples=512,
            output_prefix="eta_nu_coordinate_0",
        )

        print(diagnostic["decision"])
        print("Identifiable:", diagnostic["bounded_95_region"])
        print(
            "Touches grid boundary:",
            diagnostic["touches_grid_boundary"],
        )
        print("Eta 95% range:", diagnostic["eta_95_grid_range"])
        print("Nu 95% range:", diagnostic["nu_95_grid_range"])

elif args.method == "EM_EXACT":
    if args.diagnostics:
        parser.error("--diagnostics can only be used with --method VAE")

    result = em.run_em_exact(
        Y,
        n_iter=500,
        rho=0.05,
        err=1e-8,
        run_until_convergence=False,
    )
elif args.method == "EM_DIAGONAL":
    if args.diagnostics:
        parser.error("--diagnostics can only be used with --method VAE")

    result = em.run_em_diagonal(
        Y,
        n_iter=500,
        rho=0.05,
        err=1e-8,
        run_until_convergence=False,
    )

"""
Error analysis section. Applies to all methods. Never comment this section out.
"""
# final values
print(f"{'':>10} {'true':>30} {'estimated':>30}")
print(f"{'mu':>10} {np.round(mu_true, 5)} {np.round(result['mu'], 5)}")
print(f"{'eta':>10} {np.round(eta_true, 5)} {np.round(result['eta'], 5)}")
print(f"{'nu':>10} {np.round(nu_true, 5)} {np.round(result['nu'], 5)}")
print(f"{'Sum of mu and gamma:':>10} {np.round(mu_true + eta_true * nu_true, 5)} {np.round(result['mu'] + result['eta'] * result['nu'], 5)}")
print("\nTrue Theta:\n", np.round(Theta_true, 5))
print("\nEstimated Theta (glasso):\n", np.round(result["Theta"], 5))
print("\n")

print("Mean squared error for mu:", mean_squared_error(mu_true, result["mu"]))
print("Mean squared error for nu:", mean_squared_error(nu_true, result["nu"]))
print("Mean squared error for eta:", mean_squared_error(eta_true, result["eta"]))
print("Mean squared error for Theta:", mean_squared_error(Theta_true, result["Theta"]))
print("\n")
def relative_error(true, estimated):
    return np.linalg.norm(estimated - true) / np.linalg.norm(true)

print("Relative error for mu:", relative_error(mu_true, result["mu"]))
print("Relative error for nu:", relative_error(nu_true, result["nu"]))
print("Relative error for eta:", relative_error(eta_true, result["eta"]))
print("Relative error for Theta:", relative_error(Theta_true, result["Theta"]))



