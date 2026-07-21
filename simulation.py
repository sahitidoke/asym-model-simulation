"""
Running instructions (run in terminal):
# VAE without diagnostics
python simulation.py --method vae

# VAE with identifiability diagnostics
python simulation.py --method vae --diagnostics

# EM
python simulation.py --method em_diagonal
python simulation.py --method em_exact
python simulation.py --method em_mwg
python simulation.py --method em_importance
"""

import argparse
import numpy as np
import EM_algorithm as em
import json
from sklearn.metrics import mean_squared_error
from aat_vae import VAEConfig, fit_aat_vae
from eta_nu_profile import profile_eta_nu

rng = np.random.default_rng()

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
mu_true  = np.array([1.0, -2.0, 0.5, 0.3, 3.0])
eta_true = np.array([1.5, -1.0, 0.3, 2.0, -1.5])    
# nu_true  = np.array([0.15, 0.25, 0.35, 0.10, 0.30]) 
nu_true = rng.uniform(low=0.05, high=0.9, size=p)
Theta_true = np.eye(p) * 2.0
for j in range(p - 1):
    Theta_true[j, j + 1] = Theta_true[j + 1, j] = 0.5
assert np.all(np.linalg.eigvalsh(Theta_true) > 0)

"""
Command-line argument parsing
"""

parser = argparse.ArgumentParser(
    description="Run the asymmetric t-distribution simulation."
)

parser.add_argument(
    "--method",
    type=str.upper,
    choices=["VAE", "EM_EXACT", "EM_DIAGONAL", "EM_MWG", "EM_IMPORTANCE"],
    default="VAE",
    help="Parameter estimation method. Default: VAE.",
)

parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="Run eta-nu identifiability diagnostics after fitting the VAE.",
)

parser.add_argument(
    "--num_simulations",
    type=int,
    default=10,
    help="Number of simulations to run. Default: 10.",
)


args = parser.parse_args()
"""
Run the specified method (VAE or EM) to estimate parameters from the simulated data.
"""
NUM_SIMULATIONS = args.num_simulations
for sim in range(NUM_SIMULATIONS):
    Y, tau_true = simulate_aat_data(n, p, mu_true, eta_true, nu_true, Theta_true, rng)
    print(f"Simulated data: Y shape = {Y.shape}")
    mus, etas, nus, thetas = [], [], [], []
    if args.method == "VAE":
        config = VAEConfig(
            epochs=100,
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
            rho=0.0025,
            err=1e-8,
            run_until_convergence=False,
        )
    elif args.method == "EM_DIAGONAL":
        if args.diagnostics:
            parser.error("--diagnostics can only be used with --method VAE")

        result = em.run_em_diagonal(
            Y,
            n_iter=500,
            rho=0.0025,
            err=1e-8,
            run_until_convergence=False,
        )
    elif args.method == "EM_MWG":
        if args.diagnostics:
            parser.error("--diagnostics can only be used with --method VAE")

        result = em.run_em_MWG(
            Y,
            n_iter=100,
            rho=0.0025,
            verbose=True,
            err=1e-3,
            run_until_convergence=False,
            mcmc_samples=400,
            proposal_scale=0.35,
            random_state=42,
        )
    elif args.method == "EM_IMPORTANCE":
        if args.diagnostics:
            parser.error("--diagnostics can only be used with --method VAE")

        result = em.run_em_importance(
            Y,
            n_iter=100,
            rho=0.0025,
            verbose=True,
            err=1e-3,
            run_until_convergence=False,
            importance_samples=400,
            ess_warn_ratio=0.20,
            random_state=42,
        )
    nus.append(result["nu"])
    etas.append(result["eta"])
    mus.append(result["mu"])
    thetas.append(result["Theta"])

nus = np.array(nus)
etas = np.array(etas)
mus = np.array(mus)
thetas = np.array(thetas)
"""
Error analysis section. Applies to all methods. Never comment this section out.
"""
avg_mu = np.mean(mus, axis=0)
avg_eta = np.mean(etas, axis=0)
avg_nu = np.mean(nus, axis=0)
avg_theta = np.mean(thetas, axis=0)
# compute MSE as a p-dimensional vector. Each entry is the MSE for that entry in the vector/matrix.
mse_mu = np.mean((mus - mu_true[None, :]) ** 2, axis=0)
mse_eta = np.mean((etas - eta_true[None, :]) ** 2, axis=0)
mse_nu = np.mean((nus - nu_true[None, :]) ** 2, axis=0)
err_Theta = np.mean(
    np.linalg.norm(
        thetas - Theta_true[None, :, :],
        ord="fro",
        axis=(1, 2),
    )
)
# final values
print(f"{'mu':>10} {np.round(mu_true, 5)} {np.round(avg_mu, 5)}")
print(f"{'eta':>10} {np.round(eta_true, 5)} {np.round(avg_eta, 5)}")
print(f"{'nu':>10} {np.round(nu_true, 5)} {np.round(avg_nu, 5)}")
print(f"{'gamma':>10} {np.round(eta_true * nu_true, 5)} {np.round(avg_eta * avg_nu, 5)}")
print(f"{'mu + gamma:':>10} {np.round(mu_true + eta_true * nu_true, 5)} {np.round(avg_mu + avg_eta * avg_nu, 5)}")
print("\nTrue Theta:\n", np.round(Theta_true, 5))
print("\nEstimated Theta:\n", np.round(avg_theta, 5))
print("\n")

print("Mean squared error for mu:", mse_mu)
print("Mean squared error for nu:", mse_nu)
print("Mean squared error for eta:", mse_eta)
print("Mean squared error for Theta:", err_Theta)

# write results to a json file, if the result file does not exist, create it. Append to the file if it already exists.
results = {
    "mu": {
        "true": mu_true.tolist(),
        "estimated": avg_mu.tolist(),
        "mse": mse_mu.tolist()
    },
    "eta": {
        "true": eta_true.tolist(),
        "estimated": avg_eta.tolist(),
        "mse": mse_eta.tolist()
    },
    "nu": {
        "true": nu_true.tolist(),
        "estimated": avg_nu.tolist(),
        "mse": mse_nu.tolist()
    },
    "Theta": {
        "true": Theta_true.tolist(),
        "estimated": avg_theta.tolist(),
        "mse": err_Theta
    }
}

with open(f"results/results_{args.method}.json", "a") as f:
    json.dump(results, f)