"""
Running instructions (run in terminal, from the repository root):
# VAE without diagnostics
python -m simulation.simulation --method vae

# VAE with identifiability diagnostics
python -m simulation.simulation --method vae --diagnostics

# EM
Example:
python -m simulation.simulation --method em_diagonal --num_simulations 10 --filename large_mu
"""

import argparse
import numpy as np
import json
import os

from method import EM_algorithm as em
from method import aat_vae, tlasso
import simulation.simulation_data_generator as dg

rng = np.random.default_rng()

if __name__ == "__main__":

    """
    Command-line argument parsing
    """

    parser = argparse.ArgumentParser(
        description="Run the asymmetric t-distribution simulation."
    )
    
    parser.add_argument(
        "--p",
        type=int,
        default=100,
        help="Dimension of observations. Default: 100.",
    )
    
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of observations. Default: 50.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Name for result json file method. Default: None.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="EM_DIAGONAL",
        help="Parameter estimation method. Default: EM_DIAGONAL.",
    )

    parser.add_argument(
        "--num_simulations",
        type=int,
        default=10,
        help="Number of simulations to run. Default: 10.",
    )


    args = parser.parse_args()
    
    """
    True distribution parameters
    """

    # Parameters
    n,p = args.n, args.p
    SEED = 20260814
    rng = np.random.default_rng(SEED)

    mu_true = np.array([0.5, -0.8, 1.2, -0.4, 0.7])
    nu_true = np.array([0.8, 0.45, 0.9, 1.5, 0.3])       
    eta_true = np.array([1.0, 1.1, 0.9, 1.2, 1.05])
    
    Theta_true = dg.make_true_theta(p, rng, prob = 0.1)
    
    """
    Run the specified method (VAE or EM) to estimate parameters from the simulated data.
    """
    NUM_SIMULATIONS = args.num_simulations
    for sim in range(NUM_SIMULATIONS):
        print(f"Simulation {sim+1}")
        Y, tau_true = dg.simulate_aat_data(args.n, args.p, Theta_true,mu_true,eta_true,nu_true,rng)
        print(f"Simulated data: Y shape = {Y.shape}")
        mus, etas, nus, thetas = [], [], [], []
        if args.method == "VAE":
            config = aat_vae.VAEConfig(
                epochs=100,
                posterior_samples=16,
                theta_l1=0.0025,
                encoder_steps=5,
                flow_layers=6
            )

            model, history = aat_vae.fit_aat_vae(Y, config=config)
            result = model.decoder.estimates()
        elif args.method == "tslasso":
            result = tlasso.run_tstar_varlasso(
                Y,
                n_iter=500,
                rho=0.05,
                tol=1e-8,
            )
        elif args.method == "EM_EXACT":
            result = em.run_em_exact(
                Y,
                n_iter=500,
                rho=0.0025,
                err=1e-8,
                run_until_convergence=False,
            )
        elif args.method == "EM_DIAGONAL":
            result = em.run_em_diagonal(
                Y,
                n_iter=200,
                rho=0.20,
                tol = 1e-8,
            )
        elif args.method == "EM_MWG":
            result = em.run_em_MWG(
                Y,
                n_iter=100,
                rho=0.0025,
                verbose=True,
                err=1e-3,
                run_until_convergence=False,
                mcmc_samples=100,
                proposal_scale=0.35,
                random_state=42,
            )
        elif args.method == "EM_MWGP":
            result = em.run_em_MWGP(
                Y,
                n_iter=50,
                rho=0.1,
                verbose=True,
                mcmc_samples=500,
                mcmc_thin=1,
                mcmc_warmup=30,
                random_state=42,
            )
        elif args.method == "EM_IMPORTANCE":
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
        print("\n")

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

    filename = f"results/simulations/{args.method}.json" if args.filename is None else f"results/simulations/{args.method}_{args.filename}.json"
    os.makedirs("results/simulations", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(results, f)
