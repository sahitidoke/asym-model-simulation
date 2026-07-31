import sachs_min as data
import EM_algorithm as em
import json
import argparse

parser = argparse.ArgumentParser(
    description="Run the asymmetric t-distribution application on real data."
)

parser.add_argument(
    "--filename",
    type=str,
    default=None,
    help="Name for result json file. Default: None.",
)

args = parser.parse_args()


# Load data
Y = data.load_control()

# Run algorithm
results,_ = em.run_em_MWGP(
    Y,
    n_iter=100,
    rho=0.0025,
    verbose=True,
    err=1e-3,
    run_until_convergence=False,
    mcmc_samples=200,
    mcmc_thin=1,
    mcmc_warmup=30,
    random_state=42,
    proposal="gig",
)

# Save results (convert ndarrays to lists so the dict is JSON-serializable)
output = {
    "mu": results["mu"].tolist(),
    "eta": results["eta"].tolist(),
    "nu": results["nu"].tolist(),
    "Theta": results["Theta"].tolist(),
}

filename = f"results/application_results.json" if args.filename is None else f"results/application_results_{args.filename}.json"
with open(filename, "w") as f:
    json.dump(output, f)
