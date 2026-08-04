import sachs_min
import SNP500
import EM_algorithm as em
import json
import argparse
import os

parser = argparse.ArgumentParser(
    description="Run the asymmetric t-distribution application on real data."
)

parser.add_argument(
    "--filename",
    type=str,
    default=None,
    help="Name for result json file. Default: None.",
)

parser.add_argument(
    "--database",
    type=str,
    default="sachs_min",
    help="Database name. Default: sachs_min.",
)

args = parser.parse_args()


# Load data
if args.database == "sachs_min":
    Y = sachs_min.load_control()
elif args.database == "SNP500":
    Y = SNP500.load_snp_500()
    Y = Y[:,:50]  # Use only the first 50 stocks for n >> p

m, s = Y.mean(0), Y.std(0, ddof=1)
# Run algorithm
results,_ = em.run_em_MWGP(
    (Y - m) / s,
    n_iter=200,
    rho=0.2,
    verbose=True,
    err=1e-5,
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


filename = f"results/applications/application_results.json" if args.filename is None else f"results/applications/{args.filename}.json"
os.makedirs("results/applications", exist_ok=True)
with open(filename, "w") as f:
    json.dump(output, f)
