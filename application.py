from data import sachs_min, SNP500
from method import EM_algorithm as em, stars
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Run the asymmetric t-distribution application on real data."
)

parser.add_argument(
    "--standardize",
    type = bool,
    default = False,
    help="Standardize the data before running the algorithm. Default: False.",
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
    Y, NAMES = sachs_min.load_control()
elif args.database == "SNP500":
    Y, NAMES = SNP500.load_snp_500(50)

# Standardize data
if (args.standardize):
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

# Run algorithm
n,p = Y.shape
RHO = np.sqrt(np.log(p)/n)  # default rho for application
fitting_algorithm = em.run_em_MWGP

# Select RHO using StARS
rho_grid = stars.rho_grid(Y)
print(f"Rho grid: {rho_grid}")
RHO, rho_curve = stars.stars(Y, rho_grid, fitting_algorithm, N = 20, beta = 0.05)

# plot rho_curve, which is a python list of tuples (rho, D)
plt.figure(figsize=(10, 6))
plt.plot([rho for rho, _ in rho_curve], [D for _, D in rho_curve], 'o-')
plt.xlabel('Rho')
plt.ylabel('D')
plt.title('StARS: Rho Selection')
plt.savefig(f"results/applications/rho_curve_{args.database}.pdf")

print(f"Running EM algorithm on {args.database} data with shape {Y.shape} and rho={RHO:.4f}...\n")
print("=" * 80)

results = fitting_algorithm(
    Y,
    n_iter=200,
    rho=RHO,
    verbose=True,
    warning=True,
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
    "NAMES": NAMES,
}


filename = f"results/applications/application_results.json" if args.filename is None else f"results/applications/{args.filename}.json"
os.makedirs("results/applications", exist_ok=True)
with open(filename, "w") as f:
    json.dump(output, f)

