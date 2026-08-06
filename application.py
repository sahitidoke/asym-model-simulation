from data import sachs_min, SNP500
from method import EM_algorithm as em, stars
import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
    description="Run the asymmetric t-distribution application on real data."
)

parser.add_argument(
    "--standardize",
    type=bool,
    help="Standardize the data before running the algorithm. Default: False. "
         "rho = sqrt(log p / n) is only scale-free on standardized data: on raw "
         "SNP500 log-returns it is ~700x the largest off-diagonal covariance and "
         "glasso returns an empty graph.",
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
parser.add_argument(
    "--method", 
    type=str, 
    default="diagonal", 
    choices=["diagonal", "mwgp"]
)


args = parser.parse_args()


# Load data
if args.database == "sachs_min":
    Y, NAMES = sachs_min.load_control()
elif args.database == "SNP500":
    Y, NAMES = SNP500.load_snp_500()

# Standardize data
if (args.standardize):
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

# Run algorithm
n,p = Y.shape
if (args.method == "diagonal"):
    fitting_algorithm = em.run_em_diagonal
elif (args.method == "mwgp"):   
    fitting_algorithm = em.run_em_MWGP

# Select RHO using StARS
RHO = np.sqrt(np.log(p)/n)  # default value if StARS fails
rho_grid = stars.rho_grid(Y, k = 10)
print(f"Theoretical Rho: {RHO}, Rho grid: {rho_grid}")

# RHO, rho_curve = stars.stars(Y, rho_grid, fitting_algorithm, N = 10, beta = 0.05)

# # plot rho_curve, which is a python list of tuples (rho, D)
# plt.figure(figsize=(10, 6))
# plt.plot([rho for rho, _ in rho_curve], [D for _, D in rho_curve], 'o-')
# plt.xlabel('Rho')
# plt.ylabel('D')
# plt.title('StARS: Rho Selection')
# plt.savefig(f"results/applications/rho_curve_{args.database}.pdf")

print(f"Running EM algorithm on {args.database} data with shape {Y.shape} and rho={RHO:.5f}...\n")
print("=" * 80)

results = fitting_algorithm(
    Y,
    n_iter=200,
    rho=RHO,
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

