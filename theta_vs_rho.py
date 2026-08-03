"""Plot ||Theta(rho_i) - Theta(rho_{i-1})||_F as rho increases."""

import os

import numpy as np
from matplotlib import pyplot as plt

import sachs_min as data
import EM_algorithm as em

Y = data.load_control()
rho_min = 0
rho_max = 0.4
rhos = np.arange(rho_min, rho_max, 0.005)

thetas = []
for rho in rhos:
    print(f"Rho = {rho}\n")
    results, _ = em.run_em_MWGP(Y, rho=rho, n_iter=100, mcmc_samples=200,
                                random_state=42, proposal="gig", verbose=False)
    thetas.append(results["Theta"])
    print(results["Theta"])
    print("\n\n")

diffs = np.linalg.norm(np.diff(thetas, axis=0), ord="fro", axis=(1, 2))

plt.plot(rhos[1:], diffs, marker="o")
plt.xlabel("rho")
plt.ylabel("||Theta(rho_i) - Theta(rho_{i-1})||_F")
os.makedirs("results/applications/theta_vs_rho", exist_ok=True)
plt.savefig(f"results/applications/theta_vs_rho_{rho_min}_{rho_max}.pdf")
