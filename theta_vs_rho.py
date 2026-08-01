"""Plot ||Theta(rho_i) - Theta(rho_{i-1})||_F as rho increases."""

import numpy as np
from matplotlib import pyplot as plt

import sachs_min as data
import EM_algorithm as em

Y = data.load_control()
rhos = np.arange(0, 0.01, 0.0005)

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
plt.savefig("results/theta_vs_rho.pdf")
