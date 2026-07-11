import numpy as np
import EM_algorithm as em
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

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
n = 5000
mu_true  = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
eta_true = np.array([1.5, -1.0, 0.0, 2.0, -1.5])    
nu_true  = np.array([0.15, 0.25, 0.35, 0.10, 0.30]) 
Theta_true = np.eye(p) * 2.0
for j in range(p - 1):
    Theta_true[j, j + 1] = Theta_true[j + 1, j] = 0.5
assert np.all(np.linalg.eigvalsh(Theta_true) > 0)

Y, tau_true = simulate_aat_data(n, p, mu_true, eta_true, nu_true, Theta_true, rng)
print(f"Simulated data: Y shape = {Y.shape}")

result = em.run_em_diagonal(Y, n_iter=300, rho=0.05, err=1e-3, run_until_convergence = True)

print(f"{'':>10} {'true':>30} {'estimated':>30}")
print(f"{'mu':>10} {np.round(mu_true, 3)} {np.round(result['mu'], 3)}")
print(f"{'eta':>10} {np.round(eta_true, 3)} {np.round(result['eta'], 3)}")
print(f"{'nu':>10} {np.round(nu_true, 3)} {np.round(result['nu'], 3)}")
print("\nTrue Theta:\n", np.round(Theta_true, 3))
print("\nEstimated Theta (glasso):\n", np.round(result["Theta"], 3))


print("Mean squared error for mu:" , mean_squared_error(np.round(mu_true, 3), np.round(result['mu'], 3)))
print("Mean squared error for nu:" , mean_squared_error(np.round(nu_true, 3), np.round(result['nu'], 3)))
print("Mean squared error for eta:" , mean_squared_error(np.round(eta_true, 3), np.round(result['eta'], 3)))
print("Mean squared error for Theta:" , mean_squared_error(np.round(Theta_true, 3), np.round(result['Theta'], 3)))

print("Relative error for mu:", mean_absolute_percentage_error(np.round(mu_true, 3), np.round(result['mu'], 3)))
print("Relative error for nu:", mean_absolute_percentage_error(np.round(nu_true, 3), np.round(result['nu'], 3)))
print("Relative error for eta:", mean_absolute_percentage_error(np.round(eta_true, 3), np.round(result['eta'], 3)))
print("Relative error for Theta:", mean_absolute_percentage_error(np.round(Theta_true, 3), np.round(result['Theta'], 3)))




