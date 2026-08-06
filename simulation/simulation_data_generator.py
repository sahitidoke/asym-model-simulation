import numpy as np

def simulate_aat_data(n, p, mu, eta, nu, Theta_true, rng):
    Psi_true = np.linalg.inv(Theta_true)
    alpha = 2.0 / nu
    beta = 2.0 / nu
    G = rng.gamma(shape=alpha, scale=1.0 / beta, size=(n, p))
    tau = 1.0 / G
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    Y = mu[None, :] + eta[None, :] * nu[None, :] * tau + np.sqrt(tau) * X
    return Y, tau

def simulate_noisy_gaussian_data(n, p, Theta_true,
                  skewness=0.0, 
                  outlier_frac=0.0, 
                  outlier_scale=4.0,
                  noise_scale=0.0,
                  seed=None):
    """
    
    Parameters:
    -----------
    n : int
    p : int
    Theta_true : (p, p) array
        true precision matrix
    skewness : float, default=0.0
        skewness strength. 0=symmetric, 0.3=light, 0.5=moderate, 0.8=heavy
    outlier_frac : float, default=0.0
        outlier fraction. 0.02=2%, 0.05=5%
    outlier_scale : float, default=4.0
        outlier magnitude (multiple of data standard deviation)
    noise_scale : float, default=0.0
        additional Gaussian noise standard deviation (multiple of data standard deviation)
    seed : int, optional
        random seed
    
    Returns:
    --------
    Y : (n, p) array
        generated observed data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Generate multivariate Gaussian data
    Psi_true = np.linalg.inv(Theta_true)
    Y = np.random.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    
    # 2. Add skewness
    if skewness != 0:
        Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
        Y = np.sinh(np.arcsinh(Y_std) + skewness)
        Y = Y * Y.std(axis=0, ddof=1) + Y.mean(axis=0)
    
    # 3. Add outliers
    n_outliers = int(n * outlier_frac)
    if n_outliers > 0:
        idx = np.random.choice(n, size=n_outliers, replace=False)
        scale = outlier_scale * Y.std(axis=0, ddof=1)
        Y[idx] = np.random.normal(loc=0, scale=scale, size=(n_outliers, p))
    
    # 4. 
    if noise_scale > 0:
        noise = np.random.normal(loc=0, 
                                 scale=noise_scale * Y.std(axis=0, ddof=1), 
                                 size=(n, p))
        Y = Y + noise
    
    return Y