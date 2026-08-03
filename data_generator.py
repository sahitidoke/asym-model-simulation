import numpy as np

def generate_data(n, p, Theta_true,
                  skewness=0.0, 
                  outlier_frac=0.0, 
                  outlier_scale=4.0,
                  noise_scale=0.0,
                  seed=None):
    """
    生成数据：高斯基础 + 可选偏斜 + 可选离群值 + 可选额外噪声
    
    Parameters:
    -----------
    n : int
        样本量
    p : int
        维度（变量个数）
    Theta_true : (p, p) array
        真实精度矩阵（必须传入！）
    skewness : float, default=0.0
        偏斜强度。0=对称，0.3=轻度，0.5=中度
    outlier_frac : float, default=0.0
        离群值比例。0.02=2%，0.05=5%
    outlier_scale : float, default=4.0
        离群值幅度（相对于数据标准差的倍数）
    noise_scale : float, default=0.0
        额外高斯噪声的标准差（相对于数据标准差的倍数）
    seed : int, optional
        随机种子
    
    Returns:
    --------
    Y : (n, p) array
        生成的观测数据
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. 生成干净高斯数据
    Psi_true = np.linalg.inv(Theta_true)
    Y = np.random.multivariate_normal(mean=np.zeros(p), cov=Psi_true, size=n)
    
    # 2. 加偏斜
    if skewness != 0:
        Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
        Y = np.sinh(np.arcsinh(Y_std) + skewness)
        Y = Y * Y.std(axis=0, ddof=1) + Y.mean(axis=0)
    
    # 3. 加离群值
    n_outliers = int(n * outlier_frac)
    if n_outliers > 0:
        idx = np.random.choice(n, size=n_outliers, replace=False)
        scale = outlier_scale * Y.std(axis=0, ddof=1)
        Y[idx] = np.random.normal(loc=0, scale=scale, size=(n_outliers, p))
    
    # 4. 加额外噪声
    if noise_scale > 0:
        noise = np.random.normal(loc=0, 
                                 scale=noise_scale * Y.std(axis=0, ddof=1), 
                                 size=(n, p))
        Y = Y + noise
    
    return Y