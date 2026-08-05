"""StARS (Liu, Roeder & Wasserman 2010): pick rho by subsampling stability."""

import numpy as np


def rho_grid(Y, k=30, ratio=0.02):
    S = np.cov(Y, rowvar=False)
    np.fill_diagonal(S, 0.0)
    hi = np.abs(S).max()
    return np.exp(np.linspace(np.log(ratio * hi), np.log(hi), k))


def stars(Y, rhos, fit, N=20, beta=0.05, b=None, seed=None):
    """fit(Ysub, rho, init) -> precision matrix. Returns (selected rho, [(rho, D)...])."""
    Y = np.asarray(Y, float)
    n, p = Y.shape
    b = b or int(10 * np.sqrt(n))
    assert b < 0.7 * n, f"b/n = {b/n:.2f}: subsamples too close to the full sample"

    rng = np.random.default_rng(seed)
    subs = [Y[rng.choice(n, b, replace=False)] for _ in range(N)]
    inits = [None] * N
    i, j = np.triu_indices(p, 1)

    best, curve, runmax = max(rhos), [], 0.0
    for rho in sorted(rhos, reverse=True):          # sparsest first
        print(f"StARS: rho={rho:.4f}...", end="\n", flush=True)
        edges = np.zeros((p, p))
        for s in range(N):
            print(f"subsample {s+1}/{N}...", end="\n", flush=True)
            inits[s] = fit(Y = subs[s], rho = rho, init = inits[s], verbose = False, warning = True)  
            A = np.abs(inits[s]["Theta"]) > 1e-8
            np.fill_diagonal(A, False)
            edges += A
        th = edges / N
        D = (2 * th * (1 - th))[i, j].mean()
        curve.append((rho, D))
        runmax = max(runmax, D)                     # this is Dbar(rho)
        if runmax > beta:
            break
        best = rho

    return best, curve