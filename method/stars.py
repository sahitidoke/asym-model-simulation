"""StARS (Liu, Roeder & Wasserman 2010): pick rho by subsampling stability."""

import numpy as np

def make_rho_grid(rho_max, n_rho=50, max_ratio = 1, min_ratio=0.05):
    """Log-spaced, DESCENDING grid on [min_ratio * rho_max, rho_max].

    Log spacing because edge count is roughly geometric in rho: linear
    spacing wastes most points in the dense end where the ROC curve barely
    moves. Descending so warm starts run sparse -> dense, which is the
    numerically stable direction when p > n.
    """
    return np.logspace(np.log10(max_ratio * rho_max), np.log10(min_ratio * rho_max), n_rho)

def offdiag_max(S):
    """max_{i != j} |S_ij|. The smallest rho that zeroes every off-diagonal."""
    A = np.abs(np.asarray(S, dtype=float)).copy()
    np.fill_diagonal(A, 0.0)
    rmax = float(A.max())
    if not np.isfinite(rmax) or rmax <= 0:
        raise ValueError("rho_max is not positive; check the input matrix.")
    return rmax

def pilot_rho_max(
    Y, algorithm, rho, algorithm_kwargs=None, rho_name="rho", S_key="S_tau",
):
    """rho_max for one method on one replicate: fit at `rho`, read off its S.

    The EM M-step penalizes the working covariance S_tau, not cov(Y): S_tau is
    tau-weighted, skew-corrected and (for MWGP) Monte-Carlo averaged, so its
    off-diagonal scale is method-specific. Building every method's grid from
    cov(Y) therefore starts each path at a different point along its own
    regularization range -- some methods begin already empty, others begin
    dense, and the index-wise average across replicates mixes those.

    S_tau also depends on rho, so there is no rho-free version of it to
    calibrate against. The pilot fit at the theoretical rho = sqrt(log p / n)
    is the reference point: run the EM there to convergence, take
    max_{i != j} |S_ij| of the converged S_tau, and use that as rho_max.
    """
    kwargs = {} if algorithm_kwargs is None else dict(algorithm_kwargs)
    print(f"  pilot fit of {algorithm.__name__} at rho={rho:.5f}")
    res = algorithm(Y, **{rho_name: rho}, **kwargs)
    rho_max = offdiag_max(res[S_key])
    print(f"  -> rho_max = max|S_ij| (off-diag) = {rho_max:.5f}")
    return rho_max

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
            inits[s] = fit(Y = subs[s], rho = rho, init = inits[s], verbose = True)  
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