"""
Mean ROC/AUC curves for precision-matrix support recovery, in the style of
Fig. 1-3 / Table 1 of docs/sggm.pdf.

Running instructions:
    python roc_simulation.py --num_simulations 30 --filename diagonal

For each replicate: simulate Y from the true model, run one EM_MWGP fit to
obtain the expected sufficient statistic S_tau from its MCMC E-step, then
re-solve *only* the graphical lasso step over a grid of rho values (this is
the "sweep the precision-matrix tuning parameter with the nuisance parameters
fixed" strategy used for the mBIC/BIC split in sggm.pdf sec 6). A naive
Gaussian graphical lasso fit directly on the raw sample covariance of Y is
included as a baseline, mirroring the GGM comparison in sggm.pdf.
"""

import os
import argparse
import numpy as np
import json
from sklearn.covariance import graphical_lasso
from matplotlib import pyplot as plt

from method import EM_algorithm as em, tlasso
from simulation import make_true_theta
import simulation_data_generator as dg

rng = np.random.default_rng()

COLOR_ASYM = "#2a78d6"
COLOR_GGM = "#e34948"
COLOR_CHANCE = "#c3c2b7"

def edge_confusion(Theta_hat, true_pos_mask, true_neg_mask, tol=1e-8):
    iu = np.triu_indices(Theta_hat.shape[0], k=1)
    est_edges = np.abs(Theta_hat[iu]) > tol
    tp_rate = np.sum(est_edges & true_pos_mask) / true_pos_mask.sum()
    fp_rate = np.sum(est_edges & true_neg_mask) / true_neg_mask.sum()
    return fp_rate, tp_rate


def roc_curve_em(
    Y, rho_grid, algorithm, true_pos_mask, true_neg_mask,
    algorithm_kwargs=None, rho_name="rho", theta_key="Theta",
):
    """Re-run the whole EM at each rho, sparsest first.

    `algorithm_kwargs` holds whatever extra arguments the given algorithm takes
    (e.g. nu/n_iter for run_tlasso, n_burn/n_keep for run_em_MWGP); `rho_name`
    and `theta_key` cover algorithms that name the penalty or the returned
    precision matrix differently.
    """
    kwargs = {} if algorithm_kwargs is None else dict(algorithm_kwargs)
    fp = np.empty(len(rho_grid))
    tp = np.empty(len(rho_grid))
    Theta_prev = np.eye(Y.shape[1])
    for i in np.argsort(rho_grid)[::-1]:
        try:
            Theta_hat = algorithm(Y, **{rho_name: rho_grid[i]}, **kwargs)[theta_key]
        except Exception:
            Theta_hat = Theta_prev
        Theta_prev = Theta_hat
        fp[i], tp[i] = edge_confusion(Theta_hat, true_pos_mask, true_neg_mask)
    return fp, tp


def roc_curve_glasso(Y, rho_grid, true_pos_mask, true_neg_mask, glasso_kwargs=None):
    """roc_curve_full_em analogue for the naive Gaussian glasso baseline.

    sklearn's graphical_lasso takes an empirical covariance rather than Y,
    names the penalty `alpha`, and returns (covariance, precision), so it needs
    its own loop.
    """
    kwargs = {} if glasso_kwargs is None else dict(glasso_kwargs)
    S = np.cov(Y, rowvar=False) + 1e-10 * np.eye(Y.shape[1])
    fp = np.empty(len(rho_grid))
    tp = np.empty(len(rho_grid))
    Theta_prev = np.eye(Y.shape[1])
    for i in np.argsort(rho_grid)[::-1]:
        try:
            _, Theta_hat = graphical_lasso(S, alpha=rho_grid[i], **kwargs)
        except Exception:
            Theta_hat = Theta_prev
        Theta_prev = Theta_hat
        fp[i], tp[i] = edge_confusion(Theta_hat, true_pos_mask, true_neg_mask)
    return fp, tp


def auc_from_curve(fp, tp):
    order = np.argsort(fp)
    fp_sorted = np.concatenate([[0.0], fp[order], [1.0]])
    tp_sorted = np.concatenate([[0.0], tp[order], [1.0]])
    return np.trapezoid(tp_sorted, fp_sorted)


def main():
    parser = argparse.ArgumentParser(
        description="Mean ROC/AUC curves for precision-matrix support recovery."
    )
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--num_simulations", type=int, default=30)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--num_rho", type=int, default=30)
    parser.add_argument("--rho_min", type=float, default=2e-3)
    parser.add_argument("--rho_max", type=float, default=3.0)
    args = parser.parse_args()

    p, n = args.p, args.n
    mu_true = rng.uniform(low=-1, high=1, size=p)
    eta_true = rng.uniform(low=-1, high=1, size=p)
    nu_true = rng.uniform(low=0.15, high=0.9, size=p)
    Theta_true = make_true_theta(p, rng=rng)

    iu = np.triu_indices(p, k=1)
    true_pos_mask = Theta_true[iu] != 0
    true_neg_mask = ~true_pos_mask

    rho_grid = np.logspace(np.log10(args.rho_min), np.log10(args.rho_max), args.num_rho)

    fp_asym = np.empty((args.num_simulations, args.num_rho))
    tp_asym = np.empty((args.num_simulations, args.num_rho))
    fp_ggm = np.empty((args.num_simulations, args.num_rho))
    tp_ggm = np.empty((args.num_simulations, args.num_rho))
    fp_t = np.empty((args.num_simulations, args.num_rho))
    tp_t = np.empty((args.num_simulations, args.num_rho))
    fp_ts = np.empty((args.num_simulations, args.num_rho))
    tp_ts = np.empty((args.num_simulations, args.num_rho))
    auc_asym = np.empty(args.num_simulations)
    auc_ggm = np.empty(args.num_simulations)
    auc_t = np.empty(args.num_simulations)
    auc_ts = np.empty(args.num_simulations)

    for sim in range(args.num_simulations):
        print(f"Replicate {sim + 1}/{args.num_simulations}")

        # Generate data from an independent model (noisy skewed Gaussian)
        Y = dg.simulate_noisy_gaussian_data(n, p, Theta_true, skewness=0.3, outlier_frac=0.05, outlier_scale=4.0, noise_scale=0.1, seed=sim)

        # Asymmetric Alternative t-distribution model (EM_MWGP)
        fp_asym[sim], tp_asym[sim] = roc_curve_em(
            Y, rho_grid, em.run_em_MWGP, true_pos_mask, true_neg_mask,
            algorithm_kwargs={"n_iter": 200, 
                              "verbose": False, 
                              "warning": True, 
                              "mcmc_samples": 200,
                              "mcmc_thin": 1,
                              "mcmc_warmup": 30,
                              "proposal": "gig"}
        )
        auc_asym[sim] = auc_from_curve(fp_asym[sim], tp_asym[sim])
        
        # Classical t-distribution model (run_tlasso)
        fp_t[sim], tp_t[sim] = roc_curve_em(
            Y, rho_grid, tlasso.run_tlasso, true_pos_mask, true_neg_mask,
            algorithm_kwargs={"n_iter": 200, "verbose": False}
        )
        auc_t[sim] = auc_from_curve(fp_t[sim], tp_t[sim])
        
        # Alternative t-distribution model (run_tstar_varlasso)
        fp_ts[sim], tp_ts[sim] = roc_curve_em(
            Y, rho_grid, tlasso.run_tstar_varlasso, true_pos_mask, true_neg_mask,
            algorithm_kwargs={"n_iter": 200, "verbose": False}
        )
        auc_ts[sim] = auc_from_curve(fp_ts[sim], tp_ts[sim])

        # Naive Gaussian graphical lasso baseline
        fp_ggm[sim], tp_ggm[sim] = roc_curve_glasso(
            Y, rho_grid, true_pos_mask, true_neg_mask,
            glasso_kwargs={"max_iter": 2000, "verbose": False}
        )
        auc_ggm[sim] = auc_from_curve(fp_ggm[sim], tp_ggm[sim])

    # Average the ROC curves across replicates and compute mean AUCs
    fp_asym_mean, tp_asym_mean = fp_asym.mean(axis=0), tp_asym.mean(axis=0)
    fp_ggm_mean, tp_ggm_mean = fp_ggm.mean(axis=0), tp_ggm.mean(axis=0)
    fp_t_mean, tp_t_mean = fp_t.mean(axis=0), tp_t.mean(axis=0)
    fp_ts_mean, tp_ts_mean = fp_ts.mean(axis=0), tp_ts.mean(axis=0)

    print(f"\n(p={p}, n={n}) over {args.num_simulations} replicates")
    print(f"  Asymmetric model (EM_MWGP): AUC = {auc_asym.mean():.3f} "
          f"(SE {auc_asym.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
    print(f"  Classical t-model (TLASSO): AUC = {auc_t.mean():.3f} "
          f"(SE {auc_t.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
    print(f"  Alternative t-model (TSTAR_VARLASSO): AUC = {auc_ts.mean():.3f} "
          f"(SE {auc_ts.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
    print(f"  Naive Gaussian glasso (GGM): AUC = {auc_ggm.mean():.3f} "
          f"(SE {auc_ggm.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=COLOR_CHANCE)
    ax.plot(
        fp_asym_mean, tp_asym_mean, color=COLOR_ASYM, linewidth=2,
        label=f"Asym. model, avg AUC={auc_asym.mean():.3f}",
    )
    ax.plot(
        fp_ggm_mean, tp_ggm_mean, color=COLOR_GGM, linewidth=2, linestyle="-.",
        label=f"Naive GGM, avg AUC={auc_ggm.mean():.3f}",
    )
    ax.plot(
        fp_t_mean, tp_t_mean, color=COLOR_T, linewidth=2, linestyle="-.",
        label=f"Classical t-model, avg AUC={auc_t.mean():.3f}",
    )
    ax.plot(
        fp_ts_mean, tp_ts_mean, color=COLOR_TS, linewidth=2, linestyle="-.",
        label=f"Alternative t-model, avg AUC={auc_ts.mean():.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title(f"Precision-matrix support recovery: p={p}, n={n}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    filename = (
        "results/simulations/roc/roc_EM_MWGP" if args.filename is None
        else f"results/simulations/roc/roc_EM_MWGP_{args.filename}"
    )
    fig.savefig(f"{filename}.pdf")

    results = {
        "p": p,
        "n": n,
        "method": "EM_MWGP",
        "rho_grid": rho_grid.tolist(),
        "asym": {
            "fp_mean": fp_asym_mean.tolist(),
            "tp_mean": tp_asym_mean.tolist(),
            "auc_mean": float(auc_asym.mean()),
            "auc_se": float(auc_asym.std(ddof=1) / np.sqrt(args.num_simulations)),
        },
        "ggm": {
            "fp_mean": fp_ggm_mean.tolist(),
            "tp_mean": tp_ggm_mean.tolist(),
            "auc_mean": float(auc_ggm.mean()),
            "auc_se": float(auc_ggm.std(ddof=1) / np.sqrt(args.num_simulations)),
        },
    }
    os.makedirs("results/simulations/roc", exist_ok=True)
    with open(f"{filename}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
