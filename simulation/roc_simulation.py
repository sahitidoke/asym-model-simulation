"""
Mean ROC/AUC curves for precision-matrix support recovery, in the style of
Fig. 1-3 / Table 1 of docs/sggm.pdf.

Running instructions (from the repository root):
    python -m simulation.roc_simulation --num_simulations 50 --filename five_methods_comparison

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
from simulation.simulation import make_true_theta
from simulation import simulation_data_generator as dg

rng = np.random.default_rng()

COLOR_ASYM = "#2a78d6"
COLOR_EM_DIAG = "#f5239a"
COLOR_GGM = "#e34948"
COLOR_T = "#f5a623"
COLOR_TS = "#23f57e"
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
    print(f"Running {algorithm.__name__} over {len(rho_grid)} rho values...")
    for i in range(len(rho_grid)):
        print(f"  rho={rho_grid[i]:.5f} ({i+1}/{len(rho_grid)})")
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
    print(f"Running graphical_lasso over {len(rho_grid)} rho values...")
    for i in range(len(rho_grid)):
        print(f"  rho={rho_grid[i]:.5f} ({i+1}/{len(rho_grid)})")
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
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--num_rho", type=int, default=21) 
    parser.add_argument(
        "--width", type=float, default=0.1,
    )
    args = parser.parse_args()

    p, n = args.p, args.n
    Theta_true = make_true_theta(p, rng=rng)

    iu = np.triu_indices(p, k=1)
    true_pos_mask = Theta_true[iu] != 0
    true_neg_mask = ~true_pos_mask

    theoretical_rho = np.sqrt(np.log(p) / n)
    # make a evenly spaced grid centered at theoretical rho with half length width
    rho_grid = np.linspace(theoretical_rho - args.width, theoretical_rho + args.width, args.num_rho)

    print(f"theoretical rho = sqrt(log({p}) / {n}) = {theoretical_rho:.5g}")
    print(f"rho grid: {rho_grid}")

    fp_mwgp = np.empty((args.num_simulations, args.num_rho))
    tp_mwgp = np.empty((args.num_simulations, args.num_rho))
    fp_em_diag = np.empty((args.num_simulations, args.num_rho))
    tp_em_diag = np.empty((args.num_simulations, args.num_rho))
    fp_ggm = np.empty((args.num_simulations, args.num_rho))
    tp_ggm = np.empty((args.num_simulations, args.num_rho))
    fp_t = np.empty((args.num_simulations, args.num_rho))
    tp_t = np.empty((args.num_simulations, args.num_rho))
    fp_ts = np.empty((args.num_simulations, args.num_rho))
    tp_ts = np.empty((args.num_simulations, args.num_rho))
    auc_mwgp = np.empty(args.num_simulations)
    auc_em_diag = np.empty(args.num_simulations)
    auc_ggm = np.empty(args.num_simulations)
    auc_t = np.empty(args.num_simulations)
    auc_ts = np.empty(args.num_simulations)

    for sim in range(args.num_simulations):
        print(f"Replicate {sim + 1}/{args.num_simulations}")

        # Generate data from an independent model (noisy skewed Gaussian)
        Y = dg.simulate_noisy_gaussian_data(n, p, Theta_true, skewness=0.8, outlier_frac=0.05, outlier_scale=4.0, noise_scale=0.1, seed=sim)

        # Asymmetric Alternative t-distribution model (EM_MWGP)
        fp_mwgp[sim], tp_mwgp[sim] = roc_curve_em(
            Y, rho_grid, em.run_em_MWGP, true_pos_mask, true_neg_mask,
            algorithm_kwargs={"n_iter": 50, 
                              "verbose": True, 
                              "warning": True, 
                              "mcmc_samples": 100,
                              "mcmc_thin": 1,
                              "mcmc_warmup": 10,
                              "proposal": "gig"}
        )
        auc_mwgp[sim] = auc_from_curve(fp_mwgp[sim], tp_mwgp[sim])

        # Asymmetric Alternative t-distribution model (EM_DIAGONAL)
        fp_em_diag[sim], tp_em_diag[sim] = roc_curve_em(
            Y, rho_grid, em.run_em_diagonal, true_pos_mask, true_neg_mask,
            algorithm_kwargs={"n_iter": 200, "verbose": False}
        )
        auc_em_diag[sim] = auc_from_curve(fp_em_diag[sim], tp_em_diag[sim])

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
    fp_mwgp_mean, tp_mwgp_mean = fp_mwgp.mean(axis=0), tp_mwgp.mean(axis=0)
    fp_em_diag_mean, tp_em_diag_mean = fp_em_diag.mean(axis=0), tp_em_diag.mean(axis=0)
    fp_ggm_mean, tp_ggm_mean = fp_ggm.mean(axis=0), tp_ggm.mean(axis=0)
    fp_t_mean, tp_t_mean = fp_t.mean(axis=0), tp_t.mean(axis=0)
    fp_ts_mean, tp_ts_mean = fp_ts.mean(axis=0), tp_ts.mean(axis=0)

    print(f"\n(p={p}, n={n}) over {args.num_simulations} replicates")
    print(f"  Asymmetric model (MWGP): AUC = {auc_mwgp.mean():.3f} "
          f"(SE {auc_mwgp.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
    print(f"  Asymmetric model (Diagonal): AUC = {auc_em_diag.mean():.3f} "
          f"(SE {auc_em_diag.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
    print(f"  Classical t-model (TLASSO): AUC = {auc_t.mean():.3f} "
          f"(SE {auc_t.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
    print(f"  Alternative t-model (TSTAR_VARLASSO): AUC = {auc_ts.mean():.3f} "
          f"(SE {auc_ts.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
    print(f"  Naive Gaussian glasso (GGM): AUC = {auc_ggm.mean():.3f} "
          f"(SE {auc_ggm.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=COLOR_CHANCE)
    ax.plot(
        fp_mwgp_mean, tp_mwgp_mean, color=COLOR_ASYM, linewidth=2,
        label=f"Asym. model MWGP, avg AUC={auc_mwgp.mean():.3f}",
    )
    ax.plot(
        fp_em_diag_mean, tp_em_diag_mean, color=COLOR_EM_DIAG, linewidth=2,
        label=f"Asym. diag. model, avg AUC={auc_em_diag.mean():.3f}",
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
    # theoretical rho is the midpoint of the centred grid
    i_star = args.num_rho // 2
    for fp_mean, tp_mean, color in (
        (fp_mwgp_mean, tp_mwgp_mean, COLOR_ASYM),
        (fp_ggm_mean, tp_ggm_mean, COLOR_GGM),
        (fp_t_mean, tp_t_mean, COLOR_T),
        (fp_ts_mean, tp_ts_mean, COLOR_TS),
        (fp_em_diag_mean, tp_em_diag_mean, COLOR_EM_DIAG),
    ):
        ax.plot(fp_mean[i_star], tp_mean[i_star], "o", color=color, zorder=5)
    ax.plot([], [], "o", color="#52514e",
            label=r"$\rho=\sqrt{\log p\,/\,n}$" + f" = {theoretical_rho:.3g}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title(f"Precision-matrix support recovery: p={p}, n={n}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    os.makedirs("results/simulations/roc", exist_ok=True)
    filename = (
        "results/simulations/roc/roc" if args.filename is None
        else f"results/simulations/roc/roc_{args.filename}"
    )
    fig.savefig(f"{filename}.pdf")

    results = {
        "p": p,
        "n": n,
        "rho_grid": rho_grid.tolist(),
        "theoretical_rho": float(theoretical_rho),
        "asym_mwgp": {
            "fp_mean": fp_mwgp_mean.tolist(),
            "tp_mean": tp_mwgp_mean.tolist(),
            "auc_mean": float(auc_mwgp.mean()),
            "auc_se": float(auc_mwgp.std(ddof=1) / np.sqrt(args.num_simulations)),
        },
        "asym_em_diag": {
            "fp_mean": fp_em_diag_mean.tolist(),
            "tp_mean": tp_em_diag_mean.tolist(),
            "auc_mean": float(auc_em_diag.mean()),
            "auc_se": float(auc_em_diag.std(ddof=1) / np.sqrt(args.num_simulations)),
        },
        "ggm": {
            "fp_mean": fp_ggm_mean.tolist(),
            "tp_mean": tp_ggm_mean.tolist(),
            "auc_mean": float(auc_ggm.mean()),
            "auc_se": float(auc_ggm.std(ddof=1) / np.sqrt(args.num_simulations)),
        },
        "t": {
            "fp_mean": fp_t_mean.tolist(),
            "tp_mean": tp_t_mean.tolist(),
            "auc_mean": float(auc_t.mean()),
            "auc_se": float(auc_t.std(ddof=1) / np.sqrt(args.num_simulations)),
        },
        "ts": {
            "fp_mean": fp_ts_mean.tolist(),
            "tp_mean": tp_ts_mean.tolist(),
            "auc_mean": float(auc_ts.mean()),
            "auc_se": float(auc_ts.std(ddof=1) / np.sqrt(args.num_simulations)),
        },
    }
    
    with open(f"{filename}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
