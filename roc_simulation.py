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

import argparse
import numpy as np
import json
from sklearn.covariance import graphical_lasso
from matplotlib import pyplot as plt

import EM_algorithm as em
from simulation import simulate_aat_data, make_true_theta

rng = np.random.default_rng()

COLOR_ASYM = "#2a78d6"
COLOR_GGM = "#e34948"
COLOR_CHANCE = "#c3c2b7"


def fit_S_tau(Y, reference_rho, mcmc_random_state=42, b_min=0.5):
    """Fit EM_MWGP and return the expected sufficient statistic S_tau from its
    final MCMC E-step. S_tau is the (mu, eta, nu)-fixed covariance surrogate
    that the glasso step consumes, so sweeping rho amounts to re-solving glasso
    on this single S_tau -- no need to rerun the expensive MCMC per rho.

    b_min routes near-symmetric / small-b coordinates (b = sqrt(chi*psi)) to the
    Inverse-Gamma proposal instead of the GIG rejection sampler, which stalls
    when b is small or nu is small (|lam| large). Raised above run_em_MWGP's
    default (0.05) because the random sparse true graph produces coordinates the
    GIG envelope cannot handle."""
    result = em.run_em_MWGP(
        Y,
        n_iter=100,
        rho=reference_rho,
        verbose=False,
        err=1e-3,
        run_until_convergence=False,
        mcmc_samples=400,
        mcmc_thin=1,
        mcmc_warmup=30,
        random_state=mcmc_random_state,
        proposal="gig",
        b_min=b_min,
    )
    return result["S_tau"]


def edge_confusion(Theta_hat, true_pos_mask, true_neg_mask, tol=1e-8):
    iu = np.triu_indices(Theta_hat.shape[0], k=1)
    est_edges = np.abs(Theta_hat[iu]) > tol
    tp_rate = np.sum(est_edges & true_pos_mask) / true_pos_mask.sum()
    fp_rate = np.sum(est_edges & true_neg_mask) / true_neg_mask.sum()
    return fp_rate, tp_rate


def roc_curve_over_rho(S, rho_grid, true_pos_mask, true_neg_mask):
    fp = np.empty(len(rho_grid))
    tp = np.empty(len(rho_grid))
    Theta_prev = np.diag(1.0 / np.diag(S))
    for i, rho in enumerate(rho_grid):
        try:
            _, Theta_hat = graphical_lasso(S, alpha=rho, max_iter=1000)
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
    parser.add_argument(
        "--reference_rho", type=float, default=0.0025,
        help="rho used only to seed the EM fit that produces mu_hat, eta_hat, "
             "nu_hat, and S_tau; the ROC sweep itself re-solves glasso at "
             "--num_rho values independent of this.",
    )
    args = parser.parse_args()

    p, n = args.p, args.n
    mu_true = rng.uniform(low=-1, high=1, size=p)
    eta_true = rng.uniform(low=-1, high=1, size=p)
    nu_true = rng.uniform(low=0.15, high=0.9, size=p)
    # Random sparse true precision matrix; sparsity/weights are regulated by
    # make_true_theta's defaults in simulation.py.
    Theta_true = make_true_theta(p, rng=rng)

    iu = np.triu_indices(p, k=1)
    true_pos_mask = Theta_true[iu] != 0
    true_neg_mask = ~true_pos_mask

    rho_grid = np.logspace(np.log10(args.rho_min), np.log10(args.rho_max), args.num_rho)

    fp_asym = np.empty((args.num_simulations, args.num_rho))
    tp_asym = np.empty((args.num_simulations, args.num_rho))
    fp_ggm = np.empty((args.num_simulations, args.num_rho))
    tp_ggm = np.empty((args.num_simulations, args.num_rho))
    auc_asym = np.empty(args.num_simulations)
    auc_ggm = np.empty(args.num_simulations)

    for sim in range(args.num_simulations):
        print(f"Replicate {sim + 1}/{args.num_simulations}")
        Y, _ = simulate_aat_data(n, p, mu_true, eta_true, nu_true, Theta_true, rng)

        S_tau = fit_S_tau(
            Y, args.reference_rho, mcmc_random_state=1000 + sim
        )
        fp_asym[sim], tp_asym[sim] = roc_curve_over_rho(
            S_tau, rho_grid, true_pos_mask, true_neg_mask
        )
        auc_asym[sim] = auc_from_curve(fp_asym[sim], tp_asym[sim])

        S_raw = np.cov(Y, rowvar=False) + 1e-10 * np.eye(p)
        fp_ggm[sim], tp_ggm[sim] = roc_curve_over_rho(
            S_raw, rho_grid, true_pos_mask, true_neg_mask
        )
        auc_ggm[sim] = auc_from_curve(fp_ggm[sim], tp_ggm[sim])

    fp_asym_mean, tp_asym_mean = fp_asym.mean(axis=0), tp_asym.mean(axis=0)
    fp_ggm_mean, tp_ggm_mean = fp_ggm.mean(axis=0), tp_ggm.mean(axis=0)

    print(f"\n(p={p}, n={n}) over {args.num_simulations} replicates")
    print(f"  Asymmetric model (EM_MWGP): AUC = {auc_asym.mean():.3f} "
          f"(SE {auc_asym.std(ddof=1) / np.sqrt(args.num_simulations):.3f})")
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title(f"Precision-matrix support recovery: p={p}, n={n}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    filename = (
        "results/roc_EM_MWGP" if args.filename is None
        else f"results/roc_EM_MWGP_{args.filename}"
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
    with open(f"{filename}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
