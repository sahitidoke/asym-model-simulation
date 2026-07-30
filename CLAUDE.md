# CLAUDE.md

REU project: asymmetric alternative-t graphical model.
Y_j = mu_j + eta_j*nu_j*tau_j + sqrt(tau_j)*X_j,
tau_j ~ Inv-Gamma(2/nu_j, 2/nu_j) independent across j, X ~ N_p(0, Psi), Theta = Psi^-1.

CRITICAL: theta_jk = 0 does NOT imply Y_j ⊥ Y_k | rest in this model.
Support of Theta and the conditional independence graph are different targets.
Never conflate them in code, variable names, or plots.

rho (glasso penalty) is the only externally chosen tuning parameter.

Papers are in docs/. Read them before answering anything about the math:
- docs/reu.pdf — our writeup (model, variational EM, GCM test)
- docs/sggm.pdf — Sheng, Li & Solea, skewed Gaussian GM
- docs/tlasso.pdf — Finegold & Drton, robust t graphical models