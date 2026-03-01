# modules/monte_carlo.py
# Monte Carlo Binary Contract Engine + Importance Sampling for tail-risk contracts

import numpy as np
from scipy.stats import norm


def simulate_binary_contract(S0, K, mu, sigma, T, N_paths=100_000):
    """
    Monte Carlo simulation for a binary contract via GBM.
    S0: current asset price, K: strike, mu: annual drift,
    sigma: annual vol, T: time to expiry in years.
    Returns probability estimate + 95% CI.
    """
    Z = np.random.standard_normal(N_paths)
    S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = (S_T > K).astype(float)
    p_hat = payoffs.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / N_paths)
    return {
        "probability": round(float(p_hat), 6),
        "std_error": round(float(se), 6),
        "ci_95": (round(float(p_hat - 1.96 * se), 6), round(float(p_hat + 1.96 * se), 6)),
        "N_paths": N_paths,
        "S0": S0,
        "K": K,
        "sigma": sigma,
        "T_days": round(T * 365, 1),
    }


def rare_event_importance_sampling(S0, K_crash_pct, sigma, T, N_paths=100_000):
    """
    Importance sampling (exponential tilting) for extreme tail-risk contracts.
    K_crash_pct: fractional drop threshold (e.g. 0.20 = 20% crash).
    Returns IS estimate vs crude MC with variance reduction factor.
    """
    K = S0 * (1 - K_crash_pct)
    mu_original = -0.5 * sigma**2
    log_threshold = np.log(K / S0)
    mu_tilt = log_threshold / T

    Z = np.random.standard_normal(N_paths)
    log_returns_tilted = mu_tilt * T + sigma * np.sqrt(T) * Z
    S_T_tilted = S0 * np.exp(log_returns_tilted)

    log_LR = (
        -0.5 * ((log_returns_tilted - mu_original * T) / (sigma * np.sqrt(T))) ** 2
        + 0.5 * ((log_returns_tilted - mu_tilt * T) / (sigma * np.sqrt(T))) ** 2
    )
    LR = np.exp(log_LR)
    payoffs = (S_T_tilted < K).astype(float)
    is_estimates = payoffs * LR
    p_IS = float(is_estimates.mean())
    se_IS = float(is_estimates.std() / np.sqrt(N_paths))

    Z_crude = np.random.standard_normal(N_paths)
    S_T_crude = S0 * np.exp(mu_original * T + sigma * np.sqrt(T) * Z_crude)
    p_crude = float((S_T_crude < K).mean())
    se_crude = float(np.sqrt(p_crude * (1 - p_crude) / N_paths)) if p_crude > 0 else float("inf")

    variance_reduction = float((se_crude / se_IS) ** 2) if se_IS > 0 else float("inf")

    return {
        "p_IS": round(p_IS, 8),
        "se_IS": round(se_IS, 8),
        "p_crude": round(p_crude, 8),
        "se_crude": round(se_crude, 8),
        "variance_reduction_factor": round(variance_reduction, 1),
        "crash_threshold_pct": K_crash_pct,
        "T_days": round(T * 365, 1),
    }


def stratified_binary_mc(S0, K, sigma, T, J=10, N_total=100_000):
    """
    Stratified Monte Carlo for binary contract pricing.
    Strata defined by quantiles of terminal price distribution.
    """
    n_per_stratum = N_total // J
    estimates = []
    for j in range(J):
        U = np.random.uniform(j / J, (j + 1) / J, n_per_stratum)
        Z = norm.ppf(U)
        S_T = S0 * np.exp((-0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        estimates.append((S_T > K).mean())
    p_stratified = float(np.mean(estimates))
    se_stratified = float(np.std(estimates) / np.sqrt(J))
    return {
        "probability": round(p_stratified, 6),
        "std_error": round(se_stratified, 6),
        "strata": J,
        "N_total": N_total,
    }
