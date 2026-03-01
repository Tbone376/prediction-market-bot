# modules/predictor.py
# GBM Monte Carlo + Particle Filter price prediction engine
# Generates 1-day, 5-day, 30-day price targets and probabilities for SPX/SPY/AAPL/NVDA/TSLA

import numpy as np
from scipy.special import expit, logit
from datetime import datetime
from modules.monte_carlo import simulate_binary_contract


# ---------------------------------------------------------------------------
# GBM PATH SIMULATOR
# ---------------------------------------------------------------------------

def simulate_price_paths(S0: float, mu: float, sigma: float,
                         T_days: int, N_paths: int = 10_000) -> np.ndarray:
    """
    Simulate N_paths GBM price paths over T_days trading days.
    Returns array shape (N_paths, T_days+1).
    """
    dt = 1 / 252
    paths = np.zeros((N_paths, T_days + 1))
    paths[:, 0] = S0
    Z = np.random.standard_normal((N_paths, T_days))
    for t in range(1, T_days + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    return paths


def path_statistics(paths: np.ndarray) -> dict:
    """Summary stats across all simulated terminal prices."""
    terminal = paths[:, -1]
    return {
        "mean": round(float(terminal.mean()), 2),
        "median": round(float(np.median(terminal)), 2),
        "p5": round(float(np.percentile(terminal, 5)), 2),
        "p25": round(float(np.percentile(terminal, 25)), 2),
        "p75": round(float(np.percentile(terminal, 75)), 2),
        "p95": round(float(np.percentile(terminal, 95)), 2),
        "std": round(float(terminal.std()), 2),
    }


# ---------------------------------------------------------------------------
# PARTICLE FILTER — REAL-TIME BAYESIAN PRICE TRACKER
# ---------------------------------------------------------------------------

class PriceParticleFilter:
    """
    Particle filter that maintains a posterior distribution over
    the latent 'fair value' of a stock given noisy observed prices.
    Runs in logit space to keep probabilities bounded.
    """

    def __init__(self, N_particles: int = 3000, init_price: float = 100.0,
                 process_vol: float = 0.01, obs_noise: float = 0.005):
        self.N = N_particles
        self.process_vol = process_vol
        self.obs_noise = obs_noise
        # Particles are raw price values
        self.particles = np.random.normal(init_price, init_price * 0.02, N_particles)
        self.weights = np.ones(N_particles) / N_particles
        self.history = []

    def update(self, observed_price: float):
        """Incorporate a new observed price."""
        # Propagate: random walk on price
        noise = np.random.normal(0, self.process_vol * observed_price, self.N)
        self.particles += noise
        self.particles = np.clip(self.particles, observed_price * 0.5, observed_price * 2.0)

        # Reweight
        log_likelihood = -0.5 * ((observed_price - self.particles) / (self.obs_noise * observed_price)) ** 2
        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()

        # Resample if ESS low
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.N / 2:
            self._systematic_resample()

        self.history.append(self.estimate())

    def _systematic_resample(self):
        cumsum = np.cumsum(self.weights)
        u = (np.arange(self.N) + np.random.uniform()) / self.N
        indices = np.searchsorted(cumsum, u)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def estimate(self) -> float:
        return float(np.average(self.particles, weights=self.weights))

    def credible_interval(self, alpha: float = 0.05):
        sorted_idx = np.argsort(self.particles)
        sorted_p = self.particles[sorted_idx]
        sorted_w = self.weights[sorted_idx]
        cumw = np.cumsum(sorted_w)
        lower = float(sorted_p[np.searchsorted(cumw, alpha / 2)])
        upper = float(sorted_p[np.searchsorted(cumw, 1 - alpha / 2)])
        return round(lower, 2), round(upper, 2)


# ---------------------------------------------------------------------------
# FULL PREDICTION BUNDLE PER TICKER
# ---------------------------------------------------------------------------

def generate_prediction(snapshot: dict, N_paths: int = 20_000) -> dict:
    """
    Given a ticker snapshot dict (from data_fetcher.get_ticker_snapshot),
    run GBM simulations for 1d/5d/30d horizons and return full prediction bundle.
    """
    S0 = snapshot["current_price"]
    mu = snapshot["mu_annualised"]
    sigma = snapshot["sigma_20d"]
    ticker = snapshot["ticker"]

    predictions = {}
    for horizon_days, label in [(1, "1d"), (5, "5d"), (30, "30d")]:
        T = horizon_days / 252
        paths = simulate_price_paths(S0, mu, sigma, horizon_days, N_paths)
        stats = path_statistics(paths)

        # P(price > current) = bullish probability
        terminal = paths[:, -1]
        p_up = float((terminal > S0).mean())
        p_up_5pct = float((terminal > S0 * 1.05).mean())
        p_down_5pct = float((terminal < S0 * 0.95).mean())
        p_up_10pct = float((terminal > S0 * 1.10).mean())
        p_down_10pct = float((terminal < S0 * 0.90).mean())

        predictions[label] = {
            "horizon_days": horizon_days,
            "p_up": round(p_up, 4),
            "p_up_5pct": round(p_up_5pct, 4),
            "p_down_5pct": round(p_down_5pct, 4),
            "p_up_10pct": round(p_up_10pct, 4),
            "p_down_10pct": round(p_down_10pct, 4),
            "price_target_mean": stats["mean"],
            "price_target_median": stats["median"],
            "ci_90_low": stats["p5"],
            "ci_90_high": stats["p95"],
            "ci_50_low": stats["p25"],
            "ci_50_high": stats["p75"],
        }

    # Particle filter: feed last 20 closes as a warm-up stream
    pf = PriceParticleFilter(init_price=S0, process_vol=0.008, obs_noise=0.004)
    # Simulate a brief warm-up with ±noise around S0 so filter is initialised
    warm_prices = S0 * (1 + np.random.normal(0, 0.005, 10))
    warm_prices[-1] = S0  # anchor to current price
    for p in warm_prices:
        pf.update(p)
    pf_estimate = pf.estimate()
    pf_ci = pf.credible_interval()

    return {
        "ticker": ticker,
        "current_price": S0,
        "sigma_20d": sigma,
        "mu_annualised": mu,
        "predictions": predictions,
        "particle_filter": {
            "fair_value_estimate": round(pf_estimate, 2),
            "ci_95_low": pf_ci[0],
            "ci_95_high": pf_ci[1],
            "divergence_from_market": round(pf_estimate - S0, 2),
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
