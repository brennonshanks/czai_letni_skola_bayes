"""One slot machine: Beta beliefs, NUTS verification, and predictive play."""
import numpy as np

# Illustrative beliefs for our fictional payout, not measured casino odds.
PRIOR_CHOICES = {
    'Broad': (1., 1.),
    'Mild house edge': (2., 3.),
    'Las Vegas': (38., 62.),
}


def fit_slot(wins, plays, a=2., b=3., draws=750, tune=750, chains=2, seed=12):
    """Sample one p; exact posterior is Beta(a+wins, b+plays-wins)."""
    import pymc as pm
    if a <= 0 or b <= 0 or not 0 <= wins <= plays:
        raise ValueError('Positive Beta parameters and 0 <= wins <= plays required')
    with pm.Model() as model:
        p = pm.Beta('p', alpha=a, beta=b)
        pm.Binomial('wins', n=plays, p=p, observed=wins)
        prior = pm.sample_prior_predictive(samples=300, random_seed=seed)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=.95, random_seed=seed)
        pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=seed+1)
    trace.attrs.update(plays=int(plays), wins=int(wins), prior_a=float(a), prior_b=float(b))
    return model, trace, prior


def simulate_profit(p_samples, bets=100, payout=2.4, seed=21):
    """Draw a future win count for each possible p; payout includes the stake."""
    samples = np.asarray(p_samples, float)
    if not np.isfinite(samples).all() or np.any((samples < 0) | (samples > 1)):
        raise ValueError('Win probabilities must lie between zero and one')
    return payout * np.random.default_rng(seed).binomial(bets, samples) - bets
