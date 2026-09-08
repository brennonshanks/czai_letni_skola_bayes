"""One slot machine: Beta beliefs, NUTS verification, and predictive play."""
from functools import lru_cache

import numpy as np

# Illustrative beliefs for our fictional payout, not measured casino odds.
PRIOR_CHOICES = {
    'Broad': (1., 1.),
    'Mild house edge': (2., 3.),
    'Las Vegas': (38., 62.),
}


@lru_cache(maxsize=1)
def _compiled_slot_model():
    import pymc as pm
    import nutpie

    with pm.Model() as model:
        prior_a = pm.Data('prior_a', 2.)
        prior_b = pm.Data('prior_b', 3.)
        observed_plays = pm.Data('observed_plays', np.int64(1))
        observed_wins = pm.Data('observed_wins', np.int64(0))
        p = pm.Beta('p', alpha=prior_a, beta=prior_b)
        pm.Binomial('wins', n=observed_plays, p=p, observed=observed_wins)
    return model, nutpie.compile_pymc_model(model, backend='numba')


def fit_slot(wins, plays, a=2., b=3., draws=750, tune=750, chains=2, seed=12):
    """Sample p with cached Numba compilation; its exact posterior is Beta(a+wins, b+plays-wins)."""
    import nutpie

    if a <= 0 or b <= 0 or not 0 <= wins <= plays:
        raise ValueError('Positive Beta parameters and 0 <= wins <= plays required')
    model, compiled = _compiled_slot_model()
    data = {
        'prior_a': float(a),
        'prior_b': float(b),
        'observed_plays': np.int64(plays),
        'observed_wins': np.int64(wins),
    }
    for name, value in data.items():
        model.set_data(name, value)
    trace = nutpie.sample(
        compiled.with_data(**data), draws=draws, tune=tune, chains=chains,
        cores=1, target_accept=.95, seed=seed, save_warmup=False,
        progress_bar=False,
    )
    trace.attrs.update(plays=int(plays), wins=int(wins), prior_a=float(a), prior_b=float(b))
    return model, trace


def simulate_profit(p_samples, bets=100, payout=2.4, seed=21):
    """Draw a future win count for each possible p; payout includes the stake."""
    samples = np.asarray(p_samples, float)
    if not np.isfinite(samples).all() or np.any((samples < 0) | (samples > 1)):
        raise ValueError('Win probabilities must lie between zero and one')
    return payout * np.random.default_rng(seed).binomial(bets, samples) - bets
