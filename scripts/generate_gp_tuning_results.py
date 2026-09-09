"""Regenerate the cached training-service results for the GP exercise.

The Bayesian optimization notebook needs a fixed table of "training runs" that
behaves like a real tuning problem: a smooth latent performance curve plus
run-to-run noise. A separate reveal table holds the latent curve and the
simulator's noise level so the search table itself does not expose either.

Both properties are chosen here and generated from a fixed seed:

* ``LATENT`` defines the true average validation loss as a function of
  ``log10(learning rate)``. It has a steep underfitting branch at tiny learning
  rates, a smooth global optimum near ``log10(lr) = -3``, a broad shallow plateau
  to its right that can trap an optimizer that assumes little smoothness, and
  an instability rise at large learning rates.
* ``NOISE`` defines the true run-to-run standard deviation. It is *not*
  constant: unstable training at large learning rates disagrees more between
  repeats. This is a fixed property of the simulated service and is independent
  of whatever ``NOISE_STD`` a workshop team assumes.

Because the optimum is smooth enough for the notebook's default length scale to
resolve, the latent curve is nearly flat across the grid points around it. The
lowest *observed* run therefore need not sit exactly at the lowest *average*
loss, which is the point the notebook's reveal makes.

Run from the repository root:

    python scripts/generate_gp_tuning_results.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'data' / 'public'

SEED = 11
GRID_POINTS = 33
LOG10_RANGE = (-5.0, -1.0)

# Latent average validation loss. See module docstring for the intended shape.
LATENT = dict(
    floor=0.1037,          # asymptotic loss once the learning rate is usable
    underfit_scale=0.3937,  # extra loss at the smallest learning rate
    underfit_rate=0.4719,   # how fast that penalty decays as the rate grows
    unstable_scale=0.1835,  # extra loss at the largest learning rate
    unstable_rate=1.1300,   # how fast instability takes over
    optimum_depth=0.02695,  # depth of the good basin
    optimum_width=0.17259,  # width of the good basin, in log10 units
    plateau_height=0.03516,  # height of the shallow decoy plateau
    plateau_width=0.46193,  # width of that plateau, in log10 units
    optimum_center=-3.00,
    plateau_center=-2.2287,
)

# True run-to-run standard deviation: a logistic ramp from quiet, repeatable
# runs at small learning rates to visibly scattered runs at large ones.
NOISE = dict(low=0.004, high=0.023, center=-1.90, width=0.42)


def latent_average_loss(log10_learning_rate, **kwargs):
    """True expected validation loss at a log10 learning rate."""
    settings = {**LATENT, **kwargs}
    x = np.asarray(log10_learning_rate, dtype=float)
    low_end, high_end = LOG10_RANGE
    underfitting = settings['underfit_scale'] * np.exp(
        -settings['underfit_rate'] * (x - low_end)
    )
    instability = settings['unstable_scale'] * np.exp(
        settings['unstable_rate'] * (x - high_end)
    )
    optimum = settings['optimum_depth'] * np.exp(
        -0.5 * ((x - settings['optimum_center']) / settings['optimum_width'])**2
    )
    plateau = settings['plateau_height'] * np.exp(
        -0.5 * ((x - settings['plateau_center']) / settings['plateau_width'])**2
    )
    return settings['floor'] + underfitting + instability - optimum + plateau


def true_noise_std(log10_learning_rate, **kwargs):
    """True run-to-run standard deviation at a log10 learning rate."""
    settings = {**NOISE, **kwargs}
    x = np.asarray(log10_learning_rate, dtype=float)
    ramp = 1.0 / (1.0 + np.exp(-(x - settings['center']) / settings['width']))
    return settings['low'] + (settings['high'] - settings['low']) * ramp


def build_tables(seed=SEED):
    """Return the learner-facing run table and the end-of-search truth table."""
    x = np.round(np.linspace(*LOG10_RANGE, GRID_POINTS), 3)
    average = latent_average_loss(x)
    noise = true_noise_std(x)

    rng = np.random.default_rng(seed)
    observed = average + noise * rng.normal(size=len(x))

    runs = pd.DataFrame({
        'run_id': np.arange(1, len(x) + 1),
        'log10_learning_rate': x,
        'learning_rate': 10.0**x,
        'validation_loss': observed,
    })
    truth = pd.DataFrame({
        'run_id': np.arange(1, len(x) + 1),
        'log10_learning_rate': x,
        'learning_rate': 10.0**x,
        'true_average_loss': average,
        'true_noise_std': noise,
    })
    return runs, truth


def main():
    runs, truth = build_tables()
    rounding = {'learning_rate': 8, 'validation_loss': 5,
                'true_average_loss': 5, 'true_noise_std': 5}
    runs.round(rounding).to_csv(
        PUBLIC / 'gp_tuning_results.csv', index=False
    )
    truth.round(rounding).to_csv(
        PUBLIC / 'gp_tuning_truth.csv', index=False
    )

    observed = runs.validation_loss.to_numpy()
    average = truth.true_average_loss.to_numpy()
    x = runs.log10_learning_rate.to_numpy()
    noise = true_noise_std(x)
    print(f'seed={SEED}  grid={GRID_POINTS}')
    print(f'true optimum        : log10(lr)={x[average.argmin()]:.3f} '
          f'average loss={average.min():.5f}')
    print(f'best observed run   : log10(lr)={x[observed.argmin()]:.3f} '
          f'observed loss={observed.min():.5f}')
    print(f'noise std range     : {noise.min():.5f} .. {noise.max():.5f}')
    print(f'observed loss range : {observed.min():.5f} .. {observed.max():.5f}')


if __name__ == '__main__':
    main()
