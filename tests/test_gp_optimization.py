from pathlib import Path

import numpy as np
import pytest

from workshop.gp_optimization import (
    gp_posterior,
    lower_confidence_bound,
    run_bayesian_optimization,
    suggest_next_index,
)


def test_gp_posterior_separates_latent_and_future_run_uncertainty():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([0.5, 0.2, 0.4])
    mean, latent_std, predictive_std = gp_posterior(
        x, y, x, prior_mean=0.4, noise_std=0.03
    )

    assert mean.shape == x.shape
    assert np.all(latent_std >= 0)
    assert np.all(predictive_std > latent_std)
    assert mean[1] < mean[0]


def test_suggestion_skips_observed_candidates():
    mean = np.array([0.3, 0.2, 0.25])
    uncertainty = np.array([0.1, 0.1, 0.1])

    assert suggest_next_index(mean, uncertainty, [1]) == 2


def test_replay_finds_cached_best_candidate_with_workshop_defaults():
    data_path = (Path(__file__).parents[1] / 'data' / 'public' /
                 'gp_tuning_results.csv')
    cached = np.genfromtxt(data_path, delimiter=',', names=True)
    x = cached['log10_learning_rate']
    y = cached['validation_loss']

    assert len(x) == 33
    assert np.all(np.diff(x) > 0)
    assert np.allclose(cached['learning_rate'], 10**x, rtol=1e-3)

    selected = run_bayesian_optimization(x, y, [2, 12, 30], budget=8)

    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert int(np.argmin(y)) in selected


def test_invalid_acquisition_settings_are_rejected():
    with pytest.raises(ValueError, match='exploration'):
        lower_confidence_bound([0.2], [0.1], exploration=-1)
