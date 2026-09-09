from pathlib import Path

import numpy as np
import pytest

from scripts.generate_gp_tuning_results import build_tables, true_noise_std
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


def _cached_table():
    data_path = (Path(__file__).parents[1] / 'data' / 'public' /
                 'gp_tuning_results.csv')
    return np.genfromtxt(data_path, delimiter=',', names=True)


def _truth_table():
    data_path = (Path(__file__).parents[1] / 'data' / 'public' /
                 'gp_tuning_truth.csv')
    return np.genfromtxt(data_path, delimiter=',', names=True)


def test_run_table_keeps_truth_out_of_the_search_data():
    cached = _cached_table()
    truth = _truth_table()
    x = cached['log10_learning_rate']
    average = truth['true_average_loss']
    noise = true_noise_std(x)

    assert len(x) == 33
    assert 'true_average_loss' not in cached.dtype.names
    assert 'true_noise_std' not in cached.dtype.names
    assert np.allclose(x, truth['log10_learning_rate'])
    assert np.allclose(noise, truth['true_noise_std'], atol=1e-5)
    # Every cached run must be a plausible draw around its own latent value,
    # which is what makes the notebook's noise story honest.
    assert np.all(noise > 0)
    assert np.all(np.abs(cached['validation_loss'] - average) < 4 * noise)
    # The optimum is interior, so a search can approach it from either side.
    assert 0 < int(np.argmin(average)) < len(x) - 1


def test_true_noise_grows_with_the_learning_rate():
    cached = _cached_table()
    noise = true_noise_std(cached['log10_learning_rate'])

    # Monotone up to the rounding of the committed table, which can tie
    # neighbouring values at the flat ends of the ramp.
    assert np.all(np.diff(noise) >= 0)
    assert noise[-1] > noise[0]
    # The generator should retain the intended heteroscedastic training noise.
    assert noise[-1] / noise[0] > 3
    assert noise[0] < 0.015 < noise[-1]


def test_committed_tables_match_the_generator():
    generated_runs, generated_truth = build_tables()
    cached = _cached_table()
    truth = _truth_table()

    assert tuple(generated_runs.columns) == cached.dtype.names
    assert tuple(generated_truth.columns) == truth.dtype.names
    for column in cached.dtype.names:
        assert np.allclose(generated_runs[column], cached[column], atol=1e-5)
    for column in truth.dtype.names:
        assert np.allclose(generated_truth[column], truth[column], atol=1e-5)


def test_default_replay_reaches_the_true_optimum_and_beats_typical_random():
    cached = _cached_table()
    truth = _truth_table()
    x = cached['log10_learning_rate']
    y = cached['validation_loss']
    average = truth['true_average_loss']
    true_best = int(np.argmin(average))

    selected = run_bayesian_optimization(x, y, [2, 12, 30], budget=8)
    final_mean, _, _ = gp_posterior(x[selected], y[selected], x)
    recommended = int(np.argmin(final_mean))
    regret = average[recommended] - average[true_best]

    # The workshop defaults must actually evaluate the best learning rate,
    # while the GP's final recommendation should remain close to it.
    assert true_best in selected
    assert regret < 0.01

    rng = np.random.default_rng(0)
    available = np.setdiff1d(np.arange(len(x)), [2, 12, 30])
    random_regrets = []
    for _ in range(2000):
        extra = rng.choice(available, 5, replace=False)
        indices = np.r_[[2, 12, 30], extra]
        random_mean, _, _ = gp_posterior(x[indices], y[indices], x)
        pick = int(np.argmin(random_mean))
        random_regrets.append(average[pick] - average[true_best])
    assert regret < np.median(random_regrets)


def test_smoothness_comparison_has_distinct_instructive_outcomes():
    cached = _cached_table()
    truth = _truth_table()
    x = cached['log10_learning_rate']
    y = cached['validation_loss']
    average = truth['true_average_loss']
    true_best = int(np.argmin(average))
    recommendations = {}
    paths = []

    for length_scale in [0.10, 0.50, 1.50]:
        selected = run_bayesian_optimization(
            x, y, [2, 12, 30], budget=8, length_scale=length_scale
        )
        final_mean, _, _ = gp_posterior(
            x[selected], y[selected], x, length_scale=length_scale
        )
        recommendations[length_scale] = int(np.argmin(final_mean))
        paths.append(tuple(selected))

    # The workshop comparison should illustrate both extremes changing the
    # search, while the moderate assumption handles this constructed task.
    assert len(set(paths)) == 3
    assert recommendations[0.50] == true_best
    assert average[recommendations[0.50]] < average[recommendations[0.10]]
    assert average[recommendations[0.50]] < average[recommendations[1.50]]


def test_lowest_observed_run_can_miss_the_best_average_loss():
    cached = _cached_table()
    truth = _truth_table()
    # The teaching point of the reveal: selecting on noisy observations does
    # not identify the best average loss, so the table must show that gap.
    assert int(np.argmin(cached['validation_loss'])) != int(
        np.argmin(truth['true_average_loss'])
    )
