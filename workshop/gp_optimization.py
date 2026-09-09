"""Small Gaussian-process helpers for the Bayesian optimization exercise."""

import numpy as np


def _as_finite_vector(values, name):
    """Return values as a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    return array


def rbf_kernel(x_left, x_right, length_scale=0.50, signal_std=0.10):
    """Covariance between inputs under a radial-basis-function kernel."""
    x_left = _as_finite_vector(x_left, 'x_left')
    x_right = _as_finite_vector(x_right, 'x_right')
    if (not np.isfinite(length_scale) or not np.isfinite(signal_std)
            or length_scale <= 0 or signal_std <= 0):
        raise ValueError('length_scale and signal_std must be positive')
    distances = (x_left[:, None] - x_right[None, :]) / length_scale
    return signal_std**2 * np.exp(-0.5 * distances**2)


def gp_posterior(
    x_observed,
    y_observed,
    x_query,
    *,
    prior_mean=0.45,
    length_scale=0.50,
    signal_std=0.10,
    noise_std=0.015,
):
    """Return GP posterior mean, latent std, and new-run predictive std.

    The latent standard deviation describes uncertainty about the average
    validation loss. The predictive standard deviation additionally includes
    run-to-run observation noise.
    """
    x_observed = _as_finite_vector(x_observed, 'x_observed')
    y_observed = _as_finite_vector(y_observed, 'y_observed')
    x_query = _as_finite_vector(x_query, 'x_query')
    if len(x_observed) != len(y_observed):
        raise ValueError('x_observed and y_observed must have equal length')
    if not np.isfinite(prior_mean):
        raise ValueError('prior_mean must be finite')
    if (not np.isfinite(length_scale) or not np.isfinite(signal_std)
            or length_scale <= 0 or signal_std <= 0):
        raise ValueError('length_scale and signal_std must be positive')
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError('noise_std must be finite and nonnegative')

    prior_std = np.full(len(x_query), float(signal_std))
    if len(x_observed) == 0:
        mean = np.full(len(x_query), float(prior_mean))
        predictive_std = np.sqrt(prior_std**2 + noise_std**2)
        return mean, prior_std, predictive_std

    observed_covariance = rbf_kernel(
        x_observed, x_observed, length_scale, signal_std
    )
    # The tiny jitter protects the Cholesky decomposition from round-off error.
    observed_covariance += np.eye(len(x_observed)) * (noise_std**2 + 1e-12)
    cross_covariance = rbf_kernel(
        x_observed, x_query, length_scale, signal_std
    )

    cholesky = np.linalg.cholesky(observed_covariance)
    centered_y = y_observed - prior_mean
    weights = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, centered_y))
    posterior_mean = prior_mean + cross_covariance.T @ weights

    projected = np.linalg.solve(cholesky, cross_covariance)
    latent_variance = signal_std**2 - np.sum(projected**2, axis=0)
    latent_variance = np.maximum(latent_variance, 0.0)
    latent_std = np.sqrt(latent_variance)
    predictive_std = np.sqrt(latent_variance + noise_std**2)
    return posterior_mean, latent_std, predictive_std


def lower_confidence_bound(mean, latent_std, exploration=1.25):
    """Score candidates for minimization; smaller values are tried first."""
    mean = _as_finite_vector(mean, 'mean')
    latent_std = _as_finite_vector(latent_std, 'latent_std')
    if len(mean) != len(latent_std):
        raise ValueError('mean and latent_std must have equal length')
    if not np.isfinite(exploration) or exploration < 0:
        raise ValueError('exploration must be finite and nonnegative')
    return mean - exploration * latent_std


def suggest_next_index(mean, latent_std, observed_indices, exploration=1.25):
    """Return the unevaluated candidate with the smallest acquisition score."""
    score = lower_confidence_bound(mean, latent_std, exploration).copy()
    observed_indices = np.asarray(observed_indices, dtype=int)
    if observed_indices.ndim != 1:
        raise ValueError('observed_indices must be one-dimensional')
    if np.any((observed_indices < 0) | (observed_indices >= len(score))):
        raise ValueError('observed_indices contains an invalid candidate index')
    if len(np.unique(observed_indices)) != len(observed_indices):
        raise ValueError('observed_indices cannot contain duplicates')
    if len(observed_indices) == len(score):
        raise ValueError('all candidates have already been evaluated')
    score[observed_indices] = np.inf
    return int(np.argmin(score))


def run_bayesian_optimization(
    candidate_x,
    candidate_y,
    initial_indices,
    budget,
    *,
    prior_mean=0.45,
    length_scale=0.50,
    signal_std=0.10,
    noise_std=0.015,
    exploration=1.25,
):
    """Replay sequential GP-guided choices against a cached result table."""
    candidate_x = _as_finite_vector(candidate_x, 'candidate_x')
    candidate_y = _as_finite_vector(candidate_y, 'candidate_y')
    if len(candidate_x) != len(candidate_y):
        raise ValueError('candidate_x and candidate_y must have equal length')
    if not isinstance(budget, (int, np.integer)):
        raise ValueError('budget must be a whole number')
    selected = [int(index) for index in initial_indices]
    if len(set(selected)) != len(selected):
        raise ValueError('initial_indices cannot contain duplicates')
    if any(index < 0 or index >= len(candidate_x) for index in selected):
        raise ValueError('initial_indices contains an invalid candidate index')
    if not len(selected) <= budget <= len(candidate_x):
        raise ValueError('budget must cover the initial runs and available candidates')

    while len(selected) < budget:
        mean, latent_std, _ = gp_posterior(
            candidate_x[selected],
            candidate_y[selected],
            candidate_x,
            prior_mean=prior_mean,
            length_scale=length_scale,
            signal_std=signal_std,
            noise_std=noise_std,
        )
        selected.append(
            suggest_next_index(mean, latent_std, selected, exploration)
        )
    return selected
