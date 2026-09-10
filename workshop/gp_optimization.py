"""Small Gaussian-process helpers for the Bayesian optimization exercise."""

import numpy as np
from scipy.optimize import minimize


DEFAULT_HYPERPARAMETER_BOUNDS = {
    'prior_mean': (0.20, 0.65),
    'length_scale': (0.05, 2.00),
    'signal_std': (0.01, 0.30),
    'noise_std': (0.002, 0.08),
}

_DEFAULT_GP_SETTINGS = {
    'prior_mean': 0.45,
    'length_scale': 0.50,
    'signal_std': 0.10,
    'noise_std': 0.015,
}
_HYPERPARAMETER_NAMES = tuple(DEFAULT_HYPERPARAMETER_BOUNDS)


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


def gp_log_marginal_likelihood(
    x_observed,
    y_observed,
    *,
    prior_mean=0.45,
    length_scale=0.50,
    signal_std=0.10,
    noise_std=0.015,
):
    """Return the log probability of the observations under a GP model."""
    x_observed = _as_finite_vector(x_observed, 'x_observed')
    y_observed = _as_finite_vector(y_observed, 'y_observed')
    if len(x_observed) != len(y_observed):
        raise ValueError('x_observed and y_observed must have equal length')
    if len(x_observed) == 0:
        raise ValueError('at least one observation is required')
    if not np.isfinite(prior_mean):
        raise ValueError('prior_mean must be finite')
    if (not np.isfinite(length_scale) or not np.isfinite(signal_std)
            or length_scale <= 0 or signal_std <= 0):
        raise ValueError('length_scale and signal_std must be positive')
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError('noise_std must be finite and nonnegative')

    covariance = rbf_kernel(
        x_observed, x_observed, length_scale, signal_std
    )
    covariance += np.eye(len(x_observed)) * (noise_std**2 + 1e-12)
    cholesky = np.linalg.cholesky(covariance)
    centered_y = y_observed - prior_mean
    weights = np.linalg.solve(
        cholesky.T, np.linalg.solve(cholesky, centered_y)
    )
    return float(
        -0.5 * centered_y @ weights
        - np.log(np.diag(cholesky)).sum()
        - 0.5 * len(x_observed) * np.log(2 * np.pi)
    )


def _validated_hyperparameter_bounds(bounds):
    """Return a validated copy of the four hyperparameter bounds."""
    bounds = DEFAULT_HYPERPARAMETER_BOUNDS if bounds is None else bounds
    if set(bounds) != set(_HYPERPARAMETER_NAMES):
        names = ', '.join(_HYPERPARAMETER_NAMES)
        raise ValueError(f'bounds must contain exactly: {names}')

    validated = {}
    for name in _HYPERPARAMETER_NAMES:
        pair = np.asarray(bounds[name], dtype=float)
        if pair.shape != (2,) or not np.isfinite(pair).all():
            raise ValueError(f'{name} bounds must be two finite values')
        lower, upper = pair
        if lower >= upper:
            raise ValueError(f'{name} lower bound must be less than upper bound')
        if name != 'prior_mean' and lower <= 0:
            raise ValueError(f'{name} bounds must be positive')
        validated[name] = (float(lower), float(upper))
    return validated


def fit_gp_hyperparameters(
    x_observed,
    y_observed,
    *,
    initial_settings=None,
    bounds=None,
):
    """Maximize GP log marginal likelihood within explicit bounds.

    Positive parameters are optimized on a log scale. Several deterministic
    starting points make the tiny workshop fit less dependent on one local
    optimum. Returns the fitted settings and their log marginal likelihood.
    """
    x_observed = _as_finite_vector(x_observed, 'x_observed')
    y_observed = _as_finite_vector(y_observed, 'y_observed')
    if len(x_observed) != len(y_observed):
        raise ValueError('x_observed and y_observed must have equal length')
    if len(x_observed) == 0:
        raise ValueError('at least one observation is required')

    bounds = _validated_hyperparameter_bounds(bounds)
    settings = dict(_DEFAULT_GP_SETTINGS)
    if initial_settings is not None:
        unexpected = set(initial_settings) - set(_HYPERPARAMETER_NAMES)
        if unexpected:
            names = ', '.join(sorted(unexpected))
            raise ValueError(f'unknown initial setting(s): {names}')
        settings.update(initial_settings)

    for name, value in settings.items():
        lower, upper = bounds[name]
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'initial {name} must lie within its bounds')

    def pack(values):
        return np.array([
            values['prior_mean'],
            np.log(values['length_scale']),
            np.log(values['signal_std']),
            np.log(values['noise_std']),
        ])

    def unpack(values):
        return {
            'prior_mean': float(values[0]),
            'length_scale': float(np.exp(values[1])),
            'signal_std': float(np.exp(values[2])),
            'noise_std': float(np.exp(values[3])),
        }

    transformed_bounds = [bounds['prior_mean']]
    transformed_bounds.extend(
        tuple(np.log(bounds[name])) for name in _HYPERPARAMETER_NAMES[1:]
    )

    empirical_std = max(float(np.std(y_observed)), bounds['signal_std'][0])
    base = {
        'prior_mean': float(np.clip(
            np.mean(y_observed), *bounds['prior_mean']
        )),
        'length_scale': float(np.sqrt(np.prod(bounds['length_scale']))),
        'signal_std': float(np.clip(
            empirical_std, *bounds['signal_std']
        )),
        'noise_std': float(np.clip(
            0.15 * empirical_std, *bounds['noise_std']
        )),
    }
    starts = [settings, base]
    log_length_limits = np.log(bounds['length_scale'])
    for fraction in (0.15, 0.50, 0.85):
        start = dict(base)
        start['length_scale'] = float(np.exp(
            log_length_limits[0]
            + fraction * (log_length_limits[1] - log_length_limits[0])
        ))
        starts.append(start)

    def objective(transformed):
        try:
            return -gp_log_marginal_likelihood(
                x_observed, y_observed, **unpack(transformed)
            )
        except np.linalg.LinAlgError:
            return np.inf

    fits = [
        minimize(
            objective,
            pack(start),
            method='L-BFGS-B',
            bounds=transformed_bounds,
        )
        for start in starts
    ]
    finite_fits = [fit for fit in fits if np.isfinite(fit.fun)]
    if not finite_fits:
        raise RuntimeError('marginal-likelihood optimization failed')
    best_fit = min(finite_fits, key=lambda fit: fit.fun)
    fitted_settings = unpack(best_fit.x)
    return fitted_settings, float(-best_fit.fun)


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


def run_bayesian_optimization_with_fitted_hyperparameters(
    candidate_x,
    candidate_y,
    initial_indices,
    budget,
    *,
    initial_settings=None,
    bounds=None,
    exploration=1.25,
):
    """Run BO while refitting GP hyperparameters before every decision.

    The final refit after the budget is spent is included so the returned
    history also contains the settings for a final GP recommendation.
    """
    candidate_x = _as_finite_vector(candidate_x, 'candidate_x')
    candidate_y = _as_finite_vector(candidate_y, 'candidate_y')
    if len(candidate_x) != len(candidate_y):
        raise ValueError('candidate_x and candidate_y must have equal length')
    if not isinstance(budget, (int, np.integer)):
        raise ValueError('budget must be a whole number')
    if not np.isfinite(exploration) or exploration < 0:
        raise ValueError('exploration must be finite and nonnegative')
    selected = [int(index) for index in initial_indices]
    if len(set(selected)) != len(selected):
        raise ValueError('initial_indices cannot contain duplicates')
    if any(index < 0 or index >= len(candidate_x) for index in selected):
        raise ValueError('initial_indices contains an invalid candidate index')
    if not len(selected) <= budget <= len(candidate_x):
        raise ValueError('budget must cover the initial runs and available candidates')

    active_settings = initial_settings
    history = []
    while True:
        active_settings, log_marginal_likelihood = fit_gp_hyperparameters(
            candidate_x[selected],
            candidate_y[selected],
            initial_settings=active_settings,
            bounds=bounds,
        )
        step = {
            'observations': len(selected),
            **active_settings,
            'log_marginal_likelihood': log_marginal_likelihood,
            'next_index': None,
        }
        history.append(step)
        if len(selected) == budget:
            break

        mean, latent_std, _ = gp_posterior(
            candidate_x[selected],
            candidate_y[selected],
            candidate_x,
            **active_settings,
        )
        next_index = suggest_next_index(
            mean, latent_std, selected, exploration
        )
        step['next_index'] = next_index
        selected.append(next_index)
    return selected, history
